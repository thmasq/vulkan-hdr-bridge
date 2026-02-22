#![allow(dead_code)]

mod capture;
pub mod dispatch;
mod error;
mod ffmpeg;
mod timing;

use ash::vk;
use ash::vk::Handle;
use crossbeam_channel::{RecvTimeoutError, Sender, bounded};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::ffi::CStr;
use std::io::Read;
use std::os::raw::{c_char, c_void};
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use capture::{CapturedFrame, SwapchainCapture};
use dispatch::{DeviceTable, InstanceTable};

// ── Global state ─────────────────────────────────────────────────────────────

static IS_RECORDING: Lazy<Arc<AtomicBool>> = Lazy::new(|| Arc::new(AtomicBool::new(false)));

static INSTANCES: Lazy<Mutex<HashMap<usize, InstanceState>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

static DEVICES: Lazy<Mutex<HashMap<usize, Arc<Mutex<DeviceState>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// ── Loader ABI structures ────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "system" fn hdr_bridge_EnumerateInstanceExtensionProperties(
    _p_layer_name: *const c_char,
    p_property_count: *mut u32,
    _p_properties: *mut vk::ExtensionProperties,
) -> vk::Result {
    if !p_property_count.is_null() {
        unsafe {
            *p_property_count = 0;
        }
    }
    vk::Result::SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn hdr_bridge_EnumerateInstanceLayerProperties(
    p_property_count: *mut u32,
    p_properties: *mut vk::LayerProperties,
) -> vk::Result {
    unsafe {
        if p_properties.is_null() {
            if !p_property_count.is_null() {
                *p_property_count = 1;
            }
            return vk::Result::SUCCESS;
        }

        if p_property_count.is_null() {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }

        if *p_property_count == 0 {
            return vk::Result::INCOMPLETE;
        }

        *p_property_count = 1;

        let mut props = vk::LayerProperties::default();

        let name = b"VK_LAYER_HDR_BRIDGE_capture\0";
        for (i, &b) in name.iter().enumerate() {
            props.layer_name[i] = b as std::os::raw::c_char;
        }

        let desc = b"HDR capture bridge layer\0";
        for (i, &b) in desc.iter().enumerate() {
            props.description[i] = b as std::os::raw::c_char;
        }

        props.spec_version = vk::API_VERSION_1_3;
        props.implementation_version = 1;

        *p_properties = props;
    }

    vk::Result::SUCCESS
}

const VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO: vk::StructureType =
    vk::StructureType::from_raw(47);

const VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO: vk::StructureType =
    vk::StructureType::from_raw(48);

const VK_LAYER_LINK_INFO: u32 = 0;

#[repr(C)]
struct VkLayerInstanceLink {
    p_next: *mut VkLayerInstanceLink,
    pfn_next_get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    pfn_next_get_phys_dev_proc_addr: Option<unsafe extern "system" fn()>,
}

#[repr(C)]
struct VkLayerInstanceCreateInfo {
    s_type: vk::StructureType,
    p_next: *const c_void,
    function: u32,
    u_layer_info: *mut VkLayerInstanceLink,
}

#[repr(C)]
struct VkLayerDeviceLink {
    p_next: *mut VkLayerDeviceLink,
    pfn_next_get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    pfn_next_get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
}

#[repr(C)]
struct VkLayerDeviceCreateInfo {
    s_type: vk::StructureType,
    p_next: *const c_void,
    function: u32,
    u_layer_info: *mut VkLayerDeviceLink,
}

struct InstanceState {
    next_gipa: vk::PFN_vkGetInstanceProcAddr,
    it: Arc<InstanceTable>,
}

struct SwapchainInfo {
    images: Vec<vk::Image>,
    format: vk::Format,
    width: u32,
    height: u32,
}

struct DeviceState {
    next_gdpa: vk::PFN_vkGetDeviceProcAddr,
    dt: Arc<DeviceTable>,
    it: Arc<InstanceTable>,
    phys: vk::PhysicalDevice,
    queue_families: HashMap<u64, u32>,
    swapchains: HashMap<u64, SwapchainInfo>,
    captures: HashMap<u64, SwapchainCapture>,
    frame_tx: Sender<CapturedFrame>,
    last_capture_pts: u64,
}

unsafe fn dispatch_key(handle: u64) -> usize {
    unsafe { *(handle as *const usize) }
}

// ── Loader negotiate ─────────────────────────────────────────────────────────

#[repr(C)]
pub struct VkNegotiateLayerInterface {
    s_type: u32,
    p_next: *mut c_void,
    loader_layer_interface_version: u32,
    pfn_get_instance_proc_addr: *const c_void,
    pfn_get_device_proc_addr: *const c_void,
    pfn_get_physical_device_proc_addr: *const c_void,
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkNegotiateLoaderLayerInterfaceVersion(
    p: *mut VkNegotiateLayerInterface,
) -> vk::Result {
    let _ = env_logger::try_init();

    if p.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    unsafe {
        (*p).loader_layer_interface_version = 2;
        (*p).pfn_get_instance_proc_addr = hdr_bridge_GetInstanceProcAddr as _;
        (*p).pfn_get_device_proc_addr = hdr_bridge_GetDeviceProcAddr as _;
        (*p).pfn_get_physical_device_proc_addr = std::ptr::null();
    }
    vk::Result::SUCCESS
}

// ── Proc-addr routers ────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "system" fn hdr_bridge_GetInstanceProcAddr(
    instance: vk::Instance,
    p_name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    let name = unsafe { CStr::from_ptr(p_name).to_bytes() };
    match name {
        b"vkGetInstanceProcAddr" => {
            return unsafe {
                std::mem::transmute(hdr_bridge_GetInstanceProcAddr as *const () as usize)
            };
        }
        b"vkCreateInstance" => {
            return unsafe { std::mem::transmute(hdr_bridge_CreateInstance as *const () as usize) };
        }
        b"vkDestroyInstance" => {
            return unsafe {
                std::mem::transmute(hdr_bridge_DestroyInstance as *const () as usize)
            };
        }
        b"vkCreateDevice" => {
            return unsafe { std::mem::transmute(hdr_bridge_CreateDevice as *const () as usize) };
        }
        b"vkGetDeviceProcAddr" => {
            return unsafe {
                std::mem::transmute(hdr_bridge_GetDeviceProcAddr as *const () as usize)
            };
        }
        b"vkEnumerateInstanceExtensionProperties" => {
            return unsafe {
                std::mem::transmute(
                    hdr_bridge_EnumerateInstanceExtensionProperties as *const () as usize,
                )
            };
        }
        b"vkEnumerateInstanceLayerProperties" => {
            return unsafe {
                std::mem::transmute(
                    hdr_bridge_EnumerateInstanceLayerProperties as *const () as usize,
                )
            };
        }
        _ => {}
    }
    if instance == vk::Instance::null() {
        return unsafe { std::mem::transmute(0usize) };
    }
    let key = unsafe { dispatch_key(instance.as_raw()) };
    if let Some(state) = INSTANCES.lock().unwrap().get(&key) {
        unsafe { (state.next_gipa)(instance, p_name) }
    } else {
        unsafe { std::mem::transmute(0usize) }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn hdr_bridge_GetDeviceProcAddr(
    device: vk::Device,
    p_name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    let name = unsafe { CStr::from_ptr(p_name).to_bytes() };
    match name {
        b"vkGetDeviceProcAddr" => {
            return unsafe {
                std::mem::transmute(hdr_bridge_GetDeviceProcAddr as *const () as usize)
            };
        }
        b"vkCreateSwapchainKHR" => {
            return unsafe {
                std::mem::transmute(hdr_bridge_CreateSwapchainKHR as *const () as usize)
            };
        }
        b"vkDestroySwapchainKHR" => {
            return unsafe {
                std::mem::transmute(hdr_bridge_DestroySwapchainKHR as *const () as usize)
            };
        }
        b"vkQueuePresentKHR" => {
            return unsafe {
                std::mem::transmute(hdr_bridge_QueuePresentKHR as *const () as usize)
            };
        }
        b"vkGetDeviceQueue" => {
            return unsafe { std::mem::transmute(hdr_bridge_GetDeviceQueue as *const () as usize) };
        }
        b"vkGetDeviceQueue2" => {
            return unsafe {
                std::mem::transmute(hdr_bridge_GetDeviceQueue2 as *const () as usize)
            };
        }
        _ => {}
    }
    let key = unsafe { dispatch_key(device.as_raw()) };
    if let Some(arc) = DEVICES.lock().unwrap().get(&key).cloned() {
        let state = arc.lock().unwrap();
        unsafe { (state.next_gdpa)(device, p_name) }
    } else {
        unsafe { std::mem::transmute(0usize) }
    }
}

// ── vkCreateInstance ─────────────────────────────────────────────────────────

type PfnCreateInstance = unsafe extern "system" fn(
    *const vk::InstanceCreateInfo,
    *const vk::AllocationCallbacks,
    *mut vk::Instance,
) -> vk::Result;

#[unsafe(no_mangle)]
pub unsafe extern "system" fn hdr_bridge_CreateInstance(
    p_create_info: *const vk::InstanceCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_instance: *mut vk::Instance,
) -> vk::Result {
    let link = match unsafe { find_instance_link((*p_create_info).p_next as _) } {
        Some(l) => l,
        None => {
            log::error!("hdr_bridge_CreateInstance: no layer link found in pNext chain");
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
    };

    let next_gipa = unsafe { (*link).pfn_next_get_instance_proc_addr };

    unsafe { advance_instance_link((*p_create_info).p_next as _) };

    let create_fn: PfnCreateInstance = unsafe {
        std::mem::transmute(next_gipa(
            vk::Instance::null(),
            b"vkCreateInstance\0".as_ptr() as _,
        ))
    };
    let result = unsafe { create_fn(p_create_info, p_allocator, p_instance) };
    if result != vk::Result::SUCCESS {
        return result;
    }

    let instance = unsafe { *p_instance };
    let key = unsafe { dispatch_key(instance.as_raw()) };

    let it = Arc::new(unsafe { InstanceTable::load(instance, next_gipa) });

    INSTANCES
        .lock()
        .unwrap()
        .insert(key, InstanceState { next_gipa, it });

    log::info!("HDR Bridge: vkCreateInstance hooked (key={:#x})", key);
    vk::Result::SUCCESS
}

// ── vkDestroyInstance ────────────────────────────────────────────────────────

type PfnDestroyInstance = unsafe extern "system" fn(vk::Instance, *const vk::AllocationCallbacks);

#[unsafe(no_mangle)]
pub unsafe extern "system" fn hdr_bridge_DestroyInstance(
    instance: vk::Instance,
    p_allocator: *const vk::AllocationCallbacks,
) {
    let key = unsafe { dispatch_key(instance.as_raw()) };
    let next_gipa = INSTANCES.lock().unwrap().remove(&key).map(|s| s.next_gipa);

    if let Some(gipa) = next_gipa {
        let destroy_fn: PfnDestroyInstance =
            unsafe { std::mem::transmute(gipa(instance, b"vkDestroyInstance\0".as_ptr() as _)) };
        unsafe { destroy_fn(instance, p_allocator) };
    }
}

// ── vkCreateDevice ───────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "system" fn hdr_bridge_GetDeviceQueue(
    device: vk::Device,
    queue_family_index: u32,
    queue_index: u32,
    p_queue: *mut vk::Queue,
) {
    let key = unsafe { dispatch_key(device.as_raw()) };
    let state_arc = DEVICES.lock().unwrap().get(&key).cloned().unwrap();
    let next_gdpa = state_arc.lock().unwrap().next_gdpa;

    let real_fn: unsafe extern "system" fn(vk::Device, u32, u32, *mut vk::Queue) =
        unsafe { std::mem::transmute(next_gdpa(device, b"vkGetDeviceQueue\0".as_ptr() as _)) };
    unsafe { real_fn(device, queue_family_index, queue_index, p_queue) };

    let queue = unsafe { *p_queue };
    if queue != vk::Queue::null() {
        state_arc
            .lock()
            .unwrap()
            .queue_families
            .insert(queue.as_raw(), queue_family_index);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn hdr_bridge_GetDeviceQueue2(
    device: vk::Device,
    p_queue_info: *const vk::DeviceQueueInfo2,
    p_queue: *mut vk::Queue,
) {
    let key = unsafe { dispatch_key(device.as_raw()) };
    let state_arc = DEVICES.lock().unwrap().get(&key).cloned().unwrap();
    let next_gdpa = state_arc.lock().unwrap().next_gdpa;

    let real_fn: unsafe extern "system" fn(
        vk::Device,
        *const vk::DeviceQueueInfo2,
        *mut vk::Queue,
    ) = unsafe { std::mem::transmute(next_gdpa(device, b"vkGetDeviceQueue2\0".as_ptr() as _)) };
    unsafe { real_fn(device, p_queue_info, p_queue) };

    let queue = unsafe { *p_queue };
    if queue != vk::Queue::null() {
        let info = unsafe { &*p_queue_info };
        state_arc
            .lock()
            .unwrap()
            .queue_families
            .insert(queue.as_raw(), info.queue_family_index);
    }
}

type PfnCreateDevice = unsafe extern "system" fn(
    vk::PhysicalDevice,
    *const vk::DeviceCreateInfo,
    *const vk::AllocationCallbacks,
    *mut vk::Device,
) -> vk::Result;

#[unsafe(no_mangle)]
pub unsafe extern "system" fn hdr_bridge_CreateDevice(
    phys_device: vk::PhysicalDevice,
    p_create_info: *const vk::DeviceCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_device: *mut vk::Device,
) -> vk::Result {
    let link = match unsafe { find_device_link((*p_create_info).p_next as _) } {
        Some(l) => l,
        None => {
            log::error!("hdr_bridge_CreateDevice: no layer link found");
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
    };

    let next_gipa = unsafe { (*link).pfn_next_get_instance_proc_addr };
    let next_gdpa = unsafe { (*link).pfn_next_get_device_proc_addr };

    unsafe { advance_device_link((*p_create_info).p_next as _) };

    let create_fn: PfnCreateDevice = unsafe {
        std::mem::transmute(next_gipa(
            vk::Instance::null(),
            b"vkCreateDevice\0".as_ptr() as _,
        ))
    };
    let result = unsafe { create_fn(phys_device, p_create_info, p_allocator, p_device) };
    if result != vk::Result::SUCCESS {
        return result;
    }

    let device = unsafe { *p_device };
    let key = unsafe { dispatch_key(device.as_raw()) };

    let dt = Arc::new(unsafe { DeviceTable::load(device, next_gdpa) });

    let inst_key = unsafe { dispatch_key(phys_device.as_raw()) };
    let it = {
        let instances = INSTANCES.lock().unwrap();
        instances
            .values()
            .find(|_| true)
            .map(|s| s.it.clone())
            .unwrap_or_else(|| {
                Arc::new(unsafe { InstanceTable::load(vk::Instance::null(), next_gipa) })
            })
    };

    let (frame_tx, frame_rx) = bounded::<CapturedFrame>(120);
    let (control_tx, control_rx) = bounded::<()>(4);

    let output_base =
        std::env::var("HDR_BRIDGE_OUTPUT").unwrap_or_else(|_| "capture.mkv".to_string());

    let recording_state = Arc::clone(&IS_RECORDING);
    let control_tx_clone = control_tx.clone();
    std::thread::spawn(move || {
        let socket_path = "/run/user/1000/hdr_capture.sock";
        let _ = std::fs::remove_file(socket_path);

        if let Ok(listener) = UnixListener::bind(socket_path) {
            log::info!("HDR Bridge: IPC listening on {}", socket_path);
            for stream in listener.incoming() {
                if let Ok(mut stream) = stream {
                    let mut buf = String::new();
                    if stream.read_to_string(&mut buf).is_ok() {
                        match buf.trim() {
                            "START" => {
                                recording_state.store(true, Ordering::SeqCst);
                                log::info!("HDR Bridge: Recording STARTED");
                            }
                            "PAUSE" => {
                                recording_state.store(false, Ordering::SeqCst);
                                log::info!("HDR Bridge: Recording PAUSED");
                            }
                            "STOP" => {
                                recording_state.store(false, Ordering::SeqCst);
                                let _ = control_tx_clone.send(());
                                log::info!("HDR Bridge: Recording STOPPED");
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    });

    std::thread::spawn(move || {
        use timing::TimingController;
        let mut timer = TimingController::new();
        let mut sink: Option<ffmpeg::FfmpegSink> = None;
        let mut frames_processed = 0;

        loop {
            match frame_rx.recv_timeout(std::time::Duration::from_millis(50)) {
                Ok(frame) => {
                    frames_processed += 1;

                    let max_delay = if frames_processed < 300 {
                        5_000_000_000
                    } else {
                        33_333_334
                    };

                    let timing = timer.next(frame.pts_ns, max_delay);
                    if timing.drop {
                        log::warn!("Timing controller dropped frame");
                        continue;
                    }
                    let s = sink.get_or_insert_with(|| {
                        let next_path = get_next_filename(&output_base);
                        log::info!("HDR Bridge: Spawning FFmpeg sink to {}", next_path);
                        ffmpeg::FfmpegSink::spawn(
                            &next_path,
                            frame.width,
                            frame.height,
                            60000,
                            1001,
                            frame.format,
                        )
                        .expect("FFmpeg spawn failed")
                    });
                    let _ = s.write_raw(&frame.raw_bytes);
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    break;
                }
            }

            while let Ok(_) = control_rx.try_recv() {
                if let Some(s) = sink.take() {
                    let _ = s.finish();
                    log::info!("HDR Bridge: File finalized and saved.");
                    timer = TimingController::new();
                    frames_processed = 0;
                }
            }
        }

        if let Some(s) = sink {
            let _ = s.finish();
        }
    });

    let state = Arc::new(Mutex::new(DeviceState {
        next_gdpa,
        dt,
        it,
        phys: phys_device,
        queue_families: HashMap::new(),
        swapchains: HashMap::new(),
        captures: HashMap::new(),
        frame_tx,
        last_capture_pts: 0,
    }));

    DEVICES.lock().unwrap().insert(key, state);

    log::info!("HDR Bridge: vkCreateDevice hooked (key={:#x})", key);
    vk::Result::SUCCESS
}

// ── vkDestroyDevice ──────────────────────────────────────────────────────────

type PfnDestroyDevice = unsafe extern "system" fn(vk::Device, *const vk::AllocationCallbacks);

#[unsafe(no_mangle)]
pub unsafe extern "system" fn hdr_bridge_DestroyDevice(
    device: vk::Device,
    p_allocator: *const vk::AllocationCallbacks,
) {
    let key = unsafe { dispatch_key(device.as_raw()) };
    let next_gdpa = {
        let mut devs = DEVICES.lock().unwrap();
        devs.remove(&key).map(|arc| arc.lock().unwrap().next_gdpa)
    };
    if let Some(gdpa) = next_gdpa {
        let destroy_fn: PfnDestroyDevice =
            unsafe { std::mem::transmute(gdpa(device, b"vkDestroyDevice\0".as_ptr() as _)) };
        unsafe { destroy_fn(device, p_allocator) };
    }
}

// ── vkCreateSwapchainKHR ─────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "system" fn hdr_bridge_CreateSwapchainKHR(
    device: vk::Device,
    p_create_info: *const vk::SwapchainCreateInfoKHR,
    p_allocator: *const vk::AllocationCallbacks,
    p_swapchain: *mut vk::SwapchainKHR,
) -> vk::Result {
    let dev_key = unsafe { dispatch_key(device.as_raw()) };
    let arc = match DEVICES.lock().unwrap().get(&dev_key).cloned() {
        Some(a) => a,
        None => return vk::Result::ERROR_INITIALIZATION_FAILED,
    };

    let mut modified_info = unsafe { *p_create_info };
    modified_info.image_usage |= vk::ImageUsageFlags::TRANSFER_SRC;

    let result = {
        let state = arc.lock().unwrap();
        unsafe { (state.dt.create_swapchain_khr)(device, &modified_info, p_allocator, p_swapchain) }
    };
    if result != vk::Result::SUCCESS {
        return result;
    }

    let swapchain = unsafe { *p_swapchain };
    let sc_key = swapchain.as_raw();

    let info = &modified_info;

    if let Err(e) = capture::validate_format(info.image_format) {
        log::warn!("HDR Bridge: Capture disabled for this swapchain — {}", e);
        return vk::Result::SUCCESS;
    }

    let width = info.image_extent.width;
    let height = info.image_extent.height;
    let format = info.image_format;

    let images = {
        let state = arc.lock().unwrap();
        unsafe { get_swapchain_images(&state.dt, device, swapchain) }
    };

    let images = match images {
        Ok(v) => v,
        Err(e) => {
            log::error!("HDR Bridge: vkGetSwapchainImagesKHR failed: {:?}", e);
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
    };

    log::info!(
        "HDR Bridge: swapchain {:#x} created — {}×{} {:?} ({} images)",
        sc_key,
        width,
        height,
        format,
        images.len()
    );

    let mut state = arc.lock().unwrap();
    let capture = SwapchainCapture::new(
        &state.it,
        state.phys,
        state.dt.clone(),
        format,
        width,
        height,
        state.frame_tx.clone(),
    );

    match capture {
        Ok(cap) => {
            state.swapchains.insert(
                sc_key,
                SwapchainInfo {
                    images,
                    format,
                    width,
                    height,
                },
            );
            state.captures.insert(sc_key, cap);
        }
        Err(e) => {
            log::error!("HDR Bridge: SwapchainCapture::new failed: {}", e);
        }
    }

    vk::Result::SUCCESS
}

// ── vkDestroySwapchainKHR ────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "system" fn hdr_bridge_DestroySwapchainKHR(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    p_allocator: *const vk::AllocationCallbacks,
) {
    let dev_key = unsafe { dispatch_key(device.as_raw()) };
    let sc_key = swapchain.as_raw();

    if let Some(arc) = DEVICES.lock().unwrap().get(&dev_key).cloned() {
        let mut state = arc.lock().unwrap();
        state.captures.remove(&sc_key);
        state.swapchains.remove(&sc_key);

        log::info!("HDR Bridge: swapchain {:#x} destroyed", sc_key);

        unsafe { (state.dt.destroy_swapchain_khr)(device, swapchain, p_allocator) };
    }
}

// ── vkQueuePresentKHR ────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "system" fn hdr_bridge_QueuePresentKHR(
    queue: vk::Queue,
    p_present: *const vk::PresentInfoKHR,
) -> vk::Result {
    let queue_key = unsafe { dispatch_key(queue.as_raw()) };
    let pts_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    if let Some(arc) = DEVICES.lock().unwrap().get(&queue_key).cloned() {
        let present = unsafe { &*p_present };
        let mut next_present = *present;
        let mut new_wait_semaphores = Vec::new();

        if let Ok(mut state) = arc.try_lock() {
            let swapchains = unsafe {
                std::slice::from_raw_parts(present.p_swapchains, present.swapchain_count as usize)
            };
            let indices = unsafe {
                std::slice::from_raw_parts(
                    present.p_image_indices,
                    present.swapchain_count as usize,
                )
            };
            let app_wait_semaphores = unsafe {
                std::slice::from_raw_parts(
                    present.p_wait_semaphores,
                    present.wait_semaphore_count as usize,
                )
            };

            let q_family = state
                .queue_families
                .get(&queue.as_raw())
                .copied()
                .unwrap_or(0);

            let mut remaining_waits = app_wait_semaphores;
            let mut replaced_semaphores = false;

            if IS_RECORDING.load(Ordering::SeqCst) {
                let target_interval_ns = 1_000_000_000 / 60;

                if pts_ns.saturating_sub(state.last_capture_pts) >= target_interval_ns {
                    state.last_capture_pts = pts_ns;

                    for (&sc, &img_idx) in swapchains.iter().zip(indices.iter()) {
                        let sc_key = sc.as_raw();
                        let image = state
                            .swapchains
                            .get(&sc_key)
                            .and_then(|info| info.images.get(img_idx as usize).copied());

                        if let (Some(image), Some(cap)) = (image, state.captures.get(&sc_key)) {
                            match cap.capture_image(
                                queue,
                                q_family,
                                image,
                                vk::ImageLayout::PRESENT_SRC_KHR,
                                pts_ns,
                                remaining_waits,
                            ) {
                                Ok(Some(sem)) => {
                                    new_wait_semaphores.push(sem);
                                    remaining_waits = &[];
                                    replaced_semaphores = true;
                                }
                                Ok(None) => {}
                                Err(e) => log::warn!("HDR Bridge: capture error: {}", e),
                            }
                        }
                    }
                }
            }

            if replaced_semaphores {
                next_present.p_wait_semaphores = new_wait_semaphores.as_ptr();
                next_present.wait_semaphore_count = new_wait_semaphores.len() as u32;
            }
        }

        let dt = arc.lock().unwrap().dt.clone();
        return unsafe { (dt.queue_present_khr)(queue, &next_present) };
    }

    log::error!(
        "hdr_bridge_QueuePresentKHR: no device state for queue key {:#x}",
        queue_key
    );
    vk::Result::ERROR_DEVICE_LOST
}

// ── Chain traversal helpers ───────────────────────────────────────────────────

unsafe fn find_instance_link(mut p: *const c_void) -> Option<*mut VkLayerInstanceLink> {
    while !p.is_null() {
        let base = unsafe { &*(p as *const vk::BaseInStructure) };
        if base.s_type == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO {
            let info = unsafe { &*(p as *const VkLayerInstanceCreateInfo) };
            if info.function == VK_LAYER_LINK_INFO && !info.u_layer_info.is_null() {
                return Some(info.u_layer_info);
            }
        }
        p = base.p_next as _;
    }
    None
}

unsafe fn find_device_link(mut p: *const c_void) -> Option<*mut VkLayerDeviceLink> {
    while !p.is_null() {
        let base = unsafe { &*(p as *const vk::BaseInStructure) };
        if base.s_type == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO {
            let info = unsafe { &*(p as *const VkLayerDeviceCreateInfo) };
            if info.function == VK_LAYER_LINK_INFO && !info.u_layer_info.is_null() {
                return Some(info.u_layer_info);
            }
        }
        p = base.p_next as _;
    }
    None
}

unsafe fn advance_instance_link(mut p: *mut c_void) {
    while !p.is_null() {
        let base = unsafe { &mut *(p as *mut vk::BaseOutStructure) };
        if base.s_type == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO {
            let info = unsafe { &mut *(p as *mut VkLayerInstanceCreateInfo) };
            if info.function == VK_LAYER_LINK_INFO {
                info.u_layer_info = unsafe { (*info.u_layer_info).p_next };
                return;
            }
        }
        p = base.p_next as _;
    }
}

unsafe fn advance_device_link(mut p: *mut c_void) {
    while !p.is_null() {
        let base = unsafe { &mut *(p as *mut vk::BaseOutStructure) };
        if base.s_type == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO {
            let info = unsafe { &mut *(p as *mut VkLayerDeviceCreateInfo) };
            if info.function == VK_LAYER_LINK_INFO {
                info.u_layer_info = unsafe { (*info.u_layer_info).p_next };
                return;
            }
        }
        p = base.p_next as _;
    }
}

// ── vkGetSwapchainImagesKHR wrapper ──────────────────────────────────────────

unsafe fn get_swapchain_images(
    dt: &DeviceTable,
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
) -> Result<Vec<vk::Image>, vk::Result> {
    let mut count = 0u32;
    (unsafe {
        (dt.get_swapchain_images_khr)(device, swapchain, &mut count, std::ptr::null_mut()).result()
    })?;
    let mut images = vec![vk::Image::null(); count as usize];
    (unsafe {
        (dt.get_swapchain_images_khr)(device, swapchain, &mut count, images.as_mut_ptr()).result()
    })?;
    Ok(images)
}

// ── File Naming Utility ──────────────────────────────────────────────────────

fn get_next_filename(base: &str) -> String {
    let path = std::path::Path::new(base);
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("capture");
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("mkv");
    let dir = path.parent().unwrap_or(std::path::Path::new(""));

    let mut counter = 1;
    loop {
        let name = format!("{}_{}.{}", stem, counter, ext);
        let full_path = if dir.as_os_str().is_empty() {
            PathBuf::from(&name)
        } else {
            dir.join(&name)
        };

        if !full_path.exists() {
            return full_path.to_string_lossy().into_owned();
        }
        counter += 1;
    }
}
