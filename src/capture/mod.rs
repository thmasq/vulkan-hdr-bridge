use crate::dispatch::{DeviceTable, InstanceTable};
use crate::error::{BridgeError, Result};
use ash::vk;
use ash::vk::Handle;
use crossbeam_channel::{Receiver, Sender, bounded};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

pub const SUPPORTED_FORMATS: &[vk::Format] = &[
    vk::Format::R16G16B16A16_SFLOAT,
    vk::Format::R16G16B16A16_UNORM,
    vk::Format::A2B10G10R10_UNORM_PACK32,
];

#[derive(Debug)]
pub struct CapturedFrame {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub raw_bytes: Vec<u8>,
    pub pts_ns: u64,
}

struct CmdData {
    pool: vk::CommandPool,
    buf: vk::CommandBuffer,
    queue_family: u32,
}

struct CaptureSlot {
    fence: vk::Fence,
    present_semaphore: vk::Semaphore,
    cmd_data: Mutex<Option<CmdData>>,
}

struct CaptureShared {
    dt: Arc<DeviceTable>,
    slots: Vec<CaptureSlot>,
    format: vk::Format,
    width: u32,
    height: u32,
    staging_size: vk::DeviceSize,
    staging_buf: vk::Buffer,
    staging_mem: vk::DeviceMemory,
    mapped_ptr: *mut u8,
    out_tx: Sender<CapturedFrame>,
}

unsafe impl Send for CaptureShared {}
unsafe impl Sync for CaptureShared {}

pub struct SwapchainCapture {
    shared: Arc<CaptureShared>,
    work_tx: Sender<(usize, u64)>,
    free_rx: Receiver<usize>,
    worker_handle: Option<JoinHandle<()>>,
}

impl SwapchainCapture {
    pub fn new(
        it: &InstanceTable,
        phys: vk::PhysicalDevice,
        dt: Arc<DeviceTable>,
        format: vk::Format,
        width: u32,
        height: u32,
        out_tx: Sender<CapturedFrame>,
    ) -> Result<Self> {
        validate_format(format)?;

        let staging_size = pixel_stride(format) * (width * height) as vk::DeviceSize;
        let dev = dt.handle;

        let num_slots = 8;
        let total_staging_size = staging_size * num_slots as vk::DeviceSize;

        let staging_buf = unsafe {
            let info = vk::BufferCreateInfo::default()
                .size(total_staging_size)
                .usage(vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let mut buf = vk::Buffer::null();
            (dt.create_buffer)(dev, &info, std::ptr::null(), &mut buf)
                .result()
                .map_err(BridgeError::Vk)?;
            buf
        };

        let staging_mem = unsafe {
            let mut reqs = vk::MemoryRequirements::default();
            (dt.get_buffer_memory_requirements)(dev, staging_buf, &mut reqs);
            let mem_props = it.get_physical_device_memory_properties(phys);
            let mem_type = find_memory_type(
                &mem_props,
                reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::HOST_CACHED,
            )?;
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(reqs.size)
                .memory_type_index(mem_type);
            let mut mem = vk::DeviceMemory::null();
            (dt.allocate_memory)(dev, &alloc_info, std::ptr::null(), &mut mem)
                .result()
                .map_err(BridgeError::Vk)?;
            (dt.bind_buffer_memory)(dev, staging_buf, mem, 0)
                .result()
                .map_err(BridgeError::Vk)?;
            mem
        };

        let mapped_ptr = unsafe {
            let mut ptr: *mut std::os::raw::c_void = std::ptr::null_mut();
            (dt.map_memory)(
                dev,
                staging_mem,
                0,
                total_staging_size,
                vk::MemoryMapFlags::empty(),
                &mut ptr,
            )
            .result()
            .map_err(BridgeError::Vk)?;
            ptr as *mut u8
        };

        let mut slots = Vec::with_capacity(num_slots);

        for _ in 0..num_slots {
            let present_semaphore = unsafe {
                let mut s = vk::Semaphore::null();
                (dt.create_semaphore)(
                    dev,
                    &vk::SemaphoreCreateInfo::default(),
                    std::ptr::null(),
                    &mut s,
                )
                .result()
                .map_err(BridgeError::Vk)?;
                s
            };

            let fence = unsafe {
                let mut f = vk::Fence::null();
                (dt.create_fence)(
                    dev,
                    &vk::FenceCreateInfo::default(),
                    std::ptr::null(),
                    &mut f,
                )
                .result()
                .map_err(BridgeError::Vk)?;
                f
            };

            slots.push(CaptureSlot {
                fence,
                present_semaphore,
                cmd_data: Mutex::new(None),
            });
        }

        let shared = Arc::new(CaptureShared {
            dt,
            slots,
            format,
            width,
            height,
            staging_size,
            staging_buf,
            staging_mem,
            mapped_ptr,
            out_tx,
        });

        let (work_tx, work_rx) = bounded::<(usize, u64)>(num_slots);
        let (free_tx, free_rx) = bounded::<usize>(num_slots);

        for i in 0..num_slots {
            free_tx.send(i).unwrap();
        }

        let shared_clone = Arc::clone(&shared);
        let worker_handle = std::thread::spawn(move || {
            let dt = &*shared_clone.dt;
            let dev = dt.handle;

            for (slot_idx, pts_ns) in work_rx {
                let slot = &shared_clone.slots[slot_idx];

                unsafe {
                    let _ = (dt.wait_for_fences)(dev, 1, &slot.fence, vk::TRUE, 1_000_000_000);
                }

                let offset = (slot_idx as vk::DeviceSize * shared_clone.staging_size) as usize;
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        shared_clone.mapped_ptr.add(offset),
                        shared_clone.staging_size as usize,
                    )
                };

                let raw_bytes = bytes.to_vec();

                if shared_clone
                    .out_tx
                    .try_send(CapturedFrame {
                        width: shared_clone.width,
                        height: shared_clone.height,
                        format: shared_clone.format,
                        raw_bytes,
                        pts_ns,
                    })
                    .is_err()
                {
                    log::warn!("Frame dropped — FFmpeg encoder pipeline is full/falling behind");
                }

                let _ = free_tx.send(slot_idx);
            }
        });

        Ok(Self {
            shared,
            work_tx,
            free_rx,
            worker_handle: Some(worker_handle),
        })
    }

    pub fn capture_image(
        &self,
        queue: vk::Queue,
        queue_family: u32,
        image: vk::Image,
        layout: vk::ImageLayout,
        pts_ns: u64,
        wait_semaphores: &[vk::Semaphore],
    ) -> Result<Option<vk::Semaphore>> {
        let slot_idx = match self.free_rx.try_recv() {
            Ok(idx) => idx,
            Err(_) => {
                log::warn!("Capture dropped — no free slots pts={}", pts_ns);
                return Ok(None);
            }
        };

        let slot = &self.shared.slots[slot_idx];
        let dt = &*self.shared.dt;
        let dev = dt.handle;

        let mut cmd_guard = slot.cmd_data.lock().unwrap();

        if let Some(cmd) = cmd_guard.as_ref() {
            if cmd.queue_family != queue_family {
                unsafe {
                    (dt.free_command_buffers)(dev, cmd.pool, 1, &cmd.buf);
                    (dt.destroy_command_pool)(dev, cmd.pool, std::ptr::null());
                }
                *cmd_guard = None;
            }
        }

        if cmd_guard.is_none() {
            let (pool, buf) = unsafe {
                let pool_info = vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
                let mut pool = vk::CommandPool::null();
                (dt.create_command_pool)(dev, &pool_info, std::ptr::null(), &mut pool)
                    .result()
                    .map_err(BridgeError::Vk)?;

                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let mut cb = vk::CommandBuffer::null();
                (dt.allocate_command_buffers)(dev, &alloc_info, &mut cb)
                    .result()
                    .map_err(BridgeError::Vk)?;

                let dev_key = *(dev.as_raw() as *const usize);
                *(cb.as_raw() as *mut usize) = dev_key;

                (pool, cb)
            };
            *cmd_guard = Some(CmdData {
                pool,
                buf,
                queue_family,
            });
        }

        let cmd_buf = cmd_guard.as_ref().unwrap().buf;

        unsafe {
            (dt.reset_command_buffer)(cmd_buf, vk::CommandBufferResetFlags::empty())
                .result()
                .map_err(BridgeError::Vk)?;

            (dt.reset_fences)(dev, 1, &slot.fence)
                .result()
                .map_err(BridgeError::Vk)?;

            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            (dt.begin_command_buffer)(cmd_buf, &begin)
                .result()
                .map_err(BridgeError::Vk)?;

            let barrier_to_transfer = vk::ImageMemoryBarrier {
                s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                p_next: std::ptr::null(),
                src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                old_layout: layout,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: color_subresource_range(),
                _marker: std::marker::PhantomData,
            };

            (dt.cmd_pipeline_barrier)(
                cmd_buf,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                1,
                &barrier_to_transfer,
            );

            let copy = vk::BufferImageCopy {
                buffer_offset: slot_idx as vk::DeviceSize * self.shared.staging_size,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D {
                    width: self.shared.width,
                    height: self.shared.height,
                    depth: 1,
                },
            };

            (dt.cmd_copy_image_to_buffer)(
                cmd_buf,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.shared.staging_buf,
                1,
                &copy,
            );

            let barrier_to_original = vk::ImageMemoryBarrier {
                s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                p_next: std::ptr::null(),
                src_access_mask: vk::AccessFlags::TRANSFER_READ,
                dst_access_mask: vk::AccessFlags::MEMORY_READ,
                old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                new_layout: layout,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: color_subresource_range(),
                _marker: std::marker::PhantomData,
            };

            (dt.cmd_pipeline_barrier)(
                cmd_buf,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                1,
                &barrier_to_original,
            );

            (dt.end_command_buffer)(cmd_buf)
                .result()
                .map_err(BridgeError::Vk)?;

            let wait_dst_stage_mask =
                vec![vk::PipelineStageFlags::ALL_COMMANDS; wait_semaphores.len()];
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(&wait_dst_stage_mask)
                .command_buffers(std::slice::from_ref(&cmd_buf))
                .signal_semaphores(std::slice::from_ref(&slot.present_semaphore));

            (dt.queue_submit)(queue, 1, &submit, slot.fence)
                .result()
                .map_err(BridgeError::Vk)?;
        }

        let _ = self.work_tx.try_send((slot_idx, pts_ns));

        Ok(Some(slot.present_semaphore))
    }
}

impl Drop for SwapchainCapture {
    fn drop(&mut self) {
        let (empty_tx, _) = bounded(1);
        self.work_tx = empty_tx;

        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }

        let dt = &*self.shared.dt;
        let dev = dt.handle;
        unsafe {
            let _ = (dt.device_wait_idle)(dev);
            for slot in &self.shared.slots {
                (dt.destroy_semaphore)(dev, slot.present_semaphore, std::ptr::null());
                (dt.destroy_fence)(dev, slot.fence, std::ptr::null());
                if let Some(cmd) = slot.cmd_data.lock().unwrap().take() {
                    (dt.free_command_buffers)(dev, cmd.pool, 1, &cmd.buf);
                    (dt.destroy_command_pool)(dev, cmd.pool, std::ptr::null());
                }
            }
            (dt.unmap_memory)(dev, self.shared.staging_mem);
            (dt.free_memory)(dev, self.shared.staging_mem, std::ptr::null());
            (dt.destroy_buffer)(dev, self.shared.staging_buf, std::ptr::null());
        }
    }
}

pub fn validate_format(fmt: vk::Format) -> Result<()> {
    if SUPPORTED_FORMATS.contains(&fmt) {
        log::info!("Capture: format accepted: {:?}", fmt);
        Ok(())
    } else {
        Err(BridgeError::UnsupportedFormat(fmt))
    }
}

fn pixel_stride(fmt: vk::Format) -> vk::DeviceSize {
    match fmt {
        vk::Format::R16G16B16A16_SFLOAT | vk::Format::R16G16B16A16_UNORM => 8,
        vk::Format::A2B10G10R10_UNORM_PACK32 => 4,
        _ => unreachable!(),
    }
}

fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Result<u32> {
    for i in 0..props.memory_type_count {
        if type_bits & (1 << i) != 0
            && props.memory_types[i as usize]
                .property_flags
                .contains(required)
        {
            return Ok(i);
        }
    }
    Err(BridgeError::Ffmpeg("no suitable memory type".into()))
}
