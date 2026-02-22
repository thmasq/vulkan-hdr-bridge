use ash::vk;
use std::os::raw::c_void;

// ── Raw function pointer types ───────────────────────────────────────────────

pub type PfnCreateBuffer = unsafe extern "system" fn(
    vk::Device,
    *const vk::BufferCreateInfo,
    *const vk::AllocationCallbacks,
    *mut vk::Buffer,
) -> vk::Result;

pub type PfnGetBufferMemoryRequirements =
    unsafe extern "system" fn(vk::Device, vk::Buffer, *mut vk::MemoryRequirements);

pub type PfnAllocateMemory = unsafe extern "system" fn(
    vk::Device,
    *const vk::MemoryAllocateInfo,
    *const vk::AllocationCallbacks,
    *mut vk::DeviceMemory,
) -> vk::Result;

pub type PfnBindBufferMemory = unsafe extern "system" fn(
    vk::Device,
    vk::Buffer,
    vk::DeviceMemory,
    vk::DeviceSize,
) -> vk::Result;

pub type PfnMapMemory = unsafe extern "system" fn(
    vk::Device,
    vk::DeviceMemory,
    vk::DeviceSize,
    vk::DeviceSize,
    vk::MemoryMapFlags,
    *mut *mut c_void,
) -> vk::Result;

pub type PfnUnmapMemory = unsafe extern "system" fn(vk::Device, vk::DeviceMemory);

pub type PfnFreeMemory =
    unsafe extern "system" fn(vk::Device, vk::DeviceMemory, *const vk::AllocationCallbacks);

pub type PfnDestroyBuffer =
    unsafe extern "system" fn(vk::Device, vk::Buffer, *const vk::AllocationCallbacks);

pub type PfnCreateCommandPool = unsafe extern "system" fn(
    vk::Device,
    *const vk::CommandPoolCreateInfo,
    *const vk::AllocationCallbacks,
    *mut vk::CommandPool,
) -> vk::Result;

pub type PfnDestroyCommandPool =
    unsafe extern "system" fn(vk::Device, vk::CommandPool, *const vk::AllocationCallbacks);

pub type PfnAllocateCommandBuffers = unsafe extern "system" fn(
    vk::Device,
    *const vk::CommandBufferAllocateInfo,
    *mut vk::CommandBuffer,
) -> vk::Result;

pub type PfnFreeCommandBuffers =
    unsafe extern "system" fn(vk::Device, vk::CommandPool, u32, *const vk::CommandBuffer);

pub type PfnResetCommandBuffer =
    unsafe extern "system" fn(vk::CommandBuffer, vk::CommandBufferResetFlags) -> vk::Result;

pub type PfnBeginCommandBuffer =
    unsafe extern "system" fn(vk::CommandBuffer, *const vk::CommandBufferBeginInfo) -> vk::Result;

pub type PfnEndCommandBuffer = unsafe extern "system" fn(vk::CommandBuffer) -> vk::Result;

pub type PfnCmdPipelineBarrier = unsafe extern "system" fn(
    vk::CommandBuffer,
    vk::PipelineStageFlags,
    vk::PipelineStageFlags,
    vk::DependencyFlags,
    u32,
    *const vk::MemoryBarrier,
    u32,
    *const vk::BufferMemoryBarrier,
    u32,
    *const vk::ImageMemoryBarrier,
);

pub type PfnCmdCopyImageToBuffer = unsafe extern "system" fn(
    vk::CommandBuffer,
    vk::Image,
    vk::ImageLayout,
    vk::Buffer,
    u32,
    *const vk::BufferImageCopy,
);

pub type PfnCreateFence = unsafe extern "system" fn(
    vk::Device,
    *const vk::FenceCreateInfo,
    *const vk::AllocationCallbacks,
    *mut vk::Fence,
) -> vk::Result;

pub type PfnDestroyFence =
    unsafe extern "system" fn(vk::Device, vk::Fence, *const vk::AllocationCallbacks);

pub type PfnResetFences =
    unsafe extern "system" fn(vk::Device, u32, *const vk::Fence) -> vk::Result;

pub type PfnWaitForFences =
    unsafe extern "system" fn(vk::Device, u32, *const vk::Fence, vk::Bool32, u64) -> vk::Result;

pub type PfnQueueSubmit =
    unsafe extern "system" fn(vk::Queue, u32, *const vk::SubmitInfo, vk::Fence) -> vk::Result;

pub type PfnDeviceWaitIdle = unsafe extern "system" fn(vk::Device) -> vk::Result;

pub type PfnGetPhysDevMemProps =
    unsafe extern "system" fn(vk::PhysicalDevice, *mut vk::PhysicalDeviceMemoryProperties);

pub type PfnGetPhysDevQueueFamilyProps =
    unsafe extern "system" fn(vk::PhysicalDevice, *mut u32, *mut vk::QueueFamilyProperties);

pub type PfnCreateSwapchainKHR = unsafe extern "system" fn(
    vk::Device,
    *const vk::SwapchainCreateInfoKHR,
    *const vk::AllocationCallbacks,
    *mut vk::SwapchainKHR,
) -> vk::Result;

pub type PfnDestroySwapchainKHR =
    unsafe extern "system" fn(vk::Device, vk::SwapchainKHR, *const vk::AllocationCallbacks);

pub type PfnGetSwapchainImagesKHR =
    unsafe extern "system" fn(vk::Device, vk::SwapchainKHR, *mut u32, *mut vk::Image) -> vk::Result;

pub type PfnQueuePresentKHR =
    unsafe extern "system" fn(vk::Queue, *const vk::PresentInfoKHR) -> vk::Result;

pub type PfnInvalidateMappedMemoryRanges =
    unsafe extern "system" fn(vk::Device, u32, *const vk::MappedMemoryRange) -> vk::Result;

// ── Loader helper ─────────────────────────────────────────────────────────────

pub unsafe fn load_device<F: Copy>(
    gdpa: vk::PFN_vkGetDeviceProcAddr,
    device: vk::Device,
    name: &[u8],
) -> Option<F> {
    let raw: *const c_void = unsafe { std::mem::transmute(gdpa(device, name.as_ptr() as _)) };
    (!raw.is_null()).then(|| unsafe { std::mem::transmute_copy(&raw) })
}

pub unsafe fn load_instance<F: Copy>(
    gipa: vk::PFN_vkGetInstanceProcAddr,
    instance: vk::Instance,
    name: &[u8],
) -> Option<F> {
    let raw: *const c_void = unsafe { std::mem::transmute(gipa(instance, name.as_ptr() as _)) };
    (!raw.is_null()).then(|| unsafe { std::mem::transmute_copy(&raw) })
}

pub type PfnCreateSemaphore = unsafe extern "system" fn(
    vk::Device,
    *const vk::SemaphoreCreateInfo,
    *const vk::AllocationCallbacks,
    *mut vk::Semaphore,
) -> vk::Result;

pub type PfnDestroySemaphore =
    unsafe extern "system" fn(vk::Device, vk::Semaphore, *const vk::AllocationCallbacks);

// ── Tables ───────────────────────────────────────────────────────────────────

pub struct DeviceTable {
    pub handle: vk::Device,
    // Core 1.0
    pub create_buffer: PfnCreateBuffer,
    pub get_buffer_memory_requirements: PfnGetBufferMemoryRequirements,
    pub allocate_memory: PfnAllocateMemory,
    pub bind_buffer_memory: PfnBindBufferMemory,
    pub map_memory: PfnMapMemory,
    pub unmap_memory: PfnUnmapMemory,
    pub invalidate_mapped_memory_ranges: PfnInvalidateMappedMemoryRanges,
    pub free_memory: PfnFreeMemory,
    pub destroy_buffer: PfnDestroyBuffer,
    pub create_command_pool: PfnCreateCommandPool,
    pub destroy_command_pool: PfnDestroyCommandPool,
    pub allocate_command_buffers: PfnAllocateCommandBuffers,
    pub free_command_buffers: PfnFreeCommandBuffers,
    pub reset_command_buffer: PfnResetCommandBuffer,
    pub begin_command_buffer: PfnBeginCommandBuffer,
    pub end_command_buffer: PfnEndCommandBuffer,
    pub cmd_pipeline_barrier: PfnCmdPipelineBarrier,
    pub cmd_copy_image_to_buffer: PfnCmdCopyImageToBuffer,
    pub create_fence: PfnCreateFence,
    pub destroy_fence: PfnDestroyFence,
    pub reset_fences: PfnResetFences,
    pub wait_for_fences: PfnWaitForFences,
    pub queue_submit: PfnQueueSubmit,
    pub device_wait_idle: PfnDeviceWaitIdle,
    pub create_semaphore: PfnCreateSemaphore,
    pub destroy_semaphore: PfnDestroySemaphore,
    // Extension
    pub create_swapchain_khr: PfnCreateSwapchainKHR,
    pub destroy_swapchain_khr: PfnDestroySwapchainKHR,
    pub get_swapchain_images_khr: PfnGetSwapchainImagesKHR,
    pub queue_present_khr: PfnQueuePresentKHR,
}

macro_rules! req {
    ($opt:expr, $name:literal) => {
        $opt.expect(concat!("required Vulkan function missing: ", $name))
    };
}

impl DeviceTable {
    pub unsafe fn load(device: vk::Device, gdpa: vk::PFN_vkGetDeviceProcAddr) -> Self {
        macro_rules! ld {
            ($name:literal, $ty:ty) => {
                req!(
                    unsafe { load_device::<$ty>(gdpa, device, concat!($name, "\0").as_bytes()) },
                    $name
                )
            };
        }
        Self {
            handle: device,
            create_buffer: ld!("vkCreateBuffer", PfnCreateBuffer),
            get_buffer_memory_requirements: ld!(
                "vkGetBufferMemoryRequirements",
                PfnGetBufferMemoryRequirements
            ),
            allocate_memory: ld!("vkAllocateMemory", PfnAllocateMemory),
            bind_buffer_memory: ld!("vkBindBufferMemory", PfnBindBufferMemory),
            map_memory: ld!("vkMapMemory", PfnMapMemory),
            unmap_memory: ld!("vkUnmapMemory", PfnUnmapMemory),
            invalidate_mapped_memory_ranges: ld!(
                "vkInvalidateMappedMemoryRanges",
                PfnInvalidateMappedMemoryRanges
            ),
            free_memory: ld!("vkFreeMemory", PfnFreeMemory),
            destroy_buffer: ld!("vkDestroyBuffer", PfnDestroyBuffer),
            create_command_pool: ld!("vkCreateCommandPool", PfnCreateCommandPool),
            destroy_command_pool: ld!("vkDestroyCommandPool", PfnDestroyCommandPool),
            allocate_command_buffers: ld!("vkAllocateCommandBuffers", PfnAllocateCommandBuffers),
            free_command_buffers: ld!("vkFreeCommandBuffers", PfnFreeCommandBuffers),
            reset_command_buffer: ld!("vkResetCommandBuffer", PfnResetCommandBuffer),
            begin_command_buffer: ld!("vkBeginCommandBuffer", PfnBeginCommandBuffer),
            end_command_buffer: ld!("vkEndCommandBuffer", PfnEndCommandBuffer),
            cmd_pipeline_barrier: ld!("vkCmdPipelineBarrier", PfnCmdPipelineBarrier),
            cmd_copy_image_to_buffer: ld!("vkCmdCopyImageToBuffer", PfnCmdCopyImageToBuffer),
            create_fence: ld!("vkCreateFence", PfnCreateFence),
            destroy_fence: ld!("vkDestroyFence", PfnDestroyFence),
            reset_fences: ld!("vkResetFences", PfnResetFences),
            wait_for_fences: ld!("vkWaitForFences", PfnWaitForFences),
            queue_submit: ld!("vkQueueSubmit", PfnQueueSubmit),
            device_wait_idle: ld!("vkDeviceWaitIdle", PfnDeviceWaitIdle),
            create_semaphore: ld!("vkCreateSemaphore", PfnCreateSemaphore),
            destroy_semaphore: ld!("vkDestroySemaphore", PfnDestroySemaphore),
            create_swapchain_khr: ld!("vkCreateSwapchainKHR", PfnCreateSwapchainKHR),
            destroy_swapchain_khr: ld!("vkDestroySwapchainKHR", PfnDestroySwapchainKHR),
            get_swapchain_images_khr: ld!("vkGetSwapchainImagesKHR", PfnGetSwapchainImagesKHR),
            queue_present_khr: ld!("vkQueuePresentKHR", PfnQueuePresentKHR),
        }
    }
}

pub struct InstanceTable {
    pub get_phys_dev_memory_props: PfnGetPhysDevMemProps,
    pub get_phys_dev_queue_family_props: PfnGetPhysDevQueueFamilyProps,
}

impl InstanceTable {
    pub unsafe fn load(instance: vk::Instance, gipa: vk::PFN_vkGetInstanceProcAddr) -> Self {
        macro_rules! li {
            ($name:literal, $ty:ty) => {
                req!(
                    unsafe {
                        load_instance::<$ty>(gipa, instance, concat!($name, "\0").as_bytes())
                    },
                    $name
                )
            };
        }
        Self {
            get_phys_dev_memory_props: li!(
                "vkGetPhysicalDeviceMemoryProperties",
                PfnGetPhysDevMemProps
            ),
            get_phys_dev_queue_family_props: li!(
                "vkGetPhysicalDeviceQueueFamilyProperties",
                PfnGetPhysDevQueueFamilyProps
            ),
        }
    }

    pub unsafe fn get_physical_device_memory_properties(
        &self,
        phys: vk::PhysicalDevice,
    ) -> vk::PhysicalDeviceMemoryProperties {
        let mut props = vk::PhysicalDeviceMemoryProperties::default();
        unsafe { (self.get_phys_dev_memory_props)(phys, &mut props) };
        props
    }

    pub unsafe fn find_transfer_queue_family(&self, phys: vk::PhysicalDevice) -> u32 {
        let mut count = 0u32;
        unsafe { (self.get_phys_dev_queue_family_props)(phys, &mut count, std::ptr::null_mut()) };
        let mut props = vec![vk::QueueFamilyProperties::default(); count as usize];
        unsafe { (self.get_phys_dev_queue_family_props)(phys, &mut count, props.as_mut_ptr()) };
        props
            .iter()
            .enumerate()
            .filter(|(_, p)| p.queue_flags.contains(vk::QueueFlags::TRANSFER))
            .min_by_key(|(_, p)| p.queue_flags.as_raw().count_ones())
            .map(|(i, _)| i as u32)
            .expect("no transfer queue family found")
    }
}
