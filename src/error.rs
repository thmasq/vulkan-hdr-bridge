use thiserror::Error;

#[derive(Debug, Error)]
pub enum BridgeError {
    #[error("Unsupported Vulkan format: {0:?}")]
    UnsupportedFormat(ash::vk::Format),
    #[error("Vulkan error: {0}")]
    Vk(#[from] ash::vk::Result),
    #[error("FFmpeg process error: {0}")]
    Ffmpeg(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, BridgeError>;
