use crate::error::{BridgeError, Result};
use ash::vk;
use std::io::Write;
use std::process::{Child, ChildStdin, Command, Stdio};

const MAX_CLL: u32 = 1000;
const MAX_FALL: u32 = 400;

pub struct FfmpegSink {
    child: Child,
    stdin: ChildStdin,
}

impl FfmpegSink {
    pub fn spawn(
        output_path: &str,
        width: u32,
        height: u32,
        fps_num: u32,
        fps_den: u32,
        vk_format: vk::Format,
    ) -> Result<Self> {
        // Mastering display: Rec.2020 primaries + D65 white + 1000-nit peak
        // Chromaticity values in units of 0.00002 as required by x265/svtav1 and FFmpeg
        let master_display =
            "G(0.170,0.797)B(0.131,0.046)R(0.708,0.292)WP(0.3127,0.3290)L(1000.0,0.0001)";

        let pix_fmt = match vk_format {
            vk::Format::A2B10G10R10_UNORM_PACK32 => "x2bgr10le",
            _ => "x2rgb10le",
        };

        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-hide_banner")
            .args(["-f", "rawvideo"])
            .args(["-pixel_format", pix_fmt])
            .args(["-video_size", &format!("{width}x{height}")])
            .args(["-framerate", &format!("{fps_num}/{fps_den}")])
            .args(["-i", "pipe:0"])
            .args([
                "-vf",
                "scale=out_color_matrix=bt2020:out_range=pc,format=yuv420p10le",
            ])
            // Color metadata
            .args(["-color_primaries", "bt2020"])
            .args(["-colorspace", "bt2020nc"])
            .args(["-color_trc", "smpte2084"])
            .args(["-color_range", "pc"])
            // AV1
            .args(["-c:v", "libsvtav1"])
            .args(["-preset", "8"])
            .args(["-crf", "20"])
            .args([
                "-svtav1-params",
                &format!(
                    "tune=0:color-primaries=9:transfer-characteristics=16:matrix-coefficients=9:color-range=1:\
                     input-depth=10:\
                     mastering-display={master_display}:\
                     content-light={MAX_CLL},{MAX_FALL}"
                ),
            ])
            // Output
            .arg("-y")
            .arg(output_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());

        let mut child = cmd
            .spawn()
            .map_err(|e| BridgeError::Ffmpeg(e.to_string()))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| BridgeError::Ffmpeg("FFmpeg stdin unavailable".into()))?;

        log::info!(
            "FFmpeg spawned → {output_path} ({width}×{height} @{fps_num}/{fps_den} fps, AV1 Main10 PQ)"
        );

        Ok(Self { child, stdin })
    }

    pub fn write_raw(&mut self, bytes: &[u8]) -> Result<()> {
        self.stdin.write_all(bytes).map_err(BridgeError::Io)
    }

    pub fn finish(mut self) -> Result<()> {
        drop(self.stdin);
        let status = self.child.wait().map_err(BridgeError::Io)?;
        if status.success() {
            log::info!("FFmpeg exited successfully");
            Ok(())
        } else {
            Err(BridgeError::Ffmpeg(format!(
                "FFmpeg exited with {status}"
            )))
        }
    }
}
