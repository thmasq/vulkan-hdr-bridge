use std::time::Instant;

/// Normalises Vulkan presentation timestamps to a monotonic stream
/// anchored at the first captured frame.
pub struct TimingController {
    origin_pts: Option<u64>,
    origin_wall: Option<Instant>,
    frame_count: u64,
    last_pts_ns: u64,
}

/// Timing advice returned for each frame.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct FrameTiming {
    /// Presentation timestamp relative to stream start, in nanoseconds.
    pub pts_ns: u64,
    /// Duration of this frame (i.e., time until next frame), in nanoseconds.
    pub duration_ns: u64,
    /// Whether this frame should be dropped (encoder is falling behind).
    pub drop: bool,
}

impl TimingController {
    pub fn new() -> Self {
        Self {
            origin_pts: None,
            origin_wall: None,
            frame_count: 0,
            last_pts_ns: 0,
        }
    }

    /// Feed a raw Vulkan PTS (nanoseconds, device clock).
    /// Returns normalised timing for this frame.
    pub fn next(&mut self, raw_pts_ns: u64, max_queue_depth: u64) -> FrameTiming {
        let origin = *self.origin_pts.get_or_insert(raw_pts_ns);
        let wall_origin = *self.origin_wall.get_or_insert_with(Instant::now);

        let pts = raw_pts_ns.saturating_sub(origin);
        let pts = if self.frame_count > 0 {
            pts.max(self.last_pts_ns + 1)
        } else {
            pts
        };

        let duration_ns = if self.frame_count > 0 {
            pts.saturating_sub(self.last_pts_ns)
        } else {
            16_666_667
        };

        let wall_ns = wall_origin.elapsed().as_nanos() as u64;

        let drop = wall_ns.saturating_sub(pts) > max_queue_depth;

        self.last_pts_ns = pts;
        self.frame_count += 1;

        FrameTiming {
            pts_ns: pts,
            duration_ns,
            drop,
        }
    }
}

impl Default for TimingController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonic_normalisation() {
        let mut tc = TimingController::new();
        let t0 = tc.next(1_000_000_000, u64::MAX);
        let t1 = tc.next(1_016_666_667, u64::MAX);
        assert_eq!(t0.pts_ns, 0);
        assert!(t1.pts_ns > 0);
        assert!(t1.duration_ns > 0);
    }

    #[test]
    fn non_monotonic_clamped() {
        let mut tc = TimingController::new();
        tc.next(1_000_000_000, u64::MAX);
        let t = tc.next(999_000_000, u64::MAX);
        assert!(t.pts_ns >= 1);
    }
}
