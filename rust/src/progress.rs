//! Time-throttled phase progress on stderr.
//!
//! The hour-scale phases (Krylov, approximant basis, reconstruction)
//! announce their step total up front -- what a reader actually needs to
//! plan a wait -- and then print one line per `MF_PROGRESS_SECS` seconds
//! (default 10, 0 disables) of the form
//!
//!     [krylov] 3200/7383 (43%) elapsed 1620s, eta 2140s
//!
//! Lines go to stderr so stdout stays result-only; phase-completion
//! timings keep coming from the existing stats summary.  Phases shorter
//! than the interval print only their 0/total announce line.

use std::time::Instant;

pub struct Progress {
    label: String,
    total: usize,
    t0: Instant,
    last: Instant,
    interval: f64,
}

impl Progress {
    pub fn start(label: &str, total: usize) -> Progress {
        let interval = std::env::var("MF_PROGRESS_SECS")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(10.0)
            .max(0.0);
        if interval > 0.0 && total > 0 {
            eprintln!("[{label}] 0/{total}");
        }
        Progress {
            label: label.to_string(),
            total,
            t0: Instant::now(),
            last: Instant::now(),
            interval,
        }
    }

    /// `done` = items completed so far.  The final item is intentionally
    /// not reported here; the phase summary line reports the timing.
    pub fn tick(&mut self, done: usize) {
        if self.interval <= 0.0 || done == 0 || done >= self.total {
            return;
        }
        let now = Instant::now();
        if (now - self.last).as_secs_f64() < self.interval {
            return;
        }
        self.last = now;
        let elapsed = now.duration_since(self.t0).as_secs_f64();
        let eta = elapsed / done as f64 * (self.total - done) as f64;
        eprintln!(
            "[{}] {}/{} ({:.0}%) elapsed {:.0}s, eta {:.0}s",
            self.label,
            done,
            self.total,
            100.0 * done as f64 / self.total as f64,
            elapsed,
            eta
        );
    }
}
