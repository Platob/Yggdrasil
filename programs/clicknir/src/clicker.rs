//! Background worker that drives the actual mouse clicking.
//!
//! The worker runs in its own OS thread and is controlled through an atomic
//! state machine plus a `Condvar`, so the UI thread never blocks and we can
//! wake the worker immediately when the user presses Start / Stop.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use enigo::{Button, Enigo, Mouse, Settings};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClickButton {
    Left,
    Right,
    Middle,
}

impl ClickButton {
    pub fn label(self) -> &'static str {
        match self {
            ClickButton::Left => "Left",
            ClickButton::Right => "Right",
            ClickButton::Middle => "Middle",
        }
    }

    fn to_enigo(self) -> Button {
        match self {
            ClickButton::Left => Button::Left,
            ClickButton::Right => Button::Right,
            ClickButton::Middle => Button::Middle,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClickKind {
    Single,
    Double,
}

impl ClickKind {
    pub fn label(self) -> &'static str {
        match self {
            ClickKind::Single => "Single",
            ClickKind::Double => "Double",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ClickConfig {
    /// Desired clicks per second (0 means "as fast as possible").
    pub cps: f32,
    pub button: ClickButton,
    pub kind: ClickKind,
    /// 0 means click forever until stopped.
    pub max_clicks: u64,
    /// Delay before the first click, in milliseconds.
    pub start_delay_ms: u64,
}

impl Default for ClickConfig {
    fn default() -> Self {
        Self {
            cps: 10.0,
            button: ClickButton::Left,
            kind: ClickKind::Single,
            max_clicks: 0,
            start_delay_ms: 3000,
        }
    }
}

const STATE_IDLE: u8 = 0;
const STATE_PENDING: u8 = 1;
const STATE_RUNNING: u8 = 2;

/// Handle to the click worker thread. Cloneable; internally reference counted.
#[derive(Clone)]
pub struct ClickerHandle {
    inner: Arc<Inner>,
}

struct Inner {
    state: AtomicU8,
    stop_requested: AtomicBool,
    shutdown: AtomicBool,
    counter: AtomicU64,
    // Configuration is copied by the worker when a run starts.
    config: Mutex<ClickConfig>,
    // Lets the worker sleep until it has something to do.
    wake: Condvar,
    wake_mu: Mutex<()>,
}

impl ClickerHandle {
    pub fn spawn() -> (Self, JoinHandle<()>) {
        let inner = Arc::new(Inner {
            state: AtomicU8::new(STATE_IDLE),
            stop_requested: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            counter: AtomicU64::new(0),
            config: Mutex::new(ClickConfig::default()),
            wake: Condvar::new(),
            wake_mu: Mutex::new(()),
        });

        let worker_inner = inner.clone();
        let join = thread::Builder::new()
            .name("clicknir-worker".into())
            .spawn(move || worker_loop(worker_inner))
            .expect("spawn clicker worker");

        (Self { inner }, join)
    }

    pub fn start(&self, config: ClickConfig) {
        *self.inner.config.lock().expect("config mutex") = config;
        self.inner.stop_requested.store(false, Ordering::SeqCst);
        self.inner.counter.store(0, Ordering::SeqCst);
        self.inner.state.store(STATE_PENDING, Ordering::SeqCst);
        self.wake();
    }

    pub fn stop(&self) {
        self.inner.stop_requested.store(true, Ordering::SeqCst);
        self.wake();
    }

    pub fn shutdown(&self) {
        self.inner.shutdown.store(true, Ordering::SeqCst);
        self.inner.stop_requested.store(true, Ordering::SeqCst);
        self.wake();
    }

    pub fn is_active(&self) -> bool {
        matches!(
            self.inner.state.load(Ordering::SeqCst),
            STATE_PENDING | STATE_RUNNING
        )
    }

    pub fn is_counting_down(&self) -> bool {
        self.inner.state.load(Ordering::SeqCst) == STATE_PENDING
    }

    pub fn clicks_so_far(&self) -> u64 {
        self.inner.counter.load(Ordering::SeqCst)
    }

    fn wake(&self) {
        let guard = self.inner.wake_mu.lock().expect("wake mutex");
        self.inner.wake.notify_all();
        drop(guard);
    }
}

fn worker_loop(inner: Arc<Inner>) {
    // Build Enigo lazily on first run — on Linux this probes the X11/Wayland
    // backend, which we don't want to do until the user actually clicks Start.
    let mut enigo: Option<Enigo> = None;

    loop {
        if inner.shutdown.load(Ordering::SeqCst) {
            return;
        }

        // Wait until we're asked to start (or shut down).
        if inner.state.load(Ordering::SeqCst) == STATE_IDLE {
            let guard = inner.wake_mu.lock().expect("wake mutex");
            let _guard = inner
                .wake
                .wait_while(guard, |_| {
                    inner.state.load(Ordering::SeqCst) == STATE_IDLE
                        && !inner.shutdown.load(Ordering::SeqCst)
                })
                .expect("wake wait");
            continue;
        }

        let config = *inner.config.lock().expect("config mutex");

        // Lazily initialize Enigo on the worker thread.
        if enigo.is_none() {
            match Enigo::new(&Settings::default()) {
                Ok(e) => enigo = Some(e),
                Err(err) => {
                    eprintln!("clicknir: failed to initialize input backend: {err}");
                    inner.state.store(STATE_IDLE, Ordering::SeqCst);
                    continue;
                }
            }
        }

        // Honour the pre-run delay, but keep it interruptible.
        if config.start_delay_ms > 0 && wait_interruptible(&inner, config.start_delay_ms) {
            inner.state.store(STATE_IDLE, Ordering::SeqCst);
            continue;
        }
        if inner.stop_requested.load(Ordering::SeqCst) {
            inner.state.store(STATE_IDLE, Ordering::SeqCst);
            continue;
        }

        inner.state.store(STATE_RUNNING, Ordering::SeqCst);

        let interval = interval_from_cps(config.cps);
        let en = enigo.as_mut().expect("enigo ready");

        let mut next_tick = Instant::now();
        while !inner.stop_requested.load(Ordering::SeqCst) {
            perform_click(en, config.button, config.kind);
            let done = inner.counter.fetch_add(1, Ordering::SeqCst) + 1;

            if config.max_clicks > 0 && done >= config.max_clicks {
                break;
            }

            if let Some(delay) = interval {
                next_tick += delay;
                let now = Instant::now();
                if next_tick <= now {
                    // We're behind schedule; catch up without spinning.
                    next_tick = now;
                } else {
                    let wait_ms = (next_tick - now).as_millis() as u64;
                    if wait_interruptible(&inner, wait_ms) {
                        break;
                    }
                }
            }
        }

        inner.state.store(STATE_IDLE, Ordering::SeqCst);
    }
}

fn perform_click(enigo: &mut Enigo, button: ClickButton, kind: ClickKind) {
    let btn = button.to_enigo();
    let _ = enigo.button(btn, enigo::Direction::Click);
    if matches!(kind, ClickKind::Double) {
        // Small gap so the OS registers it as a genuine double-click.
        thread::sleep(Duration::from_millis(30));
        let _ = enigo.button(btn, enigo::Direction::Click);
    }
}

fn interval_from_cps(cps: f32) -> Option<Duration> {
    if cps <= 0.0 {
        None
    } else {
        Some(Duration::from_secs_f32(1.0 / cps))
    }
}

/// Sleep up to `millis` ms, returning early (with `true`) if a stop/shutdown
/// is requested in the meantime.
fn wait_interruptible(inner: &Inner, millis: u64) -> bool {
    let stop_or_shutdown = || {
        inner.stop_requested.load(Ordering::SeqCst) || inner.shutdown.load(Ordering::SeqCst)
    };
    if millis == 0 {
        return stop_or_shutdown();
    }
    let guard = inner.wake_mu.lock().expect("wake mutex");
    let _res = inner
        .wake
        .wait_timeout_while(guard, Duration::from_millis(millis), |_| !stop_or_shutdown())
        .expect("wait_timeout_while");
    stop_or_shutdown()
}
