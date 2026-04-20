//! egui frontend for clicknir.

use std::time::Duration;

use eframe::{CreationContext, Frame};
use egui::{Color32, Context, Key, RichText, Ui};
use serde::{Deserialize, Serialize};

use crate::clicker::{ClickButton, ClickConfig, ClickKind, ClickerHandle};

#[derive(Serialize, Deserialize)]
#[serde(default)]
struct PersistedState {
    cps: f32,
    button: ClickButton,
    kind: ClickKind,
    max_clicks: u64,
    start_delay_ms: u64,
    infinite: bool,
}

impl Default for PersistedState {
    fn default() -> Self {
        let c = ClickConfig::default();
        Self {
            cps: c.cps,
            button: c.button,
            kind: c.kind,
            max_clicks: c.max_clicks,
            start_delay_ms: c.start_delay_ms,
            infinite: c.max_clicks == 0,
        }
    }
}

pub struct ClicknirApp {
    state: PersistedState,
    handle: ClickerHandle,
    // Last `clicks_so_far` seen; used for the status line.
    last_count: u64,
}

impl ClicknirApp {
    pub fn new(cc: &CreationContext<'_>) -> Self {
        let state = cc
            .storage
            .and_then(|s| eframe::get_value::<PersistedState>(s, eframe::APP_KEY))
            .unwrap_or_default();

        let (handle, _join) = ClickerHandle::spawn();
        // The worker thread lives as long as the process; we intentionally drop
        // the join handle — the OS reclaims it on exit and `shutdown()` on
        // drop of the app signals the worker to stop cleanly.

        Self {
            state,
            handle,
            last_count: 0,
        }
    }

    fn current_config(&self) -> ClickConfig {
        ClickConfig {
            cps: self.state.cps,
            button: self.state.button,
            kind: self.state.kind,
            max_clicks: if self.state.infinite {
                0
            } else {
                self.state.max_clicks
            },
            start_delay_ms: self.state.start_delay_ms,
        }
    }

    fn toggle(&self) {
        if self.handle.is_active() {
            self.handle.stop();
        } else {
            self.handle.start(self.current_config());
        }
    }
}

impl eframe::App for ClicknirApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.state);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.handle.shutdown();
    }

    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        // F6 toggles start/stop while the window is focused.
        if ctx.input(|i| i.key_pressed(Key::F6)) {
            self.toggle();
        }

        // While active, poll the worker so the counter stays fresh.
        if self.handle.is_active() {
            ctx.request_repaint_after(Duration::from_millis(50));
        }
        self.last_count = self.handle.clicks_so_far();

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading(RichText::new("Clicknir").strong());
                ui.label(
                    RichText::new("Cross-platform auto clicker").color(Color32::from_gray(160)),
                );
            });
            ui.add_space(8.0);
            ui.separator();

            self.draw_rate_section(ui);
            ui.add_space(6.0);
            self.draw_click_section(ui);
            ui.add_space(6.0);
            self.draw_limits_section(ui);
            ui.add_space(10.0);
            ui.separator();
            self.draw_controls(ui);
            ui.add_space(6.0);
            self.draw_status(ui);
        });
    }
}

impl ClicknirApp {
    fn draw_rate_section(&mut self, ui: &mut Ui) {
        ui.group(|ui| {
            ui.label(RichText::new("Click rate").strong());
            ui.horizontal(|ui| {
                ui.add(
                    egui::Slider::new(&mut self.state.cps, 0.5..=100.0)
                        .logarithmic(true)
                        .text("clicks / sec"),
                );
            });
            let interval_ms = if self.state.cps > 0.0 {
                1000.0 / self.state.cps
            } else {
                0.0
            };
            ui.label(
                RichText::new(format!("≈ {interval_ms:.1} ms between clicks"))
                    .color(Color32::from_gray(160)),
            );
        });
    }

    fn draw_click_section(&mut self, ui: &mut Ui) {
        ui.group(|ui| {
            ui.label(RichText::new("Click type").strong());
            ui.horizontal(|ui| {
                ui.label("Button:");
                for btn in [ClickButton::Left, ClickButton::Middle, ClickButton::Right] {
                    ui.selectable_value(&mut self.state.button, btn, btn.label());
                }
            });
            ui.horizontal(|ui| {
                ui.label("Action:");
                for kind in [ClickKind::Single, ClickKind::Double] {
                    ui.selectable_value(&mut self.state.kind, kind, kind.label());
                }
            });
        });
    }

    fn draw_limits_section(&mut self, ui: &mut Ui) {
        ui.group(|ui| {
            ui.label(RichText::new("Run options").strong());
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.state.infinite, "Click until stopped");
            });
            ui.add_enabled_ui(!self.state.infinite, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Click count:");
                    let mut count = self.state.max_clicks.max(1) as u32;
                    if ui
                        .add(egui::DragValue::new(&mut count).range(1..=1_000_000))
                        .changed()
                    {
                        self.state.max_clicks = count as u64;
                    }
                });
            });
            ui.horizontal(|ui| {
                ui.label("Start delay:");
                let mut delay = self.state.start_delay_ms as u32;
                if ui
                    .add(
                        egui::DragValue::new(&mut delay)
                            .range(0..=60_000)
                            .suffix(" ms"),
                    )
                    .changed()
                {
                    self.state.start_delay_ms = delay as u64;
                }
                ui.label(
                    RichText::new("(time to move the cursor before clicking starts)")
                        .color(Color32::from_gray(160))
                        .small(),
                );
            });
        });
    }

    fn draw_controls(&mut self, ui: &mut Ui) {
        let active = self.handle.is_active();
        ui.horizontal(|ui| {
            let (text, color) = if active {
                ("■  Stop  (F6)", Color32::from_rgb(200, 70, 70))
            } else {
                ("▶  Start  (F6)", Color32::from_rgb(70, 160, 90))
            };
            let btn = egui::Button::new(RichText::new(text).strong().size(16.0))
                .fill(color)
                .min_size(egui::vec2(200.0, 36.0));
            if ui.add(btn).clicked() {
                self.toggle();
            }

            if ui
                .add_enabled(
                    !active,
                    egui::Button::new("Reset counter").min_size(egui::vec2(120.0, 36.0)),
                )
                .clicked()
            {
                // Counter resets on next start; show zero now for feedback.
                self.last_count = 0;
            }
        });
    }

    fn draw_status(&mut self, ui: &mut Ui) {
        let active = self.handle.is_active();
        let pending = self.handle.is_counting_down();
        let status = if pending {
            RichText::new("Arming — move the cursor to the target…")
                .color(Color32::from_rgb(220, 180, 60))
        } else if active {
            RichText::new("Running — press F6 to stop").color(Color32::from_rgb(120, 200, 140))
        } else {
            RichText::new("Idle").color(Color32::from_gray(170))
        };
        ui.label(status);
        ui.label(format!("Clicks this run: {}", self.last_count));
        ui.add_space(4.0);
        ui.label(
            RichText::new(
                "Tip: set a start delay so you can position the mouse before the first click.",
            )
            .small()
            .color(Color32::from_gray(140)),
        );
    }
}
