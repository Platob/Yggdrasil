#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod clicker;
mod ui;

use eframe::NativeOptions;
use egui::ViewportBuilder;

use crate::ui::ClicknirApp;

fn main() -> eframe::Result<()> {
    let options = NativeOptions {
        viewport: ViewportBuilder::default()
            .with_title("Clicknir — Auto Clicker")
            .with_inner_size([420.0, 520.0])
            .with_min_inner_size([360.0, 460.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Clicknir",
        options,
        Box::new(|cc| Ok(Box::new(ClicknirApp::new(cc)))),
    )
}
