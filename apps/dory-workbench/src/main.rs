#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod app;
mod chat;
mod config;
mod engine;
mod tools;

use eframe::egui;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1370.0, 855.0])
            .with_min_inner_size([1050.0, 620.0])
            .with_app_id("dory-workbench"),
        ..Default::default()
    };
    eframe::run_native(
        "Dory Workbench",
        options,
        Box::new(|cc| Ok(Box::new(app::Workbench::new(cc)?))),
    )
}
