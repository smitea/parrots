//! Parrots menubar app (plan 4): tray-resident GUI hosting the bidirectional
//! translation engine. The window is the settings panel; closing it hides to
//! the tray, Quit (tray menu) exits the process.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod tray;

use std::sync::Arc;

use eframe::egui;
use parrots_core::Lang;
use parrots_engine::service::{Direction, ServiceConfig, ServiceEvent};
use tray::Tray;

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([460.0, 420.0])
            .with_title("Parrots"),
        ..Default::default()
    };

    eframe::run_native(
        "Parrots",
        options,
        Box::new(|cc| {
            // macOS: tray must be created on the main thread — here we are.
            let tray = tray::Tray::new().expect("failed to create tray icon");
            Ok(Box::new(ParrotsApp::new(cc, tray)))
        }),
    )
}

struct ParrotsApp {
    tray: Tray,
    quit_requested: bool,
    status: String,
    last_events: Vec<String>,
}

impl ParrotsApp {
    fn new(_cc: &eframe::CreationContext<'_>, tray: Tray) -> Self {
        Self {
            tray,
            quit_requested: false,
            status: "Idle".into(),
            last_events: Vec::new(),
        }
    }

    fn handle_menu(&mut self, ctx: &egui::Context, action: tray::MenuAction) {
        match action {
            tray::MenuAction::Settings => {
                self.status = "Settings".into();
                ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
                ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            }
            tray::MenuAction::Quit => self.quit_requested = true,
            tray::MenuAction::StartA | tray::MenuAction::StartB | tray::MenuAction::Stop => {
                // Wired up in Task 3 (service start/stop from settings window).
                self.status = format!("{action:?} (not wired yet — Task 3)");
            }
        }
    }
}

impl eframe::App for ParrotsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(action) = self.tray.poll() {
            self.handle_menu(ctx, action);
        }

        // Closing the window hides it to the tray; only the tray Quit exits.
        if ctx.input(|i| i.viewport().close_requested()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
        }
        if self.quit_requested {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        // Keep repainting while services run (event-driven UI arrives later).
        ctx.request_repaint_after(std::time::Duration::from_millis(500));

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Parrots");
            ui.separator();
            ui.label(format!("Status: {}", self.status));
            ui.add_space(8.0);
            ui.label("Directions:");
            ui.monospace("  A  I speak -> listeners hear a cloned voice");
            ui.monospace("  B  listeners speak -> I hear the translation");
            ui.add_space(8.0);
            if !self.last_events.is_empty() {
                ui.label("Recent events:");
                for line in self.last_events.iter().rev().take(6) {
                    ui.monospace(line);
                }
            }
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // Dropping the tray removes the icon.
        let _ = &self.tray;
    }
}

// referenced by Task 3; keeps imports honest until then
#[allow(dead_code)]
fn _service_types(_: ServiceConfig, _: ServiceEvent, _: Lang, _: Arc<Direction>) {}
