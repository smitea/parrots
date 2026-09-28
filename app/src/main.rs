//! Parrots menubar app (plan 4): tray-resident GUI hosting the bidirectional
//! translation engine. The window is the settings panel; closing it hides to
//! the tray, Quit (tray menu) exits the process.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod doctor_ui;
mod enroll_ui;
mod settings;
mod tray;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

use eframe::egui;
use parrots_core::{DeviceId, Lang};
use parrots_engine::service::{self, Direction, RunningService, ServiceConfig, ServiceEvent};
use parrots_platform_macos::PARROTS_DEVICE_NAMES;
use tray::Tray;

/// Resolved paths for one app instance.
#[derive(Debug, Clone)]
pub struct AppPaths {
    pub models_root: PathBuf,
    pub profiles_dir: PathBuf,
    pub voice_name: String,
}

impl AppPaths {
    fn from_settings(settings: &settings::AppSettings) -> Self {
        let models_root = std::env::var_os("PARROTS_MODELS")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("models"));
        let profiles_dir = std::env::var_os("PARROTS_PROFILES")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("profiles"));
        Self {
            models_root,
            profiles_dir,
            voice_name: settings.voice.clone(),
        }
    }
}

enum AppEvent {
    Service(ServiceEvent),
    Enroll(String),
    EnrollDone(anyhow::Result<()>),
}

#[derive(PartialEq)]
enum Tab {
    Settings,
    Health,
}

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([560.0, 520.0])
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
    tx: Sender<AppEvent>,
    rx: Receiver<AppEvent>,
    settings: settings::AppSettings,
    devices: Vec<String>,
    svc_a: Option<RunningService>,
    svc_b: Option<RunningService>,
    events: VecDeque<String>,
    enroll_busy: bool,
    tab: Tab,
    health_checks: Option<Vec<doctor_ui::Check>>,
    quit_requested: bool,
}

impl ParrotsApp {
    fn new(_cc: &eframe::CreationContext<'_>, tray: Tray) -> Self {
        let settings = settings::AppSettings::load(&PathBuf::from(
            std::env::var_os("PARROTS_PROFILES")
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "profiles".into()),
        ));
        let devices = parrots_platform_macos::device_names();
        let (tx, rx) = std::sync::mpsc::channel();
        Self {
            tray,
            tx,
            rx,
            settings,
            devices,
            svc_a: None,
            svc_b: None,
            events: VecDeque::new(),
            enroll_busy: false,
            tab: Tab::Settings,
            health_checks: None,
            quit_requested: false,
        }
    }

    fn paths(&self) -> AppPaths {
        AppPaths::from_settings(&self.settings)
    }

    fn log(&mut self, line: impl Into<String>) {
        self.events.push_front(line.into());
        self.events.truncate(40);
    }

    fn handle_menu(&mut self, ctx: &egui::Context, action: tray::MenuAction) {
        match action {
            tray::MenuAction::Settings => {
                self.tab = Tab::Settings;
                ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
                ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            }
            tray::MenuAction::Quit => self.quit_requested = true,
            tray::MenuAction::StartA => self.start_direction(Direction::A),
            tray::MenuAction::StartB => self.start_direction(Direction::B),
            tray::MenuAction::Stop => {
                self.stop_direction(Direction::A);
                self.stop_direction(Direction::B);
            }
        }
    }

    fn resolve_input_b(&self, override_name: Option<&str>) -> (DeviceId, DeviceId) {
        // (initial, policy for recovery re-resolve): prefer the branded
        // virtual speakers, fall back to the system default input
        match override_name {
            Some(name) => (DeviceId(Some(name.into())), DeviceId(Some(name.into()))),
            None => {
                let virtual_speakers = PARROTS_DEVICE_NAMES[1];
                match parrots_platform_macos::find_device(virtual_speakers) {
                    Some(found) => (
                        DeviceId(Some(found)),
                        DeviceId(Some(virtual_speakers.into())),
                    ),
                    None => (DeviceId(None), DeviceId(None)),
                }
            }
        }
    }

    fn start_direction(&mut self, direction: Direction) {
        let already = match direction {
            Direction::A => self.svc_a.is_some(),
            Direction::B => self.svc_b.is_some(),
        };
        if already {
            self.log(format!("{direction:?} already running"));
            return;
        }
        let paths = self.paths();
        let (from, to) = match direction {
            Direction::A => (Lang::Zh, Lang::En),
            Direction::B => (Lang::En, Lang::Zh),
        };

        let recovery = match direction {
            Direction::A => service::DeviceRecovery {
                initial_input: DeviceId(None),
                resolve_input: Arc::new(|| DeviceId(None)),
                resolve_output: {
                    let out = self.settings.output_device.clone();
                    Arc::new(move || match &out {
                        Some(name) => DeviceId(Some(name.clone())),
                        None => DeviceId(None),
                    })
                },
            },
            Direction::B => {
                let (initial, policy) = self.resolve_input_b(self.settings.input_device.as_deref());
                service::DeviceRecovery {
                    initial_input: initial.clone(),
                    resolve_input: Arc::new(move || match &policy {
                        DeviceId(Some(name)) => DeviceId(Some(name.clone())),
                        DeviceId(None) => DeviceId(None),
                    }),
                    resolve_output: Arc::new(|| DeviceId(None)),
                }
            }
        };
        let input_device = match direction {
            Direction::A => DeviceId(None),
            Direction::B => recovery.initial_input.clone(),
        };

        let hotwords = Some(Arc::new(parrots_pipeline::Hotwords::load(
            &paths.profiles_dir.join("hotwords.txt"),
        )));
        let cfg = ServiceConfig {
            direction,
            from,
            to,
            voice: Some(self.settings.voice.clone()),
            voice_utterance: false,
            input_device,
            output_device: match &self.settings.output_device {
                Some(name) => DeviceId(Some(name.clone())),
                None => DeviceId(None),
            },
            hotwords,
            models_root: paths.models_root.clone(),
            profiles_dir: paths.profiles_dir.clone(),
            asr: service::AsrKind::Sensevoice,
            polisher: if self.settings.polish {
                Some(Arc::new(parrots_polish_qwen::QwenPolisher::load(
                    &PathBuf::from("models").join("polish/qwen1.5b"),
                )))
            } else {
                None
            },
            aec: self.settings.aec,
            gate_playback: self.settings.gate_playback,
            max_seconds: None,
            recovery: Some(recovery),
            prebuilt: None,
            collect: false,
        };

        let tx = self.tx.clone();
        let label = format!("{direction:?}");
        match service::start(
            cfg,
            Arc::new(move |ev| {
                let _ = tx.send(AppEvent::Service(ev));
            }),
        ) {
            Ok(handle) => {
                match direction {
                    Direction::A => self.svc_a = Some(handle),
                    Direction::B => self.svc_b = Some(handle),
                }
                self.log(format!("{label} started"));
            }
            Err(e) => self.log(format!("{label} failed to start: {e}")),
        }
    }

    fn stop_direction(&mut self, direction: Direction) {
        let handle = match direction {
            Direction::A => self.svc_a.take(),
            Direction::B => self.svc_b.take(),
        };
        let Some(handle) = handle else { return };
        match handle.stop() {
            Ok(outcome) => self.log(format!(
                "{direction:?} stopped ({} utterances)",
                outcome.timings.e2e_ms.len()
            )),
            Err(e) => self.log(format!("{direction:?} stop error: {e}")),
        }
    }

    fn drain_events(&mut self) {
        while let Ok(ev) = self.rx.try_recv() {
            match ev {
                AppEvent::Service(ServiceEvent::Transcribed(t)) => {
                    self.log(format!("recognised: {t}"));
                }
                AppEvent::Service(ServiceEvent::Translated(t)) => {
                    self.log(format!("translated: {t}"));
                }
                AppEvent::Service(ServiceEvent::Speaking) => {}
                AppEvent::Enroll(msg) => self.log(format!("enroll: {msg}")),
                AppEvent::EnrollDone(result) => {
                    self.enroll_busy = false;
                    match result {
                        Ok(()) => self.log("enroll finished".to_string()),
                        Err(e) => self.log(format!("enroll failed: {e}")),
                    }
                }
            }
        }
    }

    fn settings_tab(&mut self, ui: &mut egui::Ui) {
        let paths = self.paths();
        ui.heading("Settings");
        ui.separator();

        // --- voice profile ---
        ui.horizontal(|ui| {
            ui.label("Voice profile:");
            ui.add(egui::TextEdit::singleline(&mut self.settings.voice).desired_width(80.0));
            let have = paths
                .profiles_dir
                .join(format!("{}.wav", self.settings.voice))
                .is_file();
            ui.label(if have { "✓ recorded" } else { "not recorded" });
            let busy = self.enroll_busy;
            ui.add_enabled(
                !busy,
                egui::Button::new(if busy { "recording…" } else { "Record voice" }),
            )
            .clicked()
            .then(|| {
                self.enroll_busy = true;
                if let Err(e) = enroll_ui::spawn_enroll(
                    self.tx.clone(),
                    paths.clone(),
                    self.settings.voice.clone(),
                    15,
                ) {
                    self.log(format!("enroll failed to start: {e}"));
                    self.enroll_busy = false;
                }
            });
        });
        ui.separator();

        // --- devices ---
        ui.label("Capture device (direction B)");
        combo(
            ui,
            "input",
            &mut self.settings.input_device,
            &self.devices,
            &self.devices,
        );
        ui.label("Playback device (direction A; pick the virtual mic for meetings)");
        combo(
            ui,
            "output",
            &mut self.settings.output_device,
            &self.devices,
            &self.devices,
        );
        ui.separator();

        // --- engine toggles ---
        ui.checkbox(
            &mut self.settings.aec,
            "Echo cancellation (VPIO, direction A)",
        );
        ui.checkbox(
            &mut self.settings.polish,
            "LLM text polisher (needs the polish model)",
        );
        ui.horizontal(|ui| {
            ui.label("Playback gate (A):");
            let mut gate = self.settings.gate_playback;
            ui.selectable_value(&mut gate, None, "auto");
            ui.selectable_value(&mut gate, Some(true), "on");
            ui.selectable_value(&mut gate, Some(false), "off");
            self.settings.gate_playback = gate;
        });
        ui.separator();

        // --- start/stop ---
        ui.horizontal(|ui| {
            let a_on = self.svc_a.is_some();
            let b_on = self.svc_b.is_some();
            ui.add_enabled(!a_on, egui::Button::new("Start A (speak → clone)"))
                .clicked()
                .then(|| self.start_direction(Direction::A));
            ui.add_enabled(!b_on, egui::Button::new("Start B (translate for me)"))
                .clicked()
                .then(|| self.start_direction(Direction::B));
            ui.add_enabled(a_on || b_on, egui::Button::new("Stop"))
                .clicked()
                .then(|| {
                    self.stop_direction(Direction::A);
                    self.stop_direction(Direction::B);
                });
        });

        // --- event log ---
        ui.add_space(8.0);
        ui.label("Events");
        egui::ScrollArea::vertical()
            .max_height(160.0)
            .show(ui, |ui| {
                for line in &self.events {
                    ui.monospace(line);
                }
            });
    }
}

fn combo(
    ui: &mut egui::Ui,
    id: &str,
    value: &mut Option<String>,
    devices: &[String],
    all: &[String],
) {
    let current = value
        .clone()
        .unwrap_or_else(|| "System default".to_string());
    egui::ComboBox::from_id_salt(id)
        .selected_text(current)
        .show_ui(ui, |ui| {
            if ui
                .selectable_label(value.is_none(), "System default")
                .clicked()
            {
                *value = None;
            }
            for d in devices {
                let selected = value.as_deref() == Some(d.as_str());
                if ui.selectable_label(selected, d).clicked() {
                    *value = Some(d.clone());
                }
            }
            let _ = all;
        });
}

impl eframe::App for ParrotsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_events();
        if let Some(action) = self.tray.poll() {
            self.handle_menu(ctx, action);
        }

        // Closing the window hides it to the tray; only the tray Quit exits.
        if ctx.input(|i| i.viewport().close_requested()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
        }
        if self.quit_requested {
            if let Err(e) = self.settings.save(&self.paths().profiles_dir) {
                tracing::warn!("settings save failed: {e}");
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        // Polling keeps tray events and service events flowing.
        ctx.request_repaint_after(std::time::Duration::from_millis(250));

        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Settings, "Settings");
                ui.selectable_value(&mut self.tab, Tab::Health, "Health check");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let a = self.svc_a.is_some();
                    let b = self.svc_b.is_some();
                    ui.label(if a || b {
                        format!(
                            "running: {}",
                            if a && b {
                                "A+B"
                            } else if a {
                                "A"
                            } else {
                                "B"
                            }
                        )
                    } else {
                        "idle".to_string()
                    });
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.tab {
            Tab::Settings => self.settings_tab(ui),
            Tab::Health => {
                ui.horizontal(|ui| {
                    if ui.button("Refresh").clicked() {
                        self.health_checks = Some(doctor_ui::run_checks(&self.paths()));
                    }
                });
                if self.health_checks.is_none() {
                    self.health_checks = Some(doctor_ui::run_checks(&self.paths()));
                }
                let checks = self.health_checks.as_mut().unwrap();
                doctor_ui::show(ui, checks);
            }
        });

        // Persist on every frame (tiny file; write only on change would be nicer)
        if let Err(e) = self.settings.save(&self.paths().profiles_dir) {
            tracing::warn!("settings save failed: {e}");
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // Dropping the tray removes the icon; stop any running services.
        if let Some(h) = self.svc_a.take() {
            let _ = h.stop();
        }
        if let Some(h) = self.svc_b.take() {
            let _ = h.stop();
        }
    }
}
