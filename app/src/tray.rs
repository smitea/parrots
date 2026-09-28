//! System tray icon + menu wiring (macOS: the tray must be created on the
//! main thread, so this happens inside the eframe app creator, and menu
//! events are polled from `MenuEvent::receiver()` in the egui update loop —
//! the same pattern as the tray-icon crate's eframe example).

use crossbeam_channel::Receiver;

use tray_icon::menu::{Menu, MenuEvent, MenuId, MenuItem, PredefinedMenuItem};
use tray_icon::{TrayIcon, TrayIconBuilder};

/// Menu action identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuAction {
    StartA,
    StartB,
    Stop,
    Settings,
    Quit,
}

pub struct Tray {
    _tray: TrayIcon,
    receiver: Receiver<MenuEvent>,
    pub id_start_a: MenuId,
    pub id_start_b: MenuId,
    pub id_stop: MenuId,
    pub id_settings: MenuId,
    pub id_quit: MenuId,
}

impl Tray {
    /// Build the tray icon + menu. Must run on the main thread (macOS).
    pub fn new() -> anyhow::Result<Self> {
        let id_start_a = MenuId::new("start_a");
        let id_start_b = MenuId::new("start_b");
        let id_stop = MenuId::new("stop");
        let id_settings = MenuId::new("settings");
        let id_quit = MenuId::new("quit");

        let menu = Menu::new();
        menu.append(&MenuItem::with_id(
            id_start_a.clone(),
            "Start A — I speak (clone my voice)",
            true,
            None,
        ))?;
        menu.append(&MenuItem::with_id(
            id_start_b.clone(),
            "Start B — translate for me",
            true,
            None,
        ))?;
        menu.append(&MenuItem::with_id(id_stop.clone(), "Stop", true, None))?;

        menu.append(&MenuItem::with_id(
            id_settings.clone(),
            "Settings...",
            true,
            None,
        ))?;

        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&MenuItem::with_id(id_quit.clone(), "Quit", true, None))?;

        let icon = tray_icon::Icon::from_rgba(icon_rgba(32), 32, 32)?;
        let tray = TrayIconBuilder::new()
            .with_tooltip("Parrots — live translation")
            .with_menu(Box::new(menu))
            .with_icon(icon)
            .build()?;

        let receiver = MenuEvent::receiver().clone();
        Ok(Self {
            _tray: tray,
            receiver,
            id_start_a,
            id_start_b,
            id_stop,
            id_settings,
            id_quit,
        })
    }

    /// Poll pending menu events and map them to actions.
    pub fn poll(&self) -> Option<MenuAction> {
        let mut action = None;
        while let Ok(ev) = self.receiver.try_recv() {
            if ev.id == self.id_start_a {
                action = Some(MenuAction::StartA);
            } else if ev.id == self.id_start_b {
                action = Some(MenuAction::StartB);
            } else if ev.id == self.id_stop {
                action = Some(MenuAction::Stop);
            } else if ev.id == self.id_settings {
                action = Some(MenuAction::Settings);
            } else if ev.id == self.id_quit {
                action = Some(MenuAction::Quit);
            }
        }
        action
    }
}

/// A simple solid disc with a parrot-green accent, generated in code so the
/// binary needs no bundled asset.
fn icon_rgba(size: u32) -> Vec<u8> {
    use image::{ImageBuffer, Rgba};
    let img = ImageBuffer::from_fn(size, size, |x, y| {
        let cx = f64::from(size) / 2.0 - 0.5;
        let cy = f64::from(size) / 2.0 - 0.5;
        let d = ((f64::from(x) - cx).powi(2) + (f64::from(y) - cy).powi(2)).sqrt();
        let r = f64::from(size) / 2.0 - 1.0;
        if d <= r * 0.55 {
            Rgba([40, 170, 90, 255]) // green core
        } else if d <= r {
            Rgba([30, 30, 30, 255]) // dark ring
        } else {
            Rgba([0, 0, 0, 0]) // transparent
        }
    });
    img.into_raw()
}
