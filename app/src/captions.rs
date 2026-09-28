//! Floating caption overlay (plan 4b): borderless always-on-top viewport
//! showing the remote speaker's original text and its translation.

use std::sync::Mutex;

use eframe::egui;

#[derive(Debug, Default, Clone)]
pub struct CaptionState {
    pub enabled: bool,
    pub source: String,
    pub translated: String,
}

pub type SharedCaptions = Arc<Mutex<CaptionState>>;

use std::sync::Arc;

/// Show the caption viewport (call every frame from the main update).
pub fn show(ctx: &egui::Context, captions: &SharedCaptions) {
    let state = captions.lock().unwrap_or_else(|e| e.into_inner());
    if !state.enabled {
        return;
    }
    let source = state.source.clone();
    let translated = state.translated.clone();
    drop(state);

    ctx.show_viewport_deferred(
        egui::ViewportId(egui::Id::new("captions")),
        egui::ViewportBuilder::default()
            .with_always_on_top()
            .with_decorations(false)
            .with_resizable(false)
            .with_inner_size([520.0, 120.0])
            .with_title("Parrots captions"),
        move |ctx, _class| {
            egui::CentralPanel::default()
                .frame(
                    egui::Frame::default()
                        .fill(egui::Color32::from_rgba_unmultiplied(10, 10, 12, 235))
                        .inner_margin(12.0),
                )
                .show(ctx, |ui| {
                    ui.set_min_size(egui::vec2(496.0, 96.0));
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(&source)
                                .color(egui::Color32::from_rgb(200, 205, 215))
                                .size(16.0),
                        )
                        .wrap(),
                    );
                    ui.add_space(4.0);
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(&translated)
                                .color(egui::Color32::WHITE)
                                .size(22.0)
                                .strong(),
                        )
                        .wrap(),
                    );
                });
            // Keep the overlay alive; content updates arrive via shared state.
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        },
    );
}
