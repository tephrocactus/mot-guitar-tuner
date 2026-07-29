use std::sync::atomic::Ordering;
use std::time::Duration;

use egui::{Align, Color32, Layout, RichText};
use truce::prelude::*;
use truce_egui::EditorUi;

use crate::{
    AssetLoadStatus, GeneratorParams, P, VERSION, arm_command_is_active, generator_can_arm,
};

pub(crate) const WINDOW_SIZE: (u32, u32) = (720, 320);

pub(crate) struct GeneratorUi;

impl EditorUi<GeneratorParams> for GeneratorUi {
    fn ui(&mut self, ui: &mut egui::Ui, context: &PluginContext<GeneratorParams>) {
        ui.ctx().request_repaint_after(Duration::from_millis(33));

        mot_ui::apply(ui);
        let editor_rect = ui.max_rect();
        let content_width = (editor_rect.width() - mot_ui::OUTER_MARGIN * 2.0).max(0.0);
        ui.painter()
            .rect_filled(editor_rect, 0.0, mot_ui::BACKGROUND);

        mot_ui::background_frame().show(ui, |ui| {
            ui.set_min_width(content_width);
            mot_ui::header(ui, "MOT GENERATOR", VERSION, false, |_| {});

            let load_status = context.asset_control.status();
            let (status_label, status_color) = status_label(context, load_status);
            let normalized_status = context.get_meter(P::Status);
            let ready = generator_can_arm(load_status, normalized_status);
            let arm_command = context.arm_command.load(Ordering::Acquire);
            let armed = arm_command_is_active(arm_command);
            let status_code = (normalized_status.clamp(0.0, 1.0) * 7.0).round() as u8;
            let capture_running = matches!(status_code, 3..=5);

            let panel_height = ui.available_height();
            mot_ui::panel().show(ui, |ui| {
                ui.set_min_width(
                    (content_width - mot_ui::PANEL_MARGIN * 2.0).max(0.0),
                );
                ui.set_min_height(
                    (panel_height - mot_ui::PANEL_MARGIN * 2.0).max(0.0),
                );

                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(mot_ui::field_label("STATUS"));
                        ui.add_space(4.0);
                        ui.label(mot_ui::status_text(status_label, status_color));
                    });

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let arm_label = if armed { "ARMED" } else { "ARM" };
                        let label_color = if armed {
                            mot_ui::BACKGROUND
                        } else {
                            mot_ui::TEXT
                        };
                        let label = RichText::new(arm_label)
                            .strong()
                            .size(15.0)
                            .color(label_color);
                        let button = if armed {
                            mot_ui::accent_button(label)
                        } else {
                            mot_ui::ghost_button(label)
                        }
                        .min_size(egui::vec2(110.0, 40.0));
                        let arm = ui.add_enabled(
                            if armed { !capture_running } else { ready },
                            button,
                        );
                        if arm.clicked() {
                            if !armed {
                                let transport_was_playing = context
                                    .transport()
                                    .is_some_and(|transport| transport.playing);
                                context
                                    .arm_transport_was_playing
                                    .store(transport_was_playing, Ordering::Release);
                            }
                            context.arm_command.fetch_add(1, Ordering::AcqRel);
                        }
                    });
                });

                if armed && matches!(status_code, 0..=2) {
                    ui.add_space(mot_ui::ROW_GAP);
                    ui.label(
                        RichText::new(
                            "Now click Play in a DAW's transport panel. Stop playback if already playing",
                        )
                        .color(mot_ui::DIM),
                    );
                }

                if armed {
                    ui.add_space(mot_ui::ROW_GAP);
                    let progress = context.get_meter(P::Progress).clamp(0.0, 1.0);
                    mot_ui::progress(
                        ui,
                        progress,
                        format!("{:.0}%", progress * 100.0),
                    );
                }
            });
        });
    }
}

fn status_label(
    context: &PluginContext<GeneratorParams>,
    load_status: AssetLoadStatus,
) -> (&'static str, Color32) {
    if load_status == AssetLoadStatus::Error {
        return ("Interrupted", mot_ui::ERROR);
    }
    if load_status == AssetLoadStatus::Loading {
        return ("Ready", mot_ui::DIM);
    }

    let status = (context.get_meter(P::Status).clamp(0.0, 1.0) * 7.0).round() as u8;
    match status {
        2 => ("Waiting For Play", mot_ui::WAITING),
        3 => ("Preroll", mot_ui::WAITING),
        4 | 5 => ("Playing Capture wav", mot_ui::ACCENT),
        7 => ("Interrupted", mot_ui::ERROR),
        _ => ("Ready", mot_ui::ACCENT),
    }
}
