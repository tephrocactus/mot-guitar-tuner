use std::sync::atomic::Ordering;
use std::time::Duration;

use egui::{Color32, RichText, Stroke};
use truce::prelude::*;
use truce_egui::EditorUi;

use crate::{
    AssetLoadStatus, GeneratorParams, P, VERSION, arm_command_is_active, generator_can_arm,
};

pub(crate) const WINDOW_SIZE: (u32, u32) = (720, 320);
const VERTICAL_GAP: f32 = 16.0;

pub(crate) struct GeneratorUi;

impl EditorUi<GeneratorParams> for GeneratorUi {
    fn ui(&mut self, ui: &mut egui::Ui, context: &PluginContext<GeneratorParams>) {
        ui.ctx().request_repaint_after(Duration::from_millis(33));

        let background = Color32::from_rgb(10, 12, 14);
        let panel = Color32::from_rgb(20, 24, 28);
        let cyan = Color32::from_rgb(58, 220, 210);
        let text = Color32::from_rgb(228, 235, 238);
        let text_dim = Color32::from_rgb(135, 148, 155);
        ui.visuals_mut().panel_fill = background;
        ui.visuals_mut().override_text_color = Some(text);
        let editor_rect = ui.max_rect();
        let content_width = (editor_rect.width() - 44.0).max(0.0);
        ui.painter().rect_filled(editor_rect, 0.0, background);

        egui::Frame::new()
            .fill(background)
            .inner_margin(22.0)
            .show(ui, |ui| {
                ui.set_min_width(content_width);
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("MOT GENERATOR")
                            .size(27.0)
                            .color(cyan),
                    );
                    ui.label(
                        RichText::new(format!("{VERSION}  •  MONO  •  48 kHz"))
                            .monospace()
                            .color(text_dim),
                    );
                });
                ui.add_space(VERTICAL_GAP);

                egui::Frame::new()
                    .fill(panel)
                    .stroke(Stroke::new(1.0_f32, Color32::from_rgb(44, 52, 58)))
                    .corner_radius(10.0)
                    .inner_margin(20.0)
                    .show(ui, |ui| {
                        ui.set_min_width((content_width - 40.0).max(0.0));
                        let load_status = context.asset_control.status();
                        let (status_label, status_color) =
                            status_label(context, load_status, cyan, text_dim);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("STATUS").color(text_dim));
                            ui.label(
                                RichText::new(status_label)
                                    .strong()
                                    .monospace()
                                    .color(status_color),
                            );
                        });

                        ui.add_space(VERTICAL_GAP);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("SEND TRIM").color(text_dim));
                            let mut send_trim = context.send_trim.value();
                            let response = ui.add(
                                egui::Slider::new(&mut send_trim, -40.0..=0.0)
                                    .step_by(0.1)
                                    .suffix(" dB"),
                            );
                            if response.drag_started() {
                                context.begin_edit(P::SendTrim);
                            }
                            if response.changed() {
                                context.set_param(
                                    P::SendTrim,
                                    f64::from(((send_trim + 40.0) / 40.0).clamp(0.0, 1.0)),
                                );
                            }
                            if response.drag_stopped() {
                                context.end_edit(P::SendTrim);
                            }
                        });

                        ui.add_space(VERTICAL_GAP);
                        let normalized_status = context.get_meter(P::Status);
                        let ready = generator_can_arm(load_status, normalized_status);
                        let arm_command = context.arm_command.load(Ordering::Acquire);
                        let armed = arm_command_is_active(arm_command);
                        let status_code =
                            (normalized_status.clamp(0.0, 1.0) * 7.0).round() as u8;
                        let capture_running = matches!(status_code, 3..=5);
                        let arm_text_color = if armed { background } else { text };
                        let arm_label = if armed { "ARMED" } else { "ARM" };
                        let mut arm_button = egui::Button::new((
                            egui::Atom::grow(),
                            RichText::new(arm_label)
                                .strong()
                                .size(16.0)
                                .color(arm_text_color),
                            egui::Atom::grow(),
                        ))
                        .min_size(egui::vec2(86.0, 36.0));
                        if armed {
                            arm_button = arm_button
                                .fill(cyan)
                                .stroke(Stroke::new(1.0_f32, cyan));
                        }
                        let arm = ui.add_enabled(
                            if armed { !capture_running } else { ready },
                            arm_button,
                        );
                        if arm.clicked() {
                            if !armed {
                                let transport_was_playing =
                                    context.transport().is_some_and(|transport| transport.playing);
                                context
                                    .arm_transport_was_playing
                                    .store(transport_was_playing, Ordering::Release);
                                context.arm_send_trim_bits.store(
                                    context.send_trim.value().clamp(-40.0, 0.0).to_bits(),
                                    Ordering::Release,
                                );
                            }
                            context.arm_command.fetch_add(1, Ordering::AcqRel);
                        }
                        if armed && matches!(status_code, 0..=2) {
                            ui.add_space(VERTICAL_GAP);
                            ui.label(
                                RichText::new(
                                    "Now click Play in a DAW's transport panel. Stop playback if already playing",
                                )
                                .color(text_dim),
                            );
                        }

                        if armed {
                            ui.add_space(VERTICAL_GAP);
                            let progress = context.get_meter(P::Progress).clamp(0.0, 1.0);
                            ui.add(egui::ProgressBar::new(progress).show_percentage());
                        }
                    });
            });
    }
}

fn status_label(
    context: &PluginContext<GeneratorParams>,
    load_status: AssetLoadStatus,
    accent: Color32,
    text_dim: Color32,
) -> (&'static str, Color32) {
    if load_status == AssetLoadStatus::Error {
        return ("Interrupted", Color32::from_rgb(255, 112, 100));
    }
    if load_status == AssetLoadStatus::Loading {
        return ("Ready", text_dim);
    }

    let status = (context.get_meter(P::Status).clamp(0.0, 1.0) * 7.0).round() as u8;
    match status {
        2 => ("Waiting For Play", Color32::from_rgb(255, 196, 74)),
        3 => ("Preroll", Color32::from_rgb(255, 196, 74)),
        4 | 5 => ("Playing Capture wav", accent),
        7 => ("Interrupted", Color32::from_rgb(255, 112, 100)),
        _ => ("Ready", accent),
    }
}
