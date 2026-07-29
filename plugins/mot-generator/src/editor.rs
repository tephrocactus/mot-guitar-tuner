use std::sync::atomic::Ordering;
use std::time::Duration;

use egui::{Color32, RichText, Stroke};
use truce::prelude::*;
use truce_egui::EditorUi;

use crate::{
    AssetLoadStatus, GeneratorParams, P, VERSION, arm_command_is_active, generator_can_arm,
};

pub(crate) const WINDOW_SIZE: (u32, u32) = (720, 300);

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

        egui::Frame::new()
            .fill(background)
            .inner_margin(22.0)
            .show(ui, |ui| {
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
                ui.add_space(16.0);

                egui::Frame::new()
                    .fill(panel)
                    .stroke(Stroke::new(1.0_f32, Color32::from_rgb(44, 52, 58)))
                    .corner_radius(10.0)
                    .inner_margin(20.0)
                    .show(ui, |ui| {
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

                        ui.add_space(14.0);
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

                        ui.add_space(18.0);
                        let normalized_status = context.get_meter(P::Status);
                        let ready = generator_can_arm(load_status, normalized_status);
                        let arm_command = context.arm_command.load(Ordering::Acquire);
                        let armed = arm_command_is_active(arm_command);
                        let arm = ui.add_enabled(
                            armed || ready,
                            egui::Button::new(RichText::new("ARM").strong().size(16.0))
                                .selected(armed)
                                .min_size(egui::vec2(86.0, 36.0)),
                        );
                        if arm.clicked() {
                            if !armed {
                                let transport_was_playing =
                                    context.transport().is_none_or(|transport| transport.playing);
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
                        if armed {
                            ui.add_space(8.0);
                            ui.label(
                                RichText::new(
                                    "Now click Play in a DAW's transport panel. Stop playback if already playing",
                                )
                                .color(text_dim),
                            );
                        }

                        ui.add_space(18.0);
                        let progress = context.get_meter(P::Progress).clamp(0.0, 1.0);
                        ui.add(
                            egui::ProgressBar::new(progress)
                                .show_percentage(),
                        );
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
        return ("ASSET ERROR", Color32::from_rgb(255, 112, 100));
    }
    if load_status == AssetLoadStatus::Loading {
        return ("LOADING CAPTURE ASSET", text_dim);
    }

    let status = (context.get_meter(P::Status).clamp(0.0, 1.0) * 7.0).round() as u8;
    match status {
        1 => ("READY", accent),
        2 => ("ARMED — WAITING FOR PLAY", Color32::from_rgb(255, 196, 74)),
        3 => ("PRE-ROLL", Color32::from_rgb(255, 196, 74)),
        4 => ("PLAYING CAPTURE WAV", accent),
        5 => ("TAIL", text_dim),
        6 => ("COMPLETE", accent),
        7 => (
            "INVALID / 48 kHz REQUIRED",
            Color32::from_rgb(255, 112, 100),
        ),
        _ => ("INITIALIZING", text_dim),
    }
}
