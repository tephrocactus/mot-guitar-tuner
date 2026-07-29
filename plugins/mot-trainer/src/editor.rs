use std::sync::atomic::Ordering;
use std::time::Duration;

use egui::{Color32, RichText, Vec2};
use truce::prelude::*;
use truce_egui::EditorUi;

use crate::{MotTrainerParams, P, TrainerStatus, read_lock_string, write_lock_string};

pub const WINDOW_SIZE: (u32, u32) = (900, 760);

pub struct MotTrainerUi;

impl EditorUi<MotTrainerParams> for MotTrainerUi {
    fn ui(&mut self, ui: &mut egui::Ui, context: &PluginContext<MotTrainerParams>) {
        ui.ctx().request_repaint_after(Duration::from_millis(33));
        let background = Color32::from_rgb(10, 12, 14);
        let panel = Color32::from_rgb(20, 24, 28);
        let accent = Color32::from_rgb(58, 220, 210);
        let dim = Color32::from_rgb(135, 148, 155);
        ui.visuals_mut().panel_fill = background;
        ui.visuals_mut().override_text_color = Some(Color32::from_rgb(228, 235, 238));

        egui::Frame::new()
            .fill(background)
            .inner_margin(18.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("MOT TRAINER")
                            .size(30.0)
                            .color(accent)
                            .strong(),
                    );
                    ui.label(
                        RichText::new("0.4.0  •  MONO  •  48 kHz")
                            .monospace()
                            .color(dim),
                    );
                });
                ui.separator();
                ui.add_space(12.0);

                egui::Frame::new()
                    .fill(panel)
                    .corner_radius(8.0)
                    .inner_margin(16.0)
                    .show(ui, |ui| {
                        egui::Grid::new("trainer_settings")
                            .num_columns(2)
                            .spacing([18.0, 9.0])
                            .show(ui, |ui| {
                                ui.label(RichText::new("TARGET").color(dim));
                                let target = context.target.value_i32();
                                ui.horizontal(|ui| {
                                    if ui.selectable_label(target == 0, "SOFTWARE CHAIN").clicked()
                                    {
                                        context.automate(P::Target, 0.0);
                                    }
                                    if ui.selectable_label(target == 1, "HARDWARE AMP").clicked() {
                                        context.automate(P::Target, 1.0);
                                    }
                                });
                                ui.end_row();

                                text_field(
                                    ui,
                                    "MODEL NAME",
                                    &context.model_name,
                                    "Captured Amp",
                                    dim,
                                );

                                ui.label(RichText::new("SOURCE SEND TRIM").color(dim));
                                let mut trim = context.source_send_trim_db.value();
                                if ui
                                    .add(
                                        egui::Slider::new(&mut trim, -40.0..=0.0)
                                            .suffix(" dB")
                                            .step_by(0.1)
                                            .fixed_decimals(1),
                                    )
                                    .changed()
                                {
                                    automate_linear(context, P::SourceSendTrimDb, trim, -40.0, 0.0);
                                }
                                ui.end_row();

                                ui.label(RichText::new("MAX PASSES").color(dim));
                                let mut epochs = context.max_epochs.value_i32();
                                if ui
                                    .add(egui::DragValue::new(&mut epochs).range(1..=400).speed(1))
                                    .changed()
                                {
                                    context.automate(P::MaxEpochs, f64::from(epochs - 1) / 399.0);
                                }
                                ui.end_row();
                            });
                        ui.add_space(8.0);
                        ui.label(
                            RichText::new(
                                "SOURCE SEND TRIM must exactly match MOT GENERATOR. Both plug-ins \
                                 use the same immutable excitation WAV and transport edge.",
                            )
                            .small()
                            .color(Color32::from_rgb(235, 164, 77)),
                        );
                        ui.label(
                            RichText::new(format!(
                                "LATCHED AT ARM  {:+.1} dB",
                                context.get_meter(P::LatchedSourceTrim) * 40.0 - 40.0
                            ))
                            .small()
                            .monospace()
                            .color(dim),
                        );
                    });

                ui.add_space(12.0);
                egui::CollapsingHeader::new("CAPTURE METADATA")
                    .default_open(false)
                    .show(ui, |ui| {
                        egui::Grid::new("trainer_metadata")
                            .num_columns(2)
                            .spacing([18.0, 7.0])
                            .show(ui, |ui| {
                                text_field(ui, "AMPLIFIER", &context.amplifier, "EVH 5153", dim);
                                text_field(ui, "CHANNEL", &context.amplifier_channel, "Blue", dim);
                                text_field(
                                    ui,
                                    "CONTROLS",
                                    &context.control_positions,
                                    "Gain 5, Bass 4…",
                                    dim,
                                );
                                text_field(
                                    ui,
                                    "INTERFACE OUT",
                                    &context.interface_output,
                                    "Line Out 3",
                                    dim,
                                );
                                text_field(
                                    ui,
                                    "INTERFACE IN",
                                    &context.interface_input,
                                    "Input 1",
                                    dim,
                                );
                                text_field(ui, "REAMP BOX", &context.reamp_box, "Model", dim);
                                text_field(
                                    ui,
                                    "REACTIVE LOAD",
                                    &context.reactive_load,
                                    "Model / raw out",
                                    dim,
                                );
                                ui.label(RichText::new("IMPEDANCE").color(dim));
                                let mut impedance = context.load_impedance_ohms.load().min(64);
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut impedance)
                                            .range(0..=64)
                                            .suffix(" Ω"),
                                    )
                                    .changed()
                                {
                                    context.load_impedance_ohms.store(impedance);
                                }
                                ui.end_row();
                                text_field(
                                    ui,
                                    "RETURN GAIN",
                                    &context.return_gain_note,
                                    "Interface gain / pad",
                                    dim,
                                );
                            });
                    });

                ui.add_space(12.0);
                let status = context.control.status();
                egui::Frame::new()
                    .fill(panel)
                    .corner_radius(8.0)
                    .inner_margin(16.0)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("STATUS").color(dim));
                            ui.label(
                                RichText::new(status.label())
                                    .monospace()
                                    .strong()
                                    .color(status.color(accent)),
                            );
                        });

                        let capture_progress =
                            context.get_meter(P::CaptureProgress).clamp(0.0, 1.0);
                        ui.add(
                            egui::ProgressBar::new(capture_progress)
                                .text(format!("CAPTURE {:.1}%", capture_progress * 100.0))
                                .fill(Color32::from_rgb(36, 142, 132)),
                        );
                        let peak = context.get_meter(P::ReturnPeak).clamp(0.0, 1.0);
                        let peak_db = if peak > 0.0 {
                            20.0 * peak.log10()
                        } else {
                            f32::NEG_INFINITY
                        };
                        ui.add(
                            egui::ProgressBar::new(peak)
                                .text(if peak_db.is_finite() {
                                    format!("RETURN PEAK {peak_db:.1} dBFS")
                                } else {
                                    "RETURN PEAK −∞ dBFS".to_owned()
                                })
                                .fill(if peak > 0.891_250_9 {
                                    Color32::from_rgb(188, 55, 60)
                                } else {
                                    Color32::from_rgb(36, 142, 132)
                                }),
                        );

                        let training = context.control.training_snapshot();
                        ui.add(
                            egui::ProgressBar::new(training.progress)
                                .text(format!(
                                    "TRAINING PASS {} / {}  •  BEST ESR {:.4}",
                                    training.epoch, training.maximum_epochs, training.best_esr
                                ))
                                .fill(Color32::from_rgb(52, 115, 164)),
                        );
                        if !training.device.is_empty() {
                            ui.label(
                                RichText::new(format!(
                                    "{}  •  elapsed {:.0}s  •  pass {:.1}s",
                                    training.device,
                                    training.elapsed_seconds,
                                    training.epoch_seconds
                                ))
                                .small()
                                .monospace()
                                .color(dim),
                            );
                        }

                        ui.add_space(10.0);
                        ui.horizontal(|ui| {
                            let can_arm = matches!(
                                status,
                                TrainerStatus::Ready
                                    | TrainerStatus::Invalid
                                    | TrainerStatus::Captured
                                    | TrainerStatus::ModelSaved
                                    | TrainerStatus::Error
                            );
                            if ui
                                .add_enabled(
                                    can_arm,
                                    egui::Button::new("ARM NEXT PLAY")
                                        .min_size(Vec2::new(150.0, 34.0))
                                        .fill(Color32::from_rgb(31, 107, 100)),
                                )
                                .clicked()
                            {
                                context.arm_generation.fetch_add(1, Ordering::AcqRel);
                                context.control.clear_error();
                            }
                            if ui
                                .add_enabled(
                                    matches!(
                                        status,
                                        TrainerStatus::Aligning | TrainerStatus::Training
                                    ),
                                    egui::Button::new("CANCEL TRAINING")
                                        .min_size(Vec2::new(150.0, 34.0))
                                        .fill(Color32::from_rgb(148, 48, 55)),
                                )
                                .clicked()
                            {
                                context.control.cancel_training();
                            }
                        });
                        ui.label(
                            RichText::new(
                                "Arm MOT GENERATOR and MOT TRAINER while transport is stopped, \
                                 then press Play or Record once.",
                            )
                            .small()
                            .color(dim),
                        );

                        let error = context.control.last_error();
                        if !error.is_empty() {
                            ui.label(
                                RichText::new(error)
                                    .small()
                                    .color(Color32::from_rgb(235, 95, 95)),
                            );
                        }
                        if let Some(model) = context.control.last_saved_model() {
                            ui.label(
                                RichText::new(format!("SAVED MODEL: {}", model.model_id))
                                    .monospace()
                                    .color(accent),
                            );
                        }
                    });

                if context.target.value_i32() == 1 {
                    ui.add_space(12.0);
                    egui::Frame::new()
                        .fill(Color32::from_rgb(62, 30, 28))
                        .corner_radius(7.0)
                        .inner_margin(12.0)
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(
                                    "Never connect an amplifier SPEAKER OUT directly to an \
                                     audio interface. Use a correctly rated reactive load.",
                                )
                                .strong()
                                .color(Color32::from_rgb(255, 155, 145)),
                            );
                        });
                }
            });
    }
}

fn text_field(
    ui: &mut egui::Ui,
    label: &str,
    value: &std::sync::RwLock<String>,
    hint: &str,
    dim: Color32,
) {
    ui.label(RichText::new(label).color(dim));
    let mut edited = read_lock_string(value);
    if ui
        .add(
            egui::TextEdit::singleline(&mut edited)
                .hint_text(hint)
                .desired_width(360.0),
        )
        .changed()
    {
        write_lock_string(value, &edited);
    }
    ui.end_row();
}

fn automate_linear(
    context: &PluginContext<MotTrainerParams>,
    id: P,
    value: f32,
    minimum: f32,
    maximum: f32,
) {
    let normalized = ((value - minimum) / (maximum - minimum)).clamp(0.0, 1.0);
    context.automate(id, f64::from(normalized));
}

impl TrainerStatus {
    fn color(self, accent: Color32) -> Color32 {
        match self {
            Self::Error | Self::Invalid => Color32::from_rgb(235, 95, 95),
            Self::ModelSaved => Color32::from_rgb(77, 190, 134),
            Self::Training | Self::Aligning | Self::Recording => accent,
            _ => Color32::from_rgb(185, 196, 201),
        }
    }
}
