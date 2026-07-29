use std::sync::atomic::Ordering;
use std::time::Duration;

use egui::{Align, Color32, Layout, RichText, Stroke, Vec2};
use mot_core::capture::CaptureTarget;
use mot_core::model_library::TrainerCapturePreset;
use truce::prelude::*;
use truce_egui::EditorUi;

use crate::{
    MotTrainerParams, P, RetrainModel, ScanTrainerModelsTask, TrainerStatus, TrainingSnapshot,
    VERSION, read_lock_string, write_lock_string,
};

pub const WINDOW_SIZE: (u32, u32) = (720, 480);
const EDITOR_STATE_ID: &str = "mot_trainer_editor_state_v2";
const EDITOR_MARGIN: f32 = 22.0;
const PANEL_MARGIN: f32 = 20.0;
const SECTION_GAP: f32 = 16.0;
const ROW_GAP: f32 = 10.0;

#[derive(Clone, Debug, Default)]
struct EditorState {
    initialized: bool,
    models: Vec<RetrainModel>,
    selected_model_id: Option<String>,
    model_message: Option<String>,
    last_saved_model_id: Option<String>,
}

pub struct MotTrainerUi;

impl EditorUi<MotTrainerParams> for MotTrainerUi {
    fn ui(&mut self, ui: &mut egui::Ui, context: &PluginContext<MotTrainerParams>) {
        ui.ctx().request_repaint_after(Duration::from_millis(33));

        let state_id = egui::Id::new(EDITOR_STATE_ID);
        let mut state = ui
            .ctx()
            .data_mut(|data| data.get_temp::<EditorState>(state_id).unwrap_or_default());
        service_model_catalog(&mut state, context);

        let background = Color32::from_rgb(10, 12, 14);
        let panel = Color32::from_rgb(20, 24, 28);
        let accent = Color32::from_rgb(58, 220, 210);
        let text = Color32::from_rgb(228, 235, 238);
        let dim = Color32::from_rgb(135, 148, 155);
        let editor_rect = ui.max_rect();
        let content_width = (editor_rect.width() - EDITOR_MARGIN * 2.0).max(0.0);
        ui.visuals_mut().panel_fill = background;
        ui.visuals_mut().override_text_color = Some(text);
        ui.painter().rect_filled(editor_rect, 0.0, background);

        egui::Frame::new()
            .fill(background)
            .inner_margin(EDITOR_MARGIN)
            .show(ui, |ui| {
                ui.set_min_width(content_width);
                header(ui, context, background, accent, text, dim);
                ui.add_space(SECTION_GAP);

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_min_width(content_width);
                        settings_panel(ui, context, &mut state, content_width, panel, dim);
                        ui.add_space(SECTION_GAP);
                        metadata_editor(ui, context, dim);
                        ui.add_space(SECTION_GAP);
                        status_panel(
                            ui,
                            context,
                            content_width,
                            background,
                            panel,
                            accent,
                            text,
                            dim,
                        );
                    });
            });

        ui.ctx().data_mut(|data| data.insert_temp(state_id, state));
    }
}

fn header(
    ui: &mut egui::Ui,
    context: &PluginContext<MotTrainerParams>,
    background: Color32,
    accent: Color32,
    text: Color32,
    dim: Color32,
) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("MOT TRAINER").size(27.0).color(accent));
        ui.label(
            RichText::new(format!("{VERSION}  •  MONO  •  48 kHz"))
                .monospace()
                .color(dim),
        );
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let monitoring = context.monitor.value();
            let button = egui::Button::new(RichText::new("MONITOR").color(if monitoring {
                background
            } else {
                text
            }))
            .fill(if monitoring {
                accent
            } else {
                Color32::from_rgb(42, 48, 52)
            });
            if ui.add(button).clicked() {
                context.automate(P::Monitor, if monitoring { 0.0 } else { 1.0 });
            }
        });
    });
}

fn settings_panel(
    ui: &mut egui::Ui,
    context: &PluginContext<MotTrainerParams>,
    state: &mut EditorState,
    content_width: f32,
    panel: Color32,
    dim: Color32,
) {
    egui::Frame::new()
        .fill(panel)
        .stroke(Stroke::new(1.0_f32, Color32::from_rgb(44, 52, 58)))
        .corner_radius(10.0)
        .inner_margin(PANEL_MARGIN)
        .show(ui, |ui| {
            ui.set_min_width((content_width - PANEL_MARGIN * 2.0).max(0.0));
            egui::Grid::new("trainer_settings")
                .num_columns(2)
                .spacing([18.0, ROW_GAP])
                .show(ui, |ui| {
                    model_editor(ui, context, state, dim);

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
        });
}

fn model_editor(
    ui: &mut egui::Ui,
    context: &PluginContext<MotTrainerParams>,
    state: &mut EditorState,
    dim: Color32,
) {
    ui.label(RichText::new("MODEL").color(dim));
    let editing_enabled = model_editing_enabled(context.control.status());
    ui.horizontal(|ui| {
        let mut edited = read_lock_string(&context.model_name);
        if ui
            .add_enabled(
                editing_enabled,
                egui::TextEdit::singleline(&mut edited)
                    .hint_text("Captured Amp")
                    .desired_width(390.0),
            )
            .changed()
        {
            write_lock_string(&context.model_name, &edited);
            state.selected_model_id = None;
        }

        let picker_enabled = editing_enabled
            && !state.models.is_empty()
            && !context.model_control.is_busy();
        let picker = ui
            .add_enabled_ui(picker_enabled, |ui| {
                ui.spacing_mut().button_padding = Vec2::new(7.0, 3.0);
                ui.menu_button(RichText::new("▾").size(15.0), |ui| {
                    for model in state.models.clone() {
                        let selected =
                            state.selected_model_id.as_deref() == Some(model.model_id.as_str());
                        let response = ui
                            .selectable_label(selected, &model.display_name)
                            .on_hover_text(&model.model_id);
                        if response.clicked() {
                            apply_retrain_model(context, &model);
                            state.selected_model_id = Some(model.model_id.clone());
                            state.model_message = model.metadata_error.clone().or_else(|| {
                                model.capture.is_none().then(|| {
                                    "Capture metadata is unavailable; only the model name was loaded"
                                        .to_owned()
                                })
                            });
                            ui.close();
                        }
                    }
                })
            })
            .inner;
        if let Some(message) = &state.model_message {
            picker.response.on_hover_text(message);
        } else if state.models.is_empty() {
            picker.response.on_hover_text("No existing models");
        }
    });
    ui.end_row();
}

fn metadata_editor(ui: &mut egui::Ui, context: &PluginContext<MotTrainerParams>, dim: Color32) {
    egui::CollapsingHeader::new("CAPTURE METADATA")
        .default_open(false)
        .show(ui, |ui| {
            egui::Grid::new("trainer_metadata")
                .num_columns(2)
                .spacing([18.0, ROW_GAP])
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
                    text_field(ui, "INTERFACE IN", &context.interface_input, "Input 1", dim);
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
}

#[allow(clippy::too_many_arguments)]
fn status_panel(
    ui: &mut egui::Ui,
    context: &PluginContext<MotTrainerParams>,
    content_width: f32,
    background: Color32,
    panel: Color32,
    accent: Color32,
    text: Color32,
    dim: Color32,
) {
    let status = context.control.status();
    let capture_progress = context.get_meter(P::CaptureProgress).clamp(0.0, 1.0);
    let peak = context.get_meter(P::ReturnPeak).clamp(0.0, 1.0);
    let training = context.control.training_snapshot();

    egui::Frame::new()
        .fill(panel)
        .stroke(Stroke::new(1.0_f32, Color32::from_rgb(44, 52, 58)))
        .corner_radius(10.0)
        .inner_margin(PANEL_MARGIN)
        .show(ui, |ui| {
            ui.set_min_width((content_width - PANEL_MARGIN * 2.0).max(0.0));
            ui.horizontal(|ui| {
                ui.label(RichText::new("STATUS").color(dim));
                ui.label(
                    RichText::new(status.label())
                        .monospace()
                        .strong()
                        .color(status.color(accent)),
                );
            });

            if capture_details_visible(status, capture_progress, peak) {
                ui.add_space(ROW_GAP);
                ui.add(
                    egui::ProgressBar::new(capture_progress)
                        .text(format!("CAPTURE {:.1}%", capture_progress * 100.0))
                        .fill(Color32::from_rgb(36, 142, 132)),
                );

                ui.add_space(ROW_GAP);
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
            }

            if training_details_visible(status, &training) {
                ui.add_space(ROW_GAP);
                ui.add(
                    egui::ProgressBar::new(training.progress)
                        .text(format!(
                            "TRAINING PASS {} / {}  •  BEST ESR {:.4}",
                            training.epoch, training.maximum_epochs, training.best_esr
                        ))
                        .fill(Color32::from_rgb(52, 115, 164)),
                );
                ui.add_space(ROW_GAP);
                ui.label(
                    RichText::new(training_detail_text(&training))
                        .small()
                        .monospace()
                        .color(dim),
                );
            }

            ui.add_space(SECTION_GAP);
            action_button(ui, context, status, background, accent, text);

            let error = context.control.last_error();
            if !error.is_empty() {
                ui.add_space(ROW_GAP);
                ui.label(
                    RichText::new(error)
                        .small()
                        .color(Color32::from_rgb(235, 95, 95)),
                );
            }
            if let Some(model) = context.control.last_saved_model() {
                ui.add_space(ROW_GAP);
                ui.label(
                    RichText::new(format!("SAVED MODEL: {}", model.model_id))
                        .monospace()
                        .color(accent),
                );
            }
        });
}

fn action_button(
    ui: &mut egui::Ui,
    context: &PluginContext<MotTrainerParams>,
    status: TrainerStatus,
    background: Color32,
    accent: Color32,
    text: Color32,
) {
    enum Action {
        Arm,
        Disarm,
        CancelTraining,
        None,
    }

    let (label, enabled, fill, label_color, action) = match status {
        TrainerStatus::Ready
        | TrainerStatus::Invalid
        | TrainerStatus::ModelSaved
        | TrainerStatus::Error => (
            "ARM",
            true,
            Color32::from_rgb(66, 68, 70),
            text,
            Action::Arm,
        ),
        TrainerStatus::Armed | TrainerStatus::Waiting => {
            ("ARMED", true, accent, background, Action::Disarm)
        }
        TrainerStatus::Recording => ("ARMED", false, accent, background, Action::None),
        TrainerStatus::Captured => (
            "CANCEL TRAINING",
            false,
            Color32::from_rgb(148, 48, 55),
            text,
            Action::None,
        ),
        TrainerStatus::Aligning | TrainerStatus::Training => (
            "CANCEL TRAINING",
            true,
            Color32::from_rgb(148, 48, 55),
            text,
            Action::CancelTraining,
        ),
        TrainerStatus::Loading => (
            "ARM",
            false,
            Color32::from_rgb(66, 68, 70),
            text,
            Action::None,
        ),
    };
    let button = egui::Button::new((
        egui::Atom::grow(),
        RichText::new(label).strong().color(label_color),
        egui::Atom::grow(),
    ))
    .min_size(Vec2::new(170.0, 36.0))
    .fill(fill);
    if ui.add_enabled(enabled, button).clicked() {
        match action {
            Action::Arm => {
                context.arm_generation.fetch_add(1, Ordering::AcqRel);
                context.control.clear_error();
            }
            Action::Disarm => {
                context.arm_generation.fetch_add(1, Ordering::AcqRel);
            }
            Action::CancelTraining => context.control.cancel_training(),
            Action::None => {}
        }
    }
}

fn model_editing_enabled(status: TrainerStatus) -> bool {
    !matches!(
        status,
        TrainerStatus::Armed
            | TrainerStatus::Waiting
            | TrainerStatus::Recording
            | TrainerStatus::Captured
            | TrainerStatus::Aligning
            | TrainerStatus::Training
    )
}

fn capture_details_visible(status: TrainerStatus, progress: f32, peak: f32) -> bool {
    progress > 0.0
        || peak > 0.0
        || matches!(
            status,
            TrainerStatus::Armed
                | TrainerStatus::Waiting
                | TrainerStatus::Recording
                | TrainerStatus::Captured
                | TrainerStatus::Aligning
                | TrainerStatus::Training
                | TrainerStatus::ModelSaved
        )
}

fn training_details_visible(status: TrainerStatus, training: &TrainingSnapshot) -> bool {
    training.epoch > 0
        || matches!(
            status,
            TrainerStatus::Aligning | TrainerStatus::Training | TrainerStatus::ModelSaved
        )
}

fn training_detail_text(training: &TrainingSnapshot) -> String {
    let mut parts = Vec::with_capacity(4);
    if !training.device.is_empty() {
        parts.push(training.device.clone());
    }
    parts.push(format!(
        "elapsed {}",
        format_duration(training.elapsed_seconds)
    ));
    if training.epoch_seconds > 0.0 {
        parts.push(format!("pass {}", format_duration(training.epoch_seconds)));
    }
    let eta = estimated_training_remaining_seconds(training)
        .map_or_else(|| "—".to_owned(), format_duration);
    parts.push(format!("ETA {eta}"));
    parts.join("  •  ")
}

fn estimated_training_remaining_seconds(training: &TrainingSnapshot) -> Option<f64> {
    if training.epoch == 0
        || training.elapsed_seconds <= 0.0
        || !training.elapsed_seconds.is_finite()
    {
        return None;
    }
    let remaining = training.maximum_epochs.saturating_sub(training.epoch);
    let average_pass_seconds = training.elapsed_seconds / f64::from(training.epoch);
    Some(average_pass_seconds * f64::from(remaining))
}

fn format_duration(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "—".to_owned();
    }
    let total = seconds.round() as u64;
    let hours = total / 3_600;
    let minutes = (total % 3_600) / 60;
    let seconds = total % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

fn service_model_catalog(state: &mut EditorState, context: &PluginContext<MotTrainerParams>) {
    if !state.initialized {
        state.initialized = true;
        request_model_catalog(state, context);
    }
    while let Some(outcome) = context.model_control.take_outcome() {
        match outcome {
            Ok(models) => {
                state.models = models;
                state.model_message = None;
            }
            Err(error) => state.model_message = Some(error),
        }
    }

    let saved_model_id = context
        .control
        .last_saved_model()
        .map(|model| model.model_id);
    if saved_model_id.is_some()
        && saved_model_id != state.last_saved_model_id
        && !context.model_control.is_busy()
    {
        state.last_saved_model_id = saved_model_id;
        request_model_catalog(state, context);
    }
}

fn request_model_catalog(state: &mut EditorState, context: &PluginContext<MotTrainerParams>) {
    if !context.model_control.try_begin() {
        return;
    }
    let Some(spawner) = context.tasks::<ScanTrainerModelsTask>() else {
        context.model_control.cancel_begin();
        state.model_message = Some("Model-library worker is unavailable".to_owned());
        return;
    };
    if spawner.try_spawn(ScanTrainerModelsTask).is_err() {
        context.model_control.cancel_begin();
        state.model_message = Some("Model-library worker queue is full".to_owned());
    }
}

fn apply_retrain_model(context: &PluginContext<MotTrainerParams>, model: &RetrainModel) {
    write_lock_string(&context.model_name, &model.display_name);
    if let Some(capture) = &model.capture {
        apply_capture_preset(context, capture);
    }
}

fn apply_capture_preset(context: &PluginContext<MotTrainerParams>, capture: &TrainerCapturePreset) {
    context.automate(
        P::Target,
        match capture.target {
            CaptureTarget::SoftwarePluginChain => 0.0,
            CaptureTarget::FullAmpUnfilteredLoad => 1.0,
        },
    );
    write_lock_string(&context.amplifier, &capture.amplifier);
    write_lock_string(&context.amplifier_channel, &capture.amplifier_channel);
    write_lock_string(&context.control_positions, &capture.control_positions);
    write_lock_string(&context.interface_output, &capture.interface_output);
    write_lock_string(&context.interface_input, &capture.interface_input);
    write_lock_string(&context.reamp_box, &capture.reamp_box);
    write_lock_string(&context.reactive_load, &capture.reactive_load);
    context
        .load_impedance_ohms
        .store(capture.load_impedance_ohms.map_or(0, u64::from));
    write_lock_string(&context.return_gain_note, &capture.return_gain_note);
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
                .desired_width(390.0),
        )
        .changed()
    {
        write_lock_string(value, &edited);
    }
    ui.end_row();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eta_uses_the_average_completed_pass_duration() {
        let snapshot = TrainingSnapshot {
            epoch: 4,
            maximum_epochs: 100,
            elapsed_seconds: 135.0,
            epoch_seconds: 33.0,
            ..TrainingSnapshot::default()
        };
        assert_eq!(
            estimated_training_remaining_seconds(&snapshot),
            Some(3_240.0)
        );
        assert_eq!(
            training_detail_text(&snapshot),
            "elapsed 2:15  •  pass 0:33  •  ETA 54:00"
        );
    }

    #[test]
    fn duration_format_is_compact_and_stable() {
        assert_eq!(format_duration(0.0), "0:00");
        assert_eq!(format_duration(65.4), "1:05");
        assert_eq!(format_duration(3_661.0), "1:01:01");
    }
}
