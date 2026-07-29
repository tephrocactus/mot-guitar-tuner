use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;
use std::time::Duration;

use egui::{Align, Color32, Layout, RichText, Vec2};
use mot_core::capture::CaptureTarget;
use mot_core::capture_asset::{DAW_REFERENCE_ASSET_FILENAME, daw_reference_asset_path};
use mot_core::model_library::{ModelLibrary, TrainerCapturePreset};
use mot_ui::{
    ACCENT, BACKGROUND, DIM, ERROR, SUCCESS, TEXT, WAITING, accent_button, background_frame,
    danger_button, field_label, ghost_button, panel, raised_panel, section_label,
    selected_model_button, status_text,
};
use truce::prelude::*;
use truce_egui::EditorUi;

use crate::{
    MotTrainerParams, P, RetrainModel, ScanTrainerModelsTask, TrainerStatus, TrainingSnapshot,
    VERSION, read_lock_string, write_lock_string,
};

pub const WINDOW_SIZE: (u32, u32) = (720, 480);
const EDITOR_STATE_ID: &str = "mot_trainer_editor_state_v3";
const MODEL_BROWSER_WIDTH: f32 = 216.0;
const COLUMN_GAP: f32 = 10.0;
const CARD_HEIGHT: f32 = 48.0;
const CONFIG_FIELD_WIDTH: f32 = 280.0;

#[derive(Clone, Debug, Default)]
struct EditorState {
    initialized: bool,
    models: Vec<RetrainModel>,
    selected_model_id: Option<String>,
    model_message: Option<String>,
    reference_message: Option<String>,
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

        mot_ui::apply(ui);
        let editor_rect = ui.max_rect();
        let content_width = (editor_rect.width() - mot_ui::OUTER_MARGIN * 2.0).max(0.0);
        ui.painter().rect_filled(editor_rect, 0.0, BACKGROUND);
        background_frame().show(ui, |ui| {
            ui.set_min_width(content_width);
            mot_ui::header(ui, "MOT TRAINER", VERSION, false, |ui| {
                monitor_button(ui, context);
            });

            let content_height = ui.available_height();
            let workspace_width =
                (ui.available_width() - MODEL_BROWSER_WIDTH - COLUMN_GAP).max(0.0);
            ui.horizontal_top(|ui| {
                ui.spacing_mut().item_spacing.x = COLUMN_GAP;
                fixed_vertical_panel(ui, Vec2::new(MODEL_BROWSER_WIDTH, content_height), |ui| {
                    model_browser(ui, context, &mut state)
                });
                fixed_vertical_panel(ui, Vec2::new(workspace_width, content_height), |ui| {
                    main_workspace(ui, context, &mut state)
                });
            });
        });

        ui.ctx().data_mut(|data| data.insert_temp(state_id, state));
    }
}

fn fixed_vertical_panel<R>(
    ui: &mut egui::Ui,
    size: Vec2,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::InnerResponse<R> {
    ui.allocate_ui_with_layout(size, Layout::top_down(Align::Min), |ui| {
        let panel_rect = ui.max_rect();
        ui.set_width(size.x);
        ui.shrink_clip_rect(panel_rect);
        add_contents(ui)
    })
}

fn monitor_button(ui: &mut egui::Ui, context: &PluginContext<MotTrainerParams>) {
    let available = monitor_control_available(context.control.status());
    let requested = context.monitor.value();
    if !available && requested {
        context.automate(P::Monitor, 0.0);
    }
    let monitoring = available && requested;
    let label = RichText::new("MONITOR")
        .strong()
        .color(if monitoring { BACKGROUND } else { TEXT });
    let button = if monitoring {
        accent_button(label)
    } else {
        ghost_button(label)
    };
    if ui.add_enabled(available, button).clicked() {
        context.automate(P::Monitor, if monitoring { 0.0 } else { 1.0 });
    }
}

fn model_browser(
    ui: &mut egui::Ui,
    context: &PluginContext<MotTrainerParams>,
    state: &mut EditorState,
) {
    let height = ui.available_height();
    panel().show(ui, |ui| {
        ui.set_min_height((height - 34.0).max(0.0));
        ui.horizontal(|ui| {
            ui.label(section_label("MODELS"));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(
                    RichText::new(state.models.len().to_string())
                        .monospace()
                        .color(DIM),
                );
            });
        });

        let editing_enabled = model_editing_enabled(context.control.status());
        let new_selected = state.selected_model_id.is_none();
        let new_label = RichText::new("+ NEW MODEL")
            .strong()
            .color(if new_selected { ACCENT } else { TEXT });
        let new_button = if new_selected {
            selected_model_button(new_label)
        } else {
            ghost_button(new_label)
        }
        .min_size(Vec2::new(ui.available_width(), 36.0));
        if ui.add_enabled(editing_enabled, new_button).clicked() {
            state.selected_model_id = None;
            state.model_message = None;
            write_lock_string(&context.model_name, "");
        }

        let controls_height = 100.0;
        let list_height = (ui.available_height() - controls_height).max(96.0);
        let models = state.models.clone();
        egui::ScrollArea::vertical()
            .id_salt("mot_trainer_model_browser")
            .max_height(list_height)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if models.is_empty() {
                    ui.label(
                        RichText::new(if context.model_control.is_busy() {
                            "Scanning model library…"
                        } else {
                            "No models yet"
                        })
                        .italics()
                        .color(DIM),
                    );
                    ui.label(
                        RichText::new("Capture a new tone to create the first model.")
                            .small()
                            .color(DIM),
                    );
                }

                for model in models {
                    model_card(ui, context, state, model, editing_enabled);
                }
            });

        ui.separator();
        let busy = context.model_control.is_busy();
        if ui
            .add_enabled(
                !busy,
                browser_action_button("REFRESH", ui.available_width()),
            )
            .clicked()
        {
            request_model_catalog(state, context);
        }
        if ui
            .add(browser_action_button("OPEN FOLDER", ui.available_width()))
            .clicked()
        {
            state.model_message = open_models_folder().err();
        }
        if busy {
            ui.label(RichText::new("BUSY").small().monospace().color(DIM));
        }

        if let Some(message) = &state.model_message {
            ui.label(
                RichText::new(message)
                    .small()
                    .color(Color32::from_rgb(235, 164, 77)),
            );
        }
    });
}

fn browser_action_button(label: &'static str, width: f32) -> egui::Button<'static> {
    egui::Button::new(RichText::new(label).small())
        .corner_radius(mot_ui::CONTROL_RADIUS)
        .min_size(Vec2::new(width, 28.0))
}

fn model_card(
    ui: &mut egui::Ui,
    context: &PluginContext<MotTrainerParams>,
    state: &mut EditorState,
    model: RetrainModel,
    editing_enabled: bool,
) {
    let selected = state.selected_model_id.as_deref() == Some(model.model_id.as_str());
    let source_label = if model.capture.is_some() {
        "RETRAIN READY"
    } else {
        "MODEL ONLY"
    };
    let label = format!("{}\n{}", model.display_name, source_label);
    let text = RichText::new(label).color(if selected { ACCENT } else { TEXT });
    let button = if selected {
        selected_model_button(text)
    } else {
        ghost_button(text)
    }
    .min_size(Vec2::new(ui.available_width(), CARD_HEIGHT));

    let response = ui.add_enabled(editing_enabled, button);
    let response = if let Some(error) = &model.metadata_error {
        response.on_hover_text(format!("{}\n{error}", model.model_id))
    } else if model.capture.is_none() {
        response.on_hover_text(format!(
            "{}\nCapture metadata is unavailable; only the model name can be reused",
            model.model_id
        ))
    } else {
        response.on_hover_text(format!(
            "{}\nLoad this model's capture metadata for retraining",
            model.model_id
        ))
    };
    if response.clicked() && !selected {
        apply_retrain_model(context, &model);
        state.selected_model_id = Some(model.model_id.clone());
        state.model_message = model.metadata_error.clone().or_else(|| {
            model.capture.is_none().then(|| {
                "Capture metadata is unavailable; only the model name was loaded".to_owned()
            })
        });
    }
}

fn open_models_folder() -> Result<(), String> {
    let library = ModelLibrary::for_current_user().map_err(|error| error.to_string())?;
    library
        .ensure_directories()
        .map_err(|error| error.to_string())?;
    let status = Command::new("/usr/bin/open")
        .arg(&library.paths().models)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|error| format!("Could not open model folder: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("Could not open model folder: {status}"))
    }
}

fn show_reference_wav_in_finder() -> Result<(), String> {
    let path = daw_reference_asset_path()?;
    if path.is_file() {
        let status = Command::new("/usr/bin/open")
            .arg("-R")
            .arg(&path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|error| format!("Could not reveal the reference WAV: {error}"))?;
        if status.success() {
            return Ok(());
        }
    }

    let folder = path
        .parent()
        .ok_or_else(|| "Could not locate the reference WAV folder".to_owned())?;
    let status = Command::new("/usr/bin/open")
        .arg(folder)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|error| format!("Could not open the reference WAV folder: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "Could not reveal the reference WAV or open its folder: {status}"
        ))
    }
}

fn main_workspace(
    ui: &mut egui::Ui,
    context: &PluginContext<MotTrainerParams>,
    state: &mut EditorState,
) {
    egui::ScrollArea::vertical()
        .id_salt("mot_trainer_workspace")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            model_configuration(ui, context, state);
            add_vertical_gap(ui, mot_ui::SECTION_GAP);
            status_panel(ui, context);
            add_vertical_gap(ui, mot_ui::SECTION_GAP);
            metadata_editor(ui, context);
        });
}

fn add_vertical_gap(ui: &mut egui::Ui, total: f32) {
    ui.add_space((total - ui.spacing().item_spacing.y).max(0.0));
}

fn full_width_panel<R>(
    ui: &mut egui::Ui,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::InnerResponse<R> {
    let content_width = (ui.available_width() - mot_ui::PANEL_MARGIN * 2.0).max(0.0);
    panel().show(ui, |ui| {
        ui.set_min_width(content_width);
        add_contents(ui)
    })
}

fn model_configuration(
    ui: &mut egui::Ui,
    context: &PluginContext<MotTrainerParams>,
    state: &mut EditorState,
) {
    let editing_enabled = model_editing_enabled(context.control.status());
    full_width_panel(ui, |ui| {
        ui.label(section_label("MODEL SETUP"));
        egui::Grid::new("mot_trainer_model_setup")
            .num_columns(2)
            .spacing([22.0, mot_ui::ROW_GAP])
            .show(ui, |ui| {
                ui.label(field_label("NAME"));
                let mut edited = read_lock_string(&context.model_name);
                if ui
                    .add_enabled(
                        editing_enabled,
                        egui::TextEdit::singleline(&mut edited)
                            .hint_text("Captured Amp")
                            .desired_width(CONFIG_FIELD_WIDTH),
                    )
                    .changed()
                {
                    write_lock_string(&context.model_name, &edited);
                    state.selected_model_id = None;
                }
                ui.end_row();

                ui.label(field_label("REFERENCE WAV"));
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(DAW_REFERENCE_ASSET_FILENAME)
                            .monospace()
                            .color(DIM),
                    );
                    let response = ui.add(egui::Link::new(
                        RichText::new("SHOW IN FINDER").small().color(ACCENT),
                    ));
                    if response
                        .on_hover_text("Reveal the canonical 48 kHz mono reference WAV in Finder")
                        .clicked()
                    {
                        state.reference_message = show_reference_wav_in_finder().err();
                    }
                });
                ui.end_row();

                ui.label(field_label("MAX PASSES"));
                let mut epochs = context.max_epochs.value_i32();
                if ui
                    .add_enabled(
                        editing_enabled,
                        egui::DragValue::new(&mut epochs)
                            .range(1..=400)
                            .speed(1)
                            .suffix(" passes"),
                    )
                    .changed()
                {
                    context.automate(P::MaxEpochs, f64::from(epochs - 1) / 399.0);
                }
                ui.end_row();
            });

        if let Some(message) = &state.reference_message {
            ui.label(RichText::new(message).small().color(WAITING));
        }

        if let Some(selected) = state.selected_model_id.as_deref() {
            ui.label(
                RichText::new(format!("RETRAIN SOURCE  •  {selected}"))
                    .small()
                    .monospace()
                    .color(DIM),
            );
        }
    });
}

fn status_panel(ui: &mut egui::Ui, context: &PluginContext<MotTrainerParams>) {
    let status = context.control.status();
    let capture_progress = context.get_meter(P::CaptureProgress).clamp(0.0, 1.0);
    let training = context.control.training_snapshot();

    full_width_panel(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(field_label("STATUS"));
            ui.label(status_text(status.label(), status_color(status)));
        });
        ui.label(RichText::new(status_guidance(status)).small().color(DIM));

        if capture_details_visible(status, capture_progress) {
            mot_ui::progress(
                ui,
                capture_progress,
                format!("RECORDED {:.1}%", capture_progress * 100.0),
            );
        }

        if training_details_visible(status, &training) {
            let progress_text = if training.epoch == 0 {
                "PREPARING TRAINING".to_owned()
            } else {
                format!(
                    "PASS {} / {}  •  BEST ESR {:.4}",
                    training.epoch, training.maximum_epochs, training.best_esr
                )
            };
            mot_ui::progress(ui, training.progress, progress_text);
            ui.label(
                RichText::new(training_detail_text(&training))
                    .small()
                    .monospace()
                    .color(DIM),
            );
        }

        let error = context.control.last_error();
        if !error.is_empty() {
            raised_panel().show(ui, |ui| {
                ui.label(RichText::new(error).small().color(ERROR));
            });
        } else if let Some(model) = context.control.last_saved_model() {
            ui.label(
                RichText::new(format!("SAVED  •  {}", model.model_id))
                    .small()
                    .monospace()
                    .color(SUCCESS),
            );
        }

        action_button(ui, context, status);
    });
}

fn status_guidance(status: TrainerStatus) -> &'static str {
    match status {
        TrainerStatus::Loading => "Preparing the capture recorder.",
        TrainerStatus::Ready => {
            "Render the reference through the target chain, then arm Trainer on the aligned render."
        }
        TrainerStatus::Armed | TrainerStatus::Waiting => {
            "Waiting for Play. Stop and restart transport if it was already running."
        }
        TrainerStatus::Recording => "Recording the processed return signal.",
        TrainerStatus::Captured => "Capture complete. Preparing alignment and training.",
        TrainerStatus::Aligning => "Aligning the returned signal to the emitted reference.",
        TrainerStatus::Training => "Training the model and retaining the best validation pass.",
        TrainerStatus::ModelSaved => "The model is ready in the shared model library.",
        TrainerStatus::Invalid => "Capture was interrupted. Arm again for a fresh take.",
        TrainerStatus::Error => "Review the message below, then arm again when resolved.",
    }
}

#[derive(Clone, Copy)]
enum Action {
    Arm,
    Disarm,
    CancelTraining,
    None,
}

#[derive(Clone, Copy)]
enum ActionStyle {
    Neutral,
    Active,
    Danger,
}

fn action_button(
    ui: &mut egui::Ui,
    context: &PluginContext<MotTrainerParams>,
    status: TrainerStatus,
) {
    let (label, enabled, style, action) = match status {
        TrainerStatus::Ready
        | TrainerStatus::Invalid
        | TrainerStatus::ModelSaved
        | TrainerStatus::Error => ("ARM", true, ActionStyle::Neutral, Action::Arm),
        TrainerStatus::Armed | TrainerStatus::Waiting => {
            ("ARMED", true, ActionStyle::Active, Action::Disarm)
        }
        TrainerStatus::Recording => ("ARMED", false, ActionStyle::Active, Action::None),
        TrainerStatus::Captured => ("CANCEL TRAINING", false, ActionStyle::Danger, Action::None),
        TrainerStatus::Aligning | TrainerStatus::Training => (
            "CANCEL TRAINING",
            true,
            ActionStyle::Danger,
            Action::CancelTraining,
        ),
        TrainerStatus::Loading => ("ARM", false, ActionStyle::Neutral, Action::None),
    };

    let label =
        RichText::new(label)
            .size(15.0)
            .strong()
            .color(if matches!(style, ActionStyle::Active) {
                BACKGROUND
            } else {
                TEXT
            });
    let button = match style {
        ActionStyle::Neutral => ghost_button(label),
        ActionStyle::Active => accent_button(label),
        ActionStyle::Danger => danger_button(label),
    }
    .min_size(Vec2::new(ui.available_width(), 40.0));

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

fn metadata_editor(ui: &mut egui::Ui, context: &PluginContext<MotTrainerParams>) {
    let editing_enabled = model_editing_enabled(context.control.status());
    egui::CollapsingHeader::new(section_label("CAPTURE METADATA"))
        .default_open(false)
        .show(ui, |ui| {
            raised_panel().show(ui, |ui| {
                ui.add_enabled_ui(editing_enabled, |ui| {
                    egui::Grid::new("mot_trainer_metadata")
                        .num_columns(2)
                        .spacing([22.0, mot_ui::ROW_GAP])
                        .show(ui, |ui| {
                            text_field(ui, "AMPLIFIER", &context.amplifier, "EVH 5153");
                            text_field(ui, "CHANNEL", &context.amplifier_channel, "Blue");
                            text_field(
                                ui,
                                "CONTROLS",
                                &context.control_positions,
                                "Gain 5, Bass 4…",
                            );
                            text_field(
                                ui,
                                "INTERFACE OUT",
                                &context.interface_output,
                                "Line Out 3",
                            );
                            text_field(ui, "INTERFACE IN", &context.interface_input, "Input 1");
                            text_field(ui, "REAMP BOX", &context.reamp_box, "Model");
                            text_field(
                                ui,
                                "REACTIVE LOAD",
                                &context.reactive_load,
                                "Model / raw out",
                            );

                            ui.label(field_label("IMPEDANCE"));
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
                            );
                        });
                });
            });
        });
}

fn text_field(ui: &mut egui::Ui, label: &str, value: &std::sync::RwLock<String>, hint: &str) {
    ui.label(field_label(label));
    let mut edited = read_lock_string(value);
    if ui
        .add(
            egui::TextEdit::singleline(&mut edited)
                .hint_text(hint)
                .desired_width(CONFIG_FIELD_WIDTH),
        )
        .changed()
    {
        write_lock_string(value, &edited);
    }
    ui.end_row();
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

fn monitor_control_available(status: TrainerStatus) -> bool {
    // The DSP exposes the live return only while the recorder is actively
    // consuming audio. Captured buffers are trained offline, not replayed.
    matches!(status, TrainerStatus::Recording)
}

fn capture_details_visible(status: TrainerStatus, progress: f32) -> bool {
    progress > 0.0
        || matches!(
            status,
            TrainerStatus::Recording
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
    let mut parts = Vec::with_capacity(3);
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

fn status_color(status: TrainerStatus) -> Color32 {
    match status {
        TrainerStatus::Error | TrainerStatus::Invalid => ERROR,
        TrainerStatus::ModelSaved => SUCCESS,
        TrainerStatus::Armed | TrainerStatus::Waiting => WAITING,
        TrainerStatus::Recording | TrainerStatus::Captured | TrainerStatus::Aligning => ACCENT,
        TrainerStatus::Training => ACCENT,
        TrainerStatus::Loading => DIM,
        TrainerStatus::Ready => TEXT,
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
