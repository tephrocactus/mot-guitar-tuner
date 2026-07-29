use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use egui::{Align, Color32, FontId, Layout, RichText, Vec2};
use mot_core::model::{ModelRef, Sha256Digest};
use mot_core::model_library::{
    ImportedIr, IrImportMetadata, IrProcessingMode, IrReference, ModelEntry, ModelLibrary,
    ModelScan, ToneSettings,
};
use truce::prelude::*;
use truce_egui::EditorUi;

use mot_ui::{
    ACCENT, DIM, ERROR, KnobSpec, SUCCESS, TEXT, WAITING, accent_button, background_frame,
    danger_button, field_label, ghost_button, panel, parameter_knob, section_label, status_text,
};

use crate::{
    ImportIrTask, IrImportOutcome, LibraryOutcome, LibraryTask, LibraryTaskOperation,
    MotPlayerParams, P, RuntimeUiState, read_shared_string, write_shared_string,
};

pub(crate) const WINDOW_SIZE: (u32, u32) = (1_180, 760);
const EDITOR_STATE_ID: &str = "mot_player_editor_state_v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IrUiMode {
    MinimumPhase,
    Raw,
}

impl IrUiMode {
    fn from_param(value: i64) -> Self {
        if value == 1 {
            Self::Raw
        } else {
            Self::MinimumPhase
        }
    }

    const fn library_mode(self) -> IrProcessingMode {
        match self {
            Self::MinimumPhase => IrProcessingMode::MinimumPhaseAutoTrim,
            Self::Raw => IrProcessingMode::Raw,
        }
    }

    const fn normalized(self) -> f64 {
        match self {
            Self::MinimumPhase => 0.0,
            Self::Raw => 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct ToneView {
    input_gain_db: f32,
    tight_percent: f32,
    bite_percent: f32,
    ir_path: String,
    ir_reference: Option<IrReference>,
    ir_processing: IrUiMode,
}

impl ToneView {
    fn equivalent(&self, other: &Self) -> bool {
        (self.input_gain_db - other.input_gain_db).abs() < 0.001
            && (self.tight_percent - other.tight_percent).abs() < 0.001
            && (self.bite_percent - other.bite_percent).abs() < 0.001
            && self.ir_path == other.ir_path
            && self.ir_reference == other.ir_reference
            && self.ir_processing == other.ir_processing
    }
}

#[derive(Clone, Debug, Default)]
struct EditorState {
    initialized: bool,
    library: Option<ModelLibrary>,
    scan: ModelScan,
    ir_files: Vec<PathBuf>,
    ir_metadata: BTreeMap<PathBuf, IrImportMetadata>,
    baseline: Option<ToneView>,
    pending_model: Option<ModelEntry>,
    pending_auto_select: Option<ModelEntry>,
    refresh_pending: bool,
    switch_after_save: bool,
    message: Option<String>,
    message_after_refresh: Option<String>,
}

pub(crate) struct MotPlayerUi;

impl EditorUi<MotPlayerParams> for MotPlayerUi {
    fn ui(&mut self, ui: &mut egui::Ui, context: &PluginContext<MotPlayerParams>) {
        ui.ctx().request_repaint_after(Duration::from_millis(16));

        let state_id = egui::Id::new(EDITOR_STATE_ID);
        let mut state = ui
            .ctx()
            .data_mut(|data| data.get_temp::<EditorState>(state_id).unwrap_or_default());
        if !state.initialized {
            initialize_editor_state(&mut state, context);
        }
        poll_library_outcomes(&mut state, context);
        poll_ir_import_outcome(&mut state, context);
        service_pending_library_refresh(&mut state, context);
        service_pending_auto_selection(&mut state, context);

        mot_ui::apply(ui);
        ui.painter()
            .rect_filled(ui.max_rect(), 0.0, mot_ui::BACKGROUND);
        background_frame().show(ui, |ui| {
            header(ui, context);
            let content_height = ui.available_height();
            ui.horizontal_top(|ui| {
                fixed_vertical_panel(ui, Vec2::new(310.0, content_height), |ui| {
                    model_browser(ui, context, &mut state);
                });
                fixed_vertical_panel(ui, Vec2::new(ui.available_width(), content_height), |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("mot_player_controls")
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            player_controls(ui, context, &mut state);
                        });
                });
            });
        });

        ui.ctx().data_mut(|data| data.insert_temp(state_id, state));
    }
}

fn initialize_editor_state(state: &mut EditorState, context: &PluginContext<MotPlayerParams>) {
    state.initialized = true;
    context.library_control.invalidate_pending();
    match ModelLibrary::for_current_user() {
        Ok(library) => {
            state.library = Some(library);
            state.refresh_pending = true;
        }
        Err(error) => state.message = Some(error.to_string()),
    }
}

fn service_pending_library_refresh(
    state: &mut EditorState,
    context: &PluginContext<MotPlayerParams>,
) {
    if !state.refresh_pending || context.library_control.is_busy() {
        return;
    }
    match submit_library_task(
        context,
        LibraryTaskOperation::Scan {
            selected_model: selected_model_reference(context),
        },
    ) {
        Ok(()) => {
            state.refresh_pending = false;
            state.message = Some("Refreshing model and IR libraries…".to_owned());
        }
        Err(error) => state.message = Some(error),
    }
}

fn service_pending_auto_selection(
    state: &mut EditorState,
    context: &PluginContext<MotPlayerParams>,
) {
    if context.library_control.is_busy() {
        return;
    }
    let Some(entry) = state.pending_auto_select.take() else {
        return;
    };
    if tone_is_dirty(state, context) {
        let display_name = entry.metadata.display_name.clone();
        state.pending_model = Some(entry);
        state.message = Some(format!(
            "“{display_name}” is imported. Save or discard the current model settings to select it."
        ));
    } else if let Err(error) = request_select_model(state, context, entry.clone()) {
        state.pending_auto_select = Some(entry);
        state.message = Some(error);
    }
}

fn submit_library_task(
    context: &PluginContext<MotPlayerParams>,
    operation: LibraryTaskOperation,
) -> Result<(), String> {
    let request_id = context
        .library_control
        .try_begin()
        .ok_or_else(|| "A model-library operation is already running".to_owned())?;
    let Some(spawner) = context.tasks::<LibraryTask>() else {
        context.library_control.cancel_begin(request_id);
        return Err("Model-library worker is unavailable in this host".to_owned());
    };
    if spawner
        .try_spawn(LibraryTask {
            request_id,
            operation,
        })
        .is_err()
    {
        context.library_control.cancel_begin(request_id);
        return Err("Model-library worker queue is full".to_owned());
    }
    Ok(())
}

fn poll_library_outcomes(state: &mut EditorState, context: &PluginContext<MotPlayerParams>) {
    while let Some(outcome) = context.library_control.take_outcome() {
        if !context.library_control.is_current(outcome.request_id()) {
            continue;
        }
        match outcome {
            LibraryOutcome::Scanned { result, .. } => {
                apply_library_scan_outcome(state, context, result);
            }
            LibraryOutcome::ToneLoaded {
                entry,
                guard_model_id,
                guard_model_sha256,
                result,
                ..
            } => {
                if selected_model_identity_matches(context, &guard_model_id, &guard_model_sha256) {
                    apply_loaded_model(state, context, *entry, result);
                }
            }
            LibraryOutcome::ToneSaved {
                settings, result, ..
            } => apply_saved_tone_outcome(state, context, settings, result),
            LibraryOutcome::NamImported { result, .. } => {
                apply_nam_import_outcome(state, context, result);
            }
            LibraryOutcome::FolderOpened { result, .. } => match result {
                Ok(()) => state.message = None,
                Err(error) => state.message = Some(error),
            },
        }
    }
}

fn apply_library_scan_outcome(
    state: &mut EditorState,
    context: &PluginContext<MotPlayerParams>,
    result: Result<Box<crate::LibrarySnapshot>, String>,
) {
    let snapshot = match result {
        Ok(snapshot) => snapshot,
        Err(error) => {
            state.scan = ModelScan::default();
            state.ir_files.clear();
            state.ir_metadata.clear();
            state.baseline = None;
            state.message = Some(error);
            return;
        }
    };

    let mut first_error = None;
    match snapshot.models {
        Ok(scan) => state.scan = scan,
        Err(error) => {
            state.scan = ModelScan::default();
            first_error = Some(error);
        }
    }
    match snapshot.irs {
        Ok(scan) => {
            state.ir_files.clear();
            state.ir_metadata.clear();
            for entry in scan.entries {
                state.ir_metadata.insert(entry.path.clone(), entry.metadata);
                state.ir_files.push(entry.path);
            }
            if let Some(issue) = scan.issues.first() {
                first_error.get_or_insert_with(|| {
                    format!("{}: {}", display_filename(&issue.path), issue.message)
                });
            }
            sort_ir_files(state);
        }
        Err(error) => {
            state.ir_files.clear();
            state.ir_metadata.clear();
            first_error.get_or_insert(error);
        }
    }

    state.baseline = None;
    if let Some(selected) = snapshot.selected_tone
        && selected_reference_is_current(context, &selected.reference)
    {
        let tone = match selected.tone {
            Ok(Some(tone)) => Some(tone),
            Ok(None) => Some(ToneSettings::defaults_for(&selected.reference)),
            Err(error) => {
                first_error.get_or_insert(error);
                None
            }
        };
        if let (Some(tone), Some(library)) = (tone, state.library.as_ref()) {
            state.baseline = Some(tone_view_from_settings(&tone, library));
        }
    }
    let message_after_refresh = state.message_after_refresh.take();
    state.message = first_error.or(message_after_refresh);
}

fn apply_nam_import_outcome(
    state: &mut EditorState,
    context: &PluginContext<MotPlayerParams>,
    result: Result<Box<mot_core::model_library::ImportedNam>, String>,
) {
    let imported = match result {
        Ok(imported) => imported,
        Err(error) => {
            state.message = Some(error);
            return;
        }
    };
    let display_name = imported.entry.metadata.display_name.clone();
    let mut message = format!("Imported “{display_name}” from NAM.");
    if let Some(notice) = imported.notice {
        message.push(' ');
        message.push_str(&notice);
    }

    state.refresh_pending = true;
    if tone_is_dirty(state, context) {
        state.pending_model = Some(imported.entry);
        message.push_str(" Save or discard the current model settings to select it.");
    } else {
        state.pending_auto_select = Some(imported.entry);
    }
    state.message = Some(message.clone());
    state.message_after_refresh = Some(message);
}

fn poll_ir_import_outcome(state: &mut EditorState, context: &PluginContext<MotPlayerParams>) {
    while let Some(outcome) = context.ir_import_control.take_outcome() {
        match outcome {
            IrImportOutcome::Imported(imported) => {
                select_imported_ir(context, &imported);
                state
                    .ir_metadata
                    .insert(imported.archived_path.clone(), imported.metadata.clone());
                if !state.ir_files.contains(&imported.archived_path) {
                    state.ir_files.push(imported.archived_path.clone());
                }
                sort_ir_files(state);
                state.refresh_pending = true;
                state.message = Some(format!(
                    "Imported RAW unchanged; default auto-trim: {} samples",
                    imported.metadata.default_trim_leading_samples
                ));
            }
            IrImportOutcome::Error(error) => state.message = Some(error),
        }
    }
}

fn header(ui: &mut egui::Ui, context: &PluginContext<MotPlayerParams>) {
    mot_ui::header(ui, "MOT PLAYER", env!("CARGO_PKG_VERSION"), true, |ui| {
        let muted = context.mute.value();
        let button = if muted {
            danger_button(RichText::new("MUTE").color(TEXT))
        } else {
            ghost_button("MUTE")
        };
        if ui.add(button).clicked() {
            context.automate(P::Mute, if muted { 0.0 } else { 1.0 });
        }
    });
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

fn model_browser(
    ui: &mut egui::Ui,
    context: &PluginContext<MotPlayerParams>,
    state: &mut EditorState,
) {
    let browser_height = ui.available_height();
    let pending_height = if state.pending_model.is_some() {
        116.0
    } else {
        0.0
    };
    panel().show(ui, |ui| {
        ui.set_min_height((browser_height - mot_ui::PANEL_MARGIN * 2.0 - pending_height).max(0.0));
        ui.horizontal(|ui| {
            ui.label(section_label("MODELS"));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(
                    RichText::new(format!("{:02}", state.scan.models.len()))
                        .monospace()
                        .color(DIM),
                );
            });
        });

        let selected_id = read_shared_string(&context.selected_model_id);
        let selected_hash = read_shared_string(&context.selected_model_sha256);
        let models = state.scan.models.clone();
        let list_height = (ui.available_height() - 142.0).max(180.0);
        egui::ScrollArea::vertical()
            .id_salt("mot_player_model_browser")
            .max_height(list_height)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if models.is_empty() {
                    ui.label(
                        RichText::new("No compatible .motmodel files")
                            .italics()
                            .color(DIM),
                    );
                    ui.label(
                        RichText::new(
                            "~/Library/Application Support/Plut&Mot/MOT Guitar Plugin/Models/",
                        )
                        .small()
                        .monospace()
                        .color(DIM),
                    );
                }
                for entry in models {
                    let selected = entry.reference.model_id == selected_id
                        && entry
                            .reference
                            .sha256
                            .to_string()
                            .eq_ignore_ascii_case(&selected_hash);
                    let label = format!(
                        "{}\n{} • {} MAC/smp",
                        entry.metadata.display_name,
                        entry.reference.filename_hint,
                        entry.metadata.estimated_macs_per_sample
                    );
                    let label = RichText::new(label).monospace().color(if selected {
                        ACCENT
                    } else {
                        TEXT
                    });
                    let button = if selected {
                        mot_ui::selected_model_button(label)
                    } else {
                        ghost_button(label)
                    };
                    let response = ui.add_sized([ui.available_width(), 54.0], button);
                    if response.clicked() && !selected {
                        if tone_is_dirty(state, context) {
                            state.pending_model = Some(entry);
                        } else if let Err(error) = request_select_model(state, context, entry) {
                            state.message = Some(error);
                        }
                    }
                }
            });

        let busy = context.library_control.is_busy();
        if ui
            .add_enabled_ui(!busy, |ui| {
                ui.add_sized([ui.available_width(), 30.0], ghost_button("IMPORT NAM…"))
            })
            .inner
            .clicked()
            && let Some(source) = rfd::FileDialog::new()
                .add_filter("Neural Amp Modeler", &["nam", "NAM"])
                .pick_file()
        {
            match submit_library_task(context, LibraryTaskOperation::ImportNam { source }) {
                Ok(()) => {
                    state.message = Some("Validating and converting the NAM model…".to_owned());
                }
                Err(error) => state.message = Some(error),
            }
        }

        ui.columns(2, |columns| {
            if columns[0]
                .add_enabled_ui(!busy, |ui| {
                    ui.add_sized([ui.available_width(), 30.0], ghost_button("REFRESH"))
                })
                .inner
                .clicked()
            {
                state.refresh_pending = true;
            }
            if columns[1]
                .add_enabled_ui(!busy, |ui| {
                    ui.add_sized([ui.available_width(), 30.0], ghost_button("OPEN FOLDER"))
                })
                .inner
                .clicked()
            {
                match submit_library_task(context, LibraryTaskOperation::OpenFolder) {
                    Ok(()) => {
                        state.message = Some("Opening model library folder…".to_owned());
                    }
                    Err(error) => state.message = Some(error),
                }
            }
        });
        if context.library_control.is_busy() {
            ui.label(RichText::new("LIBRARY BUSY").small().monospace().color(DIM));
        }

        if !state.scan.issues.is_empty() {
            egui::CollapsingHeader::new(
                RichText::new(format!("{} scan issue(s)", state.scan.issues.len())).color(WAITING),
            )
            .show(ui, |ui| {
                for issue in &state.scan.issues {
                    ui.label(
                        RichText::new(format!(
                            "{}: {}",
                            display_filename(&issue.path),
                            issue.message
                        ))
                        .small()
                        .color(DIM),
                    );
                }
            });
        }
    });

    if let Some(pending) = state.pending_model.clone() {
        egui::Frame::new()
            .fill(Color32::from_rgb(47, 37, 19))
            .stroke(egui::Stroke::new(1.0_f32, WAITING.linear_multiply(0.55)))
            .corner_radius(7.0)
            .inner_margin(10.0)
            .show(ui, |ui| {
                ui.label(
                    RichText::new("UNSAVED MODEL SETTINGS")
                        .strong()
                        .color(WAITING),
                );
                ui.label(format!("Switch to “{}”?", pending.metadata.display_name));
                ui.horizontal(|ui| {
                    let busy = context.library_control.is_busy();
                    if ui.add_enabled(!busy, accent_button("SAVE")).clicked() {
                        request_save_current_tone(state, context, true);
                    }
                    if ui.add_enabled(!busy, danger_button("DISCARD")).clicked()
                        && let Err(error) = request_select_model(state, context, pending.clone())
                    {
                        state.message = Some(error);
                    }
                    if ui.add_enabled(!busy, ghost_button("CANCEL")).clicked() {
                        state.pending_model = None;
                    }
                });
            });
    }
}

fn player_controls(
    ui: &mut egui::Ui,
    context: &PluginContext<MotPlayerParams>,
    state: &mut EditorState,
) {
    let selected = selected_model_entry(state, context);
    panel().show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(section_label("ACTIVE MODEL"));
                ui.label(
                    RichText::new(selected.as_ref().map_or("NO MODEL — TRANSPARENT", |entry| {
                        entry.metadata.display_name.as_str()
                    }))
                    .font(FontId::proportional(22.0))
                    .strong()
                    .color(if selected.is_some() { ACCENT } else { WAITING }),
                );
            });
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let dirty = tone_is_dirty(state, context);
                ui.label(
                    RichText::new(if dirty { "UNSAVED *" } else { "SAVED" })
                        .monospace()
                        .color(if dirty { WAITING } else { SUCCESS }),
                );
            });
        });

        if let Some(entry) = &selected {
            ui.label(
                RichText::new(format!(
                    "{} v{}  •  causal  •  0 samples  •  {} MAC/sample",
                    entry.metadata.architecture_id,
                    entry.metadata.architecture_version,
                    entry.metadata.estimated_macs_per_sample
                ))
                .small()
                .monospace()
                .color(DIM),
            );
        }

        ui.separator();
        ui.horizontal(|ui| {
            runtime_status(ui, context);
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let has_model = selected.is_some();
                let busy = context.library_control.is_busy();
                if ui
                    .add_enabled(
                        has_model
                            && !busy
                            && state.baseline.is_some()
                            && tone_is_dirty(state, context),
                        ghost_button("REVERT"),
                    )
                    .clicked()
                {
                    revert_current_tone(state, context);
                }
                if ui
                    .add_enabled(has_model && !busy, accent_button("SAVE"))
                    .clicked()
                {
                    request_save_current_tone(state, context, false);
                }
            });
        });
    });

    panel().show(ui, |ui| {
        ui.columns(3, |columns| {
            player_parameter(
                &mut columns[0],
                context,
                PlayerParameter {
                    label: "INPUT GAIN · dB",
                    id: P::InputGain,
                    value: context.input_gain.value(),
                    minimum: -24.0,
                    maximum: 24.0,
                    percent: false,
                },
            );
            player_parameter(
                &mut columns[1],
                context,
                PlayerParameter {
                    label: "TIGHT",
                    id: P::Tight,
                    value: context.tight.value(),
                    minimum: 0.0,
                    maximum: 100.0,
                    percent: true,
                },
            );
            player_parameter(
                &mut columns[2],
                context,
                PlayerParameter {
                    label: "BITE",
                    id: P::Bite,
                    value: context.bite.value(),
                    minimum: 0.0,
                    maximum: 100.0,
                    percent: true,
                },
            );
        });
    });

    ir_section(ui, context, state);
    if let Some(message) = &state.message {
        ui.label(RichText::new(message).small().color(WAITING));
    }
}

fn runtime_status(ui: &mut egui::Ui, context: &PluginContext<MotPlayerParams>) {
    let meter = context.get_meter(P::RuntimeStatus);
    let state = context.runtime_control.get();
    let (label, detail, color) = match state {
        RuntimeUiState::Transparent => (
            "TRANSPARENT",
            "No model selected; input passes through unchanged.".to_owned(),
            DIM,
        ),
        RuntimeUiState::Loading => ("LOADING", "Preparing model and cabinet…".to_owned(), ACCENT),
        RuntimeUiState::Ready {
            model_name,
            ir_name,
        } => (
            "READY",
            ir_name.map_or_else(
                || format!("{model_name} • uncabbed"),
                |ir| format!("{model_name} • {ir}"),
            ),
            SUCCESS,
        ),
        RuntimeUiState::SafeMuted { asset, message } => {
            ("SAFE MUTE", format!("{asset:?}: {message}"), ERROR)
        }
    };
    let visible_label = if meter < 0.25 && label != "SAFE MUTE" {
        "SAFE MUTE"
    } else if (0.25..0.75).contains(&meter) {
        "LOADING"
    } else {
        label
    };
    ui.horizontal_wrapped(|ui| {
        ui.label(field_label("RUNTIME"));
        ui.label(status_text(visible_label, color));
        ui.label(RichText::new(detail).small().monospace().color(color));
    });
    if meter < 0.25
        && !matches!(
            context.runtime_control.get(),
            RuntimeUiState::SafeMuted { .. }
        )
    {
        ui.label(
            RichText::new("MOT PLAYER requires a 48 kHz host session.")
                .small()
                .color(ERROR),
        );
    }
}

struct PlayerParameter<'a> {
    label: &'a str,
    id: P,
    value: f32,
    minimum: f32,
    maximum: f32,
    percent: bool,
}

fn player_parameter(
    ui: &mut egui::Ui,
    context: &PluginContext<MotPlayerParams>,
    parameter: PlayerParameter<'_>,
) {
    ui.vertical_centered(|ui| {
        let mut edited = parameter.value;
        let value_text = if parameter.percent {
            format!("{edited:.0}%")
        } else {
            format!("{edited:+.1}")
        };
        let response = parameter_knob(
            ui,
            &mut edited,
            KnobSpec {
                label: parameter.label,
                value_text: &value_text,
                minimum: parameter.minimum,
                maximum: parameter.maximum,
                default: 0.0,
                step: 0.1,
            },
        );
        if response.changed() {
            automate_linear(
                context,
                parameter.id,
                edited,
                parameter.minimum,
                parameter.maximum,
            );
        }
    });
}

fn ir_section(
    ui: &mut egui::Ui,
    context: &PluginContext<MotPlayerParams>,
    state: &mut EditorState,
) {
    panel().show(ui, |ui| {
        let current_path = read_shared_string(&context.selected_ir_path);
        let selected_name = if current_path.is_empty() {
            "NO IR".to_owned()
        } else {
            ir_display_name(state, Path::new(&current_path))
        };

        ui.horizontal(|ui| {
            ui.label(section_label("CABINET IR"));
            egui::ComboBox::from_id_salt("mot_player_ir_selector")
                .selected_text(selected_name)
                .width(300.0)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(current_path.is_empty(), "NO IR")
                        .clicked()
                    {
                        clear_selected_ir(context);
                    }
                    for path in state.ir_files.clone() {
                        let encoded = path.to_string_lossy().into_owned();
                        if ui
                            .selectable_label(
                                encoded == current_path,
                                ir_display_name(state, &path),
                            )
                            .clicked()
                        {
                            match select_ir_path(context, state, &path) {
                                Ok(()) => state.message = None,
                                Err(error) => state.message = Some(error),
                            }
                        }
                    }
                });

            let busy = context.ir_import_control.is_busy();
            if ui.add_enabled(!busy, ghost_button("IMPORT…")).clicked()
                && let Some(source) = rfd::FileDialog::new()
                    .add_filter("WAV audio", &["wav", "WAV"])
                    .pick_file()
            {
                if !context.ir_import_control.try_begin() {
                    state.message = Some("An IR import is already running".to_owned());
                } else if let Some(spawner) = context.tasks::<ImportIrTask>() {
                    match spawner.try_spawn(ImportIrTask { source }) {
                        Ok(()) => {
                            state.message = Some("Importing IR in the background…".to_owned());
                        }
                        Err(_) => {
                            context.ir_import_control.cancel_begin();
                            state.message = Some("IR import worker queue is full".to_owned());
                        }
                    }
                } else {
                    context.ir_import_control.cancel_begin();
                    state.message = Some("IR import worker is unavailable in this host".to_owned());
                }
            }
            if ui.add(ghost_button("REFRESH")).clicked() {
                state.refresh_pending = true;
            }
            if busy {
                ui.label(RichText::new("IMPORTING").small().monospace().color(ACCENT));
            }
        });

        ui.horizontal(|ui| {
            ui.label(field_label("PROCESSING"));
            let current = IrUiMode::from_param(context.ir_processing.value());
            for (mode, label) in [
                (IrUiMode::MinimumPhase, "MINIMUM PHASE + AUTO-TRIM"),
                (IrUiMode::Raw, "RAW"),
            ] {
                let button = if current == mode {
                    mot_ui::selected_model_button(label)
                } else {
                    ghost_button(label)
                };
                if ui.add(button).clicked() {
                    context.automate(P::IrProcessing, mode.normalized());
                    bump_runtime_generation(context);
                }
            }
        });

        if current_path.is_empty() {
            ui.label(
                RichText::new("Amp output is currently uncabbed.")
                    .small()
                    .color(DIM),
            );
        } else {
            let verified = state.ir_metadata.contains_key(Path::new(&current_path));
            ui.label(
                RichText::new(&current_path)
                    .small()
                    .monospace()
                    .color(if verified { DIM } else { WAITING }),
            );
            if let Some(metadata) = state.ir_metadata.get(Path::new(&current_path)) {
                ui.label(
                    RichText::new(format!(
                        "ARCHIVED RAW • {} samples @ {} Hz • MPT auto-trim: {} samples",
                        metadata.sample_count,
                        metadata.sample_rate_hz,
                        metadata.default_trim_leading_samples
                    ))
                    .small()
                    .monospace()
                    .color(DIM),
                );
                ui.label(
                    RichText::new(format!(
                        "SOURCE: {} • SHA-256 {}",
                        metadata.original_filename, metadata.sha256
                    ))
                    .small()
                    .monospace()
                    .color(DIM),
                );
            }
        }
        if IrUiMode::from_param(context.ir_processing.value()) == IrUiMode::Raw {
            ui.label(
                RichText::new(
                    "RAW preserves the IR phase and any intrinsic delay contained in the file.",
                )
                .small()
                .color(WAITING),
            );
        }
    });
}

fn sort_ir_files(state: &mut EditorState) {
    let metadata = &state.ir_metadata;
    state.ir_files.sort_by(|left, right| {
        let left_name = metadata.get(left).map_or_else(
            || display_filename(left),
            |item| item.original_filename.clone(),
        );
        let right_name = metadata.get(right).map_or_else(
            || display_filename(right),
            |item| item.original_filename.clone(),
        );
        left_name
            .to_lowercase()
            .cmp(&right_name.to_lowercase())
            .then_with(|| left.cmp(right))
    });
}

fn ir_display_name(state: &EditorState, path: &Path) -> String {
    state.ir_metadata.get(path).map_or_else(
        || display_filename(path),
        |metadata| metadata.original_filename.clone(),
    )
}

fn current_tone_view(context: &PluginContext<MotPlayerParams>) -> ToneView {
    let ir_processing = IrUiMode::from_param(context.ir_processing.value());
    ToneView {
        input_gain_db: round_to_tenth(context.input_gain.value()),
        tight_percent: context.tight.value(),
        bite_percent: context.bite.value(),
        ir_path: read_shared_string(&context.selected_ir_path),
        ir_reference: persisted_ir_reference(context, ir_processing)
            .ok()
            .flatten(),
        ir_processing,
    }
}

fn tone_is_dirty(state: &EditorState, context: &PluginContext<MotPlayerParams>) -> bool {
    if read_shared_string(&context.selected_model_id).is_empty() {
        return false;
    }
    state
        .baseline
        .as_ref()
        .is_none_or(|baseline| !baseline.equivalent(&current_tone_view(context)))
}

fn selected_model_entry(
    state: &EditorState,
    context: &PluginContext<MotPlayerParams>,
) -> Option<ModelEntry> {
    let selected_id = read_shared_string(&context.selected_model_id);
    let selected_hash = read_shared_string(&context.selected_model_sha256);
    state
        .scan
        .models
        .iter()
        .find(|entry| {
            entry.reference.model_id == selected_id
                && entry
                    .reference
                    .sha256
                    .to_string()
                    .eq_ignore_ascii_case(&selected_hash)
        })
        .cloned()
}

fn selected_model_reference(context: &MotPlayerParams) -> Option<ModelRef> {
    let model_id = read_shared_string(&context.selected_model_id);
    if model_id.is_empty() {
        return None;
    }
    let sha256 = read_shared_string(&context.selected_model_sha256)
        .parse::<Sha256Digest>()
        .ok()?;
    let mut filename_hint = read_shared_string(&context.selected_model_filename_hint);
    if filename_hint.is_empty() {
        filename_hint = format!("{model_id}.motmodel");
    }
    Some(ModelRef {
        model_id,
        sha256,
        filename_hint,
    })
}

fn selected_reference_is_current(context: &MotPlayerParams, reference: &ModelRef) -> bool {
    read_shared_string(&context.selected_model_id) == reference.model_id
        && read_shared_string(&context.selected_model_sha256)
            .eq_ignore_ascii_case(&reference.sha256.to_string())
}

fn selected_model_identity_matches(
    context: &MotPlayerParams,
    model_id: &str,
    model_sha256: &str,
) -> bool {
    read_shared_string(&context.selected_model_id) == model_id
        && read_shared_string(&context.selected_model_sha256).eq_ignore_ascii_case(model_sha256)
}

fn request_select_model(
    state: &mut EditorState,
    context: &PluginContext<MotPlayerParams>,
    entry: ModelEntry,
) -> Result<(), String> {
    let display_name = entry.metadata.display_name.clone();
    submit_library_task(
        context,
        LibraryTaskOperation::LoadTone {
            entry: Box::new(entry),
            guard_model_id: read_shared_string(&context.selected_model_id),
            guard_model_sha256: read_shared_string(&context.selected_model_sha256),
        },
    )?;
    state.message = Some(format!("Loading “{display_name}”…"));
    Ok(())
}

fn apply_loaded_model(
    state: &mut EditorState,
    context: &PluginContext<MotPlayerParams>,
    entry: ModelEntry,
    result: Result<Option<ToneSettings>, String>,
) {
    let Some(library) = state.library.as_ref() else {
        state.message = Some("Model library is unavailable".to_owned());
        return;
    };
    let tone = match result {
        Ok(Some(tone)) => tone,
        Ok(None) => ToneSettings::defaults_for(&entry.reference),
        Err(error) => {
            state.message = Some(format!(
                "Saved tone could not be loaded; defaults used: {error}"
            ));
            ToneSettings::defaults_for(&entry.reference)
        }
    };
    let view = tone_view_from_settings(&tone, library);
    apply_tone_view(context, &view);
    write_shared_string(&context.selected_model_id, &entry.reference.model_id);
    write_shared_string(
        &context.selected_model_sha256,
        &entry.reference.sha256.to_string(),
    );
    write_shared_string(
        &context.selected_model_filename_hint,
        &entry.reference.filename_hint,
    );
    bump_runtime_generation(context);
    state.baseline = Some(view);
    state.pending_model = None;
}

fn tone_view_from_settings(settings: &ToneSettings, library: &ModelLibrary) -> ToneView {
    let (ir_path, ir_reference, ir_processing) = settings.ir.as_ref().map_or_else(
        || (String::new(), None, IrUiMode::MinimumPhase),
        |ir| {
            let mode = match ir.processing {
                IrProcessingMode::MinimumPhaseAutoTrim => IrUiMode::MinimumPhase,
                IrProcessingMode::Raw => IrUiMode::Raw,
            };
            (
                library
                    .paths()
                    .irs
                    .join(&ir.filename_hint)
                    .to_string_lossy()
                    .into_owned(),
                Some(ir.clone()),
                mode,
            )
        },
    );
    ToneView {
        input_gain_db: settings.input_gain_db,
        tight_percent: settings.tight_percent,
        bite_percent: settings.bite_percent,
        ir_path,
        ir_reference,
        ir_processing,
    }
}

fn apply_tone_view(context: &PluginContext<MotPlayerParams>, view: &ToneView) {
    automate_linear(context, P::InputGain, view.input_gain_db, -24.0, 24.0);
    automate_linear(context, P::Tight, view.tight_percent, 0.0, 100.0);
    automate_linear(context, P::Bite, view.bite_percent, 0.0, 100.0);
    context.automate(P::IrProcessing, view.ir_processing.normalized());
    write_selected_ir_state(context, &view.ir_path, view.ir_reference.as_ref());
}

fn request_save_current_tone(
    state: &mut EditorState,
    context: &PluginContext<MotPlayerParams>,
    switch_after_save: bool,
) {
    let Some(entry) = selected_model_entry(state, context) else {
        state.message = Some("Select a valid model before saving".to_owned());
        return;
    };
    let view = current_tone_view(context);
    let ir = match persisted_ir_reference(context, view.ir_processing) {
        Ok(ir) => ir,
        Err(error) => {
            state.message = Some(error);
            return;
        }
    };
    let settings = ToneSettings {
        schema_version: mot_core::model_library::TONE_SETTINGS_VERSION,
        model_id: entry.reference.model_id.clone(),
        model_sha256: entry.reference.sha256,
        input_gain_db: view.input_gain_db,
        tight_percent: view.tight_percent,
        bite_percent: view.bite_percent,
        ir,
    };
    match submit_library_task(
        context,
        LibraryTaskOperation::SaveTone {
            model_reference: entry.reference,
            settings,
        },
    ) {
        Ok(()) => {
            state.switch_after_save = switch_after_save;
            state.message = Some("Saving model settings…".to_owned());
        }
        Err(error) => state.message = Some(error),
    }
}

fn apply_saved_tone_outcome(
    state: &mut EditorState,
    context: &PluginContext<MotPlayerParams>,
    settings: ToneSettings,
    result: Result<(), String>,
) {
    match result {
        Ok(()) => {
            let current_matches = selected_model_identity_matches(
                context,
                &settings.model_id,
                &settings.model_sha256.to_string(),
            );
            if current_matches && let Some(library) = state.library.as_ref() {
                state.baseline = Some(tone_view_from_settings(&settings, library));
            }
            state.message = Some("Model settings saved".to_owned());
            let switch_target = if state.switch_after_save && current_matches {
                state.pending_model.clone()
            } else {
                None
            };
            state.switch_after_save = false;
            if let Some(target) = switch_target
                && let Err(error) = request_select_model(state, context, target)
            {
                state.message = Some(error);
            }
        }
        Err(error) => {
            state.switch_after_save = false;
            state.message = Some(error);
        }
    }
}

fn persisted_ir_reference(
    context: &MotPlayerParams,
    mode: IrUiMode,
) -> Result<Option<IrReference>, String> {
    let path = read_shared_string(&context.selected_ir_path);
    if path.is_empty() {
        return Ok(None);
    }
    let ir_id = read_shared_string(&context.selected_ir_id);
    let sha256 = read_shared_string(&context.selected_ir_sha256)
        .parse::<Sha256Digest>()
        .map_err(|error| format!("Selected IR SHA-256 is invalid: {error}"))?;
    let filename_hint = read_shared_string(&context.selected_ir_filename_hint);
    if ir_id.is_empty() || filename_hint.is_empty() {
        return Err("Selected IR identity is incomplete".to_owned());
    }
    Ok(Some(IrReference {
        ir_id,
        sha256,
        filename_hint,
        processing: mode.library_mode(),
    }))
}

fn select_ir_path(
    context: &PluginContext<MotPlayerParams>,
    state: &EditorState,
    path: &Path,
) -> Result<(), String> {
    let metadata = state
        .ir_metadata
        .get(path)
        .ok_or_else(|| "This IR is missing or failed exact-content validation".to_owned())?;
    let mut reference = metadata.reference();
    reference.processing = IrUiMode::from_param(context.ir_processing.value()).library_mode();
    write_selected_ir_state(context, &path.to_string_lossy(), Some(&reference));
    bump_runtime_generation(context);
    Ok(())
}

fn select_imported_ir(context: &PluginContext<MotPlayerParams>, imported: &ImportedIr) {
    context.automate(P::IrProcessing, IrUiMode::MinimumPhase.normalized());
    let mut reference = imported.reference.clone();
    reference.processing = IrProcessingMode::MinimumPhaseAutoTrim;
    write_selected_ir_state(
        context,
        &imported.archived_path.to_string_lossy(),
        Some(&reference),
    );
    bump_runtime_generation(context);
}

fn revert_current_tone(state: &mut EditorState, context: &PluginContext<MotPlayerParams>) {
    if let Some(baseline) = state.baseline.clone() {
        apply_tone_view(context, &baseline);
        bump_runtime_generation(context);
        state.message = Some("Saved model settings restored".to_owned());
    }
}

fn write_selected_ir_state(context: &MotPlayerParams, path: &str, reference: Option<&IrReference>) {
    write_shared_string(&context.selected_ir_path, path);
    if let Some(reference) = reference {
        write_shared_string(&context.selected_ir_id, &reference.ir_id);
        write_shared_string(&context.selected_ir_sha256, &reference.sha256.to_string());
        write_shared_string(&context.selected_ir_filename_hint, &reference.filename_hint);
    } else {
        write_shared_string(&context.selected_ir_id, "");
        write_shared_string(&context.selected_ir_sha256, "");
        write_shared_string(&context.selected_ir_filename_hint, "");
    }
}

fn clear_selected_ir(context: &PluginContext<MotPlayerParams>) {
    write_selected_ir_state(context, "", None);
    bump_runtime_generation(context);
}

fn bump_runtime_generation(context: &PluginContext<MotPlayerParams>) {
    let next = context.runtime_generation.load().wrapping_add(1).max(1);
    context.runtime_generation.store(next);
    context.runtime_control.set(RuntimeUiState::Loading);
}

fn automate_linear(
    context: &PluginContext<MotPlayerParams>,
    id: P,
    value: f32,
    minimum: f32,
    maximum: f32,
) {
    let normalized = ((value - minimum) / (maximum - minimum)).clamp(0.0, 1.0);
    context.automate(id, f64::from(normalized));
}

fn round_to_tenth(value: f32) -> f32 {
    (value * 10.0).round() / 10.0
}

fn display_filename(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}
