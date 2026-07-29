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
    refresh_pending: bool,
    switch_after_save: bool,
    message: Option<String>,
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

        let background = Color32::from_rgb(10, 12, 14);
        let panel = Color32::from_rgb(20, 24, 28);
        let panel_alt = Color32::from_rgb(25, 30, 34);
        let cyan = Color32::from_rgb(58, 220, 210);
        let text_dim = Color32::from_rgb(135, 148, 155);
        ui.visuals_mut().panel_fill = background;
        ui.visuals_mut().override_text_color = Some(Color32::from_rgb(228, 235, 238));

        egui::Frame::new()
            .fill(background)
            .inner_margin(18.0)
            .show(ui, |ui| {
                header(ui, context, cyan, text_dim);
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(12.0);
                ui.horizontal_top(|ui| {
                    fixed_vertical_panel(ui, Vec2::new(350.0, ui.available_height()), |ui| {
                        model_browser(ui, context, &mut state, panel, panel_alt, cyan, text_dim);
                    });
                    ui.add_space(10.0);
                    ui.separator();
                    ui.add_space(10.0);
                    ui.vertical(|ui| {
                        player_controls(ui, context, &mut state, panel, panel_alt, cyan, text_dim);
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
    state.message = first_error;
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

fn header(
    ui: &mut egui::Ui,
    context: &PluginContext<MotPlayerParams>,
    accent: Color32,
    text_dim: Color32,
) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new("MOT PLAYER")
                .font(FontId::proportional(24.0))
                .strong()
                .color(accent),
        );
        ui.label(
            RichText::new("0.4.0  •  MONO  •  48 kHz  •  ZERO LATENCY")
                .monospace()
                .color(text_dim),
        );
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let muted = context.mute.value();
            let button = egui::Button::new("MUTE").fill(if muted {
                Color32::from_rgb(146, 48, 55)
            } else {
                Color32::from_rgb(42, 48, 52)
            });
            if ui.add(button).clicked() {
                context.automate(P::Mute, if muted { 0.0 } else { 1.0 });
            }
        });
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
    panel: Color32,
    panel_alt: Color32,
    accent: Color32,
    text_dim: Color32,
) {
    egui::Frame::new()
        .fill(panel)
        .corner_radius(8.0)
        .inner_margin(12.0)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("MODELS").strong().color(accent));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(
                        RichText::new(state.scan.models.len().to_string())
                            .monospace()
                            .color(text_dim),
                    );
                });
            });
            ui.add_space(6.0);

            let selected_id = read_shared_string(&context.selected_model_id);
            let selected_hash = read_shared_string(&context.selected_model_sha256);
            let models = state.scan.models.clone();
            egui::ScrollArea::vertical()
                .id_salt("mot_player_model_browser")
                .max_height(440.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if models.is_empty() {
                        ui.label(
                            RichText::new("No compatible .motmodel files")
                                .italics()
                                .color(text_dim),
                        );
                        ui.label(
                            RichText::new(
                                "~/Library/Application Support/Plut&Mot/MOT Guitar Plugin/Models/",
                            )
                            .small()
                            .monospace()
                            .color(text_dim),
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
                        let response = ui.add_sized(
                            [ui.available_width(), 48.0],
                            egui::Button::new(RichText::new(label).color(if selected {
                                accent
                            } else {
                                Color32::WHITE
                            }))
                            .selected(selected)
                            .fill(if selected {
                                Color32::from_rgb(26, 70, 68)
                            } else {
                                panel_alt
                            }),
                        );
                        if response.clicked() && !selected {
                            if tone_is_dirty(state, context) {
                                state.pending_model = Some(entry);
                            } else {
                                request_select_model(state, context, entry);
                            }
                        }
                    }
                });

            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                let busy = context.library_control.is_busy();
                if ui
                    .add_enabled(!busy, egui::Button::new("REFRESH").small())
                    .clicked()
                {
                    state.refresh_pending = true;
                }
                if ui
                    .add_enabled(!busy, egui::Button::new("OPEN FOLDER").small())
                    .clicked()
                {
                    match submit_library_task(context, LibraryTaskOperation::OpenFolder) {
                        Ok(()) => {
                            state.message = Some("Opening model library folder…".to_owned());
                        }
                        Err(error) => state.message = Some(error),
                    }
                }
                if busy {
                    ui.spinner();
                }
            });

            if !state.scan.issues.is_empty() {
                egui::CollapsingHeader::new(
                    RichText::new(format!("{} scan issue(s)", state.scan.issues.len()))
                        .color(Color32::from_rgb(235, 164, 77)),
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
                            .color(text_dim),
                        );
                    }
                });
            }
        });

    if let Some(pending) = state.pending_model.clone() {
        ui.add_space(8.0);
        egui::Frame::new()
            .fill(Color32::from_rgb(54, 43, 24))
            .corner_radius(7.0)
            .inner_margin(10.0)
            .show(ui, |ui| {
                ui.label(
                    RichText::new("UNSAVED MODEL SETTINGS")
                        .strong()
                        .color(Color32::from_rgb(245, 190, 95)),
                );
                ui.label(format!("Switch to “{}”?", pending.metadata.display_name));
                ui.horizontal(|ui| {
                    let busy = context.library_control.is_busy();
                    if ui
                        .add_enabled(!busy, egui::Button::new("SAVE").small())
                        .clicked()
                    {
                        request_save_current_tone(state, context, true);
                    }
                    if ui
                        .add_enabled(!busy, egui::Button::new("DISCARD").small())
                        .clicked()
                    {
                        request_select_model(state, context, pending.clone());
                    }
                    if ui
                        .add_enabled(!busy, egui::Button::new("CANCEL").small())
                        .clicked()
                    {
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
    panel: Color32,
    panel_alt: Color32,
    accent: Color32,
    text_dim: Color32,
) {
    let selected = selected_model_entry(state, context);
    egui::Frame::new()
        .fill(panel)
        .corner_radius(8.0)
        .inner_margin(16.0)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label(RichText::new("ACTIVE MODEL").small().color(text_dim));
                    ui.label(
                        RichText::new(
                            selected.as_ref().map_or("NO MODEL — TRANSPARENT", |entry| {
                                entry.metadata.display_name.as_str()
                            }),
                        )
                        .font(FontId::proportional(22.0))
                        .strong()
                        .color(if selected.is_some() {
                            accent
                        } else {
                            Color32::from_rgb(235, 164, 77)
                        }),
                    );
                });
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let dirty = tone_is_dirty(state, context);
                    ui.label(
                        RichText::new(if dirty { "UNSAVED *" } else { "SAVED" })
                            .monospace()
                            .color(if dirty {
                                Color32::from_rgb(245, 190, 95)
                            } else {
                                text_dim
                            }),
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
                    .color(text_dim),
                );
            }

            ui.add_space(10.0);
            runtime_status(ui, context, accent, text_dim);
            ui.add_space(16.0);
            ui.horizontal(|ui| {
                player_parameter(
                    ui,
                    context,
                    "INPUT GAIN",
                    P::InputGain,
                    context.input_gain.value(),
                    -24.0,
                    24.0,
                    " dB",
                    accent,
                );
                player_parameter(
                    ui,
                    context,
                    "TIGHT",
                    P::Tight,
                    context.tight.value(),
                    0.0,
                    100.0,
                    "%",
                    accent,
                );
                player_parameter(
                    ui,
                    context,
                    "BITE",
                    P::Bite,
                    context.bite.value(),
                    0.0,
                    100.0,
                    "%",
                    accent,
                );
            });

            ui.add_space(16.0);
            ir_section(ui, context, state, panel_alt, accent, text_dim);
            ui.add_space(14.0);
            ui.horizontal(|ui| {
                let has_model = selected.is_some();
                let busy = context.library_control.is_busy();
                if ui
                    .add_enabled(has_model && !busy, egui::Button::new("SAVE"))
                    .clicked()
                {
                    request_save_current_tone(state, context, false);
                }
                if ui
                    .add_enabled(
                        has_model
                            && !busy
                            && state.baseline.is_some()
                            && tone_is_dirty(state, context),
                        egui::Button::new("REVERT"),
                    )
                    .clicked()
                {
                    revert_current_tone(state, context);
                }
                ui.separator();
                ui.label(
                    RichText::new(format!(
                        "runtime generation {}",
                        context.runtime_generation.load()
                    ))
                    .small()
                    .monospace()
                    .color(text_dim),
                );
            });
            if let Some(message) = &state.message {
                ui.add_space(8.0);
                ui.label(
                    RichText::new(message)
                        .small()
                        .color(Color32::from_rgb(235, 164, 77)),
                );
            }
        });
}

fn runtime_status(
    ui: &mut egui::Ui,
    context: &PluginContext<MotPlayerParams>,
    accent: Color32,
    text_dim: Color32,
) {
    let meter = context.get_meter(P::RuntimeStatus);
    let state = context.runtime_control.get();
    let (label, detail, color) = match state {
        RuntimeUiState::Transparent => (
            "TRANSPARENT",
            "No model selected; input passes through unchanged.".to_owned(),
            text_dim,
        ),
        RuntimeUiState::Loading => ("LOADING", "Preparing model and cabinet…".to_owned(), accent),
        RuntimeUiState::Ready {
            model_name,
            ir_name,
        } => (
            "READY",
            ir_name.map_or_else(
                || format!("{model_name} • uncabbed"),
                |ir| format!("{model_name} • {ir}"),
            ),
            Color32::from_rgb(77, 190, 134),
        ),
        RuntimeUiState::SafeMuted { asset, message } => (
            "SAFE MUTE",
            format!("{asset:?}: {message}"),
            Color32::from_rgb(235, 95, 95),
        ),
    };
    let visible_label = if meter < 0.25 && label != "SAFE MUTE" {
        "SAFE MUTE"
    } else if (0.25..0.75).contains(&meter) {
        "LOADING"
    } else {
        label
    };
    ui.horizontal(|ui| {
        ui.label(RichText::new("RUNTIME").small().color(text_dim));
        ui.label(
            RichText::new(visible_label)
                .strong()
                .monospace()
                .color(color),
        );
    });
    ui.label(RichText::new(detail).small().color(color));
    if meter < 0.25
        && !matches!(
            context.runtime_control.get(),
            RuntimeUiState::SafeMuted { .. }
        )
    {
        ui.label(
            RichText::new("MOT PLAYER requires a 48 kHz host session.")
                .small()
                .color(Color32::from_rgb(235, 95, 95)),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn player_parameter(
    ui: &mut egui::Ui,
    context: &PluginContext<MotPlayerParams>,
    label: &str,
    id: P,
    value: f32,
    minimum: f32,
    maximum: f32,
    suffix: &str,
    accent: Color32,
) {
    ui.vertical(|ui| {
        ui.set_width((ui.available_width() / 3.0).max(170.0));
        ui.label(RichText::new(label).strong().color(accent));
        let mut edited = value;
        if ui
            .add(
                egui::Slider::new(&mut edited, minimum..=maximum)
                    .suffix(suffix)
                    .step_by(0.1)
                    .fixed_decimals(1),
            )
            .changed()
        {
            automate_linear(context, id, edited, minimum, maximum);
        }
    });
}

fn ir_section(
    ui: &mut egui::Ui,
    context: &PluginContext<MotPlayerParams>,
    state: &mut EditorState,
    panel_alt: Color32,
    accent: Color32,
    text_dim: Color32,
) {
    egui::Frame::new()
        .fill(panel_alt)
        .corner_radius(7.0)
        .inner_margin(12.0)
        .show(ui, |ui| {
            ui.label(RichText::new("CABINET IR").strong().color(accent));
            let current_path = read_shared_string(&context.selected_ir_path);
            let selected_name = if current_path.is_empty() {
                "NO IR".to_owned()
            } else {
                ir_display_name(state, Path::new(&current_path))
            };

            ui.horizontal(|ui| {
                ui.label(RichText::new("IR").color(text_dim));
                egui::ComboBox::from_id_salt("mot_player_ir_selector")
                    .selected_text(selected_name)
                    .width(280.0)
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
                if ui
                    .add_enabled(!busy, egui::Button::new("IMPORT IR…").small())
                    .clicked()
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
                        state.message =
                            Some("IR import worker is unavailable in this host".to_owned());
                    }
                }
                if busy {
                    ui.spinner();
                }
                if ui.small_button("REFRESH IRs").clicked() {
                    state.refresh_pending = true;
                }
            });

            ui.horizontal(|ui| {
                ui.label(RichText::new("PROCESSING").color(text_dim));
                let current = IrUiMode::from_param(context.ir_processing.value());
                for (mode, label) in [
                    (IrUiMode::MinimumPhase, "MINIMUM PHASE + AUTO-TRIM"),
                    (IrUiMode::Raw, "RAW"),
                ] {
                    if ui.selectable_label(current == mode, label).clicked() {
                        context.automate(P::IrProcessing, mode.normalized());
                        bump_runtime_generation(context);
                    }
                }
            });

            if current_path.is_empty() {
                ui.label(
                    RichText::new("Amp output is currently uncabbed.")
                        .small()
                        .color(text_dim),
                );
            } else {
                let verified = state.ir_metadata.contains_key(Path::new(&current_path));
                ui.label(
                    RichText::new(&current_path)
                        .small()
                        .monospace()
                        .color(if verified {
                            text_dim
                        } else {
                            Color32::from_rgb(235, 164, 77)
                        }),
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
                        .color(text_dim),
                    );
                    ui.label(
                        RichText::new(format!(
                            "SOURCE: {} • SHA-256 {}",
                            metadata.original_filename, metadata.sha256
                        ))
                        .small()
                        .monospace()
                        .color(text_dim),
                    );
                }
            }
            if IrUiMode::from_param(context.ir_processing.value()) == IrUiMode::Raw {
                ui.label(
                    RichText::new(
                        "RAW preserves the IR phase and any intrinsic delay contained in the file.",
                    )
                    .small()
                    .color(Color32::from_rgb(235, 164, 77)),
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
) {
    let display_name = entry.metadata.display_name.clone();
    match submit_library_task(
        context,
        LibraryTaskOperation::LoadTone {
            entry: Box::new(entry),
            guard_model_id: read_shared_string(&context.selected_model_id),
            guard_model_sha256: read_shared_string(&context.selected_model_sha256),
        },
    ) {
        Ok(()) => state.message = Some(format!("Loading “{display_name}”…")),
        Err(error) => state.message = Some(error),
    }
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
            if let Some(target) = switch_target {
                request_select_model(state, context, target);
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
