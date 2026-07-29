use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Duration;

use egui::{Align, Align2, Color32, FontId, Layout, RichText, Sense, Stroke, StrokeKind, Vec2};
use truce::prelude::*;
use truce_egui::EditorUi;

use crate::capture::SessionCheckLevelState;
use crate::capture_runtime::CaptureWorkerStatus;
use crate::model::Sha256Digest;
use crate::model_library::{
    ImportedIr, IrImportMetadata, IrProcessingMode, IrReference, ModelEntry, ModelLibrary,
    ModelScan, ToneSettings,
};
use crate::tuner::STRING_COUNT;
use crate::{
    ImportIrTask, IrImportOutcome, LibraryOutcome, LibraryTask, LibraryTaskOperation,
    MotStrobeParams, P, notes, offsets, round_to_tenth,
};

pub(crate) const WINDOW_SIZE: (u32, u32) = (1_180, 760);
const OFFSET_MIN: f32 = -25.0;
const OFFSET_MAX: f32 = 25.0;
const EDITOR_STATE_ID: &str = "mot_guitar_plugin_editor_state_v1";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum EditorTab {
    #[default]
    Amp,
    Tuner,
    Settings,
}

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

#[derive(Clone, Debug)]
struct EditorState {
    tab: EditorTab,
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

impl Default for EditorState {
    fn default() -> Self {
        Self {
            tab: EditorTab::Amp,
            initialized: false,
            library: None,
            scan: ModelScan::default(),
            ir_files: Vec::new(),
            ir_metadata: BTreeMap::new(),
            baseline: None,
            pending_model: None,
            refresh_pending: false,
            switch_after_save: false,
            message: None,
        }
    }
}

pub(crate) struct MotStrobeUi;

impl EditorUi<MotStrobeParams> for MotStrobeUi {
    fn ui(&mut self, ui: &mut egui::Ui, context: &PluginContext<MotStrobeParams>) {
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
                ui.add_space(10.0);
                tab_bar(ui, &mut state, cyan);
                ui.add_space(12.0);

                // Every page gets the same hard viewport. Only the selected
                // page is instantiated, and oversized child content cannot
                // paint into the space of another page.
                let tab_viewport = ui.available_rect_before_wrap();
                ui.scope(|ui| {
                    ui.shrink_clip_rect(tab_viewport);
                    match state.tab {
                        EditorTab::Amp => {
                            amp_tab(ui, context, &mut state, panel, panel_alt, cyan, text_dim);
                        }
                        EditorTab::Tuner => tuner_tab(ui, context, panel, cyan, text_dim),
                        EditorTab::Settings => {
                            settings_tab(ui, context, &mut state, panel, panel_alt, cyan, text_dim);
                        }
                    }
                });
            });

        ui.ctx().data_mut(|data| data.insert_temp(state_id, state));
    }
}

fn initialize_editor_state(state: &mut EditorState, context: &PluginContext<MotStrobeParams>) {
    state.initialized = true;
    context.library_control.invalidate_pending();
    match ModelLibrary::for_current_user() {
        Ok(library) => {
            state.library = Some(library);
            request_library_refresh(state);
        }
        Err(error) => state.message = Some(error.to_string()),
    }
}

fn request_library_refresh(state: &mut EditorState) {
    state.refresh_pending = true;
}

fn service_pending_library_refresh(
    state: &mut EditorState,
    context: &PluginContext<MotStrobeParams>,
) {
    if !state.refresh_pending || context.library_control.is_busy() {
        return;
    }
    let operation = LibraryTaskOperation::Scan {
        selected_model: selected_model_reference(context),
    };
    match submit_library_task(context, operation) {
        Ok(()) => {
            state.refresh_pending = false;
            state.message = Some("Refreshing model and IR libraries…".to_owned());
        }
        Err(error) => state.message = Some(error),
    }
}

fn submit_library_task(
    context: &PluginContext<MotStrobeParams>,
    operation: LibraryTaskOperation,
) -> Result<(), String> {
    let request_id = context
        .library_control
        .try_begin()
        .ok_or_else(|| "A model-library operation is already running".to_owned())?;
    let task = LibraryTask {
        request_id,
        operation,
    };
    let Some(spawner) = context.tasks::<LibraryTask>() else {
        context.library_control.cancel_begin(request_id);
        return Err("Model-library worker is unavailable in this host".to_owned());
    };
    if spawner.try_spawn(task).is_err() {
        context.library_control.cancel_begin(request_id);
        return Err("Model-library worker queue is full".to_owned());
    }
    Ok(())
}

fn poll_library_outcomes(state: &mut EditorState, context: &PluginContext<MotStrobeParams>) {
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
                if !selected_model_identity_matches(context, &guard_model_id, &guard_model_sha256) {
                    continue;
                }
                apply_loaded_model(state, context, *entry, result);
            }
            LibraryOutcome::ToneSaved {
                settings, result, ..
            } => {
                apply_saved_tone_outcome(state, context, settings, result);
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
    context: &PluginContext<MotStrobeParams>,
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

fn poll_ir_import_outcome(state: &mut EditorState, context: &PluginContext<MotStrobeParams>) {
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
                request_library_refresh(state);
                state.message = Some(format!(
                    "Imported RAW unchanged; default auto-trim: {} samples",
                    imported.metadata.default_trim_leading_samples
                ));
            }
            IrImportOutcome::Error(error) => state.message = Some(error),
        }
    }
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

fn header(
    ui: &mut egui::Ui,
    context: &PluginContext<MotStrobeParams>,
    accent: Color32,
    text_dim: Color32,
) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new("MOT GUITAR PLUGIN")
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

fn tab_bar(ui: &mut egui::Ui, state: &mut EditorState, accent: Color32) {
    ui.horizontal(|ui| {
        for (tab, label) in [
            (EditorTab::Amp, "AMP"),
            (EditorTab::Tuner, "TUNER"),
            (EditorTab::Settings, "SETTINGS"),
        ] {
            let selected = state.tab == tab;
            let response = ui.selectable_label(
                selected,
                RichText::new(label)
                    .strong()
                    .color(if selected { accent } else { Color32::GRAY }),
            );
            if response.clicked() {
                state.tab = tab;
            }
        }
    });
    ui.separator();
}

fn fixed_vertical_panel<R>(
    ui: &mut egui::Ui,
    size: Vec2,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::InnerResponse<R> {
    ui.allocate_ui_with_layout(size, Layout::top_down(Align::Min), |ui| {
        // `allocate_ui` normally inherits the surrounding horizontal layout
        // and may grow to fit a long path. Keep this column vertical, fixed,
        // and clipped to its allocation instead.
        let panel_rect = ui.max_rect();
        ui.set_width(size.x);
        ui.shrink_clip_rect(panel_rect);
        add_contents(ui)
    })
}

fn amp_tab(
    ui: &mut egui::Ui,
    context: &PluginContext<MotStrobeParams>,
    state: &mut EditorState,
    panel: Color32,
    panel_alt: Color32,
    accent: Color32,
    text_dim: Color32,
) {
    ui.horizontal_top(|ui| {
        fixed_vertical_panel(ui, Vec2::new(350.0, ui.available_height()), |ui| {
            model_browser(ui, context, state, panel, panel_alt, accent, text_dim);
        });
        ui.add_space(10.0);
        ui.separator();
        ui.add_space(10.0);
        ui.vertical(|ui| {
            amp_controls(ui, context, state, panel, panel_alt, accent, text_dim);
        });
    });
}

fn model_browser(
    ui: &mut egui::Ui,
    context: &PluginContext<MotStrobeParams>,
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

            let selected_id = read_lock_string(&context.selected_model_id);
            let selected_hash = read_lock_string(&context.selected_model_sha256);
            let models = state.scan.models.clone();
            egui::ScrollArea::vertical()
                .id_salt("model_browser_list")
                .max_height(440.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if models.is_empty() {
                        ui.label(
                            RichText::new("No compatible .motmodel files")
                                .italics()
                                .color(text_dim),
                        );
                        ui.add_space(4.0);
                        let library_path =
                            "~/Library/Application Support/Plut&Mot/MOT Guitar Plugin/Models/";
                        ui.add(
                            egui::Label::new(
                                RichText::new(library_path)
                                    .small()
                                    .monospace()
                                    .color(text_dim),
                            )
                            .truncate(),
                        )
                        .on_hover_text(library_path);
                    }

                    for entry in models {
                        let is_selected = entry.reference.model_id == selected_id
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
                            egui::Button::new(RichText::new(label).color(if is_selected {
                                accent
                            } else {
                                Color32::WHITE
                            }))
                            .selected(is_selected)
                            .fill(if is_selected {
                                Color32::from_rgb(26, 70, 68)
                            } else {
                                panel_alt
                            }),
                        );
                        if response.clicked() && !is_selected {
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
                let library_busy = context.library_control.is_busy();
                if ui
                    .add_enabled(!library_busy, egui::Button::new("REFRESH").small())
                    .clicked()
                {
                    request_library_refresh(state);
                }
                if ui
                    .add_enabled(!library_busy, egui::Button::new("OPEN FOLDER").small())
                    .clicked()
                {
                    match submit_library_task(context, LibraryTaskOperation::OpenFolder) {
                        Ok(()) => {
                            state.message = Some("Opening model library folder…".to_owned());
                        }
                        Err(error) => state.message = Some(error),
                    }
                }
                if library_busy {
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
                ui.label(
                    RichText::new(format!("Switch to “{}”?", pending.metadata.display_name))
                        .small(),
                );
                ui.horizontal(|ui| {
                    let library_busy = context.library_control.is_busy();
                    if ui
                        .add_enabled(!library_busy, egui::Button::new("SAVE").small())
                        .clicked()
                    {
                        request_save_current_tone(state, context, true);
                    }
                    if ui
                        .add_enabled(!library_busy, egui::Button::new("DISCARD").small())
                        .clicked()
                    {
                        request_select_model(state, context, pending.clone());
                    }
                    if ui
                        .add_enabled(!library_busy, egui::Button::new("CANCEL").small())
                        .clicked()
                    {
                        state.pending_model = None;
                    }
                });
            });
    }
}

fn amp_controls(
    ui: &mut egui::Ui,
    context: &PluginContext<MotStrobeParams>,
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
                            selected
                                .as_ref()
                                .map_or("NO MODEL", |entry| entry.metadata.display_name.as_str()),
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

            ui.add_space(18.0);
            ui.horizontal(|ui| {
                amp_parameter(
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
                amp_parameter(
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
                amp_parameter(
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
                let library_busy = context.library_control.is_busy();
                if ui
                    .add_enabled(has_model && !library_busy, egui::Button::new("SAVE"))
                    .clicked()
                {
                    request_save_current_tone(state, context, false);
                }
                if ui
                    .add_enabled(
                        has_model
                            && !library_busy
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

#[allow(clippy::too_many_arguments)]
fn amp_parameter(
    ui: &mut egui::Ui,
    context: &PluginContext<MotStrobeParams>,
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
        let response = ui.add(
            egui::Slider::new(&mut edited, minimum..=maximum)
                .suffix(suffix)
                .step_by(0.1)
                .fixed_decimals(1),
        );
        if response.changed() {
            automate_linear(context, id, edited, minimum, maximum);
        }
    });
}

fn ir_section(
    ui: &mut egui::Ui,
    context: &PluginContext<MotStrobeParams>,
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
            let current_path = read_lock_string(&context.selected_ir_path);
            let selected_name = if current_path.is_empty() {
                "NO IR".to_owned()
            } else {
                ir_display_name(state, Path::new(&current_path))
            };

            ui.horizontal(|ui| {
                ui.label(RichText::new("IR").color(text_dim));
                egui::ComboBox::from_id_salt("cabinet_ir_selector")
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
                                let result = select_ir_path(context, state, &path);
                                match result {
                                    Ok(()) => state.message = None,
                                    Err(error) => state.message = Some(error),
                                }
                            }
                        }
                    });

                let import_busy = context.ir_import_control.is_busy();
                if ui
                    .add_enabled(!import_busy, egui::Button::new("IMPORT IR…").small())
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
                if import_busy {
                    ui.spinner();
                }
                if ui.small_button("REFRESH IRs").clicked() {
                    request_library_refresh(state);
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
                let selected_ir_is_verified =
                    state.ir_metadata.contains_key(Path::new(&current_path));
                ui.label(RichText::new(&current_path).small().monospace().color(
                    if selected_ir_is_verified {
                        text_dim
                    } else {
                        Color32::from_rgb(235, 164, 77)
                    },
                ));
                if let Some(metadata) = state.ir_metadata.get(Path::new(&current_path)) {
                    ui.label(
                        RichText::new(format!(
                            "ARCHIVED RAW • {} samples @ {} Hz • default MPT auto-trim: {} samples",
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

fn tuner_tab(
    ui: &mut egui::Ui,
    context: &PluginContext<MotStrobeParams>,
    panel: Color32,
    accent: Color32,
    text_dim: Color32,
) {
    strobe(ui, context, panel, accent);
    ui.add_space(10.0);
    string_editor(ui, context, panel, accent, text_dim);
}

fn settings_tab(
    ui: &mut egui::Ui,
    context: &PluginContext<MotStrobeParams>,
    state: &mut EditorState,
    panel: Color32,
    panel_alt: Color32,
    accent: Color32,
    text_dim: Color32,
) {
    ui.horizontal_top(|ui| {
        fixed_vertical_panel(ui, Vec2::new(520.0, ui.available_height()), |ui| {
            egui::Frame::new()
                .fill(panel)
                .corner_radius(8.0)
                .inner_margin(16.0)
                .show(ui, |ui| {
                    ui.label(RichText::new("CAPTURE LAB").strong().color(accent));
                    ui.add_space(10.0);

                    // Capture configuration is an explicit two-column form:
                    // one parameter per row, never a single flowing toolbar.
                    let role = context.capture_role.value().clamp(0, 2);
                    let target = context.capture_target.value().clamp(0, 1);
                    let mut session_name = read_lock_string(&context.capture_session_name);
                    let mut model_name = read_lock_string(&context.capture_model_name);
                    let mut send_trim = context.capture_send_trim.value();
                    let mut max_passes = context.max_passes.value().clamp(1, 400);
                    egui::Grid::new("capture_settings_grid")
                        .num_columns(2)
                        .spacing([16.0, 10.0])
                        .min_col_width(104.0)
                        .show(ui, |ui| {
                            ui.label(RichText::new("ROLE").color(text_dim));
                            ui.horizontal(|ui| {
                                for (value, label) in
                                    [(0_i64, "NORMAL"), (1, "SOURCE"), (2, "RETURN")]
                                {
                                    if ui.selectable_label(role == value, label).clicked() {
                                        context.automate(P::CaptureRole, value as f64 / 2.0);
                                        if value == 0 && context.capture_armed.value() {
                                            context.automate(P::CaptureArmed, 0.0);
                                        }
                                    }
                                }
                            });
                            ui.end_row();

                            ui.label(RichText::new("TARGET").color(text_dim));
                            ui.horizontal(|ui| {
                                for (value, label) in
                                    [(0_i64, "SOFTWARE CHAIN"), (1, "HARDWARE AMP")]
                                {
                                    if ui.selectable_label(target == value, label).clicked() {
                                        context.automate(P::CaptureTarget, value as f64);
                                    }
                                }
                            });
                            ui.end_row();

                            ui.label(RichText::new("SESSION").color(text_dim));
                            let response = ui.add(
                                egui::TextEdit::singleline(&mut session_name)
                                    .hint_text("e.g. amp-room-a")
                                    .desired_width(300.0),
                            );
                            if response.changed() {
                                write_lock_string(&context.capture_session_name, &session_name);
                                context
                                    .capture_session_id
                                    .store(session_id_from_name(&session_name));
                                if context.capture_armed.value() {
                                    context.automate(P::CaptureArmed, 0.0);
                                }
                            }
                            ui.end_row();

                            ui.label(RichText::new("SESSION ID").color(text_dim));
                            ui.label(
                                RichText::new(format!(
                                    "{:016x}",
                                    context.capture_session_id.load()
                                ))
                                .small()
                                .monospace()
                                .color(text_dim),
                            );
                            ui.end_row();

                            ui.label(RichText::new("MODEL NAME").color(text_dim));
                            if ui
                                .add(
                                    egui::TextEdit::singleline(&mut model_name)
                                        .hint_text("Captured Amp")
                                        .desired_width(300.0),
                                )
                                .changed()
                            {
                                write_lock_string(&context.capture_model_name, &model_name);
                            }
                            ui.end_row();

                            ui.label(RichText::new("SEND TRIM").color(text_dim));
                            if ui
                                .add_sized(
                                    [300.0, 20.0],
                                    egui::Slider::new(&mut send_trim, -40.0..=0.0)
                                        .suffix(" dB")
                                        .fixed_decimals(1),
                                )
                                .changed()
                            {
                                automate_linear(context, P::CaptureSendTrim, send_trim, -40.0, 0.0);
                            }
                            ui.end_row();

                            ui.label(RichText::new("MAX EPOCHS").color(text_dim));
                            if ui
                                .add_sized(
                                    [300.0, 20.0],
                                    egui::Slider::new(&mut max_passes, 1..=400).integer(),
                                )
                                .changed()
                            {
                                automate_discrete(context, P::MaxPasses, max_passes, 1, 400);
                            }
                            ui.end_row();
                        });

                    if !session_name.trim().is_empty() && context.capture_session_id.load() == 0 {
                        context
                            .capture_session_id
                            .store(session_id_from_name(&session_name));
                    }
                    ui.add_space(6.0);
                    egui::CollapsingHeader::new("CAPTURE METADATA")
                        .default_open(false)
                        .show(ui, |ui| {
                            capture_metadata_editor(ui, context, text_dim);
                        });

                    ui.add_space(12.0);
                    let check_level_state =
                        decode_check_level_status(context.get_meter(P::CheckLevelStatus));
                    if role == 2 {
                        ui.horizontal(|ui| {
                            let measuring = check_level_state == SessionCheckLevelState::Measuring;
                            let label = if measuring {
                                "MEASURING…"
                            } else {
                                "CHECK LEVEL"
                            };
                            let enabled = !context.capture_armed.value()
                                && !measuring
                                && context.capture_session_id.load() != 0;
                            if ui
                                .add_enabled(
                                    enabled,
                                    egui::Button::new(label)
                                        .min_size(Vec2::new(130.0, 32.0))
                                        .fill(Color32::from_rgb(40, 91, 117)),
                                )
                                .clicked()
                            {
                                context
                                    .check_level_trigger_generation
                                    .fetch_add(1, Ordering::AcqRel);
                                state.message = None;
                            }
                            ui.label(
                                RichText::new(check_level_label(check_level_state))
                                    .strong()
                                    .monospace()
                                    .color(check_level_color(check_level_state, accent, text_dim)),
                            );
                        });
                        ui.label(
                            RichText::new(
                                "SOURCE automatically loops the 1 s reference probe through \
                                 the current SEND TRIM and routing.",
                            )
                            .small()
                            .color(text_dim),
                        );
                    }

                    ui.add_space(8.0);
                    let armed = context.capture_armed.value();
                    let can_arm = role == 1
                        && context.capture_session_id.load() != 0
                        && check_level_state == SessionCheckLevelState::Passed;
                    let arm_label = if armed { "DISARM" } else { "ARM" };
                    let arm_button = egui::Button::new(arm_label)
                        .min_size(Vec2::new(110.0, 34.0))
                        .fill(if armed {
                            Color32::from_rgb(148, 48, 55)
                        } else {
                            Color32::from_rgb(31, 107, 100)
                        });
                    if ui.add_enabled(can_arm || armed, arm_button).clicked() {
                        if armed {
                            context.automate(P::CaptureArmed, 0.0);
                            context.capture_control.cancel_training();
                        } else {
                            context.automate(P::CaptureArmed, 1.0);
                            state.message = None;
                        }
                    }
                    if !can_arm && !armed {
                        ui.label(
                            RichText::new(if role == 1 {
                                "Run CHECK LEVEL on the paired RETURN before ARM."
                            } else if role == 2 {
                                "CHECK LEVEL here, then ARM the paired SOURCE."
                            } else {
                                "Choose SOURCE or RETURN and enter a shared session name."
                            })
                            .small()
                            .color(text_dim),
                        );
                    }
                });

            ui.add_space(10.0);
            capture_meters(ui, context, panel_alt, accent, text_dim);
            if let Some(message) = &state.message {
                ui.add_space(6.0);
                ui.label(
                    RichText::new(message)
                        .small()
                        .color(Color32::from_rgb(235, 164, 77)),
                );
            }
        });

        ui.add_space(14.0);
        ui.vertical(|ui| {
            egui::Frame::new()
                .fill(panel)
                .corner_radius(8.0)
                .inner_margin(16.0)
                .show(ui, |ui| {
                    ui.label(RichText::new("HARDWARE AMP SAFETY").strong().color(accent));
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new(
                            "MOT SOURCE\n\
                             ↓  interface LINE OUT\n\
                             REAMP BOX\n\
                             ↓\n\
                             AMPLIFIER\n\
                             ↓  SPEAKER OUT\n\
                             REACTIVE LOAD\n\
                             ↓  RAW / UNFILTERED LINE OUT\n\
                             interface INPUT\n\
                             ↓\n\
                             MOT RETURN",
                        )
                        .monospace()
                        .color(Color32::from_rgb(205, 218, 221)),
                    );
                    ui.add_space(12.0);
                    egui::Frame::new()
                        .fill(Color32::from_rgb(62, 30, 28))
                        .corner_radius(6.0)
                        .inner_margin(10.0)
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(
                                    "NEVER connect an amplifier SPEAKER OUT directly to an \
                                     audio interface.",
                                )
                                .strong()
                                .color(Color32::from_rgb(255, 155, 145)),
                            );
                            ui.label(
                                RichText::new(
                                    "Use a correctly rated reactive load, matching impedance, \
                                     and a speaker cable. Keep the Return track out of the \
                                     Source hardware output to prevent feedback.",
                                )
                                .small(),
                            );
                        });
                    ui.add_space(10.0);
                    ui.label(
                        RichText::new(
                            "Both instances use the same session ID, so SOURCE and RETURN may \
                             live on completely different DAW tracks.",
                        )
                        .small()
                        .color(text_dim),
                    );
                });
        });
    });
}

fn capture_metadata_editor(
    ui: &mut egui::Ui,
    context: &PluginContext<MotStrobeParams>,
    text_dim: Color32,
) {
    egui::Grid::new("capture_metadata_grid")
        .num_columns(2)
        .spacing([8.0, 4.0])
        .show(ui, |ui| {
            capture_text_field(
                ui,
                "AMPLIFIER",
                &context.capture_amplifier,
                "EVH 5153 100W",
                text_dim,
            );
            capture_text_field(
                ui,
                "CHANNEL",
                &context.capture_amplifier_channel,
                "Blue",
                text_dim,
            );
            capture_text_field(
                ui,
                "CONTROLS",
                &context.capture_control_positions,
                "Gain 5, Bass 4…",
                text_dim,
            );
            capture_text_field(
                ui,
                "INTERFACE OUT",
                &context.capture_interface_output,
                "Line Out 3",
                text_dim,
            );
            capture_text_field(
                ui,
                "INTERFACE IN",
                &context.capture_interface_input,
                "Input 1",
                text_dim,
            );
            capture_text_field(
                ui,
                "REAMP BOX",
                &context.capture_reamp_box,
                "Model / setting",
                text_dim,
            );
            capture_text_field(
                ui,
                "REACTIVE LOAD",
                &context.capture_reactive_load,
                "Model / raw output",
                text_dim,
            );

            ui.label(RichText::new("IMPEDANCE").small().color(text_dim));
            let mut impedance = context.capture_load_impedance_ohms.load().min(64);
            if ui
                .add(
                    egui::DragValue::new(&mut impedance)
                        .range(0..=64)
                        .suffix(" Ω")
                        .speed(1.0),
                )
                .changed()
            {
                context.capture_load_impedance_ohms.store(impedance);
            }
            ui.end_row();

            capture_text_field(
                ui,
                "RETURN GAIN",
                &context.capture_return_gain_note,
                "Interface gain / pad",
                text_dim,
            );
        });
}

fn capture_text_field(
    ui: &mut egui::Ui,
    label: &str,
    value: &std::sync::RwLock<String>,
    hint: &str,
    text_dim: Color32,
) {
    ui.label(RichText::new(label).small().color(text_dim));
    let mut edited = read_lock_string(value);
    if ui
        .add(
            egui::TextEdit::singleline(&mut edited)
                .hint_text(hint)
                .desired_width(270.0),
        )
        .changed()
    {
        write_lock_string(value, &edited);
    }
    ui.end_row();
}

fn decode_check_level_status(value: f32) -> SessionCheckLevelState {
    match (value.clamp(0.0, 1.0) * 3.0).round() as u8 {
        1 => SessionCheckLevelState::Measuring,
        2 => SessionCheckLevelState::Passed,
        3 => SessionCheckLevelState::Failed,
        _ => SessionCheckLevelState::Required,
    }
}

const fn check_level_label(state: SessionCheckLevelState) -> &'static str {
    match state {
        SessionCheckLevelState::Required => "REQUIRED",
        SessionCheckLevelState::Measuring => "MEASURING",
        SessionCheckLevelState::Passed => "PASS",
        SessionCheckLevelState::Failed => "FAIL",
    }
}

fn check_level_color(state: SessionCheckLevelState, accent: Color32, text_dim: Color32) -> Color32 {
    match state {
        SessionCheckLevelState::Passed => Color32::from_rgb(77, 190, 134),
        SessionCheckLevelState::Failed => Color32::from_rgb(235, 95, 95),
        SessionCheckLevelState::Measuring => accent,
        SessionCheckLevelState::Required => text_dim,
    }
}

fn capture_meters(
    ui: &mut egui::Ui,
    context: &PluginContext<MotStrobeParams>,
    panel: Color32,
    accent: Color32,
    text_dim: Color32,
) {
    egui::Frame::new()
        .fill(panel)
        .corner_radius(8.0)
        .inner_margin(14.0)
        .show(ui, |ui| {
            let capture_status_index = (context.get_meter(P::CaptureStatus) * 7.0).round() as usize;
            let capture_status = [
                "IDLE",
                "ARMED",
                "WAITING FOR TRANSPORT",
                "PRE-ROLL",
                "CAPTURING",
                "TAIL",
                "READY",
                "INVALID",
            ][capture_status_index.min(7)];
            let worker_status = match context.capture_control.status() {
                CaptureWorkerStatus::Idle => "IDLE",
                CaptureWorkerStatus::Preparing => "PREPARING",
                CaptureWorkerStatus::Ready => "READY",
                CaptureWorkerStatus::Capturing => "CAPTURING",
                CaptureWorkerStatus::Aligning => "ALIGNING",
                CaptureWorkerStatus::Training => "TRAINING",
                CaptureWorkerStatus::ModelSaved => "MODEL SAVED",
                CaptureWorkerStatus::Error => "ERROR",
            };
            ui.horizontal(|ui| {
                ui.label(RichText::new("CAPTURE").color(text_dim));
                ui.label(
                    RichText::new(capture_status)
                        .strong()
                        .monospace()
                        .color(accent),
                );
                ui.separator();
                ui.label(RichText::new("WORKER").color(text_dim));
                ui.label(
                    RichText::new(worker_status)
                        .strong()
                        .monospace()
                        .color(accent),
                );
            });

            let check_level_state =
                decode_check_level_status(context.get_meter(P::CheckLevelStatus));
            let check_level_progress = context.get_meter(P::CheckLevelProgress).clamp(0.0, 1.0);
            let check_level_peak = context.get_meter(P::CheckLevelPeak).clamp(0.0, 1.0);
            let check_level_peak_db = if check_level_peak > 0.0 {
                20.0 * check_level_peak.log10()
            } else {
                f32::NEG_INFINITY
            };
            ui.horizontal(|ui| {
                ui.label(RichText::new("CHECK LEVEL").color(text_dim));
                ui.label(
                    RichText::new(check_level_label(check_level_state))
                        .strong()
                        .monospace()
                        .color(check_level_color(check_level_state, accent, text_dim)),
                );
                if check_level_state == SessionCheckLevelState::Measuring {
                    ui.label(
                        RichText::new(format!("{:.0}%", check_level_progress * 100.0))
                            .monospace()
                            .color(text_dim),
                    );
                }
            });
            ui.add(
                egui::ProgressBar::new(check_level_peak)
                    .text(if check_level_peak_db.is_finite() {
                        format!("CHECK PEAK {check_level_peak_db:.1} dBFS")
                    } else {
                        "CHECK PEAK −∞ dBFS".to_owned()
                    })
                    .fill(
                        if check_level_state == SessionCheckLevelState::Failed
                            || check_level_peak > 0.891_250_9
                        {
                            Color32::from_rgb(188, 55, 60)
                        } else {
                            Color32::from_rgb(36, 142, 132)
                        },
                    ),
            );

            let peak = context.get_meter(P::CapturePeak).clamp(0.0, 1.0);
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
                    .fill(if peak >= 0.891_250_9 {
                        Color32::from_rgb(188, 55, 60)
                    } else {
                        Color32::from_rgb(36, 142, 132)
                    }),
            );

            let progress = context
                .get_meter(P::TrainingProgress)
                .max(context.capture_control.progress())
                .clamp(0.0, 1.0);
            ui.add(
                egui::ProgressBar::new(progress)
                    .text(format!("TRAINING {:.0}%", progress * 100.0))
                    .fill(Color32::from_rgb(52, 115, 164)),
            );
            ui.label(
                RichText::new("Capture starts when the DAW transport enters Play or Record.")
                    .small()
                    .color(text_dim),
            );
            if context.capture_control.status() == CaptureWorkerStatus::Error {
                ui.label(
                    RichText::new(context.capture_control.last_error())
                        .small()
                        .color(Color32::from_rgb(235, 95, 95)),
                );
            }
            if let Some(model) = context.capture_control.last_saved_model() {
                ui.label(
                    RichText::new(format!("SAVED MODEL: {}", model.model_id))
                        .small()
                        .monospace()
                        .color(accent),
                );
            }
        });
}

fn strobe(
    ui: &mut egui::Ui,
    context: &PluginContext<MotStrobeParams>,
    panel: Color32,
    accent: Color32,
) {
    let desired = Vec2::new(ui.available_width(), 300.0);
    let (rect, _) = ui.allocate_exact_size(desired, Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 10.0, panel);

    let detected = decode_note(context.get_meter(P::DetectedNote));
    let cents = (context.get_meter(P::Cents) - 0.5) * 100.0;
    let phase = context.get_meter(P::Phase);
    let active = detected.is_some() && !context.bypass.value();
    let period = 54.0;
    let stripe_width = period * 0.5;
    let offset = phase * period;

    let bright = if active {
        accent
    } else {
        Color32::from_rgb(45, 65, 66)
    };
    let dark = if active {
        Color32::from_rgb(20, 70, 68)
    } else {
        Color32::from_rgb(28, 36, 39)
    };

    let mut x = rect.left() - period + offset;
    let mut alternate = false;
    while x < rect.right() + period {
        let stripe = egui::Rect::from_min_max(
            egui::pos2(x, rect.top() + 12.0),
            egui::pos2(x + stripe_width, rect.bottom() - 12.0),
        );
        painter.rect_filled(stripe, 3.0, if alternate { bright } else { dark });
        alternate = !alternate;
        x += stripe_width;
    }

    painter.line_segment(
        [
            egui::pos2(rect.center().x, rect.top() + 8.0),
            egui::pos2(rect.center().x, rect.bottom() - 8.0),
        ],
        Stroke::new(1.0_f32, Color32::from_white_alpha(105)),
    );
    painter.rect_stroke(
        rect,
        10.0,
        Stroke::new(1.0_f32, Color32::from_rgb(48, 56, 61)),
        StrokeKind::Inside,
    );

    let circle_center = rect.center();
    let circle_radius = 78.0;
    painter.circle_filled(circle_center, circle_radius, Color32::from_white_alpha(215));
    painter.circle_stroke(
        circle_center,
        circle_radius,
        Stroke::new(1.5_f32, Color32::from_white_alpha(235)),
    );

    let note_text = detected.map_or_else(|| "—".to_string(), note_name);
    let cents_text = if detected.is_some() {
        format!("{cents:+.1} c")
    } else {
        "waiting".to_string()
    };
    let ink = Color32::from_rgb(18, 24, 27);
    painter.text(
        egui::pos2(circle_center.x, circle_center.y - 20.0),
        Align2::CENTER_CENTER,
        note_text,
        FontId::monospace(46.0),
        ink,
    );
    painter.text(
        egui::pos2(circle_center.x, circle_center.y + 31.0),
        Align2::CENTER_CENTER,
        cents_text,
        FontId::monospace(20.0),
        ink,
    );
}

fn string_editor(
    ui: &mut egui::Ui,
    context: &PluginContext<MotStrobeParams>,
    panel: Color32,
    accent: Color32,
    text_dim: Color32,
) {
    let note_ids = [
        P::String7Note,
        P::String6Note,
        P::String5Note,
        P::String4Note,
        P::String3Note,
        P::String2Note,
        P::String1Note,
    ];
    let offset_ids = [
        P::String7Offset,
        P::String6Offset,
        P::String5Offset,
        P::String4Offset,
        P::String3Offset,
        P::String2Offset,
        P::String1Offset,
    ];
    let note_values = notes(context);
    let mut offset_values = offsets(context);
    let matched = decode_string(context.get_meter(P::MatchedString));

    egui::Frame::new()
        .fill(panel)
        .corner_radius(8.0)
        .inner_margin(12.0)
        .show(ui, |ui| {
            offset_switch(ui, context, accent, text_dim);
            ui.add_space(6.0);

            egui::Grid::new("string_grid")
                .num_columns(3)
                .spacing([18.0, 6.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label(RichText::new("STRING").color(text_dim));
                    ui.label(RichText::new("REFERENCE NOTE").color(text_dim));
                    ui.label(RichText::new("OFFSET").color(text_dim));
                    ui.end_row();

                    for index in 0..STRING_COUNT {
                        let display_string = 7 - index;
                        let string_label = RichText::new(display_string.to_string()).strong();
                        ui.label(if matched == Some(index) {
                            string_label.color(accent)
                        } else {
                            string_label
                        });

                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(
                                    note_values[index] > 0,
                                    egui::Button::new("◀").frame(false),
                                )
                                .clicked()
                            {
                                let previous = note_values[index] - 1;
                                context.automate(note_ids[index], f64::from(previous) / 127.0);
                            }
                            ui.add_sized(
                                [50.0, 18.0],
                                egui::Label::new(
                                    RichText::new(note_name(note_values[index]))
                                        .monospace()
                                        .strong(),
                                ),
                            );
                            if ui
                                .add_enabled(
                                    note_values[index] < 127,
                                    egui::Button::new("▶").frame(false),
                                )
                                .clicked()
                            {
                                let next = note_values[index] + 1;
                                context.automate(note_ids[index], f64::from(next) / 127.0);
                            }
                        });

                        let response = ui.add(
                            egui::DragValue::new(&mut offset_values[index])
                                .range(OFFSET_MIN..=OFFSET_MAX)
                                .speed(0.1)
                                .fixed_decimals(1)
                                .suffix(" c"),
                        );
                        if response.changed() {
                            offset_values[index] = round_to_tenth(offset_values[index]);
                            let normalized =
                                (offset_values[index] - OFFSET_MIN) / (OFFSET_MAX - OFFSET_MIN);
                            context.automate(offset_ids[index], f64::from(normalized));
                        }

                        ui.end_row();
                    }
                });
        });
}

fn offset_switch(
    ui: &mut egui::Ui,
    context: &PluginContext<MotStrobeParams>,
    accent: Color32,
    text_dim: Color32,
) {
    let enabled = context.offsets_enabled.value();
    ui.horizontal(|ui| {
        ui.label(RichText::new("OFFSETS").strong().color(text_dim));
        let (rect, response) = ui.allocate_exact_size(Vec2::new(46.0, 24.0), Sense::click());
        if response.clicked() {
            context.automate(P::OffsetsEnabled, if enabled { 0.0 } else { 1.0 });
        }
        response.on_hover_text("Temporarily compare reference notes against pure 12-TET");

        let track_color = if enabled {
            Color32::from_rgb(31, 107, 100)
        } else {
            Color32::from_rgb(52, 58, 62)
        };
        ui.painter()
            .rect_filled(rect, rect.height() / 2.0, track_color);
        let knob_x = if enabled {
            rect.right() - rect.height() / 2.0
        } else {
            rect.left() + rect.height() / 2.0
        };
        ui.painter().circle_filled(
            egui::pos2(knob_x, rect.center().y),
            8.5,
            if enabled {
                accent
            } else {
                Color32::from_rgb(172, 180, 184)
            },
        );
        ui.label(
            RichText::new(if enabled { "ON" } else { "OFF" })
                .monospace()
                .color(if enabled { accent } else { text_dim }),
        );
    });
}

fn current_tone_view(context: &PluginContext<MotStrobeParams>) -> ToneView {
    let ir_processing = IrUiMode::from_param(context.ir_processing.value());
    ToneView {
        input_gain_db: round_to_tenth(context.input_gain.value()),
        tight_percent: context.tight.value(),
        bite_percent: context.bite.value(),
        ir_path: read_lock_string(&context.selected_ir_path),
        ir_reference: persisted_ir_reference(context, ir_processing)
            .ok()
            .flatten(),
        ir_processing,
    }
}

fn tone_is_dirty(state: &EditorState, context: &PluginContext<MotStrobeParams>) -> bool {
    let selected_id = read_lock_string(&context.selected_model_id);
    if selected_id.is_empty() {
        return false;
    }
    state
        .baseline
        .as_ref()
        .is_none_or(|baseline| !baseline.equivalent(&current_tone_view(context)))
}

fn selected_model_entry(
    state: &EditorState,
    context: &PluginContext<MotStrobeParams>,
) -> Option<ModelEntry> {
    let selected_id = read_lock_string(&context.selected_model_id);
    let selected_hash = read_lock_string(&context.selected_model_sha256);
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

fn selected_model_reference(context: &MotStrobeParams) -> Option<crate::model::ModelRef> {
    let model_id = read_lock_string(&context.selected_model_id);
    if model_id.is_empty() {
        return None;
    }
    let sha256 = read_lock_string(&context.selected_model_sha256)
        .parse::<Sha256Digest>()
        .ok()?;
    let mut filename_hint = read_lock_string(&context.selected_model_filename_hint);
    if filename_hint.is_empty() {
        filename_hint = format!("{model_id}.motmodel");
    }
    Some(crate::model::ModelRef {
        model_id,
        sha256,
        filename_hint,
    })
}

fn selected_reference_is_current(
    context: &MotStrobeParams,
    reference: &crate::model::ModelRef,
) -> bool {
    let current_id = read_lock_string(&context.selected_model_id);
    let current_sha256 = read_lock_string(&context.selected_model_sha256);
    current_id == reference.model_id
        && current_sha256.eq_ignore_ascii_case(&reference.sha256.to_string())
}

fn selected_model_identity_matches(
    context: &MotStrobeParams,
    model_id: &str,
    model_sha256: &str,
) -> bool {
    let current_id = read_lock_string(&context.selected_model_id);
    let current_sha256 = read_lock_string(&context.selected_model_sha256);
    current_id == model_id && current_sha256.eq_ignore_ascii_case(model_sha256)
}

fn request_select_model(
    state: &mut EditorState,
    context: &PluginContext<MotStrobeParams>,
    entry: ModelEntry,
) {
    let guard_model_id = read_lock_string(&context.selected_model_id);
    let guard_model_sha256 = read_lock_string(&context.selected_model_sha256);
    let display_name = entry.metadata.display_name.clone();
    match submit_library_task(
        context,
        LibraryTaskOperation::LoadTone {
            entry: Box::new(entry),
            guard_model_id,
            guard_model_sha256,
        },
    ) {
        Ok(()) => state.message = Some(format!("Loading “{display_name}”…")),
        Err(error) => state.message = Some(error),
    }
}

fn apply_loaded_model(
    state: &mut EditorState,
    context: &PluginContext<MotStrobeParams>,
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
    write_lock_string(&context.selected_model_id, &entry.reference.model_id);
    write_lock_string(
        &context.selected_model_sha256,
        &entry.reference.sha256.to_string(),
    );
    write_lock_string(
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
            let path = library.paths().irs.join(&ir.filename_hint);
            let mode = match ir.processing {
                IrProcessingMode::MinimumPhaseAutoTrim => IrUiMode::MinimumPhase,
                IrProcessingMode::Raw => IrUiMode::Raw,
            };
            (path.to_string_lossy().into_owned(), Some(ir.clone()), mode)
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

fn apply_tone_view(context: &PluginContext<MotStrobeParams>, view: &ToneView) {
    automate_linear(context, P::InputGain, view.input_gain_db, -24.0, 24.0);
    automate_linear(context, P::Tight, view.tight_percent, 0.0, 100.0);
    automate_linear(context, P::Bite, view.bite_percent, 0.0, 100.0);
    context.automate(P::IrProcessing, view.ir_processing.normalized());
    write_selected_ir_state(context, &view.ir_path, view.ir_reference.as_ref());
}

fn request_save_current_tone(
    state: &mut EditorState,
    context: &PluginContext<MotStrobeParams>,
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
        schema_version: crate::model_library::TONE_SETTINGS_VERSION,
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
    context: &PluginContext<MotStrobeParams>,
    settings: ToneSettings,
    result: Result<(), String>,
) {
    match result {
        Ok(()) => {
            let current_model_matches = selected_model_identity_matches(
                context,
                &settings.model_id,
                &settings.model_sha256.to_string(),
            );
            if current_model_matches && let Some(library) = state.library.as_ref() {
                state.baseline = Some(tone_view_from_settings(&settings, library));
            }
            state.message = Some("Model settings saved".to_owned());

            let switch_target = if state.switch_after_save && current_model_matches {
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
    context: &MotStrobeParams,
    mode: IrUiMode,
) -> Result<Option<IrReference>, String> {
    let path = read_lock_string(&context.selected_ir_path);
    if path.is_empty() {
        return Ok(None);
    }
    let ir_id = read_lock_string(&context.selected_ir_id);
    let sha256 = read_lock_string(&context.selected_ir_sha256)
        .parse::<Sha256Digest>()
        .map_err(|error| format!("Selected IR SHA-256 is invalid: {error}"))?;
    let filename_hint = read_lock_string(&context.selected_ir_filename_hint);
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
    context: &PluginContext<MotStrobeParams>,
    state: &EditorState,
    path: &Path,
) -> Result<(), String> {
    let mode = IrUiMode::from_param(context.ir_processing.value());
    let metadata = state
        .ir_metadata
        .get(path)
        .ok_or_else(|| "This IR is missing or failed exact-content validation".to_owned())?;
    let mut reference = metadata.reference();
    reference.processing = mode.library_mode();
    write_selected_ir_state(context, &path.to_string_lossy(), Some(&reference));
    bump_runtime_generation(context);
    Ok(())
}

fn select_imported_ir(context: &PluginContext<MotStrobeParams>, imported: &ImportedIr) {
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

fn revert_current_tone(state: &mut EditorState, context: &PluginContext<MotStrobeParams>) {
    if let Some(baseline) = state.baseline.clone() {
        apply_tone_view(context, &baseline);
        bump_runtime_generation(context);
        state.message = Some("Saved model settings restored".to_owned());
    }
}

fn write_selected_ir_state(context: &MotStrobeParams, path: &str, reference: Option<&IrReference>) {
    write_lock_string(&context.selected_ir_path, path);
    if let Some(reference) = reference {
        write_lock_string(&context.selected_ir_id, &reference.ir_id);
        write_lock_string(&context.selected_ir_sha256, &reference.sha256.to_string());
        write_lock_string(&context.selected_ir_filename_hint, &reference.filename_hint);
    } else {
        write_lock_string(&context.selected_ir_id, "");
        write_lock_string(&context.selected_ir_sha256, "");
        write_lock_string(&context.selected_ir_filename_hint, "");
    }
}

fn clear_selected_ir(context: &PluginContext<MotStrobeParams>) {
    write_selected_ir_state(context, "", None);
    bump_runtime_generation(context);
}

fn bump_runtime_generation(context: &PluginContext<MotStrobeParams>) {
    let next = context.runtime_generation.load().wrapping_add(1).max(1);
    context.runtime_generation.store(next);
}

fn automate_linear(
    context: &PluginContext<MotStrobeParams>,
    id: P,
    value: f32,
    minimum: f32,
    maximum: f32,
) {
    let normalized = ((value - minimum) / (maximum - minimum)).clamp(0.0, 1.0);
    context.automate(id, f64::from(normalized));
}

fn automate_discrete(
    context: &PluginContext<MotStrobeParams>,
    id: P,
    value: i64,
    minimum: i64,
    maximum: i64,
) {
    let normalized = (value - minimum) as f64 / (maximum - minimum) as f64;
    context.automate(id, normalized.clamp(0.0, 1.0));
}

fn read_lock_string(lock: &std::sync::RwLock<String>) -> String {
    lock.read()
        .map_or_else(|_| String::new(), |value| value.clone())
}

fn write_lock_string(lock: &std::sync::RwLock<String>, value: &str) {
    if let Ok(mut destination) = lock.write() {
        destination.clear();
        destination.push_str(value);
    }
}

fn session_id_from_name(name: &str) -> u64 {
    let name = name.trim();
    if name.is_empty() {
        return 0;
    }
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in name.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash.max(1)
}

fn display_filename(path: &Path) -> String {
    path.file_name()
        .and_then(std::ffi::OsStr::to_str)
        .map_or_else(|| path.display().to_string(), str::to_owned)
}

pub(crate) fn decode_string(encoded: f32) -> Option<usize> {
    if encoded <= 0.0 {
        return None;
    }
    let decoded = (encoded * 8.0).round() as isize - 1;
    (0..STRING_COUNT as isize)
        .contains(&decoded)
        .then_some(decoded as usize)
}

pub(crate) fn decode_note(encoded: f32) -> Option<u8> {
    if encoded <= 0.0 {
        return None;
    }
    let decoded = (encoded * 128.0).round() as i16 - 1;
    (0..=127).contains(&decoded).then_some(decoded as u8)
}

pub(crate) fn note_name(midi_note: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
    ];
    let pitch_class = usize::from(midi_note % 12);
    let octave = i16::from(midi_note / 12) - 1;
    format!("{}{octave}", NAMES[pitch_class])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tone_sidecar_ir_identity_survives_project_state_selection() {
        let params = MotStrobeParams::new();
        let reference = IrReference {
            ir_id: "ir-exact-content-id".to_owned(),
            sha256: "7e".repeat(32).parse().unwrap(),
            filename_hint: "7e7e7e7e.wav".to_owned(),
            processing: IrProcessingMode::Raw,
        };

        write_selected_ir_state(&params, "/managed/IRs/7e7e7e7e.wav", Some(&reference));
        let restored = persisted_ir_reference(&params, IrUiMode::Raw)
            .unwrap()
            .expect("selected IR reference");

        assert_eq!(restored, reference);
        assert_eq!(
            read_lock_string(&params.selected_ir_path),
            "/managed/IRs/7e7e7e7e.wav"
        );
    }

    #[test]
    fn model_selection_identity_uses_only_id_and_sha() {
        let params = MotStrobeParams::new();
        let sha256: Sha256Digest = "3a".repeat(32).parse().unwrap();
        write_lock_string(&params.selected_model_id, "amp-blue");
        write_lock_string(&params.selected_model_sha256, &sha256.to_string());
        write_lock_string(&params.selected_model_filename_hint, "old-name.motmodel");

        let renamed = crate::model::ModelRef {
            model_id: "amp-blue".to_owned(),
            sha256,
            filename_hint: "renamed.motmodel".to_owned(),
        };
        assert!(selected_reference_is_current(&params, &renamed));

        let other_revision = crate::model::ModelRef {
            sha256: "4b".repeat(32).parse().unwrap(),
            ..renamed
        };
        assert!(!selected_reference_is_current(&params, &other_revision));
    }

    #[test]
    fn fixed_panel_stays_vertical_and_within_its_requested_width() {
        let mut direction = None;
        let mut response_width = None;
        egui::__run_test_ui(|ui| {
            ui.set_width(800.0);
            ui.horizontal(|ui| {
                let response = fixed_vertical_panel(ui, Vec2::new(350.0, 120.0), |panel_ui| {
                    direction = Some(panel_ui.layout().main_dir);
                    panel_ui.add(
                        egui::Label::new("a/very/long/model/library/path/".repeat(32)).truncate(),
                    );
                });
                response_width = Some(response.response.rect.width());
            });
        });

        assert_eq!(direction, Some(egui::Direction::TopDown));
        assert!(
            response_width.is_some_and(|width| width <= 350.5),
            "fixed panel expanded to {response_width:?}"
        );
    }
}
