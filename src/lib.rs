#[cfg(test)]
mod acceptance;
mod amp;
mod cabinet;
mod capture;
mod capture_runtime;
mod editor;
mod model;
mod model_library;
mod runtime;
mod signal_chain;
mod trainer;
mod tuner;
mod wav_io;

use std::path::PathBuf;
use std::process::Command;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use crossbeam_queue::ArrayQueue;
use truce::prelude::*;
use truce_egui::EguiEditor;

use amp::AmpControls;
use capture::{
    CHECK_LEVEL_DURATION_SAMPLES, CaptureInvalidation, CaptureRole, CaptureSessionId,
    CaptureTarget, CheckLevelMeter, CheckLevelState, SessionCheckLevelSnapshot,
    SessionCheckLevelState, TransportInfo as CaptureTransportInfo,
};
use capture_runtime::{
    CaptureWorkerStatus, PrepareCaptureTask, PreparedCaptureRuntime, StartTrainingTask,
};
use editor::{MotStrobeUi, WINDOW_SIZE};
use model::{ModelRef, Sha256Digest};
use model_library::{
    ImportedIr, IrLibraryScan, IrProcessingMode, IrReference, ModelEntry, ModelLibrary, ModelScan,
    ToneSettings,
};
use runtime::{
    PreparedRuntime, RuntimeLoadRequest, RuntimeLoader, RuntimeMuteReason, RuntimeUpdate,
};
use signal_chain::{GuitarSignalChain, OutputMute, RuntimeApplyStatus};
use tuner::{PitchAnalysis, STRING_COUNT, TunerEngine, cents_ratio, midi_to_hz};

static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Params)]
pub struct MotStrobeParams {
    #[param(name = "Bypass", flags = "automatable | bypass")]
    pub bypass: BoolParam,

    #[param(name = "Mute", flags = "automatable")]
    pub mute: BoolParam,

    #[param(
        name = "Input Gain",
        range = "linear(-24, 24)",
        default = 0,
        flags = "automatable"
    )]
    pub input_gain: FloatParam,

    #[param(
        name = "Tight",
        range = "linear(0, 100)",
        default = 0,
        flags = "automatable"
    )]
    pub tight: FloatParam,

    #[param(
        name = "Bite",
        range = "linear(0, 100)",
        default = 0,
        flags = "automatable"
    )]
    pub bite: FloatParam,

    #[param(name = "Capture Role", range = "discrete(0, 2)", default = 0)]
    pub capture_role: IntParam,

    #[param(name = "Capture Target", range = "discrete(0, 1)", default = 0)]
    pub capture_target: IntParam,

    #[param(name = "Capture Armed")]
    pub capture_armed: BoolParam,

    #[param(
        name = "Capture Send Trim",
        range = "linear(-40, 0)",
        default = -20
    )]
    pub capture_send_trim: FloatParam,

    #[param(name = "Max Passes", range = "discrete(1, 400)", default = 400)]
    pub max_passes: IntParam,

    #[param(name = "IR Processing", range = "discrete(0, 1)", default = 0)]
    pub ir_processing: IntParam,

    #[param(name = "Offsets Enabled", flags = "automatable", default = true)]
    pub offsets_enabled: BoolParam,

    // Seven-string B standard: B1 E2 A2 D3 G3 B3 E4.
    #[param(name = "String 7 Note", range = "discrete(0, 127)", default = 35)]
    pub string_7_note: IntParam,
    #[param(name = "String 7 Offset", range = "linear(-25, 25)", default = 0)]
    pub string_7_offset: FloatParam,

    #[param(name = "String 6 Note", range = "discrete(0, 127)", default = 40)]
    pub string_6_note: IntParam,
    #[param(name = "String 6 Offset", range = "linear(-25, 25)", default = 0)]
    pub string_6_offset: FloatParam,

    #[param(name = "String 5 Note", range = "discrete(0, 127)", default = 45)]
    pub string_5_note: IntParam,
    #[param(name = "String 5 Offset", range = "linear(-25, 25)", default = 0)]
    pub string_5_offset: FloatParam,

    #[param(name = "String 4 Note", range = "discrete(0, 127)", default = 50)]
    pub string_4_note: IntParam,
    #[param(name = "String 4 Offset", range = "linear(-25, 25)", default = 0)]
    pub string_4_offset: FloatParam,

    #[param(name = "String 3 Note", range = "discrete(0, 127)", default = 55)]
    pub string_3_note: IntParam,
    #[param(name = "String 3 Offset", range = "linear(-25, 25)", default = 0)]
    pub string_3_offset: FloatParam,

    #[param(name = "String 2 Note", range = "discrete(0, 127)", default = 59)]
    pub string_2_note: IntParam,
    #[param(name = "String 2 Offset", range = "linear(-25, 25)", default = 0)]
    pub string_2_offset: FloatParam,

    #[param(name = "String 1 Note", range = "discrete(0, 127)", default = 64)]
    pub string_1_note: IntParam,
    #[param(name = "String 1 Offset", range = "linear(-25, 25)", default = 0)]
    pub string_1_offset: FloatParam,

    /// Non-automatable project state. Audio code only observes
    /// `runtime_generation`; model paths are read by loader/UI threads.
    #[persist]
    pub selected_model_id: RwLock<String>,
    #[persist]
    pub selected_model_sha256: RwLock<String>,
    #[persist]
    pub selected_model_filename_hint: RwLock<String>,
    #[persist]
    pub selected_ir_path: RwLock<String>,
    #[persist]
    pub selected_ir_id: RwLock<String>,
    #[persist]
    pub selected_ir_sha256: RwLock<String>,
    #[persist]
    pub selected_ir_filename_hint: RwLock<String>,
    #[persist]
    pub runtime_generation: AtomicCell<u64>,
    #[persist]
    pub capture_session_id: AtomicCell<u64>,
    #[persist]
    pub capture_session_name: RwLock<String>,
    #[persist]
    pub capture_model_name: RwLock<String>,
    #[persist]
    pub capture_amplifier: RwLock<String>,
    #[persist]
    pub capture_amplifier_channel: RwLock<String>,
    #[persist]
    pub capture_control_positions: RwLock<String>,
    #[persist]
    pub capture_interface_output: RwLock<String>,
    #[persist]
    pub capture_interface_input: RwLock<String>,
    #[persist]
    pub capture_reamp_box: RwLock<String>,
    #[persist]
    pub capture_reactive_load: RwLock<String>,
    #[persist]
    pub capture_load_impedance_ohms: AtomicCell<u64>,
    #[persist]
    pub capture_return_gain_note: RwLock<String>,

    /// Non-persistent channels shared by the UI, loader workers, and the
    /// audio callback. All audio-side operations are bounded and lock-free.
    #[skip]
    pub runtime_mailbox: Arc<runtime::RuntimeMailbox>,
    /// Invalidates an in-flight loader result when the host resets or changes
    /// its audio configuration without changing the selected model.
    #[skip]
    pub runtime_request_epoch: AtomicCell<u64>,
    #[skip]
    pub capture_control: Arc<capture_runtime::CaptureControl>,
    /// GUI-to-worker IR import handoff. WAV I/O, hashing, validation, and
    /// minimum-phase preparation never run on either the GUI or audio thread.
    #[skip]
    pub ir_import_control: Arc<IrImportControl>,
    /// GUI-to-worker model/IR library operations. Model hashing, directory
    /// scans, tone sidecar reads/writes, and folder preparation never run on
    /// either the GUI or audio thread.
    #[skip]
    pub library_control: Arc<LibraryControl>,
    #[skip]
    pub instance_id: AtomicCell<u64>,
    /// UI-to-audio one-shot command. It is deliberately non-persistent: a
    /// restored project must perform a fresh physical Return level check.
    #[skip]
    pub check_level_trigger_generation: AtomicU64,

    #[meter]
    pub detected_note: MeterSlot,
    #[meter]
    pub matched_string: MeterSlot,
    #[meter]
    pub cents: MeterSlot,
    #[meter]
    pub phase: MeterSlot,
    #[meter]
    pub runtime_status: MeterSlot,
    #[meter]
    pub capture_status: MeterSlot,
    #[meter]
    pub capture_peak: MeterSlot,
    #[meter]
    pub check_level_status: MeterSlot,
    #[meter]
    pub check_level_progress: MeterSlot,
    #[meter]
    pub check_level_peak: MeterSlot,
    #[meter]
    pub training_progress: MeterSlot,
}

pub(crate) use MotStrobeParamsParamId as P;

const IR_IMPORT_OUTCOME_CAPACITY: usize = 2;

#[derive(Debug)]
pub(crate) enum IrImportOutcome {
    Imported(Box<ImportedIr>),
    Error(String),
}

#[derive(Debug)]
pub struct IrImportControl {
    busy: AtomicBool,
    outcomes: ArrayQueue<IrImportOutcome>,
}

impl Default for IrImportControl {
    fn default() -> Self {
        Self {
            busy: AtomicBool::new(false),
            outcomes: ArrayQueue::new(IR_IMPORT_OUTCOME_CAPACITY),
        }
    }
}

impl IrImportControl {
    pub(crate) fn try_begin(&self) -> bool {
        self.busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    pub(crate) fn cancel_begin(&self) {
        self.busy.store(false, Ordering::Release);
    }

    pub(crate) fn is_busy(&self) -> bool {
        self.busy.load(Ordering::Acquire)
    }

    pub(crate) fn take_outcome(&self) -> Option<IrImportOutcome> {
        self.outcomes.pop()
    }

    fn finish(&self, outcome: IrImportOutcome) {
        let _ = self.outcomes.force_push(outcome);
        self.busy.store(false, Ordering::Release);
    }
}

#[derive(Debug)]
pub(crate) struct ImportIrTask {
    pub source: PathBuf,
}

impl BackgroundTask for ImportIrTask {
    type Params = MotStrobeParams;
    const SERIALIZED: bool = true;

    fn run(self, params: &Self::Params) {
        let outcome = ModelLibrary::for_current_user()
            .and_then(|library| library.import_ir(&self.source))
            .map_or_else(
                |error| IrImportOutcome::Error(error.to_string()),
                |imported| IrImportOutcome::Imported(Box::new(imported)),
            );
        params.ir_import_control.finish(outcome);
    }
}

const LIBRARY_OUTCOME_CAPACITY: usize = 4;

#[derive(Debug)]
pub(crate) struct SelectedToneSnapshot {
    pub reference: ModelRef,
    pub tone: Result<Option<ToneSettings>, String>,
}

#[derive(Debug)]
pub(crate) struct LibrarySnapshot {
    pub models: Result<ModelScan, String>,
    pub irs: Result<IrLibraryScan, String>,
    pub selected_tone: Option<SelectedToneSnapshot>,
}

#[derive(Debug)]
pub(crate) enum LibraryOutcome {
    Scanned {
        request_id: u64,
        result: Result<Box<LibrarySnapshot>, String>,
    },
    ToneLoaded {
        request_id: u64,
        entry: Box<ModelEntry>,
        guard_model_id: String,
        guard_model_sha256: String,
        result: Result<Option<ToneSettings>, String>,
    },
    ToneSaved {
        request_id: u64,
        settings: ToneSettings,
        result: Result<(), String>,
    },
    FolderOpened {
        request_id: u64,
        result: Result<(), String>,
    },
}

impl LibraryOutcome {
    #[must_use]
    pub(crate) const fn request_id(&self) -> u64 {
        match self {
            Self::Scanned { request_id, .. }
            | Self::ToneLoaded { request_id, .. }
            | Self::ToneSaved { request_id, .. }
            | Self::FolderOpened { request_id, .. } => *request_id,
        }
    }
}

#[derive(Debug)]
pub struct LibraryControl {
    busy: AtomicBool,
    next_request_id: AtomicU64,
    latest_request_id: AtomicU64,
    outcomes: ArrayQueue<LibraryOutcome>,
}

impl Default for LibraryControl {
    fn default() -> Self {
        Self {
            busy: AtomicBool::new(false),
            next_request_id: AtomicU64::new(0),
            latest_request_id: AtomicU64::new(0),
            outcomes: ArrayQueue::new(LIBRARY_OUTCOME_CAPACITY),
        }
    }
}

impl LibraryControl {
    fn next_request_id(&self) -> u64 {
        self.next_request_id
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1)
            .max(1)
    }

    pub(crate) fn try_begin(&self) -> Option<u64> {
        self.busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()?;
        let request_id = self.next_request_id();
        self.latest_request_id.store(request_id, Ordering::Release);
        Some(request_id)
    }

    /// Invalidates an outcome left behind by a closed editor without
    /// pretending that an already-running worker has stopped. Its eventual
    /// result remains safely drainable but can no longer mutate reopened UI or
    /// restored DAW state.
    pub(crate) fn invalidate_pending(&self) {
        let invalidation_id = self.next_request_id();
        self.latest_request_id
            .store(invalidation_id, Ordering::Release);
    }

    pub(crate) fn cancel_begin(&self, request_id: u64) {
        if self.latest_request_id.load(Ordering::Acquire) == request_id {
            self.busy.store(false, Ordering::Release);
        }
    }

    pub(crate) fn is_busy(&self) -> bool {
        self.busy.load(Ordering::Acquire)
    }

    pub(crate) fn is_current(&self, request_id: u64) -> bool {
        self.latest_request_id.load(Ordering::Acquire) == request_id
    }

    pub(crate) fn take_outcome(&self) -> Option<LibraryOutcome> {
        self.outcomes.pop()
    }

    fn finish(&self, outcome: LibraryOutcome) {
        let _ = self.outcomes.force_push(outcome);
        self.busy.store(false, Ordering::Release);
    }
}

#[derive(Debug)]
pub(crate) enum LibraryTaskOperation {
    Scan {
        selected_model: Option<ModelRef>,
    },
    LoadTone {
        entry: Box<ModelEntry>,
        guard_model_id: String,
        guard_model_sha256: String,
    },
    SaveTone {
        model_reference: ModelRef,
        settings: ToneSettings,
    },
    OpenFolder,
}

#[derive(Debug)]
pub(crate) struct LibraryTask {
    pub request_id: u64,
    pub operation: LibraryTaskOperation,
}

impl BackgroundTask for LibraryTask {
    type Params = MotStrobeParams;
    const SERIALIZED: bool = true;

    fn run(self, params: &Self::Params) {
        let outcome = run_library_task(self);
        params.library_control.finish(outcome);
    }
}

fn run_library_task(task: LibraryTask) -> LibraryOutcome {
    let request_id = task.request_id;
    let library = match ModelLibrary::for_current_user() {
        Ok(library) => library,
        Err(error) => {
            let message = error.to_string();
            return match task.operation {
                LibraryTaskOperation::Scan { .. } => LibraryOutcome::Scanned {
                    request_id,
                    result: Err(message),
                },
                LibraryTaskOperation::LoadTone {
                    entry,
                    guard_model_id,
                    guard_model_sha256,
                } => LibraryOutcome::ToneLoaded {
                    request_id,
                    entry,
                    guard_model_id,
                    guard_model_sha256,
                    result: Err(message),
                },
                LibraryTaskOperation::SaveTone {
                    model_reference: _,
                    settings,
                } => LibraryOutcome::ToneSaved {
                    request_id,
                    settings,
                    result: Err(message),
                },
                LibraryTaskOperation::OpenFolder => LibraryOutcome::FolderOpened {
                    request_id,
                    result: Err(message),
                },
            };
        }
    };

    match task.operation {
        LibraryTaskOperation::Scan { selected_model } => {
            let models = library
                .scan(&runtime::tracking_runtime_limits())
                .map_err(|error| error.to_string());
            let irs = library.scan_irs().map_err(|error| error.to_string());
            let selected_tone = selected_model.map(|reference| {
                let tone = library
                    .load_exact(&reference, &runtime::tracking_runtime_limits())
                    .map_err(|error| error.to_string())
                    .and_then(|_| {
                        library
                            .load_tone(&reference)
                            .map_err(|error| error.to_string())
                    });
                SelectedToneSnapshot { reference, tone }
            });
            let result = Ok(Box::new(LibrarySnapshot {
                models,
                irs,
                selected_tone,
            }));
            LibraryOutcome::Scanned { request_id, result }
        }
        LibraryTaskOperation::LoadTone {
            entry,
            guard_model_id,
            guard_model_sha256,
        } => {
            let result = library
                .load_exact(&entry.reference, &runtime::tracking_runtime_limits())
                .map_err(|error| error.to_string())
                .and_then(|_| {
                    library
                        .load_tone(&entry.reference)
                        .map_err(|error| error.to_string())
                });
            LibraryOutcome::ToneLoaded {
                request_id,
                entry,
                guard_model_id,
                guard_model_sha256,
                result,
            }
        }
        LibraryTaskOperation::SaveTone {
            model_reference,
            settings,
        } => {
            let result = library
                .load_exact(&model_reference, &runtime::tracking_runtime_limits())
                .map_err(|error| error.to_string())
                .and_then(|_| {
                    library
                        .save_tone(&settings)
                        .map_err(|error| error.to_string())
                });
            LibraryOutcome::ToneSaved {
                request_id,
                settings,
                result,
            }
        }
        LibraryTaskOperation::OpenFolder => {
            let result = open_library_folder_on_worker(&library);
            LibraryOutcome::FolderOpened { request_id, result }
        }
    }
}

fn open_library_folder_on_worker(library: &ModelLibrary) -> Result<(), String> {
    library
        .ensure_directories()
        .map_err(|error| error.to_string())?;

    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(&library.paths().plugin_root)
            .spawn()
            .map_err(|error| format!("Cannot open model library: {error}"))?;
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err(format!(
            "Open this folder manually: {}",
            library.paths().plugin_root.display()
        ))
    }
}

#[derive(Clone, Copy, Debug)]
struct LoadRuntimeTask {
    generation: u64,
    request_epoch: u64,
    host_sample_rate_hz: u32,
    host_max_block_size: usize,
}

impl BackgroundTask for LoadRuntimeTask {
    type Params = MotStrobeParams;
    const SERIALIZED: bool = true;

    fn run(self, params: &Self::Params) {
        params.runtime_mailbox.drain_retired();
        let model_id = read_shared_string(&params.selected_model_id);
        if model_id.is_empty() {
            // A fresh instance intentionally has no amp model. Keep the
            // startup path bit-exact instead of treating that as corruption.
            let mut runtime = PreparedRuntime::transparent();
            runtime.reset(&AudioConfig::new(
                f64::from(self.host_sample_rate_hz),
                self.host_max_block_size.max(1),
            ));
            if runtime_request_is_current(params, self.generation, self.request_epoch) {
                params.runtime_mailbox.publish_latest(RuntimeUpdate::Ready {
                    generation: self.generation,
                    runtime: Box::new(runtime),
                });
            }
            return;
        }

        let result = (|| -> Result<(), RuntimeMuteReason> {
            let digest = Sha256Digest::from_str(&read_shared_string(&params.selected_model_sha256))
                .map_err(|_| RuntimeMuteReason::CorruptConfiguration)?;
            let model_reference = ModelRef {
                model_id: model_id.clone(),
                sha256: digest,
                filename_hint: {
                    let hint = read_shared_string(&params.selected_model_filename_hint);
                    if hint.is_empty() {
                        format!("{model_id}.motmodel")
                    } else {
                        hint
                    }
                },
            };
            let mut tone = ToneSettings::defaults_for(&model_reference);
            tone.input_gain_db = params.input_gain.value();
            tone.tight_percent = params.tight.value();
            tone.bite_percent = params.bite.value();

            let ir_path_string = read_shared_string(&params.selected_ir_path);
            let ir_path = if ir_path_string.is_empty() {
                None
            } else {
                let path = PathBuf::from(&ir_path_string);
                let ir_digest =
                    Sha256Digest::from_str(&read_shared_string(&params.selected_ir_sha256))
                        .map_err(|_| RuntimeMuteReason::CorruptConfiguration)?;
                let ir_id = read_shared_string(&params.selected_ir_id);
                let filename_hint = read_shared_string(&params.selected_ir_filename_hint);
                if ir_id.is_empty() || filename_hint.is_empty() {
                    return Err(RuntimeMuteReason::CorruptConfiguration);
                }
                tone.ir = Some(IrReference {
                    ir_id,
                    sha256: ir_digest,
                    filename_hint,
                    processing: if params.ir_processing.value_i32() == 1 {
                        IrProcessingMode::Raw
                    } else {
                        IrProcessingMode::MinimumPhaseAutoTrim
                    },
                });
                Some(path)
            };

            let library = ModelLibrary::for_current_user()
                .map_err(|_| RuntimeMuteReason::CorruptConfiguration)?;
            let loader = RuntimeLoader::new(library);
            let mut request = RuntimeLoadRequest::new(self.generation, model_reference);
            request.tone = Some(tone);
            request.ir_path = ir_path;
            request.host_sample_rate_hz = self.host_sample_rate_hz;
            request.host_max_block_size = self.host_max_block_size;
            let outcome = loader.load(request);
            if runtime_request_is_current(params, self.generation, self.request_epoch) {
                params.runtime_mailbox.publish_latest(outcome.update);
            }
            Ok(())
        })();

        if let Err(reason) = result
            && runtime_request_is_current(params, self.generation, self.request_epoch)
        {
            params.runtime_mailbox.publish_latest(RuntimeUpdate::Mute {
                generation: self.generation,
                reason,
            });
        }
    }
}

fn runtime_request_is_current(
    params: &MotStrobeParams,
    generation: u64,
    request_epoch: u64,
) -> bool {
    params.runtime_generation.load() == generation
        && params.runtime_request_epoch.load() == request_epoch
}

fn advance_runtime_request_epoch(params: &MotStrobeParams) -> u64 {
    let next = params.runtime_request_epoch.load().wrapping_add(1).max(1);
    params.runtime_request_epoch.store(next);
    next
}

fn read_shared_string(value: &RwLock<String>) -> String {
    value
        .read()
        .map_or_else(|_| String::new(), |value| value.clone())
}

pub struct MotStrobe {
    tuner: TunerEngine,
    signal_chain: GuitarSignalChain,
    capture_runtime: Box<PreparedCaptureRuntime>,
    retired_capture_runtime: Option<Box<PreparedCaptureRuntime>>,
    pending_training: Option<StartTrainingTask>,
    pending_recycled: Option<capture::CompletedReturn>,
    output_mute: OutputMute,
    sample_rate: f32,
    live_path_sample_rate_compatible: bool,
    max_block_size: usize,
    strobe_phase: f32,
    displayed_note: Option<u8>,
    reference_hz: f32,
    requested_runtime_generation: u64,
    installed_capture_generation: u64,
    requested_capture_role: u8,
    requested_capture_session: u64,
    capture_request_generation: u64,
    capture_arm_started: bool,
    check_level: CheckLevelMeter,
    observed_check_level_trigger: u64,
    active_check_level_generation: u64,
    source_check_level_generation: u64,
    source_check_level_position: usize,
    runtime_status: f32,
}

impl Default for MotStrobe {
    fn default() -> Self {
        let default_session = CaptureSessionId::new(1).expect("one is a valid capture session");
        Self {
            tuner: TunerEngine::default(),
            signal_chain: GuitarSignalChain::default(),
            capture_runtime: Box::new(PreparedCaptureRuntime::normal(0, default_session)),
            retired_capture_runtime: None,
            pending_training: None,
            pending_recycled: None,
            output_mute: OutputMute::default(),
            sample_rate: 48_000.0,
            live_path_sample_rate_compatible: true,
            max_block_size: 0,
            strobe_phase: 0.0,
            displayed_note: None,
            reference_hz: 0.0,
            requested_runtime_generation: u64::MAX,
            installed_capture_generation: 0,
            requested_capture_role: CaptureRole::Normal as u8,
            requested_capture_session: 1,
            capture_request_generation: 0,
            capture_arm_started: false,
            check_level: CheckLevelMeter::default(),
            observed_check_level_trigger: 0,
            active_check_level_generation: 0,
            source_check_level_generation: 0,
            source_check_level_position: 0,
            runtime_status: 1.0,
        }
    }
}

pub(crate) fn notes(params: &MotStrobeParams) -> [u8; STRING_COUNT] {
    [
        params.string_7_note.value_u8(),
        params.string_6_note.value_u8(),
        params.string_5_note.value_u8(),
        params.string_4_note.value_u8(),
        params.string_3_note.value_u8(),
        params.string_2_note.value_u8(),
        params.string_1_note.value_u8(),
    ]
}

pub(crate) fn offsets(params: &MotStrobeParams) -> [f32; STRING_COUNT] {
    [
        round_to_tenth(params.string_7_offset.value()),
        round_to_tenth(params.string_6_offset.value()),
        round_to_tenth(params.string_5_offset.value()),
        round_to_tenth(params.string_4_offset.value()),
        round_to_tenth(params.string_3_offset.value()),
        round_to_tenth(params.string_2_offset.value()),
        round_to_tenth(params.string_1_offset.value()),
    ]
}

pub(crate) fn round_to_tenth(value: f32) -> f32 {
    (value * 10.0).round() / 10.0
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct DisplayAnalysis {
    detected_note: Option<u8>,
    matched_string: Option<usize>,
    cents: f32,
    phase: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ResolvedReference {
    frequency_hz: f32,
    matched_string: Option<usize>,
    effective_offset_cents: f32,
}

fn resolve_reference(
    detected_note: u8,
    reference_notes: &[u8; STRING_COUNT],
    note_offsets: &[f32; STRING_COUNT],
    offsets_enabled: bool,
) -> ResolvedReference {
    let equal_temperament = midi_to_hz(detected_note);
    if let Some(index) = reference_notes
        .iter()
        .position(|note| *note == detected_note)
    {
        let effective_offset = if offsets_enabled {
            note_offsets[index]
        } else {
            0.0
        };
        ResolvedReference {
            frequency_hz: equal_temperament * cents_ratio(effective_offset),
            matched_string: Some(index),
            effective_offset_cents: effective_offset,
        }
    } else {
        ResolvedReference {
            frequency_hz: equal_temperament,
            matched_string: None,
            effective_offset_cents: 0.0,
        }
    }
}

fn prepare_display(
    state: &mut MotStrobe,
    params: &MotStrobeParams,
    pitch: PitchAnalysis,
    elapsed_samples: usize,
) -> DisplayAnalysis {
    let Some(detected_note) = pitch.detected_note else {
        state.displayed_note = None;
        state.reference_hz = 0.0;
        state.strobe_phase = 0.0;
        return DisplayAnalysis::default();
    };

    let open_notes = notes(params);
    let string_offsets = offsets(params);
    let reference = resolve_reference(
        detected_note,
        &open_notes,
        &string_offsets,
        params.offsets_enabled.value(),
    );
    let reference_changed = state.displayed_note != Some(detected_note)
        || (state.reference_hz - reference.frequency_hz).abs() > 0.000_1;
    if reference_changed {
        state.strobe_phase = 0.0;
    } else {
        let drift_hz = pitch.detected_frequency_hz - reference.frequency_hz;
        state.strobe_phase = (state.strobe_phase
            + drift_hz * elapsed_samples as f32 / state.sample_rate)
            .rem_euclid(1.0);
    }
    state.displayed_note = Some(detected_note);
    state.reference_hz = reference.frequency_hz;

    let cents = 1_200.0
        * (pitch.detected_frequency_hz / reference.frequency_hz.max(f32::MIN_POSITIVE)).log2();
    DisplayAnalysis {
        detected_note: Some(detected_note),
        matched_string: reference.matched_string,
        cents,
        phase: state.strobe_phase,
    }
}

impl MotStrobe {
    /// Installs a prepared Source/Return role and returns the previous role to
    /// a worker-owned retirement queue. No allocation or destruction occurs
    /// in the callback.
    #[inline]
    fn poll_capture_runtime(&mut self, params: &MotStrobeParams) {
        if let Some(retired) = self.retired_capture_runtime.take()
            && let Err(returned) = params.capture_control.retire_runtime(retired)
        {
            self.retired_capture_runtime = Some(returned);
            return;
        }
        let Some(runtime) = params.capture_control.take_runtime() else {
            self.recycle_return_storage(params);
            return;
        };
        if runtime.request_generation < self.installed_capture_generation {
            if let Err(returned) = params.capture_control.retire_runtime(runtime) {
                self.retired_capture_runtime = Some(returned);
            }
            self.recycle_return_storage(params);
            return;
        }
        self.installed_capture_generation = runtime.request_generation;
        let previous = std::mem::replace(&mut self.capture_runtime, runtime);
        self.retired_capture_runtime = Some(previous);
        self.capture_arm_started = false;
        self.recycle_return_storage(params);
    }

    #[inline]
    fn recycle_return_storage(&mut self, params: &MotStrobeParams) {
        if self.pending_recycled.is_none() {
            self.pending_recycled = params.capture_control.take_recycled();
        }
        let Some(completed) = self.pending_recycled.take() else {
            return;
        };
        if let Err((_error, returned)) = self
            .capture_runtime
            .engine
            .recycle_completed_return(completed)
        {
            // A role/session swap may temporarily make the storage
            // inapplicable. Retain ownership and retry; never deallocate it
            // on the audio thread.
            self.pending_recycled = Some(returned);
        }
    }

    #[inline]
    fn try_schedule_training(&mut self, context: &ProcessContext) {
        let Some(task) = self.pending_training.take() else {
            return;
        };
        let Some(spawner) = context.tasks::<StartTrainingTask>() else {
            self.pending_training = Some(task);
            return;
        };
        if let Err(returned) = spawner.try_spawn(task) {
            self.pending_training = Some(returned);
        }
    }

    #[inline]
    fn poll_check_level_trigger(&mut self, params: &MotStrobeParams, context: &ProcessContext) {
        if self.capture_runtime.role != CaptureRole::Return || params.capture_armed.value() {
            return;
        }
        let requested = params
            .check_level_trigger_generation
            .load(Ordering::Acquire);
        if requested == self.observed_check_level_trigger {
            return;
        }
        let Some(binding) = self.capture_runtime.binding.as_ref() else {
            return;
        };
        let Ok(generation) = binding.request_check_level() else {
            // Pair preparation may still be in flight. Leave the UI token
            // pending and retry without allocating on a later callback.
            return;
        };
        self.observed_check_level_trigger = requested;
        self.active_check_level_generation = generation;
        self.check_level = CheckLevelMeter::default();
        if self
            .check_level
            .start(context.sample_rate.round() as u32)
            .is_err()
        {
            let _ =
                binding.publish_check_level(generation, SessionCheckLevelState::Failed, 0.0, 0.0);
        }
    }

    /// Emits the immutable, precomputed probe while the paired Return is
    /// measuring. The fragment loops across arbitrary host block sizes and
    /// passes through the exact same Source Send Trim as the real excitation.
    #[inline]
    fn process_source_check_level_probe(
        &mut self,
        params: &MotStrobeParams,
        output: &mut [f32],
    ) -> bool {
        if self.capture_runtime.role != CaptureRole::Source {
            return false;
        }
        let Some(binding) = self.capture_runtime.binding.as_ref() else {
            return false;
        };
        let snapshot = binding.check_level_snapshot();
        if snapshot.state != SessionCheckLevelState::Measuring {
            self.source_check_level_generation = 0;
            self.source_check_level_position = 0;
            return false;
        }
        let Some(program) = self.capture_runtime.program.as_ref() else {
            // A Source without the validated capture asset must never allow a
            // silent check to pass.
            binding.invalidate_check_level();
            output.fill(0.0);
            return true;
        };
        let probe = program.check_level_probe();
        debug_assert_eq!(probe.len(), CHECK_LEVEL_DURATION_SAMPLES);
        if snapshot.generation != self.source_check_level_generation {
            self.source_check_level_generation = snapshot.generation;
            self.source_check_level_position = 0;
        }
        let send_gain = 10.0_f32.powf(params.capture_send_trim.value().clamp(-40.0, 0.0) / 20.0);
        for sample in output {
            *sample = probe[self.source_check_level_position] * send_gain;
            self.source_check_level_position += 1;
            if self.source_check_level_position == probe.len() {
                self.source_check_level_position = 0;
            }
        }
        true
    }

    /// Returns true while CHECK LEVEL owns this Return block.
    #[inline]
    fn process_check_level(&mut self, input: &[f32], output: &mut [f32]) -> bool {
        if self.capture_runtime.role != CaptureRole::Return
            || !matches!(self.check_level.state(), CheckLevelState::Measuring { .. })
        {
            return false;
        }
        output.fill(0.0);
        self.check_level.process_block(input);
        let result = self.check_level.result();
        let progress = result.measured_samples as f32 / CHECK_LEVEL_DURATION_SAMPLES.max(1) as f32;
        let session_state = match self.check_level.state() {
            CheckLevelState::Measuring { .. } => SessionCheckLevelState::Measuring,
            CheckLevelState::Passed => SessionCheckLevelState::Passed,
            CheckLevelState::Failed(_) => SessionCheckLevelState::Failed,
            CheckLevelState::Idle => SessionCheckLevelState::Required,
        };
        let Some(binding) = self.capture_runtime.binding.as_ref() else {
            self.check_level.interrupt();
            return true;
        };
        if !binding.publish_check_level(
            self.active_check_level_generation,
            session_state,
            progress,
            result.peak_linear,
        ) {
            // The pair changed while this callback was measuring. Its old
            // result is no longer authoritative.
            self.check_level.interrupt();
            self.active_check_level_generation = 0;
        }
        true
    }

    #[inline]
    fn process_capture(
        &mut self,
        params: &MotStrobeParams,
        input: &[f32],
        output: &mut [f32],
        context: &ProcessContext,
    ) {
        let role = self.capture_runtime.role;
        let armed = params.capture_armed.value();
        self.poll_check_level_trigger(params, context);
        if let Some(binding) = &self.capture_runtime.binding {
            if role == CaptureRole::Source {
                let send_trim_db = params.capture_send_trim.value().clamp(-40.0, 0.0);
                binding.publish_source_send_trim_db(send_trim_db);
                if armed && !self.capture_arm_started {
                    if binding.arm_pair(context.transport.playing).is_ok() {
                        self.capture_arm_started = true;
                    }
                } else if !armed && self.capture_arm_started {
                    binding.abort_pair(CaptureInvalidation::CoordinatorAbort);
                    self.capture_arm_started = false;
                }
            }
            let _ = self.capture_runtime.engine.synchronize(binding);
        }

        if self.process_source_check_level_probe(params, output) {
            return;
        }
        if self.process_check_level(input, output) {
            return;
        }

        let transport = CaptureTransportInfo {
            playing: context.transport.playing,
            recording: context.transport.recording,
            timeline_sample: Some(context.transport.position_samples),
            loop_active: context.transport.loop_active,
            discontinuity: false,
            sample_rate_hz: context.sample_rate.round() as u32,
        };
        self.capture_runtime
            .engine
            .process_block(input, output, transport);

        if role == CaptureRole::Source {
            let send_gain =
                10.0_f32.powf(params.capture_send_trim.value().clamp(-40.0, 0.0) / 20.0);
            for sample in output {
                *sample *= send_gain;
            }
        } else if role == CaptureRole::Return
            && self.pending_training.is_none()
            && let Some(completed) = self.capture_runtime.engine.take_completed_return()
            && let Some(program) = self.capture_runtime.program.as_ref()
        {
            let send_trim_db = self
                .capture_runtime
                .binding
                .as_ref()
                .map_or(params.capture_send_trim.value(), |binding| {
                    binding.source_send_trim_db()
                });
            params.capture_control.prepare_training();
            self.pending_training = Some(StartTrainingTask {
                completed,
                program: Arc::clone(program),
                source_send_trim_db: send_trim_db,
                target: if params.capture_target.value_i32() == 1 {
                    CaptureTarget::FullAmpUnfilteredLoad
                } else {
                    CaptureTarget::SoftwarePluginChain
                },
            });
        }
    }
}

impl PluginLogic for MotStrobe {
    type Params = MotStrobeParams;
    type DspState = Self;

    fn init(params: &Self::Params, _context: &InitContext) -> Self::DspState {
        let instance_id = NEXT_INSTANCE_ID.fetch_add(1, Ordering::Relaxed).max(1);
        params.instance_id.store(instance_id);
        if params.capture_session_id.load() == 0 {
            params.capture_session_id.store(1);
            if let Ok(mut name) = params.capture_session_name.write()
                && name.is_empty()
            {
                *name = "AUTO".to_owned();
            }
        }
        Self::default()
    }

    fn supports_in_place() -> bool {
        // The wrapper snapshots host-aliased input into preallocated scratch.
        // This keeps the mono block API simple without allocating in process().
        false
    }

    fn bus_layouts() -> Vec<BusLayout> {
        vec![BusLayout::mono()]
    }

    fn reset(state: &mut Self::DspState, params: &Self::Params, config: &AudioConfig) {
        state.sample_rate = config.sample_rate as f32;
        state.live_path_sample_rate_compatible =
            config.sample_rate.round() as u32 == model::REQUIRED_SAMPLE_RATE_HZ;
        state.max_block_size = config.max_block_size;
        state.tuner.reset(state.sample_rate, &notes(params));
        state.signal_chain.reset(config);
        state
            .output_mute
            .reset(state.sample_rate, params.mute.value());
        state.strobe_phase = 0.0;
        state.displayed_note = None;
        state.reference_hz = 0.0;
        // A host reset can change sample rate while an old loader task is
        // still running. Its prepared runtime must never enter the callback.
        advance_runtime_request_epoch(params);
        state.requested_runtime_generation = u64::MAX;
        state.capture_arm_started = false;
        state.check_level = CheckLevelMeter::default();
        state.observed_check_level_trigger = params
            .check_level_trigger_generation
            .load(Ordering::Acquire);
        state.active_check_level_generation = 0;
        state.source_check_level_generation = 0;
        state.source_check_level_position = 0;
        state.runtime_status = if state.live_path_sample_rate_compatible {
            0.5
        } else {
            0.0
        };
        state.pending_training = None;
        state.pending_recycled = None;
        state.retired_capture_runtime = None;
        if let Some(binding) = state.capture_runtime.binding.as_ref() {
            binding.invalidate_check_level();
        }
        if state.capture_runtime.role != CaptureRole::Normal {
            state
                .capture_runtime
                .engine
                .invalidate(CaptureInvalidation::CoordinatorAbort);
        }
    }

    fn process(
        state: &mut Self::DspState,
        params: &Self::Params,
        buffer: &mut AudioBuffer,
        _events: &EventList,
        context: &mut ProcessContext,
    ) -> ProcessStatus {
        let runtime_generation = params.runtime_generation.load();
        if runtime_generation != state.requested_runtime_generation {
            if let Some(spawner) = context.tasks::<LoadRuntimeTask>() {
                let request_epoch = advance_runtime_request_epoch(params);
                spawner.spawn_coalescing(LoadRuntimeTask {
                    generation: runtime_generation,
                    request_epoch,
                    host_sample_rate_hz: state.sample_rate.round() as u32,
                    host_max_block_size: state.max_block_size.max(1),
                });
                state.runtime_status = 0.5;
            }
            state.requested_runtime_generation = runtime_generation;
        }
        if let Some(status) = state.signal_chain.poll_runtime(&params.runtime_mailbox) {
            state.runtime_status = match status {
                RuntimeApplyStatus::Ready { .. } => 1.0,
                RuntimeApplyStatus::SafeMuted { .. } => 0.0,
            };
        }
        state.signal_chain.set_controls(AmpControls {
            input_gain_db: params.input_gain.value(),
            tight: params.tight.value() / 100.0,
            bite: params.bite.value() / 100.0,
        });

        let requested_capture_role = params.capture_role.value_i32().clamp(0, 2) as u8;
        let requested_capture_session = params.capture_session_id.load().max(1);
        if requested_capture_role != state.requested_capture_role
            || requested_capture_session != state.requested_capture_session
        {
            state.check_level.interrupt();
            state.check_level = CheckLevelMeter::default();
            state.active_check_level_generation = 0;
            state.source_check_level_generation = 0;
            state.source_check_level_position = 0;
            state.observed_check_level_trigger = params
                .check_level_trigger_generation
                .load(Ordering::Acquire);
            state.capture_request_generation =
                state.capture_request_generation.wrapping_add(1).max(1);
            if let Some(spawner) = context.tasks::<PrepareCaptureTask>() {
                params
                    .capture_control
                    .set_status(CaptureWorkerStatus::Preparing);
                spawner.spawn_coalescing(PrepareCaptureTask {
                    request_generation: state.capture_request_generation,
                    role: requested_capture_role,
                    session_id: requested_capture_session,
                    instance_id: params.instance_id.load().max(1),
                });
            }
            state.requested_capture_role = requested_capture_role;
            state.requested_capture_session = requested_capture_session;
        }
        state.poll_capture_runtime(params);
        state.try_schedule_training(context);

        let open_notes = notes(params);
        state.tuner.configure_range(&open_notes);

        let input_channels = buffer.num_input_channels();
        let samples = buffer.num_samples();
        let bypassed = params.bypass.value();
        let mute_requested = params.mute.value();

        if !bypassed && input_channels > 0 {
            // Analyze a mono sum while leaving every host sample untouched.
            for sample_index in 0..samples {
                let mut mono = 0.0;
                for channel in 0..input_channels {
                    mono += buffer.input(channel)[sample_index];
                }
                state.tuner.push_sample(mono / input_channels as f32);
            }
        }

        if buffer.num_input_channels() > 0 && buffer.num_output_channels() > 0 {
            // The dry tuner tap above is intentionally independent from the
            // processed branch. The branch keeps running under Mute and Bypass
            // so future nonlinear/convolution state cannot resume from stale
            // audio. Mute is a short click-free output ramp; host bypass wins.
            let (input, output) = buffer.io_pair(0, 0);
            if !state.live_path_sample_rate_compatible {
                // Amp models, cabinet IRs, Capture Source, and Capture Return
                // are fixed to native 48 kHz. Fail closed immediately on the
                // very first callback after a host-rate change; do not wait
                // for the asynchronous loader to publish its rejection.
                output.fill(0.0);
                state.runtime_status = 0.0;
            } else if state.capture_runtime.role == CaptureRole::Normal {
                state.signal_chain.process_block(input, output);
            } else {
                state.process_capture(params, input, output, context);
            }
            for sample_index in 0..samples {
                let mute_gain = state.output_mute.next_gain(mute_requested);
                if bypassed {
                    output[sample_index] = input[sample_index];
                } else {
                    output[sample_index] *= mute_gain;
                }
            }
        }
        state.try_schedule_training(context);

        let analysis = if bypassed {
            state.strobe_phase = 0.0;
            state.displayed_note = None;
            state.reference_hz = 0.0;
            DisplayAnalysis::default()
        } else {
            let pitch = state.tuner.finish_block();
            prepare_display(state, params, pitch, samples)
        };
        publish_analysis(context, analysis);
        publish_runtime_and_capture_state(state, params, context);

        ProcessStatus::Normal
    }

    fn latency(state: &Self::DspState) -> u32 {
        state.signal_chain.latency_samples()
    }

    fn tail(state: &Self::DspState) -> u32 {
        state.signal_chain.tail_samples()
    }

    fn editor(params: Arc<MotStrobeParams>) -> Box<dyn Editor> {
        EguiEditor::with_ui(params, WINDOW_SIZE, MotStrobeUi).into_editor()
    }
}

fn publish_analysis(context: &ProcessContext, analysis: DisplayAnalysis) {
    let encoded_note = analysis
        .detected_note
        .map_or(0.0, |note| (f32::from(note) + 1.0) / 128.0);
    let encoded_string = analysis
        .matched_string
        .map_or(0.0, |index| (index as f32 + 1.0) / 8.0);
    context.set_meter(P::DetectedNote, encoded_note);
    context.set_meter(P::MatchedString, encoded_string);
    context.set_meter(P::Cents, (analysis.cents.clamp(-50.0, 50.0) + 50.0) / 100.0);
    context.set_meter(P::Phase, analysis.phase.clamp(0.0, 1.0));
}

fn publish_runtime_and_capture_state(
    state: &MotStrobe,
    params: &MotStrobeParams,
    context: &ProcessContext,
) {
    context.set_meter(P::RuntimeStatus, state.runtime_status.clamp(0.0, 1.0));
    let phase = state.capture_runtime.engine.state().phase();
    context.set_meter(P::CaptureStatus, f32::from(phase as u8) / 7.0);
    context.set_meter(
        P::CapturePeak,
        state
            .capture_runtime
            .engine
            .return_peak_linear()
            .clamp(0.0, 1.0),
    );
    let check_level = state
        .capture_runtime
        .binding
        .as_ref()
        .map_or_else(SessionCheckLevelSnapshot::default, |binding| {
            binding.check_level_snapshot()
        });
    context.set_meter(
        P::CheckLevelStatus,
        f32::from(check_level.state as u8) / 3.0,
    );
    context.set_meter(P::CheckLevelProgress, check_level.progress.clamp(0.0, 1.0));
    context.set_meter(P::CheckLevelPeak, check_level.peak_linear.clamp(0.0, 1.0));
    context.set_meter(
        P::TrainingProgress,
        params.capture_control.progress().clamp(0.0, 1.0),
    );
}

truce::plugin! {
    logic: MotStrobe,
    params: MotStrobeParams,
    tasks: [
        LoadRuntimeTask,
        ImportIrTask,
        LibraryTask,
        PrepareCaptureTask,
        StartTrainingTask
    ],
}

truce::enable_rt_paranoid!();

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::{decode_note, decode_string, note_name};

    #[test]
    fn meter_string_encoding_round_trips() {
        for index in 0..STRING_COUNT {
            let encoded = (index as f32 + 1.0) / 8.0;
            assert_eq!(decode_string(encoded), Some(index));
        }
        assert_eq!(decode_string(0.0), None);
    }

    #[test]
    fn meter_note_encoding_round_trips_the_full_midi_range() {
        for note in 0..=127 {
            let encoded = (f32::from(note) + 1.0) / 128.0;
            assert_eq!(decode_note(encoded), Some(note));
        }
        assert_eq!(decode_note(0.0), None);
    }

    #[test]
    fn note_names_follow_midi_octaves() {
        assert_eq!(note_name(35), "B1");
        assert_eq!(note_name(40), "E2");
        assert_eq!(note_name(64), "E4");
    }

    #[test]
    fn default_parameters_are_b_standard() {
        let params = MotStrobeParams::new();
        assert_eq!(notes(&params), tuner::DEFAULT_TUNING);
        assert!(params.offsets_enabled.value());
        assert!(!params.mute.value());
    }

    #[test]
    fn library_control_rejects_an_unconsumed_stale_outcome() {
        let control = LibraryControl::default();
        let first = control.try_begin().unwrap();
        control.finish(LibraryOutcome::FolderOpened {
            request_id: first,
            result: Ok(()),
        });
        let second = control.try_begin().unwrap();
        let stale = control.take_outcome().unwrap();
        assert_eq!(stale.request_id(), first);
        assert!(!control.is_current(stale.request_id()));
        assert!(control.is_current(second));
        control.invalidate_pending();
        assert!(!control.is_current(second));
        assert!(control.is_busy());
        control.cancel_begin(second);
        // The invalidation deliberately does not let a stale caller clear a
        // still-running request. Its worker completion owns that transition.
        assert!(control.is_busy());
        control.finish(LibraryOutcome::FolderOpened {
            request_id: second,
            result: Ok(()),
        });
        assert!(!control.is_busy());
        assert!(!control.is_current(second));
    }

    #[test]
    fn offsets_are_quantized_to_tenths() {
        assert_eq!(round_to_tenth(0.46), 0.5);
        assert_eq!(round_to_tenth(0.96), 1.0);
        assert_eq!(round_to_tenth(-2.54), -2.5);
    }

    #[test]
    fn custom_offsets_apply_only_to_the_seven_reference_notes() {
        let reference_notes = tuner::DEFAULT_TUNING;
        let note_offsets = [2.0, -1.0, 0.5, 0.0, -2.5, 1.25, 3.0];

        for note in 0..=127 {
            let reference = resolve_reference(note, &reference_notes, &note_offsets, true);
            let expected = reference_notes
                .iter()
                .position(|reference_note| *reference_note == note);
            assert_eq!(reference.matched_string, expected, "MIDI {note}");
            if let Some(index) = expected {
                assert!(
                    (reference.frequency_hz - midi_to_hz(note) * cents_ratio(note_offsets[index]))
                        .abs()
                        < 1.0e-4
                );
                assert!((reference.effective_offset_cents - note_offsets[index]).abs() < 1.0e-6);
            } else {
                assert_eq!(reference.frequency_hz, midi_to_hz(note));
                assert_eq!(reference.effective_offset_cents, 0.0);
            }
        }
    }

    #[test]
    fn disabling_offsets_uses_twelve_tet_but_keeps_the_matched_row() {
        let reference = resolve_reference(35, &tuner::DEFAULT_TUNING, &[4.0; STRING_COUNT], false);
        assert_eq!(reference.matched_string, Some(0));
        assert_eq!(reference.frequency_hz, midi_to_hz(35));
        assert_eq!(reference.effective_offset_cents, 0.0);
    }

    #[test]
    fn duplicate_notes_resolve_to_the_first_table_row() {
        let reference_notes = [40, 40, 45, 50, 55, 59, 64];
        let offsets = [1.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let reference = resolve_reference(40, &reference_notes, &offsets, true);
        assert_eq!(reference.matched_string, Some(0));
        assert!((reference.effective_offset_cents - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn editor_has_a_headless_render_path() {
        let params = Arc::new(MotStrobeParams::new());
        let mut editor = EguiEditor::with_ui(Arc::clone(&params), WINDOW_SIZE, MotStrobeUi);
        let erased: Arc<dyn truce::params::Params> = params;
        assert_eq!(Editor::size(&editor), WINDOW_SIZE);
        // A sandboxed or headless runner may have no Metal adapter, in which
        // case Truce deliberately returns `None`.
        if let Some((_, width, height)) = Editor::screenshot(&mut editor, erased) {
            assert_eq!((width, height), (WINDOW_SIZE.0 * 2, WINDOW_SIZE.1 * 2));
        }
    }

    #[test]
    fn return_ui_trigger_measures_exactly_one_second_and_unlocks_pair_arm() {
        let session = CaptureSessionId::new(7001).unwrap();
        let coordinator = capture::CaptureCoordinator::with_capacity(1);
        let source_binding = coordinator
            .bind(session, CaptureRole::Source, 1001)
            .unwrap();
        let return_binding = coordinator
            .bind(session, CaptureRole::Return, 1002)
            .unwrap();

        let params = MotStrobeParams::new();
        params
            .check_level_trigger_generation
            .store(1, Ordering::Release);
        let mut state = MotStrobe {
            sample_rate: 48_000.0,
            capture_runtime: Box::new(PreparedCaptureRuntime {
                request_generation: 1,
                role: CaptureRole::Return,
                session_id: session,
                engine: capture::CaptureEngine::new(CaptureRole::Return, session),
                binding: Some(return_binding),
                program: None,
            }),
            ..MotStrobe::default()
        };

        let transport = TransportInfo::default();
        let mut output_events = EventList::with_capacity(0);
        let context = ProcessContext::new(&transport, 48_000.0, 257, &mut output_events);
        let mut remaining = CHECK_LEVEL_DURATION_SAMPLES;
        while remaining > 0 {
            let block_size = remaining.min(257);
            let input = vec![0.5; block_size];
            let mut output = vec![1.0; block_size];
            state.process_capture(&params, &input, &mut output, &context);
            assert!(output.iter().all(|sample| *sample == 0.0));
            remaining -= block_size;
        }

        assert_eq!(state.check_level.state(), CheckLevelState::Passed);
        assert_eq!(
            state.check_level.result().measured_samples,
            CHECK_LEVEL_DURATION_SAMPLES
        );
        let snapshot = source_binding.check_level_snapshot();
        assert_eq!(snapshot.state, SessionCheckLevelState::Passed);
        assert_eq!(snapshot.progress, 1.0);
        assert_eq!(snapshot.peak_linear, 0.5);
        source_binding.arm_pair(false).unwrap();
    }

    #[test]
    fn clipped_return_check_fails_and_keeps_pair_arm_locked() {
        let session = CaptureSessionId::new(7002).unwrap();
        let coordinator = capture::CaptureCoordinator::with_capacity(1);
        let source_binding = coordinator
            .bind(session, CaptureRole::Source, 2001)
            .unwrap();
        let return_binding = coordinator
            .bind(session, CaptureRole::Return, 2002)
            .unwrap();

        let params = MotStrobeParams::new();
        params
            .check_level_trigger_generation
            .store(1, Ordering::Release);
        let mut state = MotStrobe {
            sample_rate: 48_000.0,
            capture_runtime: Box::new(PreparedCaptureRuntime {
                request_generation: 1,
                role: CaptureRole::Return,
                session_id: session,
                engine: capture::CaptureEngine::new(CaptureRole::Return, session),
                binding: Some(return_binding),
                program: None,
            }),
            ..MotStrobe::default()
        };

        let transport = TransportInfo::default();
        let mut output_events = EventList::with_capacity(0);
        let context = ProcessContext::new(&transport, 48_000.0, 16, &mut output_events);
        let input = [0.0, capture::RETURN_CLIP_THRESHOLD_LINEAR + 0.001, 0.0];
        let mut output = [1.0; 3];
        state.process_capture(&params, &input, &mut output, &context);

        assert!(matches!(
            state.check_level.state(),
            CheckLevelState::Failed(_)
        ));
        assert_eq!(
            source_binding.check_level_snapshot().state,
            SessionCheckLevelState::Failed
        );
        assert_eq!(
            source_binding.arm_pair(false),
            Err(capture::CoordinatorError::CheckLevelNotPassed(
                SessionCheckLevelState::Failed
            ))
        );
    }

    #[test]
    fn source_probe_reaches_return_and_level_result_tracks_routed_gain() {
        let session = CaptureSessionId::new(7003).unwrap();
        let coordinator = capture::CaptureCoordinator::with_capacity(1);
        let source_binding = coordinator
            .bind(session, CaptureRole::Source, 3001)
            .unwrap();
        let return_binding = coordinator
            .bind(session, CaptureRole::Return, 3002)
            .unwrap();
        let program =
            Arc::new(capture::CaptureProgram::new(vec![0.1; 16], vec![0.5; 128]).unwrap());

        let source_params = MotStrobeParams::new();
        source_params.capture_send_trim.set_value(-6.0);
        let mut source = MotStrobe {
            sample_rate: 48_000.0,
            capture_runtime: Box::new(PreparedCaptureRuntime {
                request_generation: 1,
                role: CaptureRole::Source,
                session_id: session,
                engine: capture::CaptureEngine::new(CaptureRole::Source, session),
                binding: Some(source_binding),
                program: Some(Arc::clone(&program)),
            }),
            ..MotStrobe::default()
        };
        let return_params = MotStrobeParams::new();
        let mut returned = MotStrobe {
            sample_rate: 48_000.0,
            capture_runtime: Box::new(PreparedCaptureRuntime {
                request_generation: 1,
                role: CaptureRole::Return,
                session_id: session,
                engine: capture::CaptureEngine::new(CaptureRole::Return, session),
                binding: Some(return_binding),
                program: Some(program),
            }),
            ..MotStrobe::default()
        };

        let transport = TransportInfo::default();
        let mut output_events = EventList::with_capacity(0);
        let context = ProcessContext::new(&transport, 48_000.0, 257, &mut output_events);

        // Let Source publish the exact trim before Return starts the check.
        let mut source_output = [1.0; 1];
        source.process_capture(&source_params, &[0.0], &mut source_output, &context);
        assert_eq!(source_output, [0.0]);

        return_params
            .check_level_trigger_generation
            .store(1, Ordering::Release);
        let mut return_output = [1.0; 1];
        returned.process_capture(&return_params, &[0.0], &mut return_output, &context);
        assert_eq!(return_output, [0.0]);

        let routed_gain = 1.4_f32;
        let mut remaining = CHECK_LEVEL_DURATION_SAMPLES - 1;
        let mut audited_callbacks = false;
        while remaining > 0 {
            let block_size = remaining.min(257);
            let silence = vec![0.0; block_size];
            let mut emitted = vec![0.0; block_size];
            if audited_callbacks {
                source.process_capture(&source_params, &silence, &mut emitted, &context);
            } else {
                let (_, allocations) = truce::rt::audit(|| {
                    source.process_capture(&source_params, &silence, &mut emitted, &context);
                });
                assert_eq!(allocations, 0);
            }
            let routed: Vec<f32> = emitted.iter().map(|sample| sample * routed_gain).collect();
            let mut sink = vec![1.0; block_size];
            if audited_callbacks {
                returned.process_capture(&return_params, &routed, &mut sink, &context);
            } else {
                let (_, allocations) = truce::rt::audit(|| {
                    returned.process_capture(&return_params, &routed, &mut sink, &context);
                });
                assert_eq!(allocations, 0);
                audited_callbacks = true;
            }
            assert!(sink.iter().all(|sample| *sample == 0.0));
            remaining -= block_size;
        }

        let expected_peak = 0.5 * 10.0_f32.powf(-6.0 / 20.0) * routed_gain;
        let passed = returned
            .capture_runtime
            .binding
            .as_ref()
            .unwrap()
            .check_level_snapshot();
        assert_eq!(passed.state, SessionCheckLevelState::Passed);
        assert!((passed.peak_linear - expected_peak).abs() < 1.0e-6);

        // The same probe through a hotter routed chain must fail, proving the
        // Return result is based on routed audio rather than a local estimate.
        return_params
            .check_level_trigger_generation
            .store(2, Ordering::Release);
        returned.process_capture(&return_params, &[0.0], &mut return_output, &context);
        let mut emitted = [0.0; 8];
        source.process_capture(&source_params, &[0.0; 8], &mut emitted, &context);
        assert!(emitted.iter().any(|sample| *sample != 0.0));
        let clipped = emitted.map(|sample| sample * 4.1);
        let mut sink = [1.0; 8];
        returned.process_capture(&return_params, &clipped, &mut sink, &context);
        assert_eq!(
            returned
                .capture_runtime
                .binding
                .as_ref()
                .unwrap()
                .check_level_snapshot()
                .state,
            SessionCheckLevelState::Failed
        );
    }

    #[test]
    fn audio_is_bit_exact_in_active_and_bypass_modes() {
        const FRAMES: usize = 8;
        let input = [0.0, -0.5, 0.25, 1.0, -1.0, 0.125, -0.25, 0.75];

        for bypassed in [false, true] {
            let params = MotStrobeParams::new();
            params.bypass.set_value(bypassed);
            let mut state = MotStrobe::default();
            MotStrobe::reset(&mut state, &params, &AudioConfig::new(48_000.0, FRAMES));

            let inputs: [&[f32]; 1] = [&input];
            let mut output = [f32::NAN; FRAMES];
            let mut outputs: [&mut [f32]; 1] = [&mut output];
            let mut buffer = AudioBuffer::from_slices_checked(&inputs, &mut outputs, FRAMES);
            let events = EventList::default();
            let mut output_events = EventList::with_capacity(0);
            let transport = TransportInfo::default();
            let mut context = ProcessContext::new(&transport, 48_000.0, FRAMES, &mut output_events);

            MotStrobe::process(&mut state, &params, &mut buffer, &events, &mut context);

            assert_eq!(output, input);
        }
    }

    #[test]
    fn wrapper_owns_the_in_place_copy_path() {
        assert!(!MotStrobe::supports_in_place());
    }

    #[test]
    fn plugin_declares_only_a_mono_bus_layout() {
        let widths: Vec<_> = MotStrobe::bus_layouts()
            .iter()
            .map(|layout| {
                (
                    layout.total_input_channels(),
                    layout.total_output_channels(),
                )
            })
            .collect();
        assert_eq!(widths, vec![(1, 1)]);
    }

    #[test]
    fn mono_signal_runs_through_the_processed_branch() {
        const FRAMES: usize = 8;
        let input = [0.0, -0.5, 0.25, 1.0, -1.0, 0.125, -0.25, 0.75];
        let mut output = [f32::NAN; FRAMES];

        let params = MotStrobeParams::new();
        let mut state = MotStrobe::default();
        MotStrobe::reset(&mut state, &params, &AudioConfig::new(48_000.0, FRAMES));

        let inputs: [&[f32]; 1] = [&input];
        let mut outputs: [&mut [f32]; 1] = [&mut output];
        let mut buffer = AudioBuffer::from_slices_checked(&inputs, &mut outputs, FRAMES);
        let events = EventList::default();
        let mut output_events = EventList::with_capacity(0);
        let transport = TransportInfo::default();
        let mut context = ProcessContext::new(&transport, 48_000.0, FRAMES, &mut output_events);

        MotStrobe::process(&mut state, &params, &mut buffer, &events, &mut context);

        assert_eq!(output, input);
        assert_eq!(state.signal_chain.processed_samples(), FRAMES as u64);
    }

    #[test]
    fn unsupported_host_rate_immediately_safe_mutes_live_path_but_not_bypass() {
        const FRAMES: usize = 8;
        let input = [0.0, -0.5, 0.25, 1.0, -1.0, 0.125, -0.25, 0.75];

        for (bypassed, expected) in [(false, [0.0; FRAMES]), (true, input)] {
            let params = MotStrobeParams::new();
            params.bypass.set_value(bypassed);
            let mut state = MotStrobe::default();
            MotStrobe::reset(&mut state, &params, &AudioConfig::new(44_100.0, FRAMES));

            let inputs: [&[f32]; 1] = [&input];
            let mut output = [f32::NAN; FRAMES];
            let mut outputs: [&mut [f32]; 1] = [&mut output];
            let mut buffer = AudioBuffer::from_slices_checked(&inputs, &mut outputs, FRAMES);
            let events = EventList::default();
            let mut output_events = EventList::with_capacity(0);
            let transport = TransportInfo::default();
            let mut context = ProcessContext::new(&transport, 44_100.0, FRAMES, &mut output_events);

            MotStrobe::process(&mut state, &params, &mut buffer, &events, &mut context);

            assert_eq!(output, expected);
            assert_eq!(state.runtime_status, 0.0);
            assert_eq!(state.signal_chain.processed_samples(), 0);
        }
    }

    #[test]
    fn mute_zeros_output_but_pitch_detection_keeps_running() {
        const FRAMES: usize = 4_096;
        let input: [f32; FRAMES] = std::array::from_fn(|index| {
            let time = index as f32 / 48_000.0;
            0.25 * (std::f32::consts::TAU * midi_to_hz(35) * time).sin()
        });
        let mut output = [f32::NAN; FRAMES];

        let params = MotStrobeParams::new();
        params.mute.set_value(true);
        let mut state = MotStrobe::default();
        MotStrobe::reset(&mut state, &params, &AudioConfig::new(48_000.0, FRAMES));

        let inputs: [&[f32]; 1] = [&input];
        let mut outputs: [&mut [f32]; 1] = [&mut output];
        let mut buffer = AudioBuffer::from_slices_checked(&inputs, &mut outputs, FRAMES);
        let events = EventList::default();
        let mut output_events = EventList::with_capacity(0);
        let transport = TransportInfo::default();
        let mut context = ProcessContext::new(&transport, 48_000.0, FRAMES, &mut output_events);

        MotStrobe::process(&mut state, &params, &mut buffer, &events, &mut context);

        assert!(output.iter().all(|sample| *sample == 0.0));
        assert_eq!(state.displayed_note, Some(35));
        assert_eq!(state.signal_chain.processed_samples(), FRAMES as u64);
    }

    #[test]
    fn changing_mute_uses_a_short_output_ramp() {
        const FRAMES: usize = 192;
        let input = [1.0; FRAMES];
        let mut output = [f32::NAN; FRAMES];

        let params = MotStrobeParams::new();
        let mut state = MotStrobe::default();
        MotStrobe::reset(&mut state, &params, &AudioConfig::new(48_000.0, FRAMES));
        params.mute.set_value(true);

        let inputs: [&[f32]; 1] = [&input];
        let mut outputs: [&mut [f32]; 1] = [&mut output];
        let mut buffer = AudioBuffer::from_slices_checked(&inputs, &mut outputs, FRAMES);
        let events = EventList::default();
        let mut output_events = EventList::with_capacity(0);
        let transport = TransportInfo::default();
        let mut context = ProcessContext::new(&transport, 48_000.0, FRAMES, &mut output_events);

        MotStrobe::process(&mut state, &params, &mut buffer, &events, &mut context);

        assert!(output[0] < 1.0 && output[0] > 0.0);
        assert_eq!(output[143], 0.0);
        assert!(output[143..].iter().all(|sample| *sample == 0.0));
        assert_eq!(state.signal_chain.processed_samples(), FRAMES as u64);
    }

    #[test]
    fn host_bypass_has_priority_over_mute() {
        const FRAMES: usize = 8;
        let input = [0.0, -0.5, 0.25, 1.0, -1.0, 0.125, -0.25, 0.75];
        let mut output = [f32::NAN; FRAMES];

        let params = MotStrobeParams::new();
        params.mute.set_value(true);
        params.bypass.set_value(true);
        let mut state = MotStrobe::default();
        MotStrobe::reset(&mut state, &params, &AudioConfig::new(48_000.0, FRAMES));

        let inputs: [&[f32]; 1] = [&input];
        let mut outputs: [&mut [f32]; 1] = [&mut output];
        let mut buffer = AudioBuffer::from_slices_checked(&inputs, &mut outputs, FRAMES);
        let events = EventList::default();
        let mut output_events = EventList::with_capacity(0);
        let transport = TransportInfo::default();
        let mut context = ProcessContext::new(&transport, 48_000.0, FRAMES, &mut output_events);

        MotStrobe::process(&mut state, &params, &mut buffer, &events, &mut context);

        assert_eq!(output, input);
        assert_eq!(state.signal_chain.processed_samples(), FRAMES as u64);
    }

    #[cfg(feature = "rt-paranoid")]
    #[test]
    fn full_chromatic_scan_is_allocation_free_and_bit_exact() {
        const FRAMES: usize = 2_048;
        let input: [f32; FRAMES] = std::array::from_fn(|index| {
            let time = index as f32 / 48_000.0;
            0.25 * (std::f32::consts::TAU * midi_to_hz(35) * time).sin()
        });
        let mut output = [f32::NAN; FRAMES];

        let params = MotStrobeParams::new();
        let mut state = MotStrobe::default();
        MotStrobe::reset(&mut state, &params, &AudioConfig::new(48_000.0, FRAMES));

        let inputs: [&[f32]; 1] = [&input];
        let mut outputs: [&mut [f32]; 1] = [&mut output];
        let mut buffer = AudioBuffer::from_slices_checked(&inputs, &mut outputs, FRAMES);
        let events = EventList::default();
        let mut output_events = EventList::with_capacity(0);
        let transport = TransportInfo::default();
        let mut context = ProcessContext::new(&transport, 48_000.0, FRAMES, &mut output_events);

        let (_, allocations) = truce::rt::audit(|| {
            let _section = truce::rt::RtSection::enter();
            MotStrobe::process(&mut state, &params, &mut buffer, &events, &mut context)
        });

        assert_eq!(allocations, 0, "audio-thread allocation detected");
        assert_eq!(output, input);
    }

    #[test]
    fn host_state_round_trip_preserves_amp_tuner_capture_and_exact_model_reference() {
        let original = <Plugin as PluginExport>::create();
        original.params().bypass.set_value(true);
        original.params().mute.set_value(true);
        original.params().input_gain.set_value(3.7);
        original.params().tight.set_value(42.5);
        original.params().bite.set_value(61.2);
        original.params().ir_processing.set_value(1);
        original.params().offsets_enabled.set_value(false);
        original.params().string_7_note.set_value(36);
        original.params().string_7_offset.set_value(-3.5);
        original.params().string_1_note.set_value(65);
        original.params().string_1_offset.set_value(2.5);
        original.params().capture_role.set_value(2);
        original.params().capture_target.set_value(1);
        original.params().capture_send_trim.set_value(-17.3);
        original.params().max_passes.set_value(321);
        original.params().capture_session_id.store(0x1234);
        original.params().runtime_generation.store(29);
        original.params().capture_load_impedance_ohms.store(8);
        *original.params().selected_model_id.write().unwrap() = "amp-blue".to_owned();
        *original.params().selected_model_sha256.write().unwrap() = "ab".repeat(32);
        *original
            .params()
            .selected_model_filename_hint
            .write()
            .unwrap() = "renamed blue model.motmodel".to_owned();
        *original.params().selected_ir_path.write().unwrap() = "/IRs/pasadena cab.wav".to_owned();
        *original.params().selected_ir_id.write().unwrap() = "ir-pasadena".to_owned();
        *original.params().selected_ir_sha256.write().unwrap() = "cd".repeat(32);
        *original.params().selected_ir_filename_hint.write().unwrap() =
            "pasadena cab.wav".to_owned();
        *original.params().capture_session_name.write().unwrap() = "amp-room-a".to_owned();
        *original.params().capture_model_name.write().unwrap() = "5153 Blue".to_owned();
        *original.params().capture_amplifier.write().unwrap() = "EVH 5153 100W".to_owned();
        *original.params().capture_amplifier_channel.write().unwrap() = "Blue".to_owned();
        *original.params().capture_control_positions.write().unwrap() = "Gain 5".to_owned();
        *original.params().capture_interface_output.write().unwrap() = "Line Out 3".to_owned();
        *original.params().capture_interface_input.write().unwrap() = "Input 1".to_owned();
        *original.params().capture_reamp_box.write().unwrap() = "Reamp".to_owned();
        *original.params().capture_reactive_load.write().unwrap() = "Captor X".to_owned();
        *original.params().capture_return_gain_note.write().unwrap() = "Pad off".to_owned();
        original.params().runtime_request_epoch.store(91);
        original
            .params()
            .check_level_trigger_generation
            .store(92, Ordering::Release);

        let state = truce::core::state::snapshot_plugin(&original);
        let mut restored = <Plugin as PluginExport>::create();
        truce::core::state::restore_plugin(&mut restored, &state).expect("state must restore");

        assert!(restored.params().bypass.value());
        assert!(restored.params().mute.value());
        assert_eq!(restored.params().input_gain.value(), 3.7);
        assert_eq!(restored.params().tight.value(), 42.5);
        assert_eq!(restored.params().bite.value(), 61.2);
        assert_eq!(restored.params().ir_processing.value(), 1);
        assert!(!restored.params().offsets_enabled.value());
        assert_eq!(restored.params().string_7_note.value(), 36);
        assert_eq!(restored.params().string_7_offset.value(), -3.5);
        assert_eq!(restored.params().string_1_note.value(), 65);
        assert_eq!(restored.params().string_1_offset.value(), 2.5);
        assert_eq!(restored.params().capture_role.value(), 2);
        assert_eq!(restored.params().capture_target.value(), 1);
        assert_eq!(restored.params().capture_send_trim.value(), -17.3);
        assert_eq!(restored.params().max_passes.value(), 321);
        assert_eq!(restored.params().capture_session_id.load(), 0x1234);
        assert_eq!(restored.params().runtime_generation.load(), 29);
        assert_eq!(restored.params().capture_load_impedance_ohms.load(), 8);
        assert_eq!(
            read_shared_string(&restored.params().selected_model_id),
            "amp-blue"
        );
        assert_eq!(
            read_shared_string(&restored.params().selected_model_sha256),
            "ab".repeat(32)
        );
        assert_eq!(
            read_shared_string(&restored.params().selected_model_filename_hint),
            "renamed blue model.motmodel"
        );
        assert_eq!(
            read_shared_string(&restored.params().selected_ir_path),
            "/IRs/pasadena cab.wav"
        );
        assert_eq!(
            read_shared_string(&restored.params().selected_ir_id),
            "ir-pasadena"
        );
        assert_eq!(
            read_shared_string(&restored.params().selected_ir_sha256),
            "cd".repeat(32)
        );
        assert_eq!(
            read_shared_string(&restored.params().selected_ir_filename_hint),
            "pasadena cab.wav"
        );
        assert_eq!(
            read_shared_string(&restored.params().capture_session_name),
            "amp-room-a"
        );
        assert_eq!(
            read_shared_string(&restored.params().capture_model_name),
            "5153 Blue"
        );
        assert_eq!(
            read_shared_string(&restored.params().capture_amplifier),
            "EVH 5153 100W"
        );
        assert_eq!(
            read_shared_string(&restored.params().capture_amplifier_channel),
            "Blue"
        );
        assert_eq!(
            read_shared_string(&restored.params().capture_control_positions),
            "Gain 5"
        );
        assert_eq!(
            read_shared_string(&restored.params().capture_interface_output),
            "Line Out 3"
        );
        assert_eq!(
            read_shared_string(&restored.params().capture_interface_input),
            "Input 1"
        );
        assert_eq!(
            read_shared_string(&restored.params().capture_reamp_box),
            "Reamp"
        );
        assert_eq!(
            read_shared_string(&restored.params().capture_reactive_load),
            "Captor X"
        );
        assert_eq!(
            read_shared_string(&restored.params().capture_return_gain_note),
            "Pad off"
        );
        assert_eq!(restored.params().runtime_request_epoch.load(), 0);
        assert_eq!(
            restored
                .params()
                .check_level_trigger_generation
                .load(Ordering::Acquire),
            0
        );
    }
}
