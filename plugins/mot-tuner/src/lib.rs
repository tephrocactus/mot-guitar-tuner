mod editor;

use std::sync::Arc;

use mot_core::tuner::{PitchAnalysis, STRING_COUNT, TunerEngine, cents_ratio, midi_to_hz};
use truce::prelude::*;
use truce_egui::EguiEditor;

use editor::{MotTunerUi, WINDOW_SIZE};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Params)]
pub struct MotTunerParams {
    #[param(name = "Bypass", flags = "automatable | bypass")]
    pub bypass: BoolParam,

    #[param(name = "Mute", flags = "automatable")]
    pub mute: BoolParam,

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

    #[meter]
    pub detected_note: MeterSlot,
    #[meter]
    pub matched_string: MeterSlot,
    #[meter]
    pub cents: MeterSlot,
    #[meter]
    pub phase: MeterSlot,
}

pub(crate) use MotTunerParamsParamId as P;

pub(crate) fn notes(params: &MotTunerParams) -> [u8; STRING_COUNT] {
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

pub(crate) fn offsets(params: &MotTunerParams) -> [f32; STRING_COUNT] {
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
        }
    } else {
        ResolvedReference {
            frequency_hz: equal_temperament,
            matched_string: None,
        }
    }
}

pub struct TunerDspState {
    tuner: TunerEngine,
    sample_rate: f32,
    strobe_phase: f32,
    displayed_note: Option<u8>,
    reference_hz: f32,
}

impl Default for TunerDspState {
    fn default() -> Self {
        Self {
            tuner: TunerEngine::default(),
            sample_rate: 48_000.0,
            strobe_phase: 0.0,
            displayed_note: None,
            reference_hz: 0.0,
        }
    }
}

fn prepare_display(
    state: &mut TunerDspState,
    params: &MotTunerParams,
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

pub struct MotTuner;

impl PluginLogic for MotTuner {
    type Params = MotTunerParams;
    type DspState = TunerDspState;

    fn supports_in_place() -> bool {
        // Disjoint host buffers make the exact passthrough branch explicit.
        false
    }

    fn bus_layouts() -> Vec<BusLayout> {
        vec![BusLayout::mono()]
    }

    fn reset(state: &mut Self::DspState, params: &Self::Params, config: &AudioConfig) {
        state.sample_rate = config.sample_rate.max(1.0) as f32;
        state.tuner.reset(state.sample_rate, &notes(params));
        state.strobe_phase = 0.0;
        state.displayed_note = None;
        state.reference_hz = 0.0;
    }

    fn process(
        state: &mut Self::DspState,
        params: &Self::Params,
        buffer: &mut AudioBuffer,
        _events: &EventList,
        context: &mut ProcessContext,
    ) -> ProcessStatus {
        let samples = buffer.num_samples();
        let bypassed = params.bypass.value();

        if buffer.num_input_channels() > 0 && buffer.num_output_channels() > 0 {
            let (input, output) = buffer.io_pair(0, 0);
            if !bypassed {
                state.tuner.configure_range(&notes(params));
                for &sample in input {
                    state.tuner.push_sample(sample);
                }
            }

            // Host bypass wins. In every unmuted state this is an exact copy:
            // the tuner never places a gain, smoother, or denormal guard in
            // the monitored audio path.
            if bypassed || !params.mute.value() {
                output.copy_from_slice(input);
            } else {
                output.fill(0.0);
            }
        }

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

        ProcessStatus::Normal
    }

    fn latency(_state: &Self::DspState) -> u32 {
        0
    }

    fn tail(_state: &Self::DspState) -> u32 {
        0
    }

    fn editor(params: Arc<Self::Params>) -> Box<dyn Editor> {
        EguiEditor::with_ui(params, WINDOW_SIZE, MotTunerUi).into_editor()
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

truce::plugin! {
    logic: MotTuner,
    params: MotTunerParams,
}

truce::enable_rt_paranoid!();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unmuted_audio_is_bit_exact_at_common_host_sample_rates() {
        const FRAMES: usize = 16;
        let input = [
            0.0, -0.75, 0.5, -0.25, 0.125, -0.0625, 1.0, -1.0, 0.3, -0.2, 0.1, -0.05, 0.025,
            -0.0125, 0.00625, -0.003125,
        ];

        for sample_rate in [44_100.0, 48_000.0, 88_200.0, 96_000.0, 192_000.0] {
            let params = MotTunerParams::new();
            let mut state = TunerDspState::default();
            MotTuner::reset(&mut state, &params, &AudioConfig::new(sample_rate, FRAMES));

            let mut output = [f32::NAN; FRAMES];
            let inputs: [&[f32]; 1] = [&input];
            let mut outputs: [&mut [f32]; 1] = [&mut output];
            let mut buffer = AudioBuffer::from_slices_checked(&inputs, &mut outputs, FRAMES);
            let events = EventList::default();
            let mut output_events = EventList::with_capacity(0);
            let transport = TransportInfo::default();
            let mut context =
                ProcessContext::new(&transport, sample_rate, FRAMES, &mut output_events);

            MotTuner::process(&mut state, &params, &mut buffer, &events, &mut context);
            assert_eq!(output, input, "sample rate {sample_rate}");
        }
    }

    #[test]
    fn mute_silences_output_and_host_bypass_wins() {
        const FRAMES: usize = 4;
        let input = [0.25, -0.5, 0.75, -1.0];
        let params = MotTunerParams::new();
        params.mute.set_value(true);
        let mut state = TunerDspState::default();
        MotTuner::reset(&mut state, &params, &AudioConfig::new(48_000.0, FRAMES));

        let mut render = |output: &mut [f32; FRAMES]| {
            let inputs: [&[f32]; 1] = [&input];
            let mut outputs: [&mut [f32]; 1] = [output];
            let mut buffer = AudioBuffer::from_slices_checked(&inputs, &mut outputs, FRAMES);
            let events = EventList::default();
            let mut output_events = EventList::with_capacity(0);
            let transport = TransportInfo::default();
            let mut context = ProcessContext::new(&transport, 48_000.0, FRAMES, &mut output_events);
            MotTuner::process(&mut state, &params, &mut buffer, &events, &mut context);
        };

        let mut output = [f32::NAN; FRAMES];
        render(&mut output);
        assert_eq!(output, [0.0; FRAMES]);

        params.bypass.set_value(true);
        render(&mut output);
        assert_eq!(output, input);
    }

    #[test]
    fn duplicate_reference_notes_use_the_first_string() {
        let notes = [35, 40, 45, 50, 55, 64, 64];
        let offsets = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -3.0];
        let reference = resolve_reference(64, &notes, &offsets, true);
        assert_eq!(reference.matched_string, Some(5));
        assert!((reference.frequency_hz - midi_to_hz(64) * cents_ratio(2.0)).abs() < 0.000_1);
    }

    #[test]
    fn disabled_offsets_use_equal_temperament() {
        let reference = resolve_reference(35, &[35, 40, 45, 50, 55, 59, 64], &[9.0; 7], false);
        assert_eq!(reference.matched_string, Some(0));
        assert!((reference.frequency_hz - midi_to_hz(35)).abs() < 0.000_1);
    }

    #[test]
    fn editor_has_a_headless_render_path() {
        let params = Arc::new(MotTunerParams::new());
        let mut editor = EguiEditor::with_ui(Arc::clone(&params), WINDOW_SIZE, MotTunerUi);
        let erased: Arc<dyn truce::params::Params> = params;
        assert_eq!(Editor::size(&editor), WINDOW_SIZE);
        if let Some((_, width, height)) = Editor::screenshot(&mut editor, erased) {
            assert_eq!((width, height), (WINDOW_SIZE.0 * 2, WINDOW_SIZE.1 * 2));
        }
    }
}
