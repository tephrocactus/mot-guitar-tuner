mod amp;
mod cabinet;
mod editor;
mod signal_chain;
mod tuner;

use std::sync::Arc;

use truce::prelude::*;
use truce_egui::EguiEditor;

use editor::{MotStrobeUi, WINDOW_SIZE};
use signal_chain::{GuitarSignalChain, OutputMute};
use tuner::{PitchAnalysis, STRING_COUNT, TunerEngine, cents_ratio, midi_to_hz};

#[derive(Params)]
pub struct MotStrobeParams {
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

pub(crate) use MotStrobeParamsParamId as P;

pub struct MotStrobe {
    tuner: TunerEngine,
    signal_chain: GuitarSignalChain,
    output_mute: OutputMute,
    sample_rate: f32,
    strobe_phase: f32,
    displayed_note: Option<u8>,
    reference_hz: f32,
}

impl Default for MotStrobe {
    fn default() -> Self {
        Self {
            tuner: TunerEngine::default(),
            signal_chain: GuitarSignalChain::default(),
            output_mute: OutputMute::default(),
            sample_rate: 48_000.0,
            strobe_phase: 0.0,
            displayed_note: None,
            reference_hz: 0.0,
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

impl PluginLogic for MotStrobe {
    type Params = MotStrobeParams;
    type DspState = Self;

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
        state.tuner.reset(state.sample_rate, &notes(params));
        state.signal_chain.reset(config);
        state
            .output_mute
            .reset(state.sample_rate, params.mute.value());
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
            state.signal_chain.process_block(input, output);
            for sample_index in 0..samples {
                let mute_gain = state.output_mute.next_gain(mute_requested);
                if bypassed {
                    output[sample_index] = input[sample_index];
                } else {
                    output[sample_index] *= mute_gain;
                }
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

truce::plugin! {
    logic: MotStrobe,
    params: MotStrobeParams,
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
    fn host_state_round_trip_preserves_all_user_tuning_controls() {
        let original = <Plugin as PluginExport>::create();
        original.params().bypass.set_value(true);
        original.params().mute.set_value(true);
        original.params().offsets_enabled.set_value(false);
        original.params().string_7_note.set_value(36);
        original.params().string_7_offset.set_value(-3.5);
        original.params().string_1_note.set_value(65);
        original.params().string_1_offset.set_value(2.5);

        let state = truce::core::state::snapshot_plugin(&original);
        let mut restored = <Plugin as PluginExport>::create();
        truce::core::state::restore_plugin(&mut restored, &state).expect("state must restore");

        assert!(restored.params().bypass.value());
        assert!(restored.params().mute.value());
        assert!(!restored.params().offsets_enabled.value());
        assert_eq!(restored.params().string_7_note.value(), 36);
        assert_eq!(restored.params().string_7_offset.value(), -3.5);
        assert_eq!(restored.params().string_1_note.value(), 65);
        assert_eq!(restored.params().string_1_offset.value(), 2.5);
    }
}
