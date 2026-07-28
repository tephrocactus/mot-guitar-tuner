mod tuner;

use std::sync::Arc;
use std::time::Duration;

use egui::{Align, Align2, Color32, FontId, Layout, RichText, Sense, Stroke, StrokeKind, Vec2};
use truce::prelude::*;
use truce_egui::{EditorUi, EguiEditor};

use tuner::{PitchAnalysis, STRING_COUNT, TunerEngine, cents_ratio, midi_to_hz};

const WINDOW_SIZE: (u32, u32) = (920, 640);
const OFFSET_MIN: f32 = -25.0;
const OFFSET_MAX: f32 = 25.0;

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

use MotStrobeParamsParamId as P;

pub struct MotStrobe {
    tuner: TunerEngine,
    sample_rate: f32,
    strobe_phase: f32,
    displayed_note: Option<u8>,
    reference_hz: f32,
}

impl Default for MotStrobe {
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

fn notes(params: &MotStrobeParams) -> [u8; STRING_COUNT] {
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

fn offsets(params: &MotStrobeParams) -> [f32; STRING_COUNT] {
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

fn round_to_tenth(value: f32) -> f32 {
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
        true
    }

    fn reset(state: &mut Self::DspState, params: &Self::Params, config: &AudioConfig) {
        state.sample_rate = config.sample_rate as f32;
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
        let open_notes = notes(params);
        state.tuner.configure_range(&open_notes);

        let channels = buffer.channels();
        let samples = buffer.num_samples();
        let bypassed = params.bypass.value();
        let muted = params.mute.value() && !bypassed;

        if !bypassed && channels > 0 {
            // Analyze a mono sum while leaving every host sample untouched.
            for sample_index in 0..samples {
                let mut mono = 0.0;
                for channel in 0..channels {
                    let sample = if buffer.is_in_place(channel) {
                        buffer.in_out_mut(channel)[sample_index]
                    } else {
                        buffer.input(channel)[sample_index]
                    };
                    mono += sample;
                }
                state.tuner.push_sample(mono / channels as f32);
            }
        }

        // Muting affects only the output: pitch analysis above continues
        // from the live input. Host bypass has priority over mute.
        for channel in 0..channels {
            if muted {
                if buffer.is_in_place(channel) {
                    buffer.in_out_mut(channel).fill(0.0);
                } else {
                    let (_, output) = buffer.io(channel);
                    output.fill(0.0);
                }
            } else if !buffer.is_in_place(channel) {
                let (input, output) = buffer.io(channel);
                output.copy_from_slice(input);
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

struct MotStrobeUi;

impl EditorUi<MotStrobeParams> for MotStrobeUi {
    fn ui(&mut self, ui: &mut egui::Ui, context: &PluginContext<MotStrobeParams>) {
        ui.ctx().request_repaint_after(Duration::from_millis(8));

        let background = Color32::from_rgb(10, 12, 14);
        let panel = Color32::from_rgb(20, 24, 28);
        let cyan = Color32::from_rgb(58, 220, 210);
        let text_dim = Color32::from_rgb(135, 148, 155);
        ui.visuals_mut().panel_fill = background;
        ui.visuals_mut().override_text_color = Some(Color32::from_rgb(228, 235, 238));

        egui::Frame::new()
            .fill(background)
            .inner_margin(18.0)
            .show(ui, |ui| {
                header(ui, context, cyan);
                ui.add_space(14.0);
                strobe(ui, context, panel, cyan);
                ui.add_space(10.0);
                string_editor(ui, context, panel, cyan, text_dim);
            });
    }
}

fn header(ui: &mut egui::Ui, context: &PluginContext<MotStrobeParams>, accent: Color32) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new("MOT GUITAR TUNER")
                .font(FontId::proportional(24.0))
                .strong()
                .color(accent),
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

fn decode_string(encoded: f32) -> Option<usize> {
    if encoded <= 0.0 {
        return None;
    }
    let decoded = (encoded * 8.0).round() as isize - 1;
    (0..STRING_COUNT as isize)
        .contains(&decoded)
        .then_some(decoded as usize)
}

fn decode_note(encoded: f32) -> Option<u8> {
    if encoded <= 0.0 {
        return None;
    }
    let decoded = (encoded * 128.0).round() as i16 - 1;
    (0..=127).contains(&decoded).then_some(decoded as u8)
}

fn note_name(midi_note: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
    ];
    let pitch_class = usize::from(midi_note % 12);
    let octave = i16::from(midi_note / 12) - 1;
    format!("{}{octave}", NAMES[pitch_class])
}

truce::plugin! {
    logic: MotStrobe,
    params: MotStrobeParams,
}

truce::enable_rt_paranoid!();

#[cfg(test)]
mod tests {
    use super::*;

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
        let left = [0.0, -0.5, 0.25, 1.0, -1.0, 0.125, -0.25, 0.75];
        let right = [0.75, -0.25, 0.125, -1.0, 1.0, 0.25, -0.5, 0.0];

        for bypassed in [false, true] {
            let params = MotStrobeParams::new();
            params.bypass.set_value(bypassed);
            let mut state = MotStrobe::default();
            MotStrobe::reset(&mut state, &params, &AudioConfig::new(48_000.0, FRAMES));

            let inputs: [&[f32]; 2] = [&left, &right];
            let mut output_left = [f32::NAN; FRAMES];
            let mut output_right = [f32::NAN; FRAMES];
            let mut outputs: [&mut [f32]; 2] = [&mut output_left, &mut output_right];
            let mut buffer = AudioBuffer::from_slices_checked(&inputs, &mut outputs, FRAMES);
            let events = EventList::default();
            let mut output_events = EventList::with_capacity(0);
            let transport = TransportInfo::default();
            let mut context = ProcessContext::new(&transport, 48_000.0, FRAMES, &mut output_events);

            MotStrobe::process(&mut state, &params, &mut buffer, &events, &mut context);

            assert_eq!(output_left, left);
            assert_eq!(output_right, right);
        }
    }

    #[test]
    fn in_place_audio_is_never_modified() {
        const FRAMES: usize = 8;
        let original_left = [0.0, -0.5, 0.25, 1.0, -1.0, 0.125, -0.25, 0.75];
        let original_right = [0.75, -0.25, 0.125, -1.0, 1.0, 0.25, -0.5, 0.0];
        let mut left = original_left;
        let mut right = original_right;

        let params = MotStrobeParams::new();
        let mut state = MotStrobe::default();
        MotStrobe::reset(&mut state, &params, &AudioConfig::new(48_000.0, FRAMES));

        let inputs: [&[f32]; 2] = [&[], &[]];
        let mut outputs: [&mut [f32]; 2] = [&mut left, &mut right];
        let mut buffer = AudioBuffer::from_slices_checked(&inputs, &mut outputs, FRAMES);
        buffer.set_in_place_mask(0b11);
        let events = EventList::default();
        let mut output_events = EventList::with_capacity(0);
        let transport = TransportInfo::default();
        let mut context = ProcessContext::new(&transport, 48_000.0, FRAMES, &mut output_events);

        MotStrobe::process(&mut state, &params, &mut buffer, &events, &mut context);

        assert_eq!(left, original_left);
        assert_eq!(right, original_right);
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
    }

    #[test]
    fn mute_zeros_in_place_output() {
        const FRAMES: usize = 8;
        let mut audio = [0.0, -0.5, 0.25, 1.0, -1.0, 0.125, -0.25, 0.75];

        let params = MotStrobeParams::new();
        params.mute.set_value(true);
        let mut state = MotStrobe::default();
        MotStrobe::reset(&mut state, &params, &AudioConfig::new(48_000.0, FRAMES));

        let inputs: [&[f32]; 1] = [&[]];
        let mut outputs: [&mut [f32]; 1] = [&mut audio];
        let mut buffer = AudioBuffer::from_slices_checked(&inputs, &mut outputs, FRAMES);
        buffer.set_in_place_mask(0b1);
        let events = EventList::default();
        let mut output_events = EventList::with_capacity(0);
        let transport = TransportInfo::default();
        let mut context = ProcessContext::new(&transport, 48_000.0, FRAMES, &mut output_events);

        MotStrobe::process(&mut state, &params, &mut buffer, &events, &mut context);

        assert_eq!(audio, [0.0; FRAMES]);
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
