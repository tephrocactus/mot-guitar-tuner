use std::time::Duration;

use egui::{Align2, Color32, FontId, RichText, Sense, Stroke, StrokeKind, Vec2};
use mot_core::tuner::STRING_COUNT;
use truce::prelude::*;
use truce_egui::EditorUi;

use crate::{MotTunerParams, P, VERSION, notes, offsets, round_to_tenth};

pub(crate) const WINDOW_SIZE: (u32, u32) = (900, 720);
const STRING_EDITOR_HEIGHT: f32 = 292.0;
const MIN_STROBE_HEIGHT: f32 = 240.0;
const OFFSET_MIN: f32 = -25.0;
const OFFSET_MAX: f32 = 25.0;

pub(crate) struct MotTunerUi;

impl EditorUi<MotTunerParams> for MotTunerUi {
    fn ui(&mut self, ui: &mut egui::Ui, context: &PluginContext<MotTunerParams>) {
        ui.ctx().request_repaint_after(Duration::from_millis(16));
        mot_ui::apply(ui);
        ui.painter()
            .rect_filled(ui.max_rect(), 0.0, mot_ui::BACKGROUND);

        mot_ui::background_frame().show(ui, |ui| {
            mot_ui::header(ui, "MOT TUNER", VERSION, false, |ui| {
                let muted = context.mute.value();
                let button = if muted {
                    mot_ui::danger_button(RichText::new("MUTE").color(mot_ui::BACKGROUND))
                } else {
                    mot_ui::ghost_button("MUTE")
                };
                if ui.add(button).clicked() {
                    context.automate(P::Mute, if muted { 0.0 } else { 1.0 });
                }
            });

            let body_height = ui.available_height();
            egui::ScrollArea::vertical()
                .id_salt("mot_tuner_body")
                .auto_shrink([false, false])
                .max_height(body_height)
                .show(ui, |ui| {
                    // Keep the complete seven-string editor visible. Its title, header,
                    // seven control rows, grid gaps and panel margins need about 290 px.
                    // On a short host surface the body scrolls instead of clipping either
                    // the string editor or the useful minimum height of the strobe.
                    let strobe_height = (body_height - STRING_EDITOR_HEIGHT - mot_ui::SECTION_GAP)
                        .max(MIN_STROBE_HEIGHT);
                    strobe(ui, context, strobe_height);
                    let extra_section_gap =
                        (mot_ui::SECTION_GAP - ui.spacing().item_spacing.y).max(0.0);
                    ui.add_space(extra_section_gap);
                    string_editor(ui, context);
                });
        });
    }
}

fn strobe(ui: &mut egui::Ui, context: &PluginContext<MotTunerParams>, height: f32) {
    let desired = Vec2::new(ui.available_width(), height);
    let (rect, _) = ui.allocate_exact_size(desired, Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, mot_ui::PANEL_RADIUS, mot_ui::SURFACE);

    let detected = decode_note(context.get_meter(P::DetectedNote));
    let cents = (context.get_meter(P::Cents) - 0.5) * 100.0;
    let phase = context.get_meter(P::Phase);
    let active = detected.is_some() && !context.bypass.value();
    let period = 54.0;
    let stripe_width = period * 0.5;
    let offset = phase * period;

    let bright =
        mot_ui::strobe_stripe_color(active, true).linear_multiply(if active { 0.42 } else { 1.0 });
    let dark = mot_ui::strobe_stripe_color(active, false);

    let mut x = rect.left() - period + offset;
    let mut alternate = false;
    while x < rect.right() + period {
        let stripe = egui::Rect::from_min_max(
            egui::pos2(x, rect.top() + mot_ui::PANEL_MARGIN),
            egui::pos2(x + stripe_width, rect.bottom() - mot_ui::PANEL_MARGIN),
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
        mot_ui::PANEL_RADIUS,
        Stroke::new(1.0_f32, mot_ui::LINE),
        StrokeKind::Inside,
    );

    let circle_center = rect.center();
    let circle_radius = height.mul_add(0.24, 0.0).clamp(72.0, 92.0);
    painter.circle_filled(circle_center, circle_radius, Color32::from_white_alpha(220));
    painter.circle_stroke(
        circle_center,
        circle_radius,
        Stroke::new(1.5_f32, Color32::from_white_alpha(235)),
    );

    let note_text = detected.map_or_else(|| "—".to_owned(), note_name);
    let cents_text = if detected.is_some() {
        format!("{cents:+.1} c")
    } else {
        "waiting".to_owned()
    };
    let ink = Color32::from_rgb(18, 24, 27);
    painter.text(
        egui::pos2(circle_center.x, circle_center.y - 20.0),
        Align2::CENTER_CENTER,
        note_text,
        FontId::monospace(48.0),
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

fn string_editor(ui: &mut egui::Ui, context: &PluginContext<MotTunerParams>) {
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

    mot_ui::panel().show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(mot_ui::section_label("STRING OFFSETS"));
            offset_switch(ui, context);
        });

        egui::Grid::new("mot_tuner_string_grid")
            .num_columns(3)
            .spacing([28.0, 5.0])
            .min_col_width(92.0)
            .show(ui, |ui| {
                ui.label(mot_ui::field_label("STRING"));
                ui.label(mot_ui::field_label("REFERENCE NOTE"));
                ui.label(mot_ui::field_label("OFFSET"));
                ui.end_row();

                for index in 0..STRING_COUNT {
                    let display_string = 7 - index;
                    let is_matched = matched == Some(index);
                    let string_label = RichText::new(display_string.to_string())
                        .monospace()
                        .strong()
                        .color(if is_matched {
                            mot_ui::ACCENT
                        } else {
                            mot_ui::TEXT
                        });
                    ui.label(string_label);

                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        if ui
                            .add_enabled(note_values[index] > 0, note_step_button("◀"))
                            .clicked()
                        {
                            let previous = note_values[index] - 1;
                            context.automate(note_ids[index], f64::from(previous) / 127.0);
                        }
                        ui.add_sized(
                            [54.0, 18.0],
                            egui::Label::new(
                                RichText::new(note_name(note_values[index]))
                                    .monospace()
                                    .strong()
                                    .color(if is_matched {
                                        mot_ui::ACCENT
                                    } else {
                                        mot_ui::TEXT
                                    }),
                            ),
                        );
                        if ui
                            .add_enabled(note_values[index] < 127, note_step_button("▶"))
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

fn offset_switch(ui: &mut egui::Ui, context: &PluginContext<MotTunerParams>) {
    let enabled = context.offsets_enabled.value();
    ui.horizontal(|ui| {
        let (rect, response) = ui.allocate_exact_size(Vec2::new(46.0, 24.0), Sense::click());
        if response.clicked() {
            context.automate(P::OffsetsEnabled, if enabled { 0.0 } else { 1.0 });
        }
        let response =
            response.on_hover_text("Temporarily compare reference notes against pure 12-TET");
        let hovered = response.hovered();
        let pressed = response.is_pointer_button_down_on();
        if hovered {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }

        let resting_track = if enabled {
            Color32::from_rgb(31, 107, 100)
        } else {
            mot_ui::LINE
        };
        let track_color = if pressed {
            Color32::from_rgb(16, 80, 78)
        } else if hovered {
            resting_track.linear_multiply(1.22)
        } else {
            resting_track
        };
        ui.painter()
            .rect_filled(rect, rect.height() / 2.0, track_color);
        ui.painter().rect_stroke(
            rect,
            rect.height() / 2.0,
            Stroke::new(
                1.0_f32,
                if pressed {
                    mot_ui::ACCENT
                } else if hovered {
                    mot_ui::ACCENT.linear_multiply(0.72)
                } else {
                    mot_ui::LINE
                },
            ),
            StrokeKind::Inside,
        );
        let knob_x = if enabled {
            rect.right() - rect.height() / 2.0
        } else {
            rect.left() + rect.height() / 2.0
        };
        ui.painter().circle_filled(
            egui::pos2(knob_x, rect.center().y),
            8.5,
            if pressed {
                mot_ui::TEXT
            } else if enabled {
                mot_ui::ACCENT
            } else if hovered {
                mot_ui::TEXT
            } else {
                mot_ui::DIM
            },
        );
        ui.label(
            RichText::new(if enabled { "ON" } else { "OFF" })
                .monospace()
                .color(if enabled { mot_ui::ACCENT } else { mot_ui::DIM }),
        );
    });
}

fn note_step_button(label: &'static str) -> egui::Button<'static> {
    egui::Button::new(RichText::new(label).monospace().strong())
        .frame_when_inactive(false)
        .corner_radius(mot_ui::CONTROL_RADIUS)
        .min_size(Vec2::new(28.0, 24.0))
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
