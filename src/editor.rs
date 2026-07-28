use std::time::Duration;

use egui::{Align, Align2, Color32, FontId, Layout, RichText, Sense, Stroke, StrokeKind, Vec2};
use truce::prelude::*;
use truce_egui::EditorUi;

use crate::tuner::STRING_COUNT;
use crate::{MotStrobeParams, P, notes, offsets, round_to_tenth};

pub(crate) const WINDOW_SIZE: (u32, u32) = (920, 640);
const OFFSET_MIN: f32 = -25.0;
const OFFSET_MAX: f32 = 25.0;

pub(crate) struct MotStrobeUi;

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
