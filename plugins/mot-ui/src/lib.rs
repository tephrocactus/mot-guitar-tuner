//! Shared Signal Lab visual language for the MOT plug-in suite.
//!
//! This crate is intentionally UI-only. It contains no audio state and is
//! never called from a plug-in's real-time callback.

use egui::{
    Align, Align2, Button, Color32, CursorIcon, FontId, Frame, Layout, Response, RichText, Sense,
    Shape, Stroke, Ui, Vec2, Widget, WidgetText, epaint::PathStroke,
};

pub const BACKGROUND: Color32 = Color32::from_rgb(8, 11, 13);
pub const SURFACE: Color32 = Color32::from_rgb(17, 23, 27);
pub const RAISED: Color32 = Color32::from_rgb(23, 32, 38);
pub const DEEP: Color32 = Color32::from_rgb(10, 15, 18);
pub const LINE: Color32 = Color32::from_rgb(42, 54, 61);
pub const TEXT: Color32 = Color32::from_rgb(231, 238, 240);
pub const DIM: Color32 = Color32::from_rgb(132, 147, 154);
pub const ACCENT: Color32 = Color32::from_rgb(57, 213, 208);
pub const WAITING: Color32 = Color32::from_rgb(240, 185, 76);
pub const SUCCESS: Color32 = Color32::from_rgb(67, 214, 159);
pub const ERROR: Color32 = Color32::from_rgb(244, 91, 105);

pub const OUTER_MARGIN: f32 = 20.0;
pub const PANEL_MARGIN: f32 = 16.0;
pub const ROW_GAP: f32 = 12.0;
pub const SECTION_GAP: f32 = 18.0;
pub const PANEL_RADIUS: f32 = 10.0;
pub const CONTROL_RADIUS: f32 = 6.0;
pub const HEADER_TITLE_SIZE: f32 = 27.0;

pub fn apply(ui: &mut Ui) {
    let visuals = ui.visuals_mut();
    visuals.panel_fill = BACKGROUND;
    visuals.override_text_color = Some(TEXT);
    visuals.widgets.noninteractive.bg_fill = SURFACE;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, LINE);
    visuals.widgets.inactive.weak_bg_fill = RAISED;
    visuals.widgets.inactive.bg_fill = RAISED;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, LINE);
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, TEXT);
    visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(28, 45, 50);
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(28, 45, 50);
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, ACCENT.linear_multiply(0.65));
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0_f32, TEXT);
    visuals.widgets.active.weak_bg_fill = Color32::from_rgb(16, 55, 57);
    visuals.widgets.active.bg_fill = Color32::from_rgb(16, 55, 57);
    visuals.widgets.active.bg_stroke = Stroke::new(1.0_f32, ACCENT);
    visuals.widgets.active.fg_stroke = Stroke::new(1.0_f32, TEXT);
    visuals.selection.bg_fill = Color32::from_rgb(16, 55, 57);
    visuals.selection.stroke = Stroke::new(1.0_f32, ACCENT);
    visuals.extreme_bg_color = DEEP;
    visuals.interact_cursor = Some(CursorIcon::PointingHand);

    ui.spacing_mut().item_spacing = Vec2::new(ROW_GAP, ROW_GAP);
    ui.spacing_mut().button_padding = Vec2::new(12.0, 6.0);
}

pub fn background_frame() -> Frame {
    Frame::new().fill(BACKGROUND).inner_margin(OUTER_MARGIN)
}

pub fn panel() -> Frame {
    Frame::new()
        .fill(SURFACE)
        .stroke(Stroke::new(1.0_f32, LINE))
        .corner_radius(PANEL_RADIUS)
        .inner_margin(PANEL_MARGIN)
}

pub fn raised_panel() -> Frame {
    Frame::new()
        .fill(RAISED)
        .stroke(Stroke::new(1.0_f32, LINE))
        .corner_radius(CONTROL_RADIUS)
        .inner_margin(12.0)
}

pub fn header(
    ui: &mut Ui,
    title: &str,
    version: &str,
    zero_latency: bool,
    add_right: impl FnOnce(&mut Ui),
) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(title)
                .font(FontId::proportional(HEADER_TITLE_SIZE))
                .color(ACCENT),
        );
        let suffix = if zero_latency {
            format!("{version}  •  MONO  •  48 kHz  •  ZERO LATENCY")
        } else {
            format!("{version}  •  MONO  •  48 kHz")
        };
        ui.label(RichText::new(suffix).monospace().color(DIM));
        ui.with_layout(Layout::right_to_left(Align::Center), add_right);
    });
    ui.separator();
}

pub fn section_label(label: impl Into<String>) -> RichText {
    RichText::new(label.into())
        .small()
        .strong()
        .monospace()
        .color(ACCENT)
}

pub fn field_label(label: impl Into<String>) -> RichText {
    RichText::new(label.into())
        .small()
        .strong()
        .monospace()
        .color(DIM)
}

pub fn status_text(label: impl Into<String>, color: Color32) -> RichText {
    RichText::new(label.into())
        .strong()
        .monospace()
        .color(color)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ButtonTone {
    Neutral,
    Accent,
    Danger,
    Selected,
}

/// A Signal Lab button whose idle, hover and pressed states all remain visible.
///
/// `egui::Button::fill` and `egui::Button::stroke` intentionally disable the
/// widget's state-dependent visuals. Keeping the tone in this wrapper lets the
/// whole suite use colored buttons without losing hover or pressed feedback.
pub struct SignalButton {
    label: WidgetText,
    tone: ButtonTone,
    min_size: Vec2,
    fill_override: Option<Color32>,
}

impl SignalButton {
    fn new(label: impl Into<WidgetText>, tone: ButtonTone) -> Self {
        Self {
            label: label.into(),
            tone,
            min_size: Vec2::ZERO,
            fill_override: None,
        }
    }

    pub fn min_size(mut self, min_size: Vec2) -> Self {
        self.min_size = min_size;
        self
    }

    /// Set a custom idle fill while retaining derived hover/pressed colors.
    pub fn fill(mut self, fill: impl Into<Color32>) -> Self {
        self.fill_override = Some(fill.into());
        self
    }
}

impl Widget for SignalButton {
    fn ui(self, ui: &mut Ui) -> Response {
        ui.scope(|ui| {
            configure_button_tone(ui, self.tone, self.fill_override);
            ui.add(
                Button::new(self.label)
                    .corner_radius(CONTROL_RADIUS)
                    .min_size(self.min_size),
            )
        })
        .inner
    }
}

fn configure_button_tone(ui: &mut Ui, tone: ButtonTone, fill_override: Option<Color32>) {
    let visuals = ui.visuals_mut();
    let (idle, hovered, active, idle_stroke, text_color) = match tone {
        ButtonTone::Neutral => {
            let idle = fill_override.unwrap_or(RAISED);
            (
                idle,
                mix_color(idle, ACCENT, 0.16),
                mix_color(idle, ACCENT, 0.32),
                LINE,
                TEXT,
            )
        }
        ButtonTone::Accent => (
            ACCENT,
            mix_color(ACCENT, Color32::WHITE, 0.18),
            mix_color(ACCENT, BACKGROUND, 0.24),
            ACCENT.linear_multiply(0.75),
            BACKGROUND,
        ),
        ButtonTone::Danger => {
            let idle = ERROR.linear_multiply(0.78);
            (
                idle,
                mix_color(idle, Color32::WHITE, 0.14),
                mix_color(idle, BACKGROUND, 0.22),
                ERROR.linear_multiply(0.72),
                TEXT,
            )
        }
        ButtonTone::Selected => {
            let idle = Color32::from_rgb(16, 37, 39);
            (
                idle,
                Color32::from_rgb(21, 49, 51),
                Color32::from_rgb(12, 58, 58),
                ACCENT.linear_multiply(0.55),
                TEXT,
            )
        }
    };

    visuals.override_text_color = Some(text_color);
    visuals.widgets.inactive.weak_bg_fill = idle;
    visuals.widgets.inactive.bg_fill = idle;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, idle_stroke);
    visuals.widgets.hovered.weak_bg_fill = hovered;
    visuals.widgets.hovered.bg_fill = hovered;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, ACCENT);
    visuals.widgets.active.weak_bg_fill = active;
    visuals.widgets.active.bg_fill = active;
    visuals.widgets.active.bg_stroke = Stroke::new(1.0_f32, ACCENT);
}

fn mix_color(base: Color32, tint: Color32, amount: f32) -> Color32 {
    let amount = amount.clamp(0.0, 1.0);
    let base = base.to_array();
    let tint = tint.to_array();
    Color32::from_rgba_unmultiplied(
        (f32::from(base[0]) + (f32::from(tint[0]) - f32::from(base[0])) * amount).round() as u8,
        (f32::from(base[1]) + (f32::from(tint[1]) - f32::from(base[1])) * amount).round() as u8,
        (f32::from(base[2]) + (f32::from(tint[2]) - f32::from(base[2])) * amount).round() as u8,
        (f32::from(base[3]) + (f32::from(tint[3]) - f32::from(base[3])) * amount).round() as u8,
    )
}

pub fn ghost_button(label: impl Into<WidgetText>) -> SignalButton {
    SignalButton::new(label, ButtonTone::Neutral)
}

pub fn accent_button(label: impl Into<WidgetText>) -> SignalButton {
    SignalButton::new(label, ButtonTone::Accent)
}

pub fn danger_button(label: impl Into<WidgetText>) -> SignalButton {
    SignalButton::new(label, ButtonTone::Danger)
}

pub fn selected_model_button(label: impl Into<WidgetText>) -> SignalButton {
    SignalButton::new(label, ButtonTone::Selected)
}

pub fn progress(ui: &mut Ui, fraction: f32, text: impl Into<WidgetText>) -> Response {
    ui.add(
        egui::ProgressBar::new(fraction.clamp(0.0, 1.0))
            .text(text)
            .fill(ACCENT)
            .corner_radius(CONTROL_RADIUS),
    )
}

#[derive(Clone, Copy, Debug)]
pub struct KnobSpec<'a> {
    pub label: &'a str,
    pub value_text: &'a str,
    pub minimum: f32,
    pub maximum: f32,
    pub default: f32,
    pub step: f32,
}

/// Signal Lab's parameter knob: an external value arc with an unobstructed
/// numeric value exactly in the geometric center.
pub fn parameter_knob(ui: &mut Ui, value: &mut f32, spec: KnobSpec<'_>) -> Response {
    let desired = Vec2::new(142.0, 150.0);
    let (rect, mut response) = ui.allocate_exact_size(desired, Sense::click_and_drag());
    let span = (spec.maximum - spec.minimum).max(f32::EPSILON);

    if response.dragged() {
        let delta = ui.input(|input| input.pointer.delta());
        let raw = *value + (delta.x - delta.y) * span / 240.0;
        let stepped = if spec.step > 0.0 {
            (raw / spec.step).round() * spec.step
        } else {
            raw
        };
        let next = stepped.clamp(spec.minimum, spec.maximum);
        if next.to_bits() != value.to_bits() {
            *value = next;
            response.mark_changed();
        }
    }
    if response.double_clicked() {
        let next = spec.default.clamp(spec.minimum, spec.maximum);
        if next.to_bits() != value.to_bits() {
            *value = next;
            response.mark_changed();
        }
    }

    let painter = ui.painter_at(rect);
    let center = egui::pos2(rect.center().x, rect.top() + 59.0);
    let radius = 48.0;
    painter.circle_filled(center, radius, RAISED);
    painter.circle_stroke(center, radius, Stroke::new(1.0_f32, LINE));

    let start = 135.0_f32.to_radians();
    let sweep = 270.0_f32.to_radians();
    let normalized = ((*value - spec.minimum) / span).clamp(0.0, 1.0);
    let background = arc_points(center, radius + 4.0, start, start + sweep, 48);
    painter.add(Shape::line(background, PathStroke::new(3.0_f32, LINE)));
    let active = arc_points(
        center,
        radius + 4.0,
        start,
        start + sweep * normalized,
        (48.0 * normalized).ceil().max(2.0) as usize,
    );
    painter.add(Shape::line(active, PathStroke::new(3.0_f32, ACCENT)));

    painter.text(
        center,
        Align2::CENTER_CENTER,
        spec.value_text,
        FontId::monospace(16.0),
        TEXT,
    );
    painter.text(
        egui::pos2(rect.center().x, center.y + radius + 22.0),
        Align2::CENTER_CENTER,
        spec.label,
        FontId::monospace(10.0),
        DIM,
    );

    response.on_hover_text("Drag to adjust · double-click to reset")
}

fn arc_points(
    center: egui::Pos2,
    radius: f32,
    start: f32,
    end: f32,
    segment_count: usize,
) -> Vec<egui::Pos2> {
    let segments = segment_count.max(2);
    (0..=segments)
        .map(|index| {
            let mix = index as f32 / segments as f32;
            let angle = start + (end - start) * mix;
            center + Vec2::new(angle.cos(), angle.sin()) * radius
        })
        .collect()
}

pub fn strobe_stripe_color(active: bool, alternate: bool) -> Color32 {
    match (active, alternate) {
        (true, true) => ACCENT,
        (true, false) => Color32::from_rgb(20, 70, 68),
        (false, true) => Color32::from_rgb(45, 65, 66),
        (false, false) => Color32::from_rgb(28, 36, 39),
    }
}

pub fn format_percent(value: f32) -> String {
    format!("{value:.0}%")
}

pub fn format_db(value: f32) -> String {
    format!("{value:+.1}")
}
