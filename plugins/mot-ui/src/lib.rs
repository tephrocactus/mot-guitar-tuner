//! Minimal visual language shared by the MOT TUNER editor.
//!
//! This crate is UI-only and is never called from the audio callback.

use egui::{
    Align, Button, Color32, CursorIcon, FontId, Frame, Layout, Response, RichText, Stroke, Ui,
    Vec2, Widget, WidgetText,
};

pub const BACKGROUND: Color32 = Color32::from_rgb(8, 11, 13);
pub const SURFACE: Color32 = Color32::from_rgb(17, 23, 27);
pub const RAISED: Color32 = Color32::from_rgb(23, 32, 38);
pub const DEEP: Color32 = Color32::from_rgb(10, 15, 18);
pub const LINE: Color32 = Color32::from_rgb(42, 54, 61);
pub const TEXT: Color32 = Color32::from_rgb(231, 238, 240);
pub const DIM: Color32 = Color32::from_rgb(132, 147, 154);
pub const ACCENT: Color32 = Color32::from_rgb(57, 213, 208);
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ButtonTone {
    Neutral,
    Danger,
}

pub struct SignalButton {
    label: WidgetText,
    tone: ButtonTone,
}

impl SignalButton {
    fn new(label: impl Into<WidgetText>, tone: ButtonTone) -> Self {
        Self {
            label: label.into(),
            tone,
        }
    }
}

impl Widget for SignalButton {
    fn ui(self, ui: &mut Ui) -> Response {
        ui.scope(|ui| {
            configure_button_tone(ui, self.tone);
            ui.add(Button::new(self.label).corner_radius(CONTROL_RADIUS))
        })
        .inner
    }
}

fn configure_button_tone(ui: &mut Ui, tone: ButtonTone) {
    let visuals = ui.visuals_mut();
    let (idle, hovered, active, idle_stroke, text_color) = match tone {
        ButtonTone::Neutral => (
            RAISED,
            mix_color(RAISED, ACCENT, 0.16),
            mix_color(RAISED, ACCENT, 0.32),
            LINE,
            TEXT,
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

pub fn danger_button(label: impl Into<WidgetText>) -> SignalButton {
    SignalButton::new(label, ButtonTone::Danger)
}

pub fn strobe_stripe_color(active: bool, alternate: bool) -> Color32 {
    match (active, alternate) {
        (true, true) => ACCENT,
        (true, false) => Color32::from_rgb(20, 70, 68),
        (false, true) => Color32::from_rgb(45, 65, 66),
        (false, false) => Color32::from_rgb(28, 36, 39),
    }
}
