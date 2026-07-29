import AppKit

let canvasWidth: CGFloat = 1600
let canvasHeight: CGFloat = 1000

struct Theme {
    let background: NSColor
    let surface: NSColor
    let raised: NSColor
    let line: NSColor
    let text: NSColor
    let dim: NSColor
    let accent: NSColor
    let secondary: NSColor
    let warning: NSColor
    let error: NSColor
}

func rgb(_ value: UInt32, alpha: CGFloat = 1) -> NSColor {
    NSColor(
        calibratedRed: CGFloat((value >> 16) & 0xff) / 255,
        green: CGFloat((value >> 8) & 0xff) / 255,
        blue: CGFloat(value & 0xff) / 255,
        alpha: alpha
    )
}

func rect(_ x: CGFloat, _ y: CGFloat, _ width: CGFloat, _ height: CGFloat) -> NSRect {
    NSRect(x: x, y: canvasHeight - y - height, width: width, height: height)
}

func fillRect(
    _ x: CGFloat,
    _ y: CGFloat,
    _ width: CGFloat,
    _ height: CGFloat,
    _ color: NSColor,
    radius: CGFloat = 0
) {
    color.setFill()
    NSBezierPath(roundedRect: rect(x, y, width, height), xRadius: radius, yRadius: radius).fill()
}

func strokeRect(
    _ x: CGFloat,
    _ y: CGFloat,
    _ width: CGFloat,
    _ height: CGFloat,
    _ color: NSColor,
    radius: CGFloat = 0,
    lineWidth: CGFloat = 1
) {
    color.setStroke()
    let path = NSBezierPath(roundedRect: rect(x, y, width, height), xRadius: radius, yRadius: radius)
    path.lineWidth = lineWidth
    path.stroke()
}

func line(
    _ x1: CGFloat,
    _ y1: CGFloat,
    _ x2: CGFloat,
    _ y2: CGFloat,
    _ color: NSColor,
    width: CGFloat = 1
) {
    color.setStroke()
    let path = NSBezierPath()
    path.move(to: NSPoint(x: x1, y: canvasHeight - y1))
    path.line(to: NSPoint(x: x2, y: canvasHeight - y2))
    path.lineWidth = width
    path.stroke()
}

func text(
    _ value: String,
    _ x: CGFloat,
    _ y: CGFloat,
    _ width: CGFloat,
    _ height: CGFloat,
    size: CGFloat,
    color: NSColor,
    weight: NSFont.Weight = .regular,
    mono: Bool = false,
    align: NSTextAlignment = .left
) {
    let paragraph = NSMutableParagraphStyle()
    paragraph.alignment = align
    paragraph.lineBreakMode = .byTruncatingTail
    let font = mono
        ? NSFont.monospacedSystemFont(ofSize: size, weight: weight)
        : NSFont.systemFont(ofSize: size, weight: weight)
    (value as NSString).draw(
        in: rect(x, y, width, height),
        withAttributes: [
            .font: font,
            .foregroundColor: color,
            .paragraphStyle: paragraph,
        ]
    )
}

func circle(_ centerX: CGFloat, _ centerY: CGFloat, _ radius: CGFloat, fill: NSColor, stroke: NSColor?) {
    let circleRect = NSRect(
        x: centerX - radius,
        y: canvasHeight - centerY - radius,
        width: radius * 2,
        height: radius * 2
    )
    fill.setFill()
    NSBezierPath(ovalIn: circleRect).fill()
    if let stroke {
        stroke.setStroke()
        let path = NSBezierPath(ovalIn: circleRect)
        path.lineWidth = 1
        path.stroke()
    }
}

func button(
    _ label: String,
    _ x: CGFloat,
    _ y: CGFloat,
    _ width: CGFloat,
    _ height: CGFloat,
    fill: NSColor,
    stroke: NSColor?,
    textColor: NSColor,
    radius: CGFloat
) {
    fillRect(x, y, width, height, fill, radius: radius)
    if let stroke {
        strokeRect(x, y, width, height, stroke, radius: radius)
    }
    text(label, x, y + (height - 14) / 2 - 1, width, 18, size: 10, color: textColor, weight: .semibold, mono: true, align: .center)
}

func progress(
    _ x: CGFloat,
    _ y: CGFloat,
    _ width: CGFloat,
    _ fraction: CGFloat,
    background: NSColor,
    foreground: NSColor,
    height: CGFloat = 6,
    radius: CGFloat = 3
) {
    fillRect(x, y, width, height, background, radius: radius)
    fillRect(x, y, width * fraction, height, foreground, radius: radius)
}

func knob(
    _ centerX: CGFloat,
    _ centerY: CGFloat,
    _ radius: CGFloat,
    fraction: CGFloat,
    theme: Theme,
    value: String,
    label: String
) {
    circle(centerX, centerY, radius, fill: theme.surface, stroke: theme.line)
    circle(centerX, centerY, radius - 10, fill: theme.raised, stroke: nil)
    theme.accent.setStroke()
    let arc = NSBezierPath()
    arc.appendArc(
        withCenter: NSPoint(x: centerX, y: canvasHeight - centerY),
        radius: radius - 4,
        startAngle: 225,
        endAngle: 225 + 270 * fraction
    )
    arc.lineWidth = 3
    arc.stroke()
    text(value, centerX - radius, centerY - 10, radius * 2, 20, size: 15, color: theme.text, weight: .semibold, mono: true, align: .center)
    text(label, centerX - radius - 18, centerY + radius + 12, radius * 2 + 36, 18, size: 9, color: theme.dim, weight: .semibold, mono: true, align: .center)
}

func boardTitle(
    index: String,
    title: String,
    note: String,
    theme: Theme
) {
    text(index.uppercased(), 38, 27, 420, 18, size: 11, color: theme.accent, weight: .bold, mono: true)
    text(title, 38, 46, 520, 28, size: 22, color: theme.accent, weight: .medium)
    text(note, 880, 28, 682, 44, size: 12, color: theme.dim, align: .right)
}

func pluginFrame(
    _ x: CGFloat,
    _ y: CGFloat,
    _ width: CGFloat,
    _ height: CGFloat,
    title: String,
    meta: String,
    theme: Theme,
    radius: CGFloat,
    headerHeight: CGFloat,
    accentTitle: Bool = true,
    squareAccent: Bool = false
) {
    fillRect(x, y, width, height, theme.background.blended(withFraction: 0.35, of: theme.surface) ?? theme.background, radius: radius)
    strokeRect(x, y, width, height, theme.line, radius: radius)
    line(x, y + headerHeight, x + width, y + headerHeight, theme.line)
    if squareAccent {
        fillRect(x, y, 3, headerHeight, theme.accent)
    }
    text(title, x + 22, y + 22, 270, 29, size: headerHeight > 60 ? 24 : 17, color: accentTitle ? theme.accent : theme.text, weight: .medium)
    text(meta, x + (headerHeight > 60 ? 226 : 183), y + (headerHeight > 60 ? 27 : 23), width - 330, 20, size: headerHeight > 60 ? 11 : 8, color: theme.dim, weight: .medium, mono: true)
}

func drawStrobe(
    _ x: CGFloat,
    _ y: CGFloat,
    _ width: CGFloat,
    _ height: CGFloat,
    theme: Theme,
    disc: Bool
) {
    fillRect(x, y, width, height, theme.surface, radius: 7)
    strokeRect(x, y, width, height, theme.line, radius: 7)
    var stripeX = x + 8
    var alternate = false
    while stripeX < x + width - 8 {
        fillRect(
            stripeX,
            y + 8,
            18,
            height - 16,
            alternate ? theme.accent.withAlphaComponent(0.20) : theme.raised,
            radius: 2
        )
        stripeX += 18
        alternate.toggle()
    }
    line(x + width / 2, y + 5, x + width / 2, y + height - 5, theme.text.withAlphaComponent(0.5))
    if disc {
        circle(x + width / 2, y + height / 2, 43, fill: theme.text.withAlphaComponent(0.91), stroke: theme.text)
        text("Eb4", x + width / 2 - 43, y + height / 2 - 21, 86, 32, size: 26, color: theme.background, weight: .bold, mono: true, align: .center)
        text("+0.3 c", x + width / 2 - 43, y + height / 2 + 17, 86, 18, size: 10, color: theme.background, weight: .semibold, mono: true, align: .center)
    }
}

func renderSignalLab() {
    let theme = Theme(
        background: rgb(0x080b0d),
        surface: rgb(0x11171b),
        raised: rgb(0x172026),
        line: rgb(0x2a363d),
        text: rgb(0xe7eef0),
        dim: rgb(0x84939a),
        accent: rgb(0x39d5d0),
        secondary: rgb(0x43d69f),
        warning: rgb(0xf0b94c),
        error: rgb(0xf45b69)
    )
    theme.background.setFill()
    NSBezierPath(rect: rect(0, 0, canvasWidth, canvasHeight)).fill()
    boardTitle(
        index: "Concept 01 / Recommended",
        title: "SIGNAL LAB",
        note: "Precise studio instrument. Calm hierarchy, cyan accents and no hardware metaphors.",
        theme: theme
    )

    let mainX: CGFloat = 38, mainY: CGFloat = 88, mainW: CGFloat = 1080, mainH: CGFloat = 876
    pluginFrame(mainX, mainY, mainW, mainH, title: "MOT PLAYER", meta: "0.5.0  •  MONO  •  48 kHz  •  0 SAMPLES", theme: theme, radius: 14, headerHeight: 72)
    button("MUTE", mainX + mainW - 90, mainY + 20, 62, 32, fill: theme.raised, stroke: theme.line, textColor: theme.text, radius: 6)

    let bodyY = mainY + 92
    fillRect(mainX + 20, bodyY, 300, 752, theme.surface, radius: 10)
    strokeRect(mainX + 20, bodyY, 300, 752, theme.line, radius: 10)
    text("MODELS", mainX + 38, bodyY + 20, 180, 18, size: 10, color: theme.accent, weight: .bold, mono: true)
    text("04", mainX + 258, bodyY + 20, 42, 18, size: 10, color: theme.dim, mono: true, align: .right)
    let models = ["EVH 5153 Red", "Pasadena Tight", "Recto Modern", "Clean Edge"]
    for (index, model) in models.enumerated() {
        let rowY = bodyY + 58 + CGFloat(index) * 72
        fillRect(mainX + 36, rowY, 268, 60, index == 0 ? rgb(0x102527) : rgb(0x0d1317), radius: 6)
        if index == 0 { strokeRect(mainX + 36, rowY, 268, 60, theme.accent.withAlphaComponent(0.45), radius: 6) }
        text(model, mainX + 50, rowY + 11, 230, 20, size: 13, color: index == 0 ? theme.accent : theme.text, weight: .medium)
        text("1731 MAC/smp · .motmodel", mainX + 50, rowY + 36, 230, 14, size: 8, color: theme.dim, mono: true)
    }
    button("REFRESH", mainX + 36, bodyY + 694, 124, 34, fill: theme.raised, stroke: theme.line, textColor: theme.text, radius: 5)
    button("OPEN FOLDER", mainX + 172, bodyY + 694, 132, 34, fill: theme.raised, stroke: theme.line, textColor: theme.text, radius: 5)

    let rightX: CGFloat = mainX + 338
    let rightW: CGFloat = 722
    fillRect(rightX, bodyY, rightW, 154, theme.surface, radius: 10)
    strokeRect(rightX, bodyY, rightW, 154, theme.line, radius: 10)
    text("ACTIVE MODEL", rightX + 20, bodyY + 19, 200, 18, size: 9, color: theme.dim, weight: .bold, mono: true)
    text("EVH 5153 Red", rightX + 20, bodyY + 47, 430, 34, size: 25, color: theme.accent, weight: .medium)
    text("SAVED", rightX + rightW - 100, bodyY + 22, 78, 18, size: 9, color: theme.secondary, weight: .bold, mono: true, align: .right)
    line(rightX + 20, bodyY + 101, rightX + rightW - 20, bodyY + 101, theme.line)
    text("RUNTIME", rightX + 20, bodyY + 119, 90, 16, size: 9, color: theme.dim, weight: .bold, mono: true)
    text("READY", rightX + 104, bodyY + 118, 74, 18, size: 10, color: theme.secondary, weight: .bold, mono: true)
    text("A2-C3 · causal · IR loaded", rightX + 184, bodyY + 119, 260, 16, size: 9, color: theme.dim, mono: true)

    let toneY = bodyY + 172
    fillRect(rightX, toneY, rightW, 376, theme.surface, radius: 10)
    strokeRect(rightX, toneY, rightW, 376, theme.line, radius: 10)
    knob(rightX + 140, toneY + 163, 63, fraction: 0.55, theme: theme, value: "+1.8", label: "INPUT GAIN · dB")
    knob(rightX + 361, toneY + 163, 63, fraction: 0.64, theme: theme, value: "64%", label: "TIGHT")
    knob(rightX + 582, toneY + 163, 63, fraction: 0.42, theme: theme, value: "42%", label: "BITE")
    text("Three model-bound controls · exact values · 5 ms smoothing", rightX + 26, toneY + 328, rightW - 52, 18, size: 9, color: theme.dim, mono: true, align: .center)

    let irY = toneY + 394
    fillRect(rightX, irY, rightW, 186, theme.surface, radius: 10)
    strokeRect(rightX, irY, rightW, 186, theme.line, radius: 10)
    text("CABINET IR", rightX + 20, irY + 18, 118, 18, size: 9, color: theme.accent, weight: .bold, mono: true)
    text("York Audio — MRSH 412 M25", rightX + 142, irY + 16, 330, 21, size: 13, color: theme.text, weight: .medium)
    button("MINIMUM PHASE", rightX + 500, irY + 13, 122, 28, fill: theme.raised, stroke: theme.line, textColor: theme.text, radius: 5)
    button("CHANGE", rightX + 632, irY + 13, 70, 28, fill: theme.raised, stroke: theme.line, textColor: theme.text, radius: 5)
    let bars: [CGFloat] = [7, 19, 34, 58, 31, 22, 47, 26, 18, 41, 23, 14, 29, 15, 10, 20, 12, 8, 15, 7, 11, 5]
    for (index, bar) in bars.enumerated() {
        let bx = rightX + 24 + CGFloat(index) * 30
        fillRect(bx, irY + 104 - bar / 2, 3, bar, theme.accent.withAlphaComponent(0.72), radius: 1)
    }
    line(rightX + 20, irY + 104, rightX + rightW - 20, irY + 104, theme.line)

    drawSignalSide(theme: theme)
    saveCurrentImage("concept-01-signal-lab.png")
}

func drawSignalSide(theme: Theme) {
    let x: CGFloat = 1142, width: CGFloat = 420, height: CGFloat = 280
    let ys: [CGFloat] = [88, 386, 684]
    pluginFrame(x, ys[0], width, height, title: "MOT GENERATOR", meta: "0.5.0 · 48 kHz", theme: theme, radius: 12, headerHeight: 55)
    text("STATUS", x + 20, ys[0] + 90, 90, 18, size: 9, color: theme.dim, weight: .bold, mono: true)
    text("WAITING FOR PLAY", x + 20, ys[0] + 116, 220, 24, size: 15, color: theme.warning, weight: .bold, mono: true)
    text("Ready on the next transport edge", x + 20, ys[0] + 155, 250, 18, size: 9, color: theme.dim, mono: true)
    button("ARMED", x + 283, ys[0] + 109, 112, 43, fill: theme.accent, stroke: nil, textColor: theme.background, radius: 6)

    pluginFrame(x, ys[1], width, height, title: "MOT TRAINER", meta: "0.5.0 · 48 kHz", theme: theme, radius: 12, headerHeight: 55)
    button("MONITOR", x + 324, ys[1] + 15, 76, 26, fill: theme.raised, stroke: theme.line, textColor: theme.text, radius: 5)
    fillRect(x + 18, ys[1] + 72, 132, 182, theme.surface, radius: 6)
    strokeRect(x + 18, ys[1] + 72, 132, 182, theme.line, radius: 6)
    text("MODELS · 04", x + 28, ys[1] + 85, 112, 14, size: 7, color: theme.accent, weight: .bold, mono: true)
    let trainerModels = ["+ NEW MODEL", "EVH 5153 Red", "Pasadena Tight", "Recto Modern"]
    for (index, model) in trainerModels.enumerated() {
        let rowY = ys[1] + 108 + CGFloat(index) * 31
        fillRect(x + 27, rowY, 114, 25, index == 1 ? rgb(0x102527) : rgb(0x0d1317), radius: 3)
        if index == 1 {
            strokeRect(x + 27, rowY, 114, 25, theme.accent.withAlphaComponent(0.45), radius: 3)
        }
        text(model, x + 33, rowY + 7, 102, 12, size: 7, color: index == 1 ? theme.accent : theme.dim, weight: .medium)
    }
    text("STATUS", x + 170, ys[1] + 84, 82, 18, size: 8, color: theme.dim, weight: .bold, mono: true)
    text("TRAINING", x + 263, ys[1] + 83, 120, 18, size: 10, color: theme.accent, weight: .bold, mono: true)
    text("EVH 5153 Red", x + 170, ys[1] + 115, 210, 20, size: 12, color: theme.text, weight: .medium)
    progress(x + 170, ys[1] + 151, 230, 0.43, background: rgb(0x0a0f12), foreground: theme.accent)
    text("PASS 43 / 100", x + 170, ys[1] + 173, 116, 18, size: 8, color: theme.dim, mono: true)
    text("ESR 0.0148", x + 286, ys[1] + 173, 114, 18, size: 8, color: theme.dim, mono: true, align: .right)
    text("ELAPSED 11:24", x + 170, ys[1] + 207, 116, 18, size: 8, color: theme.dim, mono: true)
    text("ETA 15:06", x + 286, ys[1] + 207, 114, 18, size: 8, color: theme.dim, mono: true, align: .right)

    pluginFrame(x, ys[2], width, height, title: "MOT TUNER", meta: "0.5.0 · 48 kHz", theme: theme, radius: 12, headerHeight: 55)
    button("MUTE", x + 344, ys[2] + 15, 56, 26, fill: theme.raised, stroke: theme.line, textColor: theme.text, radius: 5)
    drawStrobe(x + 18, ys[2] + 72, 250, 168, theme: theme, disc: true)
    text("OFFSETS · ON", x + 286, ys[2] + 75, 114, 16, size: 8, color: theme.accent, weight: .bold, mono: true)
    let cells = [("7  B1", "+0.5"), ("6  E2", "-0.3"), ("5  A2", "+0.1"), ("1  Eb4", "+0.7")]
    for (index, cell) in cells.enumerated() {
        let cy = ys[2] + 103 + CGFloat(index) * 31
        fillRect(x + 284, cy, 116, 25, theme.surface, radius: 4)
        text(cell.0, x + 292, cy + 7, 58, 12, size: 8, color: theme.dim, mono: true)
        text(cell.1, x + 350, cy + 7, 40, 12, size: 8, color: theme.text, mono: true, align: .right)
    }
}

func renderSignalTrainerDetail() {
    let theme = Theme(
        background: rgb(0x080b0d),
        surface: rgb(0x11171b),
        raised: rgb(0x172026),
        line: rgb(0x2a363d),
        text: rgb(0xe7eef0),
        dim: rgb(0x84939a),
        accent: rgb(0x39d5d0),
        secondary: rgb(0x43d69f),
        warning: rgb(0xf0b94c),
        error: rgb(0xf45b69)
    )
    theme.background.setFill()
    NSBezierPath(rect: rect(0, 0, canvasWidth, canvasHeight)).fill()
    boardTitle(
        index: "Signal Lab / Trainer detail",
        title: "MODEL BROWSER WORKFLOW",
        note: "Logical editor size remains compact at 720 × 480. Existing model selection restores its capture metadata.",
        theme: theme
    )

    let x: CGFloat = 130
    let y: CGFloat = 100
    let width: CGFloat = 1340
    let height: CGFloat = 800
    pluginFrame(
        x,
        y,
        width,
        height,
        title: "MOT TRAINER",
        meta: "0.5.0  •  MONO  •  48 kHz",
        theme: theme,
        radius: 14,
        headerHeight: 72
    )
    button(
        "MONITOR",
        x + width - 118,
        y + 20,
        88,
        32,
        fill: theme.raised,
        stroke: theme.line,
        textColor: theme.dim,
        radius: 6
    )

    let bodyY = y + 92
    let browserX = x + 20
    let browserW: CGFloat = 326
    let bodyH: CGFloat = 688
    fillRect(browserX, bodyY, browserW, bodyH, theme.surface, radius: 10)
    strokeRect(browserX, bodyY, browserW, bodyH, theme.line, radius: 10)
    text("MODELS", browserX + 18, bodyY + 20, 180, 18, size: 10, color: theme.accent, weight: .bold, mono: true)
    text("04", browserX + 268, bodyY + 20, 38, 18, size: 10, color: theme.dim, mono: true, align: .right)
    button(
        "+ NEW MODEL",
        browserX + 18,
        bodyY + 55,
        browserW - 36,
        38,
        fill: theme.raised,
        stroke: theme.line,
        textColor: theme.text,
        radius: 5
    )
    let models = [
        ("EVH 5153 Red", "capture metadata available"),
        ("Pasadena Tight", "capture metadata available"),
        ("Recto Modern", "capture metadata available"),
        ("Clean Edge", "imported model"),
    ]
    for (index, model) in models.enumerated() {
        let rowY = bodyY + 111 + CGFloat(index) * 77
        fillRect(
            browserX + 18,
            rowY,
            browserW - 36,
            65,
            index == 0 ? rgb(0x102527) : rgb(0x0d1317),
            radius: 6
        )
        if index == 0 {
            strokeRect(
                browserX + 18,
                rowY,
                browserW - 36,
                65,
                theme.accent.withAlphaComponent(0.45),
                radius: 6
            )
        }
        text(
            model.0,
            browserX + 32,
            rowY + 12,
            browserW - 64,
            20,
            size: 13,
            color: index == 0 ? theme.accent : theme.text,
            weight: .medium
        )
        text(
            model.1,
            browserX + 32,
            rowY + 39,
            browserW - 64,
            15,
            size: 8,
            color: theme.dim,
            mono: true
        )
    }
    button(
        "REFRESH",
        browserX + 18,
        bodyY + bodyH - 52,
        138,
        34,
        fill: theme.raised,
        stroke: theme.line,
        textColor: theme.text,
        radius: 5
    )
    button(
        "OPEN FOLDER",
        browserX + 168,
        bodyY + bodyH - 52,
        140,
        34,
        fill: theme.raised,
        stroke: theme.line,
        textColor: theme.text,
        radius: 5
    )

    let mainX = browserX + browserW + 18
    let mainW = width - (mainX - x) - 20
    fillRect(mainX, bodyY, mainW, 228, theme.surface, radius: 10)
    strokeRect(mainX, bodyY, mainW, 228, theme.line, radius: 10)
    text("SELECTED MODEL", mainX + 20, bodyY + 20, 190, 18, size: 9, color: theme.dim, weight: .bold, mono: true)
    text("EVH 5153 Red", mainX + 20, bodyY + 49, 440, 34, size: 25, color: theme.accent, weight: .medium)
    fillRect(mainX + mainW - 138, bodyY + 22, 116, 28, theme.accent.withAlphaComponent(0.08), radius: 4)
    strokeRect(mainX + mainW - 138, bodyY + 22, 116, 28, theme.accent.withAlphaComponent(0.35), radius: 4)
    text("RETRAIN MODE", mainX + mainW - 138, bodyY + 31, 116, 13, size: 8, color: theme.accent, weight: .bold, mono: true, align: .center)
    text(
        "The selected capture metadata is loaded. Training publishes a new immutable model.",
        mainX + 20,
        bodyY + 91,
        mainW - 40,
        20,
        size: 9,
        color: theme.dim,
        mono: true
    )
    line(mainX + 20, bodyY + 125, mainX + mainW - 20, bodyY + 125, theme.line)
    text("MAX PASSES", mainX + 20, bodyY + 151, 150, 18, size: 9, color: theme.dim, weight: .bold, mono: true)
    fillRect(mainX + 167, bodyY + 143, 100, 34, theme.raised, radius: 5)
    text("100", mainX + 167, bodyY + 151, 100, 18, size: 12, color: theme.text, weight: .semibold, mono: true, align: .center)
    text("CAPTURE METADATA", mainX + 20, bodyY + 196, 200, 18, size: 9, color: theme.text, weight: .bold, mono: true)
    text("AMPLIFIER · CHANNEL · CONTROLS · ROUTING", mainX + 218, bodyY + 196, 390, 18, size: 8, color: theme.dim, mono: true)
    text("›", mainX + mainW - 39, bodyY + 192, 18, 22, size: 17, color: theme.dim, weight: .medium, align: .center)

    let statusY = bodyY + 246
    let statusH = bodyH - 246
    fillRect(mainX, statusY, mainW, statusH, theme.surface, radius: 10)
    strokeRect(mainX, statusY, mainW, statusH, theme.line, radius: 10)
    text("STATUS", mainX + 22, statusY + 24, 100, 18, size: 10, color: theme.dim, weight: .bold, mono: true)
    text("TRAINING", mainX + 127, statusY + 23, 180, 18, size: 12, color: theme.accent, weight: .bold, mono: true)
    progress(
        mainX + 22,
        statusY + 70,
        mainW - 44,
        0.43,
        background: rgb(0x0a0f12),
        foreground: theme.accent,
        height: 10,
        radius: 5
    )
    text("TRAINING PASS 43 / 100", mainX + 22, statusY + 98, 240, 18, size: 10, color: theme.text, weight: .semibold, mono: true)
    text("BEST ESR 0.0148", mainX + mainW - 252, statusY + 98, 230, 18, size: 10, color: theme.text, weight: .semibold, mono: true, align: .right)
    line(mainX + 22, statusY + 139, mainX + mainW - 22, statusY + 139, theme.line)
    text("ELAPSED", mainX + 22, statusY + 166, 90, 16, size: 9, color: theme.dim, weight: .bold, mono: true)
    text("11:24", mainX + 111, statusY + 165, 80, 18, size: 11, color: theme.text, mono: true)
    text("PASS", mainX + 235, statusY + 166, 58, 16, size: 9, color: theme.dim, weight: .bold, mono: true)
    text("00:16", mainX + 294, statusY + 165, 80, 18, size: 11, color: theme.text, mono: true)
    text("ETA", mainX + 421, statusY + 166, 48, 16, size: 9, color: theme.dim, weight: .bold, mono: true)
    text("15:06", mainX + 469, statusY + 165, 80, 18, size: 11, color: theme.text, mono: true)
    button(
        "CANCEL TRAINING",
        mainX + 22,
        statusY + statusH - 61,
        174,
        39,
        fill: theme.error.withAlphaComponent(0.76),
        stroke: nil,
        textColor: theme.text,
        radius: 6
    )
    text(
        "MONITOR unlocks only while a return signal is available.",
        mainX + 220,
        statusY + statusH - 48,
        mainW - 242,
        18,
        size: 8,
        color: theme.dim,
        mono: true
    )
    saveCurrentImage("concept-01b-signal-lab-trainer.png")
}

func renderEditorial() {
    let theme = Theme(
        background: rgb(0x090a0a),
        surface: rgb(0x151614),
        raised: rgb(0x1b1c19),
        line: rgb(0x302f2a),
        text: rgb(0xeeeae1),
        dim: rgb(0x96938b),
        accent: rgb(0xd7a85b),
        secondary: rgb(0x67d1af),
        warning: rgb(0xe7b75c),
        error: rgb(0xe26862)
    )
    theme.background.setFill()
    NSBezierPath(rect: rect(0, 0, canvasWidth, canvasHeight)).fill()
    text("MOT / 02", 1060, 808, 520, 190, size: 142, color: theme.accent.withAlphaComponent(0.035), weight: .bold, mono: true, align: .right)
    boardTitle(
        index: "Concept 02 / Boutique",
        title: "OBSIDIAN EDITORIAL",
        note: "Boutique software: warm typography, one brass keyline and editorial composition.",
        theme: theme
    )
    let x: CGFloat = 38, y: CGFloat = 88, width: CGFloat = 1080, height: CGFloat = 876
    pluginFrame(x, y, width, height, title: "MOT PLAYER", meta: "0.5.0  •  MONO  •  48 kHz  •  ZERO LATENCY", theme: theme, radius: 3, headerHeight: 72, accentTitle: false, squareAccent: true)
    button("MUTE", x + width - 90, y + 20, 62, 32, fill: theme.background, stroke: theme.line, textColor: theme.dim, radius: 2)
    let bodyX = x + 31, bodyY = y + 72, bodyW = width - 62
    line(bodyX, bodyY + 235, bodyX + bodyW, bodyY + 235, theme.line)
    text("01 / ACTIVE MODEL", bodyX, bodyY + 40, 240, 18, size: 10, color: theme.accent, weight: .bold, mono: true)
    text("EVH 5153", bodyX, bodyY + 74, 520, 70, size: 56, color: theme.text, weight: .light)
    text("Red", bodyX, bodyY + 132, 520, 70, size: 56, color: theme.text, weight: .light)
    text("A2-C3 · causal · saved tone · runtime ready", bodyX, bodyY + 204, 520, 18, size: 10, color: theme.dim, mono: true)
    let browserX = bodyX + 747
    line(browserX - 25, bodyY + 28, browserX - 25, bodyY + 210, theme.line)
    text("LIBRARY / 04", browserX, bodyY + 31, 230, 18, size: 9, color: theme.dim, weight: .bold, mono: true)
    let models = ["EVH 5153 Red", "Pasadena Tight", "Recto Modern", "Clean Edge"]
    for (index, model) in models.enumerated() {
        let rowY = bodyY + 63 + CGFloat(index) * 31
        text(model, browserX, rowY, 210, 18, size: 11, color: index == 0 ? theme.accent : theme.dim, weight: index == 0 ? .medium : .regular)
        line(browserX, rowY + 22, browserX + 230, rowY + 22, theme.line.withAlphaComponent(0.7))
    }

    let railsY = bodyY + 292
    let labels = [("INPUT GAIN", "+1.8 dB", CGFloat(0.56)), ("TIGHT", "64%", CGFloat(0.64)), ("BITE", "42%", CGFloat(0.42))]
    for (index, item) in labels.enumerated() {
        let rx = bodyX + CGFloat(index) * 342
        text(item.0, rx, railsY, 145, 18, size: 9, color: theme.dim, weight: .bold, mono: true)
        text(item.1, rx + 150, railsY - 8, 165, 28, size: 22, color: theme.text, weight: .medium, mono: true, align: .right)
        line(rx, railsY + 48, rx + 312, railsY + 48, theme.line, width: 2)
        line(rx, railsY + 48, rx + 312 * item.2, railsY + 48, theme.accent, width: 2)
        fillRect(rx + 312 * item.2 - 1, railsY + 41, 2, 14, theme.text)
    }
    line(bodyX, bodyY + 650, bodyX + bodyW, bodyY + 650, theme.line)
    text("02 / CABINET", bodyX, bodyY + 683, 160, 18, size: 10, color: theme.accent, weight: .bold, mono: true)
    text("MRSH 412 M25", bodyX, bodyY + 708, 210, 20, size: 13, color: theme.text, weight: .medium)
    let heights: [CGFloat] = [6, 12, 25, 49, 28, 17, 37, 20, 14, 31, 19, 11, 22, 12, 8, 16, 10, 7]
    for (index, h) in heights.enumerated() {
        fillRect(bodyX + 247 + CGFloat(index) * 26, bodyY + 715 - h / 2, 2, h, theme.accent.withAlphaComponent(0.7))
    }
    button("MINIMUM PHASE", bodyX + 767, bodyY + 690, 122, 30, fill: theme.background, stroke: theme.line, textColor: theme.dim, radius: 2)
    button("CHANGE IR", bodyX + 898, bodyY + 690, 100, 30, fill: theme.background, stroke: theme.line, textColor: theme.dim, radius: 2)
    drawEditorialSide(theme: theme)
    saveCurrentImage("concept-02-obsidian-editorial.png")
}

func drawEditorialSide(theme: Theme) {
    let x: CGFloat = 1142, width: CGFloat = 420, height: CGFloat = 280
    let ys: [CGFloat] = [88, 386, 684]
    pluginFrame(x, ys[0], width, height, title: "MOT GENERATOR", meta: "0.5.0 · 48 kHz", theme: theme, radius: 3, headerHeight: 55, accentTitle: false, squareAccent: true)
    text("01 / STATUS", x + 22, ys[0] + 84, 140, 18, size: 9, color: theme.accent, weight: .bold, mono: true)
    text("Waiting", x + 22, ys[0] + 112, 210, 32, size: 25, color: theme.accent, weight: .light)
    text("for Play", x + 22, ys[0] + 142, 210, 32, size: 25, color: theme.accent, weight: .light)
    text("Brass line becomes capture progress", x + 22, ys[0] + 196, 250, 18, size: 8, color: theme.dim, mono: true)
    button("ARMED", x + 284, ys[0] + 122, 112, 44, fill: theme.accent, stroke: nil, textColor: theme.background, radius: 2)

    pluginFrame(x, ys[1], width, height, title: "MOT TRAINER", meta: "0.5.0 · 48 kHz", theme: theme, radius: 3, headerHeight: 55, accentTitle: false, squareAccent: true)
    button("MONITOR", x + 324, ys[1] + 15, 76, 26, fill: theme.background, stroke: theme.line, textColor: theme.dim, radius: 2)
    text("01 · MODEL", x + 20, ys[1] + 86, 110, 16, size: 8, color: theme.dim, weight: .bold, mono: true)
    text("02 · CAPTURE", x + 20, ys[1] + 136, 110, 16, size: 8, color: theme.dim, weight: .bold, mono: true)
    text("03 · TRAIN", x + 20, ys[1] + 186, 110, 16, size: 8, color: theme.accent, weight: .bold, mono: true)
    line(x + 135, ys[1] + 77, x + 135, ys[1] + 224, theme.line)
    text("43 / 100", x + 157, ys[1] + 112, 220, 34, size: 27, color: theme.text, weight: .light, mono: true)
    progress(x + 157, ys[1] + 164, 223, 0.43, background: theme.line, foreground: theme.accent, height: 2, radius: 0)
    text("BEST ESR 0.0148  •  ETA 15:06", x + 157, ys[1] + 187, 230, 18, size: 8, color: theme.dim, mono: true)

    pluginFrame(x, ys[2], width, height, title: "MOT TUNER", meta: "0.5.0 · 48 kHz", theme: theme, radius: 3, headerHeight: 55, accentTitle: false, squareAccent: true)
    button("MUTE", x + 344, ys[2] + 15, 56, 26, fill: theme.background, stroke: theme.line, textColor: theme.dim, radius: 2)
    for index in 0..<14 {
        line(x + 16 + CGFloat(index) * 29, ys[2] + 77, x + 16 + CGFloat(index) * 29, ys[2] + 245, theme.accent.withAlphaComponent(0.08))
    }
    line(x + width / 2, ys[2] + 70, x + width / 2, ys[2] + 249, theme.accent)
    text("Eb4", x + 112, ys[2] + 108, 196, 70, size: 62, color: theme.text, weight: .light, mono: true, align: .center)
    text("+0.3 CENT", x + 112, ys[2] + 190, 196, 20, size: 11, color: theme.secondary, weight: .bold, mono: true, align: .center)
}

func renderTelemetry() {
    let theme = Theme(
        background: rgb(0x080b14),
        surface: rgb(0x101623),
        raised: rgb(0x171f2e),
        line: rgb(0x273247),
        text: rgb(0xeef3fa),
        dim: rgb(0x718097),
        accent: rgb(0xb9ef5b),
        secondary: rgb(0x5cb8ff),
        warning: rgb(0xffbf55),
        error: rgb(0xff647c)
    )
    theme.background.setFill()
    NSBezierPath(rect: rect(0, 0, canvasWidth, canvasHeight)).fill()
    for gridX in stride(from: CGFloat(0), through: canvasWidth, by: 32) {
        line(gridX, 0, gridX, canvasHeight, theme.line.withAlphaComponent(0.10))
    }
    for gridY in stride(from: CGFloat(0), through: canvasHeight, by: 32) {
        line(0, gridY, canvasWidth, gridY, theme.line.withAlphaComponent(0.10))
    }
    boardTitle(
        index: "Concept 03 / Digital",
        title: "TELEMETRY GRID",
        note: "A modular real-time control surface. Technical and scalable, with segmented state cells and one active color per screen.",
        theme: theme
    )
    let x: CGFloat = 38, y: CGFloat = 88, width: CGFloat = 1080, height: CGFloat = 876
    pluginFrame(x, y, width, height, title: "MOT PLAYER", meta: "0.5.0 / MONO / 48 kHz / 0 SAMPLES", theme: theme, radius: 8, headerHeight: 72, accentTitle: false)
    fillRect(x + 22, y + 30, 7, 7, theme.accent, radius: 2)
    button("MUTE", x + width - 90, y + 20, 62, 32, fill: theme.raised, stroke: theme.line, textColor: theme.text, radius: 5)
    let bodyY = y + 90
    fillRect(x + 18, bodyY, 292, 756, theme.surface, radius: 6)
    strokeRect(x + 18, bodyY, 292, 756, theme.line, radius: 6)
    text("MODEL LIBRARY", x + 35, bodyY + 18, 180, 18, size: 9, color: theme.dim, weight: .bold, mono: true)
    text("04 ONLINE", x + 213, bodyY + 18, 72, 18, size: 8, color: theme.dim, mono: true, align: .right)
    let models = ["EVH 5153 Red", "Pasadena Tight", "Recto Modern", "Clean Edge"]
    for (index, model) in models.enumerated() {
        let rowY = bodyY + 55 + CGFloat(index) * 68
        fillRect(x + 34, rowY, 260, 57, index == 0 ? theme.accent.withAlphaComponent(0.055) : rgb(0x0c121e), radius: 4)
        strokeRect(x + 34, rowY, 260, 57, index == 0 ? theme.accent.withAlphaComponent(0.55) : theme.line, radius: 4)
        text(model, x + 46, rowY + 11, 230, 18, size: 12, color: index == 0 ? theme.accent : theme.text, weight: .medium)
        text("1731 MAC/SMP", x + 46, rowY + 36, 180, 14, size: 8, color: theme.dim, mono: true)
    }
    button("REFRESH", x + 34, bodyY + 701, 124, 32, fill: theme.raised, stroke: theme.line, textColor: theme.text, radius: 4)
    button("FOLDER", x + 170, bodyY + 701, 124, 32, fill: theme.raised, stroke: theme.line, textColor: theme.text, radius: 4)

    let rightX: CGFloat = x + 324
    let rightW: CGFloat = 738
    let tileY = bodyY
    fillRect(rightX, tileY, 270, 478, theme.surface, radius: 6)
    strokeRect(rightX, tileY, 270, 478, theme.line, radius: 6)
    text("ACTIVE MODEL", rightX + 18, tileY + 18, 180, 18, size: 9, color: theme.dim, weight: .bold, mono: true)
    text("EVH 5153", rightX + 18, tileY + 146, 238, 34, size: 28, color: theme.text, weight: .semibold)
    text("Red", rightX + 18, tileY + 179, 238, 34, size: 28, color: theme.text, weight: .semibold)
    fillRect(rightX + 18, tileY + 418, 118, 28, theme.accent.withAlphaComponent(0.08), radius: 4)
    strokeRect(rightX + 18, tileY + 418, 118, 28, theme.accent.withAlphaComponent(0.35), radius: 4)
    text("RUNTIME READY", rightX + 18, tileY + 427, 118, 13, size: 8, color: theme.accent, weight: .bold, mono: true, align: .center)
    let paramX = rightX + 284
    let paramWidth: CGFloat = 142
    let params = [("INPUT GAIN", "+1.8", CGFloat(0.60)), ("TIGHT", "64", CGFloat(0.64)), ("BITE", "42", CGFloat(0.42))]
    for (index, param) in params.enumerated() {
        let px = paramX + CGFloat(index) * 154
        fillRect(px, tileY, paramWidth, 478, theme.surface, radius: 6)
        strokeRect(px, tileY, paramWidth, 478, theme.line, radius: 6)
        text(param.0, px + 14, tileY + 18, paramWidth - 28, 34, size: 8, color: theme.dim, weight: .bold, mono: true)
        text(param.1, px + 14, tileY + 174, paramWidth - 28, 42, size: 27, color: theme.text, weight: .bold, mono: true, align: .center)
        for segment in 0..<10 {
            let sy = tileY + 300 + CGFloat(9 - segment) * 12
            fillRect(px + 26, sy, paramWidth - 52, 7, CGFloat(segment) / 10 < param.2 ? theme.accent : theme.line, radius: 2)
        }
    }
    let lowerY = tileY + 492
    fillRect(rightX, lowerY, 500, 264, theme.surface, radius: 6)
    strokeRect(rightX, lowerY, 500, 264, theme.line, radius: 6)
    text("CABINET IR", rightX + 17, lowerY + 17, 120, 18, size: 9, color: theme.dim, weight: .bold, mono: true)
    text("York Audio — MRSH 412 M25", rightX + 139, lowerY + 16, 270, 20, size: 11, color: theme.text)
    button("IMPORT", rightX + 418, lowerY + 12, 65, 28, fill: theme.raised, stroke: theme.line, textColor: theme.text, radius: 4)
    let wave: [CGFloat] = [7, 15, 33, 62, 35, 22, 48, 28, 18, 39, 24, 15, 27, 16, 11]
    for (index, h) in wave.enumerated() {
        fillRect(rightX + 24 + CGFloat(index) * 28, lowerY + 130 - h / 2, 3, h, theme.secondary.withAlphaComponent(0.72), radius: 1)
    }
    button("MINIMUM PHASE + AUTO-TRIM", rightX + 17, lowerY + 210, 187, 29, fill: theme.secondary.withAlphaComponent(0.08), stroke: theme.secondary.withAlphaComponent(0.45), textColor: theme.secondary, radius: 3)
    button("RAW", rightX + 213, lowerY + 210, 54, 29, fill: theme.raised, stroke: theme.line, textColor: theme.dim, radius: 3)
    button("CHANGE IR", rightX + 276, lowerY + 210, 92, 29, fill: theme.raised, stroke: theme.line, textColor: theme.dim, radius: 3)

    fillRect(rightX + 514, lowerY, rightW - 514, 264, theme.surface, radius: 6)
    strokeRect(rightX + 514, lowerY, rightW - 514, 264, theme.line, radius: 6)
    text("OUTPUT TELEMETRY", rightX + 531, lowerY + 17, 180, 18, size: 9, color: theme.dim, weight: .bold, mono: true)
    let meterHeights: [CGFloat] = [37, 61, 101, 147, 117, 78, 130, 92, 64, 107, 48, 25]
    for (index, h) in meterHeights.enumerated() {
        fillRect(rightX + 531 + CGFloat(index) * 15, lowerY + 221 - h, 9, h, index < 7 ? theme.accent : theme.line, radius: 2)
    }
    drawTelemetrySide(theme: theme)
    saveCurrentImage("concept-03-telemetry-grid.png")
}

func drawTelemetrySide(theme: Theme) {
    let x: CGFloat = 1142, width: CGFloat = 420, height: CGFloat = 280
    let ys: [CGFloat] = [88, 386, 684]
    pluginFrame(x, ys[0], width, height, title: "MOT GENERATOR", meta: "0.5.0 / 48 kHz", theme: theme, radius: 8, headerHeight: 55, accentTitle: false)
    fillRect(x + 18, ys[0] + 25, 6, 6, theme.accent, radius: 2)
    for index in 0..<4 {
        fillRect(x + 20, ys[0] + 82 + CGFloat(index) * 35, 48, 26, index == 1 ? theme.warning : theme.line, radius: 2)
    }
    text("STATE 02 / 04", x + 87, ys[0] + 90, 180, 18, size: 8, color: theme.dim, weight: .bold, mono: true)
    text("WAITING FOR PLAY", x + 87, ys[0] + 120, 205, 22, size: 14, color: theme.warning, weight: .bold, mono: true)
    text("NEXT TRANSPORT EDGE", x + 87, ys[0] + 159, 185, 18, size: 8, color: theme.dim, mono: true)
    button("ARMED", x + 295, ys[0] + 114, 105, 42, fill: theme.accent, stroke: nil, textColor: theme.background, radius: 5)

    pluginFrame(x, ys[1], width, height, title: "MOT TRAINER", meta: "0.5.0 / 48 kHz", theme: theme, radius: 8, headerHeight: 55, accentTitle: false)
    fillRect(x + 18, ys[1] + 25, 6, 6, theme.accent, radius: 2)
    button("MONITOR", x + 324, ys[1] + 15, 76, 26, fill: theme.raised, stroke: theme.line, textColor: theme.text, radius: 4)
    text("TRAINING / EVH 5153 RED", x + 20, ys[1] + 82, 250, 18, size: 8, color: theme.dim, weight: .bold, mono: true)
    text("43 / 100", x + 20, ys[1] + 112, 200, 30, size: 21, color: theme.text, weight: .bold, mono: true)
    for segment in 0..<10 {
        fillRect(x + 20 + CGFloat(segment) * 22, ys[1] + 156, 17, 13, segment < 4 ? theme.accent : theme.line, radius: 2)
    }
    text("ESR 0.0148 · ETA 15:06", x + 20, ys[1] + 186, 220, 18, size: 8, color: theme.dim, mono: true)
    let stages = ["01 / CAPTURE", "02 / ALIGN", "03 / TRAIN"]
    for (index, stage) in stages.enumerated() {
        let sy = ys[1] + 79 + CGFloat(index) * 48
        fillRect(x + 265, sy, 135, 38, theme.surface, radius: 3)
        strokeRect(x + 265, sy, 135, 38, index == 2 ? theme.accent.withAlphaComponent(0.45) : theme.line, radius: 3)
        text(stage, x + 273, sy + 13, 119, 14, size: 8, color: index == 2 ? theme.accent : theme.dim, weight: .bold, mono: true)
    }

    pluginFrame(x, ys[2], width, height, title: "MOT TUNER", meta: "0.5.0 / 48 kHz", theme: theme, radius: 8, headerHeight: 55, accentTitle: false)
    fillRect(x + 18, ys[2] + 25, 6, 6, theme.accent, radius: 2)
    button("MUTE", x + 344, ys[2] + 15, 56, 26, fill: theme.raised, stroke: theme.line, textColor: theme.text, radius: 4)
    drawStrobe(x + 18, ys[2] + 72, 250, 168, theme: theme, disc: true)
    let cells = ["7\nB1", "6\nE2", "5\nA2", "4\nD3", "3\nG3", "2\nB3", "1\nEb4", "OFFSET\nON"]
    for (index, cell) in cells.enumerated() {
        let column = index % 2
        let row = index / 2
        let cx = x + 282 + CGFloat(column) * 60
        let cy = ys[2] + 72 + CGFloat(row) * 43
        fillRect(cx, cy, 54, 37, theme.surface, radius: 3)
        strokeRect(cx, cy, 54, 37, index == 6 ? theme.accent.withAlphaComponent(0.55) : theme.line, radius: 3)
        text(cell.replacingOccurrences(of: "\n", with: " · "), cx + 4, cy + 12, 46, 14, size: 7, color: index == 6 ? theme.accent : theme.dim, weight: .bold, mono: true, align: .center)
    }
}

func saveCurrentImage(_ filename: String) {
    guard
        let image = NSGraphicsContext.current?.cgContext.makeImage(),
        let destination = CGImageDestinationCreateWithURL(
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("design/ui-concepts")
                .appendingPathComponent(filename) as CFURL,
            "public.png" as CFString,
            1,
            nil
        )
    else {
        fatalError("Unable to create image destination")
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
        fatalError("Unable to write \(filename)")
    }
}

func render(_ body: () -> Void) {
    let bitmap = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: Int(canvasWidth),
        pixelsHigh: Int(canvasHeight),
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    )!
    let context = NSGraphicsContext(bitmapImageRep: bitmap)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = context
    body()
    NSGraphicsContext.restoreGraphicsState()
}

render(renderSignalLab)
render(renderSignalTrainerDetail)
render(renderEditorial)
render(renderTelemetry)
