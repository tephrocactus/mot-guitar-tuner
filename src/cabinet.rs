use std::fmt;

use truce::prelude::AudioConfig;

pub const MAX_IR_SAMPLES: usize = 8192;
pub const DIRECT_HEAD_SAMPLES: usize = 64;
pub const CABINET_SAMPLE_RATE_HZ: u32 = 48_000;
const EARLY_BLOCK_SAMPLES: usize = 64;
const MID_BLOCK_SAMPLES: usize = 256;
const LATE_BLOCK_SAMPLES: usize = 1024;
const EARLY_FFT_SIZE: usize = EARLY_BLOCK_SAMPLES * 2;
const MID_FFT_SIZE: usize = MID_BLOCK_SAMPLES * 2;
const LATE_FFT_SIZE: usize = LATE_BLOCK_SAMPLES * 2;
const MID_TAIL_START: usize = MID_BLOCK_SAMPLES;
const LATE_TAIL_START: usize = LATE_BLOCK_SAMPLES;
const DEFAULT_TRIM_THRESHOLD_DB: f32 = -80.0;
const MAGNITUDE_FLOOR: f32 = 1.0e-12;
const DENORMAL_LIMIT: f32 = 1.0e-20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CabinetIrMode {
    /// Auto-trim followed by a real-cepstrum minimum-phase transform.
    MinimumPhase,
    /// Preserve the IR's phase. Auto-trim remains independently selectable.
    Raw,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CabinetIrImportOptions {
    pub mode: CabinetIrMode,
    pub trim_leading_silence: bool,
    /// Relative to the IR's absolute peak. Valid values are clamped to
    /// `-160..=0 dB`.
    pub trim_threshold_db: f32,
}

impl Default for CabinetIrImportOptions {
    fn default() -> Self {
        Self {
            mode: CabinetIrMode::MinimumPhase,
            trim_leading_silence: true,
            trim_threshold_db: DEFAULT_TRIM_THRESHOLD_DB,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CabinetIrError {
    Empty,
    TooLong(usize),
    Silent,
    NonFiniteSample,
    UnsupportedSampleRate(u32),
}

impl fmt::Display for CabinetIrError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("cabinet IR is empty"),
            Self::TooLong(length) => write!(
                formatter,
                "cabinet IR is {length} samples; maximum is {MAX_IR_SAMPLES}"
            ),
            Self::Silent => formatter.write_str("cabinet IR contains no usable signal"),
            Self::NonFiniteSample => formatter.write_str("cabinet IR contains a non-finite sample"),
            Self::UnsupportedSampleRate(rate) => {
                write!(formatter, "unsupported cabinet IR sample rate {rate} Hz")
            }
        }
    }
}

impl std::error::Error for CabinetIrError {}

#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f32,
    im: f32,
}

impl Complex {
    const ZERO: Self = Self { re: 0.0, im: 0.0 };

    #[inline]
    fn from_polar(radius: f32, phase: f32) -> Self {
        let (sin, cos) = phase.sin_cos();
        Self {
            re: radius * cos,
            im: radius * sin,
        }
    }

    #[inline]
    fn conjugate(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline]
    fn multiply(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl std::ops::AddAssign for Complex {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.re += other.re;
        self.im += other.im;
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

#[derive(Clone)]
struct FixedTwiddles<const FFT_LEN: usize> {
    forward: Vec<Complex>,
}

impl<const FFT_LEN: usize> Default for FixedTwiddles<FFT_LEN> {
    fn default() -> Self {
        Self {
            forward: (0..FFT_LEN / 2)
                .map(|index| {
                    Complex::from_polar(1.0, -std::f32::consts::TAU * index as f32 / FFT_LEN as f32)
                })
                .collect(),
        }
    }
}

/// Fully prepared immutable IR data.
///
/// `prepare()` performs trimming, optional minimum-phase conversion, and all
/// tail FFT work. It is intentionally not a real-time operation. Move the
/// result into a new [`CabinetProcessor`] on the loader thread, then swap the
/// complete runtime at a host block boundary.
#[derive(Clone)]
pub struct PreparedCabinetIr {
    sample_rate_hz: u32,
    mode: CabinetIrMode,
    original_len: usize,
    processed_len: usize,
    trimmed_leading_samples: usize,
    intrinsic_delay_samples: usize,
    direct_head: [f32; DIRECT_HEAD_SAMPLES],
    early_partitions: Vec<[Complex; EARLY_FFT_SIZE]>,
    mid_partitions: Vec<[Complex; MID_FFT_SIZE]>,
    late_partitions: Vec<[Complex; LATE_FFT_SIZE]>,
}

impl fmt::Debug for PreparedCabinetIr {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedCabinetIr")
            .field("sample_rate_hz", &self.sample_rate_hz)
            .field("mode", &self.mode)
            .field("original_len", &self.original_len)
            .field("processed_len", &self.processed_len)
            .field("trimmed_leading_samples", &self.trimmed_leading_samples)
            .field("intrinsic_delay_samples", &self.intrinsic_delay_samples)
            .field("early_partition_count", &self.early_partitions.len())
            .field("mid_partition_count", &self.mid_partitions.len())
            .field("late_partition_count", &self.late_partitions.len())
            .finish()
    }
}

impl PreparedCabinetIr {
    pub fn prepare(
        raw_ir: &[f32],
        sample_rate_hz: u32,
        options: CabinetIrImportOptions,
    ) -> Result<Self, CabinetIrError> {
        if raw_ir.is_empty() {
            return Err(CabinetIrError::Empty);
        }
        if raw_ir.len() > MAX_IR_SAMPLES {
            return Err(CabinetIrError::TooLong(raw_ir.len()));
        }
        if sample_rate_hz != CABINET_SAMPLE_RATE_HZ {
            return Err(CabinetIrError::UnsupportedSampleRate(sample_rate_hz));
        }
        if raw_ir.iter().any(|sample| !sample.is_finite()) {
            return Err(CabinetIrError::NonFiniteSample);
        }

        let peak = raw_ir
            .iter()
            .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
        if peak <= f32::MIN_POSITIVE {
            return Err(CabinetIrError::Silent);
        }
        let trim_db = if options.trim_threshold_db.is_finite() {
            options.trim_threshold_db.clamp(-160.0, 0.0)
        } else {
            DEFAULT_TRIM_THRESHOLD_DB
        };
        let trim_threshold = peak * 10.0_f32.powf(trim_db / 20.0);
        let detected_leading = raw_ir
            .iter()
            .position(|sample| sample.abs() >= trim_threshold)
            .ok_or(CabinetIrError::Silent)?;
        let trimmed_leading_samples = if options.trim_leading_silence {
            detected_leading
        } else {
            0
        };
        let trimmed = &raw_ir[trimmed_leading_samples..];

        let processed = match options.mode {
            CabinetIrMode::Raw => trimmed.to_vec(),
            CabinetIrMode::MinimumPhase => minimum_phase_ir(trimmed),
        };
        if processed.is_empty() {
            return Err(CabinetIrError::Silent);
        }

        let mut direct_head = [0.0; DIRECT_HEAD_SAMPLES];
        let direct_len = processed.len().min(DIRECT_HEAD_SAMPLES);
        direct_head[..direct_len].copy_from_slice(&processed[..direct_len]);

        // Each stage begins exactly one of its own blocks into the IR. That
        // inherent FIR delay gives the stage a complete input block to work
        // with without adding plugin latency:
        //
        //   direct:   0..64
        //   early:   64..256   (64-sample partitions)
        //   middle: 256..1024  (256-sample partitions)
        //   late:  1024..8192  (1024-sample partitions)
        //
        // The costly late FFT consequently runs only once per 1024 samples.
        let early_partitions = prepare_partitions::<EARLY_BLOCK_SAMPLES, EARLY_FFT_SIZE>(
            &processed,
            DIRECT_HEAD_SAMPLES,
            MID_TAIL_START,
        );
        let mid_partitions = prepare_partitions::<MID_BLOCK_SAMPLES, MID_FFT_SIZE>(
            &processed,
            MID_TAIL_START,
            LATE_TAIL_START,
        );
        let late_partitions = prepare_partitions::<LATE_BLOCK_SAMPLES, LATE_FFT_SIZE>(
            &processed,
            LATE_TAIL_START,
            MAX_IR_SAMPLES,
        );

        Ok(Self {
            sample_rate_hz,
            mode: options.mode,
            original_len: raw_ir.len(),
            processed_len: processed.len(),
            trimmed_leading_samples,
            intrinsic_delay_samples: match options.mode {
                CabinetIrMode::MinimumPhase => 0,
                CabinetIrMode::Raw if options.trim_leading_silence => 0,
                CabinetIrMode::Raw => detected_leading,
            },
            direct_head,
            early_partitions,
            mid_partitions,
            late_partitions,
        })
    }

    #[must_use]
    pub const fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    #[must_use]
    pub const fn mode(&self) -> CabinetIrMode {
        self.mode
    }

    #[must_use]
    pub const fn original_len(&self) -> usize {
        self.original_len
    }

    #[must_use]
    pub const fn processed_len(&self) -> usize {
        self.processed_len
    }

    #[must_use]
    pub const fn trimmed_leading_samples(&self) -> usize {
        self.trimmed_leading_samples
    }

    /// Delay contained in RAW IR data. It is metadata, not plugin/PDC latency.
    #[must_use]
    pub const fn intrinsic_delay_samples(&self) -> usize {
        self.intrinsic_delay_samples
    }
}

fn prepare_partitions<const BLOCK_SIZE: usize, const FFT_LEN: usize>(
    processed: &[f32],
    range_start: usize,
    range_end: usize,
) -> Vec<[Complex; FFT_LEN]> {
    debug_assert_eq!(FFT_LEN, BLOCK_SIZE * 2);
    debug_assert_eq!(range_start, BLOCK_SIZE);
    let source_end = processed.len().min(range_end);
    if source_end <= range_start {
        return Vec::new();
    }

    let partition_count = (source_end - range_start).div_ceil(BLOCK_SIZE);
    let twiddles = FixedTwiddles::<FFT_LEN>::default();
    let mut partitions = Vec::with_capacity(partition_count);
    for partition_index in 0..partition_count {
        let source_start = range_start + partition_index * BLOCK_SIZE;
        let partition_end = (source_start + BLOCK_SIZE).min(source_end);
        let mut spectrum = [Complex::ZERO; FFT_LEN];
        for (destination, source) in spectrum[..partition_end - source_start]
            .iter_mut()
            .zip(&processed[source_start..partition_end])
        {
            destination.re = *source;
        }
        fft_fixed(&mut spectrum, false, &twiddles);
        partitions.push(spectrum);
    }
    partitions
}

/// One uniform overlap-add stage within the non-uniform tail.
///
/// The first IR partition represented by a stage is delayed by exactly
/// `BLOCK_SIZE` samples. The input FFT is therefore completed at the end of a
/// block and its first output block is ready before the next sample is read.
/// All vectors and workspaces are created by the loader thread.
#[derive(Clone)]
struct UniformTailStage<const BLOCK_SIZE: usize, const FFT_LEN: usize> {
    partitions: Vec<[Complex; FFT_LEN]>,
    input_spectra: Vec<[Complex; FFT_LEN]>,
    spectra_write: usize,
    spectra_seen: usize,
    input_block: [f32; BLOCK_SIZE],
    block_position: usize,
    output: [f32; BLOCK_SIZE],
    overlap: [f32; BLOCK_SIZE],
    fft_work: [Complex; FFT_LEN],
    twiddles: FixedTwiddles<FFT_LEN>,
}

impl<const BLOCK_SIZE: usize, const FFT_LEN: usize> Default
    for UniformTailStage<BLOCK_SIZE, FFT_LEN>
{
    fn default() -> Self {
        debug_assert_eq!(FFT_LEN, BLOCK_SIZE * 2);
        Self {
            partitions: Vec::new(),
            input_spectra: Vec::new(),
            spectra_write: 0,
            spectra_seen: 0,
            input_block: [0.0; BLOCK_SIZE],
            block_position: 0,
            output: [0.0; BLOCK_SIZE],
            overlap: [0.0; BLOCK_SIZE],
            fft_work: [Complex::ZERO; FFT_LEN],
            twiddles: FixedTwiddles::default(),
        }
    }
}

impl<const BLOCK_SIZE: usize, const FFT_LEN: usize> UniformTailStage<BLOCK_SIZE, FFT_LEN> {
    fn install(&mut self, partitions: Vec<[Complex; FFT_LEN]>) {
        self.partitions = partitions;
        self.input_spectra = vec![[Complex::ZERO; FFT_LEN]; self.partitions.len()];
        self.clear_runtime_state();
    }

    fn unload(&mut self) {
        self.partitions.clear();
        self.input_spectra.clear();
        self.clear_runtime_state();
    }

    fn clear_runtime_state(&mut self) {
        for spectrum in &mut self.input_spectra {
            spectrum.fill(Complex::ZERO);
        }
        self.spectra_write = 0;
        self.spectra_seen = 0;
        self.input_block.fill(0.0);
        self.block_position = 0;
        self.output.fill(0.0);
        self.overlap.fill(0.0);
        self.fft_work.fill(Complex::ZERO);
    }

    #[inline]
    fn process_sample(&mut self, input_sample: f32) -> f32 {
        if self.partitions.is_empty() {
            return 0.0;
        }

        let output_sample = self.output[self.block_position];
        self.input_block[self.block_position] = input_sample;
        self.block_position += 1;
        if self.block_position == BLOCK_SIZE {
            self.prepare_next_output_block();
            self.block_position = 0;
        }
        output_sample
    }

    #[inline(never)]
    fn prepare_next_output_block(&mut self) {
        self.fft_work.fill(Complex::ZERO);
        for (destination, source) in self.fft_work[..BLOCK_SIZE]
            .iter_mut()
            .zip(self.input_block.iter().copied())
        {
            destination.re = source;
        }
        fft_fixed(&mut self.fft_work, false, &self.twiddles);
        self.input_spectra[self.spectra_write] = self.fft_work;
        self.spectra_seen = (self.spectra_seen + 1).min(self.input_spectra.len());

        self.fft_work.fill(Complex::ZERO);
        for partition in 0..self.spectra_seen {
            let input_index = (self.spectra_write + self.input_spectra.len() - partition)
                % self.input_spectra.len();
            let input_spectrum = &self.input_spectra[input_index];
            let ir_spectrum = &self.partitions[partition];
            for bin in 0..FFT_LEN {
                self.fft_work[bin] += input_spectrum[bin].multiply(ir_spectrum[bin]);
            }
        }
        fft_fixed(&mut self.fft_work, true, &self.twiddles);
        for index in 0..BLOCK_SIZE {
            self.output[index] = flush_denormal(self.fft_work[index].re + self.overlap[index]);
            self.overlap[index] = flush_denormal(self.fft_work[index + BLOCK_SIZE].re);
        }

        self.spectra_write = (self.spectra_write + 1) % self.input_spectra.len();
    }

    #[inline]
    fn partition_count(&self) -> usize {
        self.partitions.len()
    }
}

/// Zero-reported-latency hybrid FIR convolver.
///
/// Taps 0..63 are evaluated directly for every sample. The remaining taps are
/// split into 64-, 256-, and 1024-sample overlap-add stages. Every stage begins
/// one full stage block into the IR, so no partition delays the direct head.
/// Internal segmentation is sample-clock based and independent of host block
/// boundaries.
#[derive(Clone)]
pub struct CabinetProcessor {
    sample_rate: f32,
    ir_sample_rate_hz: u32,
    ir_len: usize,
    compatible: bool,
    direct_head: [f32; DIRECT_HEAD_SAMPLES],
    direct_history: [f32; DIRECT_HEAD_SAMPLES],
    direct_write: usize,
    early_tail: UniformTailStage<EARLY_BLOCK_SAMPLES, EARLY_FFT_SIZE>,
    mid_tail: UniformTailStage<MID_BLOCK_SAMPLES, MID_FFT_SIZE>,
    late_tail: UniformTailStage<LATE_BLOCK_SAMPLES, LATE_FFT_SIZE>,
    loaded: bool,
}

impl fmt::Debug for CabinetProcessor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CabinetProcessor")
            .field("sample_rate", &self.sample_rate)
            .field("ir_sample_rate_hz", &self.ir_sample_rate_hz)
            .field("ir_len", &self.ir_len)
            .field("compatible", &self.compatible)
            .field(
                "tail_partition_counts",
                &(
                    self.early_tail.partition_count(),
                    self.mid_tail.partition_count(),
                    self.late_tail.partition_count(),
                ),
            )
            .field("loaded", &self.loaded)
            .finish()
    }
}

impl Default for CabinetProcessor {
    fn default() -> Self {
        Self {
            sample_rate: 48_000.0,
            ir_sample_rate_hz: 48_000,
            ir_len: 0,
            compatible: true,
            direct_head: [0.0; DIRECT_HEAD_SAMPLES],
            direct_history: [0.0; DIRECT_HEAD_SAMPLES],
            direct_write: 0,
            early_tail: UniformTailStage::default(),
            mid_tail: UniformTailStage::default(),
            late_tail: UniformTailStage::default(),
            loaded: false,
        }
    }
}

impl CabinetProcessor {
    /// Builds a complete runtime. Call on the loader/control thread.
    pub fn from_prepared(ir: PreparedCabinetIr) -> Self {
        let mut processor = Self::default();
        processor.install_prepared(ir);
        processor
    }

    /// Installs prepared data and allocates the frequency-domain history.
    /// Never call while `process_block()` is running.
    pub fn install_prepared(&mut self, ir: PreparedCabinetIr) {
        let PreparedCabinetIr {
            sample_rate_hz,
            processed_len,
            direct_head,
            early_partitions,
            mid_partitions,
            late_partitions,
            ..
        } = ir;
        self.ir_sample_rate_hz = sample_rate_hz;
        self.ir_len = processed_len;
        self.direct_head = direct_head;
        self.early_tail.install(early_partitions);
        self.mid_tail.install(mid_partitions);
        self.late_tail.install(late_partitions);
        self.loaded = true;
        self.compatible = self.sample_rate.round() as u32 == self.ir_sample_rate_hz;
        self.clear_runtime_state();
    }

    pub fn unload_ir(&mut self) {
        self.loaded = false;
        self.compatible = true;
        self.ir_len = 0;
        self.direct_head.fill(0.0);
        self.early_tail.unload();
        self.mid_tail.unload();
        self.late_tail.unload();
        self.clear_runtime_state();
    }

    pub fn reset(&mut self, config: &AudioConfig) {
        self.sample_rate = (config.sample_rate as f32).max(1.0);
        self.compatible = !self.loaded || self.sample_rate.round() as u32 == self.ir_sample_rate_hz;
        self.clear_runtime_state();
    }

    fn clear_runtime_state(&mut self) {
        self.direct_history.fill(0.0);
        self.direct_write = 0;
        self.early_tail.clear_runtime_state();
        self.mid_tail.clear_runtime_state();
        self.late_tail.clear_runtime_state();
    }

    #[must_use]
    pub const fn is_loaded(&self) -> bool {
        self.loaded
    }

    #[must_use]
    pub const fn is_sample_rate_compatible(&self) -> bool {
        self.compatible
    }

    #[inline]
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len());
        if !self.loaded {
            output.copy_from_slice(input);
            return;
        }
        if !self.compatible {
            output.fill(0.0);
            return;
        }

        for (&input_sample, output_sample) in input.iter().zip(output) {
            self.direct_history[self.direct_write] = input_sample;
            let mut direct = 0.0;
            let direct_len = self.ir_len.min(DIRECT_HEAD_SAMPLES);
            for tap in 0..direct_len {
                let history_index =
                    (self.direct_write + DIRECT_HEAD_SAMPLES - tap) % DIRECT_HEAD_SAMPLES;
                direct += self.direct_head[tap] * self.direct_history[history_index];
            }
            self.direct_write = (self.direct_write + 1) % DIRECT_HEAD_SAMPLES;

            let tail = self.early_tail.process_sample(input_sample)
                + self.mid_tail.process_sample(input_sample)
                + self.late_tail.process_sample(input_sample);
            *output_sample = flush_denormal(direct + tail);
        }
    }

    #[must_use]
    pub const fn latency_samples(&self) -> u32 {
        0
    }

    #[must_use]
    pub fn tail_samples(&self) -> u32 {
        self.ir_len.saturating_sub(1) as u32
    }
}

#[inline]
fn flush_denormal(sample: f32) -> f32 {
    if sample.abs() < DENORMAL_LIMIT {
        0.0
    } else {
        sample
    }
}

fn fft_fixed<const FFT_LEN: usize>(
    data: &mut [Complex; FFT_LEN],
    inverse: bool,
    twiddles: &FixedTwiddles<FFT_LEN>,
) {
    debug_assert!(FFT_LEN.is_power_of_two());
    bit_reverse_permute(data);
    let mut length = 2;
    while length <= FFT_LEN {
        let half = length / 2;
        let twiddle_stride = FFT_LEN / length;
        for block_start in (0..FFT_LEN).step_by(length) {
            for offset in 0..half {
                let twiddle = twiddles.forward[offset * twiddle_stride];
                let twiddle = if inverse {
                    twiddle.conjugate()
                } else {
                    twiddle
                };
                let even = data[block_start + offset];
                let odd = data[block_start + offset + half].multiply(twiddle);
                data[block_start + offset] = even + odd;
                data[block_start + offset + half] = even - odd;
            }
        }
        length *= 2;
    }
    if inverse {
        let scale = 1.0 / FFT_LEN as f32;
        for value in data {
            value.re *= scale;
            value.im *= scale;
        }
    }
}

fn bit_reverse_permute<const FFT_LEN: usize>(data: &mut [Complex; FFT_LEN]) {
    let mut reversed = 0;
    for index in 1..FFT_LEN {
        let mut bit = FFT_LEN >> 1;
        while reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if index < reversed {
            data.swap(index, reversed);
        }
    }
}

/// Offline real-cepstrum minimum-phase conversion.
fn minimum_phase_ir(input: &[f32]) -> Vec<f32> {
    if input.len() == 1 {
        return input.to_vec();
    }
    let fft_len = (input.len() * 2).next_power_of_two();
    let mut spectrum = vec![Complex::ZERO; fft_len];
    for (destination, source) in spectrum.iter_mut().zip(input) {
        destination.re = *source;
    }
    fft_dynamic(&mut spectrum, false);

    for bin in &mut spectrum {
        let magnitude = (bin.re * bin.re + bin.im * bin.im)
            .sqrt()
            .max(MAGNITUDE_FLOOR);
        bin.re = magnitude.ln();
        bin.im = 0.0;
    }
    fft_dynamic(&mut spectrum, true);

    let nyquist = fft_len / 2;
    for (index, cepstrum) in spectrum.iter_mut().enumerate() {
        if index == 0 || index == nyquist {
            cepstrum.im = 0.0;
        } else if index < nyquist {
            cepstrum.re *= 2.0;
            cepstrum.im = 0.0;
        } else {
            *cepstrum = Complex::ZERO;
        }
    }
    fft_dynamic(&mut spectrum, false);
    for bin in &mut spectrum {
        let radius = bin.re.exp();
        *bin = Complex::from_polar(radius, bin.im);
    }
    fft_dynamic(&mut spectrum, true);

    let mut output: Vec<f32> = spectrum[..input.len()]
        .iter()
        .map(|value| value.re)
        .collect();
    let input_energy = input.iter().map(|sample| sample * sample).sum::<f32>();
    let output_energy = output.iter().map(|sample| sample * sample).sum::<f32>();
    if input_energy > 0.0 && output_energy > f32::MIN_POSITIVE {
        let correction = (input_energy / output_energy).sqrt();
        for sample in &mut output {
            *sample *= correction;
        }
    }
    output
}

fn fft_dynamic(data: &mut [Complex], inverse: bool) {
    debug_assert!(data.len().is_power_of_two());
    let size = data.len();
    let mut reversed = 0;
    for index in 1..size {
        let mut bit = size >> 1;
        while reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if index < reversed {
            data.swap(index, reversed);
        }
    }

    let mut length = 2;
    while length <= size {
        let phase = if inverse {
            std::f32::consts::TAU / length as f32
        } else {
            -std::f32::consts::TAU / length as f32
        };
        let step = Complex::from_polar(1.0, phase);
        for block_start in (0..size).step_by(length) {
            let half = length / 2;
            let mut twiddle = Complex { re: 1.0, im: 0.0 };
            for offset in 0..half {
                let even = data[block_start + offset];
                let odd = data[block_start + offset + half].multiply(twiddle);
                data[block_start + offset] = even + odd;
                data[block_start + offset + half] = even - odd;
                twiddle = twiddle.multiply(step);
            }
        }
        length *= 2;
    }
    if inverse {
        let scale = 1.0 / size as f32;
        for value in data {
            value.re *= scale;
            value.im *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_options() -> CabinetIrImportOptions {
        CabinetIrImportOptions {
            mode: CabinetIrMode::Raw,
            trim_leading_silence: true,
            trim_threshold_db: -80.0,
        }
    }

    fn process_with_partitions(
        processor: &mut CabinetProcessor,
        input: &[f32],
        partitions: &[usize],
    ) -> Vec<f32> {
        let mut output = vec![0.0; input.len()];
        let mut start = 0;
        let mut partition_index = 0;
        while start < input.len() {
            let end = (start + partitions[partition_index % partitions.len()]).min(input.len());
            processor.process_block(&input[start..end], &mut output[start..end]);
            start = end;
            partition_index += 1;
        }
        output
    }

    #[test]
    fn unloaded_cabinet_is_bit_exact() {
        let mut cabinet = CabinetProcessor::default();
        cabinet.reset(&AudioConfig::new(44_100.0, 512));
        let input = [0.0, -1.0, 0.125, 0.5, 1.0];
        let mut output = [f32::NAN; 5];
        cabinet.process_block(&input, &mut output);
        assert_eq!(output, input);
        assert_eq!(cabinet.latency_samples(), 0);
        assert_eq!(cabinet.tail_samples(), 0);
    }

    #[test]
    fn raw_import_trims_leading_silence_and_preserves_onset() {
        let prepared =
            PreparedCabinetIr::prepare(&[0.0, 0.0, 1.0, 0.5], 48_000, raw_options()).unwrap();
        assert_eq!(prepared.original_len(), 4);
        assert_eq!(prepared.processed_len(), 2);
        assert_eq!(prepared.trimmed_leading_samples(), 2);
        assert_eq!(prepared.intrinsic_delay_samples(), 0);

        let mut cabinet = CabinetProcessor::from_prepared(prepared);
        cabinet.reset(&AudioConfig::new(48_000.0, 32));
        let input = [1.0, 0.0, 0.0, 0.0];
        let mut output = [0.0; 4];
        cabinet.process_block(&input, &mut output);
        assert_eq!(output, [1.0, 0.5, 0.0, 0.0]);
        assert_eq!(cabinet.latency_samples(), 0);
        assert_eq!(cabinet.tail_samples(), 1);
    }

    #[test]
    fn partitioned_tail_matches_direct_convolution() {
        let mut ir = vec![0.0; 193];
        for (index, tap) in ir.iter_mut().enumerate() {
            *tap = (-(index as f32) / 47.0).exp() * ((index as f32 * 0.31).cos() + 0.2);
        }
        let prepared = PreparedCabinetIr::prepare(&ir, 48_000, raw_options()).unwrap();
        let mut cabinet = CabinetProcessor::from_prepared(prepared);
        cabinet.reset(&AudioConfig::new(48_000.0, 512));

        let input: Vec<f32> = (0..321)
            .map(|index| (index as f32 * 0.071).sin() * 0.4)
            .collect();
        let mut padded_input = input.clone();
        padded_input.resize(input.len() + ir.len() - 1, 0.0);
        let output = process_with_partitions(&mut cabinet, &padded_input, &[257]);

        let mut expected = vec![0.0; padded_input.len()];
        for (output_index, expected_sample) in expected.iter_mut().enumerate() {
            let tap_end = ir.len().min(output_index + 1);
            for (tap, ir_sample) in ir.iter().enumerate().take(tap_end) {
                let input_index = output_index - tap;
                if input_index < input.len() {
                    *expected_sample += ir_sample * input[input_index];
                }
            }
        }
        for (actual, expected) in output.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 2.5e-4,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn output_is_independent_of_host_block_partitioning() {
        let mut ir = vec![0.0; MAX_IR_SAMPLES];
        for (index, tap) in ir.iter_mut().enumerate() {
            *tap = (-(index as f32) / 1300.0).exp() * (index as f32 * 0.173).cos() * 0.025;
        }
        ir[0] += 0.9;
        let prepared = PreparedCabinetIr::prepare(&ir, 48_000, raw_options()).unwrap();
        let input: Vec<f32> = (0..10_000)
            .map(|index| (index as f32 * 0.119).sin() * 0.6)
            .collect();

        let mut whole = CabinetProcessor::from_prepared(prepared.clone());
        whole.reset(&AudioConfig::new(48_000.0, 1024));
        let whole_output = process_with_partitions(&mut whole, &input, &[input.len()]);

        for block_size in [1, 7, 16, 32, 64, 257, 512] {
            let mut split = CabinetProcessor::from_prepared(prepared.clone());
            split.reset(&AudioConfig::new(48_000.0, 1024));
            let split_output = process_with_partitions(&mut split, &input, &[block_size]);
            assert_eq!(
                whole_output, split_output,
                "output changed for host block size {block_size}"
            );
        }
    }

    #[test]
    fn all_non_uniform_stage_boundaries_preserve_impulse_timing() {
        let mut ir = vec![0.0; MAX_IR_SAMPLES];
        for (index, amplitude) in [
            (0, 0.9),
            (63, -0.8),
            (64, 0.7),
            (255, -0.6),
            (256, 0.5),
            (1023, -0.4),
            (1024, 0.3),
            (4095, -0.2),
            (8191, 0.1),
        ] {
            ir[index] = amplitude;
        }
        let prepared = PreparedCabinetIr::prepare(&ir, 48_000, raw_options()).unwrap();
        let mut cabinet = CabinetProcessor::from_prepared(prepared);
        cabinet.reset(&AudioConfig::new(48_000.0, 32));

        let mut input = vec![0.0; MAX_IR_SAMPLES];
        input[0] = 1.0;
        let output = process_with_partitions(&mut cabinet, &input, &[1, 7, 16, 32, 64, 257, 512]);
        for (index, (&actual, &expected)) in output.iter().zip(&ir).enumerate() {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "impulse mismatch at {index}: actual={actual}, expected={expected}"
            );
        }
        assert_eq!(output[0], ir[0]);
        assert_eq!(cabinet.latency_samples(), 0);
        assert_eq!(cabinet.tail_samples(), (MAX_IR_SAMPLES - 1) as u32);
    }

    #[test]
    fn max_ir_non_uniform_tail_matches_direct_convolution() {
        let mut ir = vec![0.0; MAX_IR_SAMPLES];
        for (index, tap) in ir.iter_mut().enumerate() {
            *tap = (-(index as f32) / 1700.0).exp()
                * ((index as f32 * 0.137).cos() + (index as f32 * 0.031).sin())
                * 0.003;
        }
        ir[0] += 0.7;
        let input: Vec<f32> = (0..257)
            .map(|index| ((index as f32 * 0.071).sin() + (index as f32 * 0.193).cos()) * 0.15)
            .collect();
        let mut padded_input = input.clone();
        padded_input.resize(input.len() + ir.len() - 1, 0.0);

        let prepared = PreparedCabinetIr::prepare(&ir, 48_000, raw_options()).unwrap();
        let mut cabinet = CabinetProcessor::from_prepared(prepared);
        cabinet.reset(&AudioConfig::new(48_000.0, 32));
        let output =
            process_with_partitions(&mut cabinet, &padded_input, &[1, 7, 16, 32, 64, 257, 512]);

        let mut expected = vec![0.0; padded_input.len()];
        for (input_index, input_sample) in input.iter().copied().enumerate() {
            for (tap, ir_sample) in ir.iter().copied().enumerate() {
                expected[input_index + tap] += input_sample * ir_sample;
            }
        }
        for (index, (&actual, &expected)) in output.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 5.0e-4,
                "direct convolution mismatch at {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn minimum_phase_conversion_has_sample_zero_onset() {
        let options = CabinetIrImportOptions {
            mode: CabinetIrMode::MinimumPhase,
            trim_leading_silence: false,
            trim_threshold_db: -100.0,
        };
        let prepared =
            PreparedCabinetIr::prepare(&[0.0, 0.0, 1.0, -0.4, 0.2], 48_000, options).unwrap();
        assert_eq!(prepared.mode(), CabinetIrMode::MinimumPhase);
        assert_eq!(prepared.intrinsic_delay_samples(), 0);
        let mut cabinet = CabinetProcessor::from_prepared(prepared);
        cabinet.reset(&AudioConfig::new(48_000.0, 32));
        let input = [1.0, 0.0, 0.0, 0.0, 0.0];
        let mut output = [0.0; 5];
        cabinet.process_block(&input, &mut output);
        assert!(output[0].abs() > 1.0e-4);
        assert_eq!(cabinet.latency_samples(), 0);
    }

    #[test]
    fn raw_untrimmed_ir_reports_intrinsic_not_plugin_delay() {
        let options = CabinetIrImportOptions {
            mode: CabinetIrMode::Raw,
            trim_leading_silence: false,
            trim_threshold_db: -80.0,
        };
        let prepared = PreparedCabinetIr::prepare(&[0.0, 0.0, 1.0], 48_000, options).unwrap();
        assert_eq!(prepared.intrinsic_delay_samples(), 2);
        let cabinet = CabinetProcessor::from_prepared(prepared);
        assert_eq!(cabinet.latency_samples(), 0);
        assert_eq!(cabinet.tail_samples(), 2);
    }

    #[test]
    fn sample_rate_mismatch_fails_safe() {
        let prepared = PreparedCabinetIr::prepare(&[1.0, 0.5], 48_000, raw_options()).unwrap();
        let mut cabinet = CabinetProcessor::from_prepared(prepared);
        cabinet.reset(&AudioConfig::new(44_100.0, 32));
        assert!(!cabinet.is_sample_rate_compatible());
        let input = [1.0, 0.5];
        let mut output = [f32::NAN; 2];
        cabinet.process_block(&input, &mut output);
        assert_eq!(output, [0.0, 0.0]);
    }

    #[cfg(feature = "rt-paranoid")]
    #[test]
    fn processing_allocates_nothing() {
        let ir: Vec<f32> = (0..MAX_IR_SAMPLES)
            .map(|index| (-(index as f32) / 800.0).exp() * (index as f32 * 0.17).cos())
            .collect();
        let prepared = PreparedCabinetIr::prepare(&ir, 48_000, raw_options()).unwrap();
        let mut cabinet = CabinetProcessor::from_prepared(prepared);
        cabinet.reset(&AudioConfig::new(48_000.0, 32));
        let input = [0.25; 257];
        let mut output = [0.0; 257];
        let (_, allocations) = truce::rt::audit(|| {
            let _section = truce::rt::RtSection::enter();
            cabinet.process_block(&input, &mut output);
        });
        assert_eq!(allocations, 0);
    }

    /// Machine-local diagnostic for the 48 kHz / 32-sample release gate.
    ///
    /// This deliberately remains ignored in normal test runs because timing
    /// assertions are only meaningful in a release build on the target Mac.
    #[test]
    #[ignore = "run with cargo test --release cabinet_callback_profile -- --ignored --nocapture"]
    fn cabinet_callback_profile() {
        assert!(
            !std::hint::black_box(cfg!(debug_assertions)),
            "callback profiling requires a release build"
        );
        let ir: Vec<f32> = (0..MAX_IR_SAMPLES)
            .map(|index| (-(index as f32) / 1600.0).exp() * (index as f32 * 0.113).cos() * 0.03)
            .collect();
        let prepared = PreparedCabinetIr::prepare(&ir, 48_000, raw_options()).unwrap();
        let mut cabinet = CabinetProcessor::from_prepared(prepared);
        cabinet.reset(&AudioConfig::new(48_000.0, 32));
        let input = [0.25; 32];
        let mut output = [0.0; 32];

        for _ in 0..4096 {
            cabinet.process_block(&input, &mut output);
        }

        let mut timings_ns = Vec::with_capacity(20_000);
        for _ in 0..20_000 {
            let started = std::time::Instant::now();
            cabinet.process_block(&input, &mut output);
            timings_ns.push(started.elapsed().as_nanos() as u64);
        }
        std::hint::black_box(output);
        timings_ns.sort_unstable();
        let p99_ns = timings_ns[timings_ns.len() * 990 / 1000];
        let p999_ns = timings_ns[timings_ns.len() * 999 / 1000];
        eprintln!(
            "cabinet 32-sample callback: p99={:.3} ms, p99.9={:.3} ms",
            p99_ns as f64 / 1_000_000.0,
            p999_ns as f64 / 1_000_000.0
        );
        assert!(p99_ns <= 167_000, "p99 exceeded 0.167 ms");
        assert!(p999_ns <= 333_000, "p99.9 exceeded 0.333 ms");
    }
}
