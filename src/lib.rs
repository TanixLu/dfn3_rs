//! Real-time DeepFilterNet3 speech enhancement for 48 kHz mono audio.
//!
//! Construction allocates; [`Denoiser::process`] and [`Denoiser::reset`]
//! perform no heap allocation, locking, or I/O. Weights are embedded and shared.
//!
//! ```
//! let mut denoiser = dfn3_rs::Denoiser::new();
//! let input = [0.0_f32; dfn3_rs::FRAME_SIZE];
//! let mut output = [0.0_f32; dfn3_rs::FRAME_SIZE];
//! let lsnr = denoiser.process(&input, &mut output);
//! ```

#![deny(unsafe_op_in_unsafe_fn)]

mod dsp;
mod net;
mod simd;
mod weights;

use dsp::{BINS, Features, HOP, Stft};
use net::{DF_ORDER, NB_DF, NB_ERB, Network};
use realfft::num_complex::Complex32;

pub const SAMPLE_RATE: u32 = 48_000;
/// Samples consumed and produced per call (10 ms).
pub const FRAME_SIZE: usize = HOP;
/// STFT delay plus two frames of model lookahead (30 ms).
pub const ALGORITHM_DELAY: usize = dsp::FFT - HOP + 2 * HOP;
/// Spectra kept for deep filtering: two past, the target, and two lookahead frames.
const LOOKAHEAD: usize = 2;

/// One independent mono stream. Input must be finite, normalized PCM.
/// Construct outside the audio callback; use one instance per channel.
pub struct Denoiser {
    stft: Stft,
    features: Features,
    net: Box<Network>,
    spectra: [[Complex32; BINS]; DF_ORDER], // oldest first; the target is LOOKAHEAD frames old
    enhanced: [Complex32; BINS],
    gains: [f32; NB_ERB],
    coefs: [f32; NB_DF * DF_ORDER * 2],
    atten_lim: f32,
    thresholds: [f32; 3],
    silent_frames: usize,
}

impl Default for Denoiser {
    fn default() -> Self {
        Self::new()
    }
}

impl Denoiser {
    pub fn new() -> Self {
        Self {
            stft: Stft::new(),
            features: Features::new(),
            net: Box::new(Network::new()),
            spectra: [[Complex32::default(); BINS]; DF_ORDER],
            enhanced: [Complex32::default(); BINS],
            gains: [0.0; NB_ERB],
            coefs: [0.0; NB_DF * DF_ORDER * 2],
            atten_lim: 0.0,
            thresholds: [-10.0, 30.0, 20.0],
            silent_frames: 0,
        }
    }

    /// Clears all streaming state without allocating; retains attenuation and thresholds.
    pub fn reset(&mut self) {
        self.stft.reset();
        self.features.reset();
        self.net.reset();
        self.spectra = [[Complex32::default(); BINS]; DF_ORDER];
        self.silent_frames = 0;
    }

    /// Sets `atten-lim-db` live, without resetting or rebuilding the model.
    ///
    /// Call between frames on the processing thread. The next processing call
    /// uses the new limit; neural history, overlap-add state and delay are preserved.
    /// This setter performs no allocations, locks or I/O.
    ///
    /// Pass finite dB values: 0 gives delayed dry audio, larger magnitudes give
    /// stronger suppression, and 100 or more disables the limit (the default).
    pub fn set_attenuation_limit(&mut self, db: f32) {
        let db = db.abs();
        self.atten_lim = if db >= 100.0 {
            0.0
        } else {
            10f32.powf(-db / 20.0)
        };
    }

    /// Local-SNR thresholds (dB) for skipping work, as in DeepFilterNet: below `min` the frame
    /// is muted, above `max_erb` it passes through, above `max_df` deep filtering is skipped.
    pub fn set_thresholds(&mut self, min: f32, max_erb: f32, max_df: f32) {
        self.thresholds = [min, max_erb, max_df];
    }

    /// Enhances in place, with the same delay and local-SNR result as [`Self::process`].
    pub fn process_frame(&mut self, frame: &mut [f32; FRAME_SIZE]) -> f32 {
        let input = *frame;
        self.process(&input, frame)
    }

    /// Enhances one frame; the output is delayed by [`ALGORITHM_DELAY`] samples.
    /// Returns the estimated local SNR in dB.
    pub fn process(&mut self, input: &[f32; FRAME_SIZE], output: &mut [f32; FRAME_SIZE]) -> f32 {
        // DeepFilterNet: output silence and hold all state once this counter exceeds 5. Masked frames
        // reset it, so it only fires for near-silent frames that would otherwise pass through.
        let power = input.iter().map(|x| x * x).sum::<f32>() / FRAME_SIZE as f32;
        self.silent_frames = if power < 1e-7 {
            self.silent_frames.saturating_add(1)
        } else {
            0
        };
        if self.silent_frames > 5 {
            output.fill(0.0);
            return -15.0;
        }

        self.spectra.copy_within(1.., 0);
        let newest = &mut self.spectra[DF_ORDER - 1];
        self.stft.analyze(input, newest);
        self.features.compute(newest);
        let f = &self.features;
        let lsnr = self.net.encode(&f.erb, &f.re, &f.im);

        let [min, max_erb, max_df] = self.thresholds;
        let target = &self.spectra[DF_ORDER - 1 - LOOKAHEAD];
        let mask = lsnr >= min && lsnr <= max_erb;
        let deep_filter = mask && lsnr <= max_df;
        if mask {
            self.silent_frames = 0;
            self.net.erb_decode(&mut self.gains);
            let mut bins = self.enhanced.iter_mut().zip(target);
            for (&n, &g) in f.erb_widths.iter().zip(&self.gains) {
                bins.by_ref().take(n).for_each(|(e, &x)| *e = x * g);
            }
        } else if lsnr < min {
            self.silent_frames = 0;
            self.enhanced.fill(Complex32::default());
        } else {
            self.enhanced.copy_from_slice(target);
            self.silent_frames += 1;
        }
        if deep_filter {
            self.net.df_decode(&mut self.coefs);
            for (f, e) in self.enhanced[..NB_DF].iter_mut().enumerate() {
                let c = &self.coefs[f * DF_ORDER * 2..];
                *e = (0..DF_ORDER).fold(Complex32::default(), |sum, o| {
                    sum + self.spectra[o][f] * Complex32::new(c[o * 2], c[o * 2 + 1])
                });
            }
        }
        if self.atten_lim > 0.0 {
            let lim = self.atten_lim;
            let target = &self.spectra[DF_ORDER - 1 - LOOKAHEAD];
            self.enhanced
                .iter_mut()
                .zip(target)
                .for_each(|(e, &x)| *e = *e * (1.0 - lim) + x * lim);
        }
        self.stft.synthesize(&mut self.enhanced, output);
        lsnr
    }
}
