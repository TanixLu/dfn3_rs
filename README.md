# dfn3_rs

A standalone Rust library for streaming speech enhancement, derived from
[DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) and optimized for
48 kHz mono audio. The embedded model uses FP32 encoder matrices and FP16 decoder
matrices with FP32 computation, recurrent state, normalization, biases, and DSP.

- 480 samples per frame (10 ms); algorithmic delay is 1,440 samples (30 ms).
- Shared, 64-byte-aligned weights: 5,260,224 bytes.
- AVX2/FMA/F16C kernels on supported x86-64 CPUs, with a scalar fallback.
- Construction allocates. Processing, parameter updates, and reset perform no
  heap allocations, locks, or file I/O.

```rust
use dfn3_rs::{Denoiser, FRAME_SIZE};

let mut denoiser = Denoiser::new(); // Outside the audio callback.
let input = [0.0_f32; FRAME_SIZE];  // Finite, normalized 48 kHz mono PCM.
let mut output = [0.0_f32; FRAME_SIZE];

denoiser.set_attenuation_limit(12.0);
denoiser.process(&input, &mut output);

// Adjust strength between frames using the same model instance.
denoiser.set_attenuation_limit(24.0);
denoiser.process(&input, &mut output);
```

`set_attenuation_limit(db)` is the runtime `atten-lim-db` control. Zero gives
delayed dry audio; larger magnitudes give stronger suppression; 100 dB or more
disables the limit (the default). Pass finite values and call the setter between
frames on the processing thread. Updates take effect on the next processing
call and preserve neural history, overlap-add state, and latency. The network
continues processing at 0 dB, so increasing strength requires no restart or warm-up.

`process_frame(&mut frame)` provides in-place processing. Both processing APIs
return the local SNR estimate. `reset()` starts an unrelated stream and retains
parameter settings. `set_thresholds(min, max_erb, max_df)` controls stage skipping;
defaults are -10, 30, and 20 dB.

Audio I/O, resampling, channel routing, gain, limiting, and input buffering belong
to the caller. Use one instance per independently enhanced channel. Output is
not clipped. Zero-pad a partial final frame and process three more zero frames
to emit the delayed tail. For offline delay compensation, discard the first
`ALGORITHM_DELAY` samples and retain the original input length.

Build and check with `cargo build --release`, `cargo clippy --all-targets -- -D warnings`,
and `cargo test --release`. `realfft` is the only direct dependency; FFT plans and
scratch buffers are allocated during construction. Construct, warm up, and reset
the denoiser before starting real-time processing. To warm all neural stages,
temporarily set thresholds to -100, 100, 100, process a few zero frames, reset,
and restore the intended thresholds.

MIT OR Apache-2.0. See [NOTICE](NOTICE), [LICENSE-MIT](LICENSE-MIT), and
[LICENSE-APACHE](LICENSE-APACHE) for model provenance and attribution.
