//! Embedded DFN3 weights: FP32 encoder and FP16 decoder matrices.
//! The blob lives in the image's read-only data and is shared by all instances.

#[cfg(not(target_endian = "little"))]
compile_error!("dfn3_rs weights require little-endian storage");

#[repr(C, align(64))]
struct Aligned<T: ?Sized>(T);

static BLOB: &Aligned<[u8]> = &Aligned(*include_bytes!("../weights/dfn3.bin"));

#[derive(Clone, Copy)]
pub(crate) enum Data {
    F32(&'static [f32]),
    F16(&'static [u16]),
}

impl Data {
    pub fn len(self) -> usize {
        match self {
            Self::F32(w) => w.len(),
            Self::F16(w) => w.len(),
        }
    }

    fn slice(self, start: usize, len: usize) -> Self {
        match self {
            Self::F32(w) => Self::F32(&w[start..][..len]),
            Self::F16(w) => Self::F16(&w[start..][..len]),
        }
    }
}

pub(crate) fn expand(w: u16) -> f32 {
    let sign = ((w & 0x8000) as u32) << 16;
    let exp = (w >> 10) & 31;
    let mantissa = (w & 1023) as u32;
    let value = match exp {
        0 => (mantissa as f32 * (1.0 / 16_777_216.0)).to_bits(),
        31 => 0x7f80_0000 | (mantissa << 13),
        _ => ((exp as u32 + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(sign | value)
}

/// Matrix packed as `[n / 16][k][16]`.
/// `n` is padded to a multiple of 16.
#[derive(Clone, Copy)]
pub(crate) struct Mat {
    pub k: usize,
    pub n: usize,
    pub w: Data,
}

/// `groups` independent matrices stored back to back (block-diagonal linear).
#[derive(Clone, Copy)]
pub(crate) struct Grouped {
    pub groups: usize,
    pub mat: Mat,
}

impl Grouped {
    pub fn group(&self, g: usize) -> Mat {
        let len = self.mat.k * self.mat.n;
        Mat {
            w: self.mat.w.slice(g * len, len),
            ..self.mat
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Gru {
    pub w: Mat,
    pub r: Mat,
    pub bias_w: &'static [f32],
    pub bias_r: &'static [f32],
}

/// Sequential reader; every tensor starts on a 64-byte boundary.
pub(crate) struct Reader {
    bytes: &'static [u8],
    pub decoder: bool,
}

impl Reader {
    pub fn new() -> Self {
        Self {
            bytes: &BLOB.0,
            decoder: false,
        }
    }

    fn tensor<T>(&mut self, len: usize) -> &'static [T] {
        let size = len * std::mem::size_of::<T>();
        let (head, rest) = self.bytes.split_at(size.next_multiple_of(64));
        self.bytes = rest;
        // SAFETY: embedded tensors start at 64-byte boundaries; T is f32 or u16,
        // both accept every bit pattern. The exporter writes little-endian data.
        unsafe { std::slice::from_raw_parts(head.as_ptr().cast(), len) }
    }

    /// Returns a padded FP32 tensor, including padding used by SIMD bias loads.
    pub fn take(&mut self, len: usize) -> &'static [f32] {
        self.tensor(len.next_multiple_of(16))
    }

    fn matrix(&mut self, len: usize) -> Data {
        if self.decoder {
            return Data::F16(self.tensor(len));
        }
        Data::F32(self.tensor(len))
    }

    pub fn mat(&mut self, k: usize, n: usize) -> Mat {
        let n = n.next_multiple_of(16);
        Mat {
            k,
            n,
            w: self.matrix(k * n),
        }
    }

    pub fn grouped(&mut self, groups: usize, k: usize, n: usize) -> Grouped {
        let n = n.next_multiple_of(16);
        let w = self.matrix(groups * k * n);
        Grouped {
            groups,
            mat: Mat { k, n, w },
        }
    }

    pub fn gru(&mut self, inputs: usize) -> Gru {
        Gru {
            w: self.mat(inputs, 768),
            r: self.mat(256, 768),
            bias_w: self.take(768),
            bias_r: self.take(768),
        }
    }

    pub fn finish(self) {
        assert!(self.bytes.is_empty(), "weight blob layout mismatch");
    }
}
