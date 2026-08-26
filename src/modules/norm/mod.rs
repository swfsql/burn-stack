/// Root-mean-square normalisation (last-dim, fp16-safe); doubles as a QK-Norm.
pub mod rms_norm;
/// RMSNorm followed by a SiLU(z) gate (a gated block's output norm).
pub mod rms_norm_gated;
/// The RMSNorm-then-dot score shared by the two gate-mixing modules.
pub mod rms_score;
