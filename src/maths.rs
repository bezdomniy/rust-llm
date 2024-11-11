use rayon::prelude::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

struct CPUFeatures {
    has_avx2: bool,
    has_neon: bool,
}

pub fn mat_mul(o: &mut [f32], x: &[f32], w: &[f32], n: usize) {
    let cpu_features = get_cpu_features();
    const PARALLEL_THRESHOLD: usize = 64;

    if n >= PARALLEL_THRESHOLD {
        o.par_chunks_mut(4)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let row_start = chunk_idx * 4;
                chunk.iter_mut().enumerate().for_each(|(i, e)| {
                    let row = row_start + i;
                    unsafe {
                        *e = if cpu_features.has_avx2 || cpu_features.has_neon {
                            simd_dot_product(&w[row * n..], x, n)
                        } else {
                            dot_product_fallback(&w[row * n..], x, n)
                        }
                    }
                });
            });
    } else {
        for i in 0..o.len() {
            o[i] = dot_product_fallback(&w[i * n..], x, n);
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn get_cpu_features() -> CPUFeatures {
    CPUFeatures {
        has_avx2: is_x86_feature_detected!("avx2"),
        has_neon: false,
    }
}

#[cfg(target_arch = "aarch64")]
fn get_cpu_features() -> CPUFeatures {
    CPUFeatures {
        has_avx2: false,
        has_neon: true,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_dot_product(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let chunks = n / 8;

    for i in 0..chunks {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vec, b_vec));
    }

    let sum_arr = std::mem::transmute::<__m256, [f32; 8]>(sum);
    let mut final_sum = sum_arr.iter().sum();

    for i in (chunks * 8)..n {
        final_sum += a[i] * b[i];
    }
    final_sum
}

#[cfg(target_arch = "aarch64")]
unsafe fn simd_dot_product(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut sum = vdupq_n_f32(0f32);
    let chunks = n / 4;

    for i in 0..chunks {
        let a_vec = vld1q_f32(a.as_ptr().add(i * 4));
        let b_vec = vld1q_f32(b.as_ptr().add(i * 4));
        sum = vfmaq_f32(sum, a_vec, b_vec);
    }

    let sum_arr = std::mem::transmute::<float32x4_t, [f32; 4]>(sum);
    let mut final_sum = sum_arr.iter().sum();

    for i in (chunks * 4)..n {
        final_sum += a[i] * b[i];
    }
    final_sum
}

#[inline(always)]
fn dot_product_fallback(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut sum = 0f32;
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_square_mul() {
        let x = vec![1f32, 2f32, 3f32, 4f32];
        let w = vec![
            0f32, 1f32, 2f32, 3f32, 3f32, 3f32, 3f32, 3f32, 4f32, 4f32, 4f32, 4f32, 5f32, 5f32,
            5f32, 5f32,
        ];

        let mut o = vec![0f32, 0f32, 0f32, 0f32];

        let excepted_o = vec![20f32, 30f32, 40f32, 50f32];

        mat_mul(o.as_mut_slice(), x.as_slice(), w.as_slice(), 4);

        assert_eq!(o, excepted_o.as_slice());
    }

    #[test]
    fn test_nonsquare_mul() {
        let x = vec![1f32, 2f32, 3f32, 4f32];
        let w = vec![
            0f32, 1f32, 2f32, 3f32, 3f32, 3f32, 3f32, 3f32, 4f32, 4f32, 4f32, 4f32, 5f32, 5f32,
            5f32, 5f32, 6f32, 6f32, 6f32, 6f32,
        ];

        let mut o = vec![0f32, 0f32, 0f32, 0f32, 0f32];

        let excepted_o = vec![20f32, 30f32, 40f32, 50f32, 60f32];

        mat_mul(o.as_mut_slice(), x.as_slice(), w.as_slice(), 4);

        assert_eq!(o, excepted_o.as_slice());
    }
}
