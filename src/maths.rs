use rayon::prelude::*;
use std::simd::f32x4;

struct CPUFeatures {
    has_avx2: bool,
    has_neon: bool,
}

pub fn mat_mul(o: &mut [f32], x: &[f32], w: &[f32], n: usize) {
    let cpu_features = get_cpu_features();
    let chunk_size = 8;
    o.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let row_start = chunk_idx * chunk_size;
            chunk.iter_mut().enumerate().for_each(|(i, e)| {
                let row = row_start + i;
                *e = if cpu_features.has_avx2 || cpu_features.has_neon {
                    simd_dot_product(&w[row * n..], x)
                } else {
                    dot_product_fallback(&w[row * n..], x)
                }
            });
        });
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

fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    use std::simd::{num::SimdFloat, StdFloat};

    // println!("{} {}", a.len(), b.len());

    a.array_chunks::<4>()
        .map(|&a| f32x4::from_array(a))
        .zip(b.array_chunks::<4>().map(|&b| f32x4::from_array(b)))
        .fold(f32x4::splat(0f32), |acc, (a, b)| a.mul_add(b, acc))
        .reduce_sum()
}

#[inline(always)]
fn dot_product_fallback(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0f32, |acc, (a, &b)| a.mul_add(b, acc))
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
