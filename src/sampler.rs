use crate::transformer::Transformer;
use rand::Rng;

#[derive(Default, Clone, Copy)]
pub struct ProbIndex {
    pub prob: f32,
    pub index: usize,
}

// #[derive(Clone, Copy)]
pub struct Sampler {
    pub temperature: f32,
    pub topp: f32,
    pub rng_state: u64,
    pub vocab_size: i32,
    pub prob_index: Box<[ProbIndex]>,
}

impl Sampler {
    pub fn sample(self: &Self, logits: &mut [f32]) -> usize {
        let mut rng = rand::thread_rng();

        if self.temperature == 0f32 {
            Sampler::sample_argmax(logits)
        } else {
            logits.iter_mut().for_each(|e| {
                *e /= self.temperature;
            });
            Transformer::softmax(logits);

            let coin: f32 = rng.gen();

            if self.topp <= 0f32 || self.topp >= 1f32 {
                Sampler::sample_mult(logits, coin)
            } else {
                Sampler::sample_topp(logits, self.topp, &self.prob_index[..], coin)
            }
        }
    }

    fn sample_argmax(logits: &[f32]) -> usize {
        let mut max_i = 0usize;
        let mut max_p = logits[0];

        logits.iter().enumerate().for_each(|(i, e)| {
            if *e > max_p {
                max_i = i;
                max_p = *e;
            }
        });

        max_i
    }

    fn sample_topp(logits: &[f32], topp: f32, prob_index: &[ProbIndex], coin: f32) -> usize {
        0
    }

    fn sample_mult(logits: &[f32], coin: f32) -> usize {
        0
    }
}
