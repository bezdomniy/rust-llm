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
    pub fn sample(self: &Self, logits: &[f32]) -> usize {
        self.sample_argmax(logits)
    }

    fn sample_argmax(self: &Self, logits: &[f32]) -> usize {
        let mut max_i = 0usize;
        let mut max_p = logits[0];

        for i in 1..self.vocab_size as usize {
            if logits[i] > max_p {
                max_i = i;
                max_p = logits[i];
            }
        }

        max_i

        // int sample_argmax(float* probabilities, int n) {
        //     // return the index that has the highest probability
        //     int max_i = 0;
        //     float max_p = probabilities[0];
        //     for (int i = 1; i < n; i++) {
        //         if (probabilities[i] > max_p) {
        //             max_i = i;
        //             max_p = probabilities[i];
        //         }
        //     }
        //     return max_i;
        // }
    }
}
