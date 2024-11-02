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
