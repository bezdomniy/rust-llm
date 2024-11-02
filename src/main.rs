mod sampler;
mod tokenizer;
mod transformer;
mod utils;

use std::io;

use sampler::{ProbIndex, Sampler};
use tokenizer::Tokenizer;
use transformer::Transformer;

fn main() -> io::Result<()> {
    let transformer = transformer::Transformer::new("assets/stories15M.bin")?;
    // println!("{:?}", transformer?.transformer_weights.rms_final_weight);

    let vocab_size = transformer.config.vocab_size;
    let tokenizer = tokenizer::Tokenizer::new("assets/tokenizer.bin", vocab_size as u32)?;
    // println!("{:?}", tokenizer?.vocab);
    // tokenizer.encode(
    //     &"\x00\x01\x02ыцö加y̆a\r\nb🇷🇺🇸🇹".to_string(),
    //     &transformer.config,
    // );

    tokenizer.encode(&"\x00\x01\x02ыцö加y̆a\r\nb".to_string(), &transformer.config);

    let sampler = Sampler {
        rng_state: 0,
        temperature: 0.0,
        topp: 0.9,
        vocab_size,
        prob_index: vec![ProbIndex::default(); vocab_size as usize].into_boxed_slice(),
    };

    Ok(())
}

fn generate(
    transformer: &Transformer,
    tokenizer: &Tokenizer,
    sampler: &Sampler,
    prompt: &str,
    steps: u32,
) {
}
