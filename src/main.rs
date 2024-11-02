mod sampler;
mod tokenizer;
mod transformer;
mod utils;
use std::io;

use sampler::{ProbIndex, Sampler};
use tokenizer::Tokenizer;
use transformer::Transformer;

fn generate(
    transformer: &Transformer,
    tokenizer: &Tokenizer,
    sampler: &Sampler,
    prompt: &str,
    steps: u32,
) -> Result<bool, String> {
    let tokens = tokenizer.encode(prompt, true, false)?;

    let decoded_tokens = tokens
        .windows(2)
        .map(|pair| {
            if let &[prev_token, token] = pair {
                tokenizer.decode(token, prev_token).unwrap()
            } else {
                '\0'.to_string()
            }
        })
        .collect::<Vec<String>>();
    println!("{:?}", decoded_tokens);

    Ok(true)
}

fn main() -> io::Result<()> {
    let transformer = transformer::Transformer::new("assets/stories15M.bin")?;

    let vocab_size = transformer.config.vocab_size;
    let tokenizer = tokenizer::Tokenizer::new("assets/tokenizer.bin", vocab_size as u32)?;

    let sampler = Sampler {
        rng_state: 0,
        temperature: 0.0,
        topp: 0.9,
        vocab_size,
        prob_index: vec![ProbIndex::default(); vocab_size as usize].into_boxed_slice(),
    };

    generate(&transformer, &tokenizer, &sampler, "\x03 abcdef 🐻\x1f", 16);

    // println!("Enter you prompt:");
    // let stdin = io::stdin();
    // for line in stdin.lock().lines() {
    //     let p = line?;
    //     let prompt = p.as_str();
    //     generate(&transformer, &tokenizer, &sampler, prompt, 16);
    //     break;
    // }

    Ok(())
}
