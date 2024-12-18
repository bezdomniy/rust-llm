#![feature(array_chunks)]
#![feature(slice_as_chunks)]
#![feature(portable_simd)]

mod maths;
mod sampler;
mod tokenizer;
mod transformer;
mod utils;
use std::io::{self, Write};

use sampler::{ProbIndex, Sampler};
use tokenizer::Tokenizer;
use transformer::Transformer;

fn generate(
    transformer: &mut Transformer,
    tokenizer: &Tokenizer,
    sampler: &Sampler,
    prompt: &str,
    steps: i32,
) -> Result<bool, String> {
    let prompt_tokens = tokenizer.encode(prompt, true, false)?;

    let mut pos = 0;
    let mut next;
    let mut token = prompt_tokens[0];
    let mut prev_token = token;

    let mut out_tokens = vec![];
    while pos < steps {
        transformer.forward(token, pos);

        if pos < prompt_tokens.len() as i32 - 1 {
            next = prompt_tokens[(pos + 1) as usize] as usize;
        } else {
            next = sampler.sample(&mut transformer.state.logits[..]);
        }

        if next == 1 {
            break;
        }

        out_tokens.push(next as u32);

        if pos > 0 {
            let word = tokenizer.decode(token, prev_token).unwrap();
            print!("{}", word);
            let _ = io::stdout().flush();
        }

        prev_token = token;
        token = next as u32;
        pos += 1;
    }
    println!("");

    // let decoded_tokens = out_tokens
    //     .windows(2)
    //     .map(|pair| {
    //         if let &[prev_token, token] = pair {
    //             tokenizer.decode(token, prev_token).unwrap()
    //         } else {
    //             '\0'.to_string()
    //         }
    //     })
    //     .collect::<Vec<String>>();
    // println!("{:?}", decoded_tokens);

    Ok(true)
}

fn main() -> io::Result<()> {
    let mut transformer = transformer::Transformer::new(
        // "assets/stories15M.bin"
        // "assets/stories42M.bin"
        // "assets/llama-3.2-1B-Instruct2.bin",
        "assets/stories110M.bin",
    )?;

    let vocab_size = transformer.config.vocab_size;
    let tokenizer = tokenizer::Tokenizer::new("assets/tokenizer.bin", vocab_size as u32)?;

    let temperature = 0f32;
    let topp = 0.9f32;
    let steps = 256;
    let rng_state = 0;

    let sampler = Sampler {
        rng_state,
        temperature,
        topp,
        vocab_size,
        prob_index: vec![ProbIndex::default(); vocab_size as usize].into_boxed_slice(),
    };

    let _res = generate(
        &mut transformer,
        &tokenizer,
        &sampler,
        // "Today I went",
        // "Why?",
        "One",
        // "One day, Lily met a Shoggoth",
        // "\x03 abcdef 🐻\x1f",
        steps,
    );

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
