mod tokenizer;
mod transformer;
mod utils;

use std::io;

fn main() -> io::Result<()> {
    let transformer = transformer::Transformer::new("assets/stories15M.bin");
    // println!("{:?}", transformer?.transformer_weights.rms_final_weight);
    let tokenizer = tokenizer::Tokenizer::new(
        "assets/tokenizer.bin",
        transformer?.config.vocab_size as usize,
    );
    // println!("{:?}", tokenizer?.vocab);
    Ok(())
}
