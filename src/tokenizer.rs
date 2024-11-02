use crate::utils;
use std::error::Error;
use std::fs::File;
use std::io::{self, Read};

#[derive(Debug)]
pub struct Tokenizer {
    pub vocab: Box<[String]>,
    pub vocab_scores: Box<[f32]>,
    pub vocab_size: usize,
    pub vocab_sorted: Box<[usize]>,
    pub max_token_length: usize,
    pub byte_pieces: [u8; 256],
}

impl Tokenizer {
    pub fn new(tokenizer_file_path: &str, vocab_size: usize) -> io::Result<Self> {
        let mut tokenizer_file = File::open(tokenizer_file_path)?;

        let byte_pieces: [u8; 256] = (0..=255).collect::<Vec<u8>>().try_into().unwrap();
        let max_token_length = utils::read_variable_length_data::<u32>(&mut tokenizer_file, 1)?[0];

        let (vocab_scores, vocab): (Vec<_>, Vec<_>) = (0..vocab_size)
            .map(|_| {
                let vocab_score =
                    utils::read_variable_length_data::<f32>(&mut tokenizer_file, 1).unwrap()[0];
                let len =
                    utils::read_variable_length_data::<u32>(&mut tokenizer_file, 1).unwrap()[0];

                let token =
                    utils::read_variable_length_string(&mut tokenizer_file, len as usize).unwrap();

                (vocab_score as f32, token)
            })
            .unzip();

        let mut vocab_sorted = (0..vocab_size).collect::<Vec<usize>>();
        vocab_sorted.sort_unstable_by(|a, b| vocab[*a].cmp(&vocab[*b]));

        Ok(Self {
            byte_pieces,
            max_token_length: max_token_length as usize,
            vocab: vocab.into_boxed_slice(),
            vocab_scores: vocab_scores.into_boxed_slice(),
            vocab_size,
            vocab_sorted: vocab_sorted.into_boxed_slice(),
        })
    }

    pub fn encode(prompt: &String) -> Result<Vec<usize>, String> {
        todo!()
    }
}
