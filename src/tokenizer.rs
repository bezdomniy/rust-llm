use crate::transformer::Config;
use crate::utils;
use core::f32;
use std::error::Error;
use std::fs::File;
use std::io::{self, Read};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug)]
pub struct Tokenizer {
    pub vocab: Box<[String]>,
    pub vocab_scores: Box<[f32]>,
    pub vocab_size: u32,
    pub vocab_sorted: Box<[u32]>,
    pub max_token_length: usize,
    pub byte_pieces: [u8; 256],
}

impl Tokenizer {
    pub fn new(tokenizer_file_path: &str, vocab_size: u32) -> io::Result<Self> {
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

        let mut vocab_sorted = (0..vocab_size).collect::<Vec<u32>>();
        vocab_sorted.sort_unstable_by(|a, b| vocab[*a as usize].cmp(&vocab[*b as usize]));

        Ok(Self {
            byte_pieces,
            max_token_length: max_token_length as usize,
            vocab: vocab.into_boxed_slice(),
            vocab_scores: vocab_scores.into_boxed_slice(),
            vocab_size,
            vocab_sorted: vocab_sorted.into_boxed_slice(),
        })
    }

    pub fn token_lookup(self: &Self, token: &String) -> Option<u32> {
        let res = self.vocab_sorted.binary_search_by(|&probe| {
            let tok = &self.vocab[probe as usize];
            tok.cmp(token)
        });

        match res {
            Ok(i) => Some(self.vocab_sorted[i]),
            Err(_) => None,
        }
    }

    pub fn encode(
        self: &Self,
        prompt: &String,
        bos: bool,
        eos: bool,
        config: &Config,
    ) -> Result<Vec<u32>, String> {
        let mut prompt_tokens = vec![];

        if bos {
            prompt_tokens.push(1);
        }

        if prompt.len() > 0 {
            let dummy_token = self.token_lookup(&" ".to_string());
            prompt_tokens.push(dummy_token.unwrap());
        }

        prompt_tokens.extend(prompt.graphemes(true).into_iter().flat_map(|x| {
            let id = self.token_lookup(&x.to_string());
            if id.is_some() {
                vec![id.unwrap()]
            } else {
                x.as_bytes()
                    .iter()
                    .map(|&b| (b + 3) as u32)
                    .collect::<Vec<u32>>()
            }
        }));

        loop {
            let mut best_score = f32::MIN;
            let mut best_id = -1i32;
            let mut best_idx = -1i32;

            prompt_tokens.windows(2).enumerate().for_each(|(i, pair)| {
                if let &[a, b] = pair {
                    let merged_str =
                        format!("{}{}", self.vocab[a as usize], self.vocab[b as usize]);
                    let id = self.token_lookup(&merged_str.to_string());

                    match id {
                        Some(id) => {
                            best_score = self.vocab_scores[id as usize];
                            best_id = id as i32;
                            best_idx = i as i32;
                        }
                        None => {}
                    }
                }
            });

            if best_idx == -1 {
                break;
            }

            prompt_tokens[best_idx as usize] = best_id as u32;
            prompt_tokens.remove((best_idx + 1) as usize);
        }

        println!(
            "{:?}",
            prompt_tokens
                .iter()
                .map(|&i| { self.vocab.get(i as usize).unwrap() })
                .collect::<Vec<_>>()
        );

        if eos {
            prompt_tokens.push(2);
        }

        Ok(prompt_tokens)
    }
}
