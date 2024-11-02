use crate::transformer::Config;
use crate::utils;
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

    pub fn encode(self: &Self, prompt: &String, config: &Config) -> Result<Vec<usize>, String> {
        // let mut prompt_tokens = vec![0u32; prompt.len() + 3];
        // let mut prompt_tokens = Vec::
        // let str_buffer = vec![0u8; self.max_token_length * 2 + 3];

        let prompt_tokens: Vec<u32> = prompt
            .graphemes(true)
            .into_iter()
            .flat_map(|x| {
                let i = self.token_lookup(&x.to_string());
                if i.is_some() {
                    // println!("{:?}  {:?}", i, self.vocab.get(i.unwrap() as usize));
                    vec![i.unwrap()]
                } else {
                    x.as_bytes()
                        .iter()
                        .map(|&b| {
                            // println!("None {:?}", self.vocab.get((b + 3) as usize));
                            (b + 3) as u32
                        })
                        .collect::<Vec<u32>>()
                }
            })
            .collect();
        println!("{:?}", prompt_tokens);

        // println!("{:?}", self.token_lookup(&"\n</s>\n".to_string()));
        // println!("{:?}", self.token_lookup(&"<0x00>".to_string()));
        // println!("{:?}", self.token_lookup(&"<0x05>".to_string()));
        // println!("{:?}", self.token_lookup(&"d".to_string()));
        Ok(vec![])
    }
}
