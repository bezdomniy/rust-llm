use crate::maths::mat_mul;
use crate::utils::{read_file_to_struct, read_variable_length_data};
use bytemuck::{Pod, Zeroable};
use std::fs::File;
use std::io::{self, Seek};
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Config {
    pub dim: i32,
    pub hidden_dim: i32,
    pub n_layers: i32,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub vocab_size: i32,
    pub seq_len: i32,
}

#[derive(Debug)]
pub struct TransformerWeights {
    // token embedding table
    pub token_embedding_table: Arc<[f32]>, // (vocab_size, dim)
    // weights for rmsnorms
    pub rms_att_weight: Box<[f32]>, // (layer, dim) rmsnorm weights
    pub rms_ffn_weight: Box<[f32]>, // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    pub wq: Box<[f32]>, // (layer, dim, n_heads * head_size)
    pub wk: Box<[f32]>, // (layer, dim, n_kv_heads * head_size)
    pub wv: Box<[f32]>, // (layer, dim, n_kv_heads * head_size)
    pub wo: Box<[f32]>, // (layer, n_heads * head_size, dim)
    // weights for ffn
    pub w1: Box<[f32]>, // (layer, hidden_dim, dim)
    pub w2: Box<[f32]>, // (layer, dim, hidden_dim)
    pub w3: Box<[f32]>, // (layer, hidden_dim, dim)
    // final rmsnorm
    pub rms_final_weight: Box<[f32]>, // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    pub wcls: Arc<[f32]>,
}

#[derive(Debug)]
pub struct RunState {
    // current wave of activations
    pub x: Box<[f32]>,      // activation at current time stamp (dim,)
    pub xb: Box<[f32]>,     // same, but inside a residual branch (dim,)
    pub xb2: Box<[f32]>,    // an additional buffer just for convenience (dim,)
    pub hb: Box<[f32]>,     // buffer for hidden dimension in the ffn (hidden_dim,)
    pub hb2: Box<[f32]>,    // buffer for hidden dimension in the ffn (hidden_dim,)
    pub q: Box<[f32]>,      // query (dim,)
    pub att: Box<[f32]>,    // buffer for scores/attention values (n_heads, seq_len)
    pub logits: Box<[f32]>, // output logits
    // kv cache
    pub key_cache: Box<[f32]>,   // (layer, seq_len, dim)
    pub value_cache: Box<[f32]>, // (layer, seq_len, dim)
}

#[derive(Debug)]
pub struct Transformer {
    pub config: Config,
    pub transformer_weights: TransformerWeights,
    pub state: RunState,
}

impl RunState {
    pub fn new(config: &Config) -> io::Result<Self> {
        let kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        let x = vec![0f32; config.dim as usize].into_boxed_slice();
        let xb = vec![0f32; config.dim as usize].into_boxed_slice();
        let xb2 = vec![0f32; config.dim as usize].into_boxed_slice();
        let hb = vec![0f32; config.hidden_dim as usize].into_boxed_slice();
        let hb2 = vec![0f32; config.hidden_dim as usize].into_boxed_slice();
        let q = vec![0f32; config.dim as usize].into_boxed_slice();
        let key_cache =
            vec![0f32; (config.n_layers * config.seq_len * kv_dim) as usize].into_boxed_slice();
        let value_cache =
            vec![0f32; (config.n_layers * config.seq_len * kv_dim) as usize].into_boxed_slice();
        let att = vec![0f32; (config.n_heads * config.seq_len) as usize].into_boxed_slice();
        let logits = vec![0f32; config.vocab_size as usize].into_boxed_slice();

        Ok(Self {
            att,
            hb,
            hb2,
            key_cache,
            logits,
            q,
            value_cache,
            x,
            xb,
            xb2,
        })
    }
}

impl TransformerWeights {
    pub fn new(model_file: &mut std::fs::File, config: &mut Config) -> io::Result<Self> {
        let model_file_size = model_file.metadata()?.len();
        // println!("{:?}", model_file_size);
        let shared_weights = config.vocab_size > 0;
        config.vocab_size = config.vocab_size.abs();

        let head_size = config.dim / config.n_heads;

        println!(
            "Reading token embeddings - size: {}...",
            config.vocab_size * config.dim
        );
        let token_embedding_table: Arc<[f32]> = Arc::from(read_variable_length_data::<f32>(
            model_file,
            (config.vocab_size * config.dim) as usize,
        )?);

        println!(
            "Reading attention weights - size: {}...",
            config.n_layers * config.dim
        );
        let rms_att_weight =
            read_variable_length_data::<f32>(model_file, (config.n_layers * config.dim) as usize)?;

        println!(
            "Reading wq - size: {}...",
            config.n_layers * config.dim * config.n_heads * head_size
        );
        let wq = read_variable_length_data::<f32>(
            model_file,
            (config.n_layers * config.dim * config.n_heads * head_size) as usize,
        )?;
        println!(
            "Reading wk - size: {}...",
            config.n_layers * config.dim * config.n_kv_heads * head_size
        );
        let wk = read_variable_length_data::<f32>(
            model_file,
            (config.n_layers * config.dim * config.n_kv_heads * head_size) as usize,
        )?;
        println!(
            "Reading wv - size: {}...",
            config.n_layers * config.dim * config.n_kv_heads * head_size
        );
        let wv = read_variable_length_data::<f32>(
            model_file,
            (config.n_layers * config.dim * config.n_kv_heads * head_size) as usize,
        )?;
        println!(
            "Reading wo - size: {}...",
            config.n_layers * config.dim * config.n_heads * head_size
        );
        let wo = read_variable_length_data::<f32>(
            model_file,
            (config.n_layers * config.dim * config.n_heads * head_size) as usize,
        )?;

        println!(
            "Reading rms ffn weights - size: {}...",
            config.n_layers * config.dim
        );
        let rms_ffn_weight =
            read_variable_length_data::<f32>(model_file, (config.n_layers * config.dim) as usize)?;

        println!(
            "Reading w1 - size: {}...",
            config.n_layers * config.dim * config.hidden_dim
        );
        let w1 = read_variable_length_data::<f32>(
            model_file,
            (config.n_layers * config.dim * config.hidden_dim) as usize,
        )?;

        println!(
            "Reading w2 - size: {}...",
            config.n_layers * config.dim * config.hidden_dim
        );
        let w2 = read_variable_length_data::<f32>(
            model_file,
            (config.n_layers * config.dim * config.hidden_dim) as usize,
        )?;

        println!(
            "Reading w3 - size: {}...",
            config.n_layers * config.dim * config.hidden_dim
        );
        let w3 = read_variable_length_data::<f32>(
            model_file,
            (config.n_layers * config.dim * config.hidden_dim) as usize,
        )?;

        println!("Reading rms final weight - size: {}...", config.dim);
        let rms_final_weight = read_variable_length_data::<f32>(model_file, (config.dim) as usize)?;

        model_file.seek_relative(head_size as i64)?; //skip what used to be freq_cis_real and freq_cis_imag (for RoPE)

        let wcls = if shared_weights {
            token_embedding_table.clone()
        } else {
            let stream_position = model_file.stream_position()?;
            println!(
                "Reading wcls - size: {}...",
                model_file_size - stream_position
            );
            Arc::from(read_variable_length_data::<f32>(
                model_file,
                (model_file_size - stream_position) as usize,
            )?)
        };

        // println!("{:?}", wcls);

        Ok(TransformerWeights {
            rms_att_weight,
            rms_ffn_weight,
            rms_final_weight,
            token_embedding_table,
            w1,
            w2,
            w3,
            wcls,
            wk,
            wo,
            wq,
            wv,
        })
    }
}

impl Transformer {
    pub fn new(model_file_path: &str) -> io::Result<Self> {
        let mut model_file = File::open(model_file_path)?;

        println!("Loading config...");
        let mut config = read_file_to_struct::<Config>(&mut model_file)?;
        println!("{:?}", config);

        println!("Loading weights...");
        let transformer_weights = TransformerWeights::new(&mut model_file, &mut config)?;

        println!("Initialising state...");
        let state = RunState::new(&config)?;

        println!("Done.");

        Ok(Transformer {
            config,
            transformer_weights,
            state,
        })
    }

    pub fn forward(self: &mut Self, token: u32, pos: i32) {
        let kv_dim = (self.config.dim * self.config.n_kv_heads) / self.config.n_heads;
        let kv_mul = self.config.n_heads / self.config.n_kv_heads;
        let head_size = self.config.dim / self.config.n_heads;

        let content_row = &self.transformer_weights.token_embedding_table[(token as i32
            * self.config.dim)
            as usize
            ..((token as i32 * self.config.dim) + self.config.dim) as usize];
        self.state.x.copy_from_slice(content_row);

        for l in 0..self.config.n_layers {
            Transformer::rms_norm(
                &mut self.state.xb,
                &self.state.x,
                &self.transformer_weights.rms_att_weight[(l * self.config.dim) as usize..],
            );

            let loff = l * self.config.seq_len * kv_dim;
            let kv_start = (loff + (pos * kv_dim)) as usize;

            mat_mul(
                &mut self.state.q,
                &self.state.xb,
                &self.transformer_weights.wq[(l * self.config.dim * self.config.dim) as usize..],
                self.config.dim as usize,
            );

            mat_mul(
                &mut self.state.key_cache[kv_start..kv_start + kv_dim as usize],
                &self.state.xb,
                &self.transformer_weights.wk[(l * self.config.dim * kv_dim) as usize..],
                self.config.dim as usize,
            );

            mat_mul(
                &mut self.state.value_cache[kv_start..kv_start + kv_dim as usize],
                &self.state.xb,
                &self.transformer_weights.wv[(l * self.config.dim * kv_dim) as usize..],
                self.config.dim as usize,
            );

            for i in (0..self.config.dim as usize).step_by(2) {
                let head_dim = i as i32 % head_size;
                let freq = 1f32 / 10000f32.powf((head_dim as f32 / head_size as f32) as f32);
                let val = pos as f32 * freq;
                let fcr = val.cos();
                let fci = val.sin();
                let rotn = if i < kv_dim as usize { 2 } else { 1 };
                for v in 0..rotn {
                    let vec = if v == 0 {
                        self.state.q.as_mut()
                    } else {
                        &mut self.state.key_cache[kv_start..]
                    };
                    let v0 = vec[i];
                    let v1 = vec[i + 1];
                    vec[i] = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            // for h in 0..self.config.n_heads {
            (0..self.config.n_heads).into_iter().for_each(|h| {
                let q =
                    &self.state.q[(h * head_size) as usize..((h * head_size) + head_size) as usize];
                let att = &mut self.state.att[(h * self.config.seq_len) as usize
                    ..((h * self.config.seq_len) + pos + 1) as usize];
                for t in 0..=pos {
                    let start = (loff + (t * kv_dim) + (h / kv_mul) * head_size) as usize;
                    let end = start + head_size as usize;
                    let k: &[f32] = &self.state.key_cache[start..end];
                    let score = q.iter().zip(k).map(|(v1, v2)| v1 * v2).sum::<f32>()
                        / (head_size as f32).sqrt();
                    att[t as usize] = score;
                }
                Transformer::softmax(&mut att[..]);

                let xb = &mut self.state.xb
                    [(h * head_size) as usize..((h * head_size) + head_size) as usize];
                xb.fill(0f32);

                for t in 0..=pos {
                    let v = &self.state.value_cache
                        [(loff + (t * kv_dim) + (h / kv_mul) * head_size) as usize..];
                    let a = att[t as usize];

                    for i in 0..head_size as usize {
                        xb[i] += a * v[i];
                    }
                }
            });

            mat_mul(
                &mut self.state.xb2,
                &self.state.xb,
                &self.transformer_weights.wo[(l * self.config.dim * self.config.dim) as usize..],
                self.config.dim as usize,
            );

            for i in 0..self.config.dim as usize {
                self.state.x[i] += self.state.xb2[i];
            }

            Transformer::rms_norm(
                &mut self.state.xb,
                &self.state.x,
                &self.transformer_weights.rms_ffn_weight[(l * self.config.dim) as usize..],
            );

            mat_mul(
                &mut self.state.hb,
                &self.state.xb,
                &self.transformer_weights.w1
                    [(l * self.config.dim * self.config.hidden_dim) as usize..],
                self.config.dim as usize,
            );

            mat_mul(
                &mut self.state.hb2,
                &self.state.xb,
                &self.transformer_weights.w3
                    [(l * self.config.dim * self.config.hidden_dim) as usize..],
                self.config.dim as usize,
            );

            // silu
            for i in 0..self.config.hidden_dim as usize {
                let mut val = self.state.hb[i];
                val *= 1f32 / (1f32 + (-val).exp());
                val *= self.state.hb2[i];
                self.state.hb[i] = val;
            }

            mat_mul(
                &mut self.state.xb,
                &self.state.hb,
                &self.transformer_weights.w2
                    [(l * self.config.dim * self.config.hidden_dim) as usize..],
                self.config.hidden_dim as usize,
            );

            for i in 0..self.config.dim as usize {
                self.state.x[i] += self.state.xb[i];
            }
        }

        // TODO: find a nicer way
        Transformer::_rms_norm_self(
            &mut self.state.x,
            &self.transformer_weights.rms_final_weight,
        );

        mat_mul(
            &mut self.state.logits,
            &self.state.x,
            &self.transformer_weights.wcls,
            self.config.dim as usize,
        );
    }

    fn rms_norm(o: &mut [f32], x: &[f32], weight: &[f32]) {
        let mut ss = x.iter().fold(0.0, |acc, &val| acc + (val * val));
        ss /= x.len() as f32;
        ss += 1e-5f32;
        ss = 1f32 / ss.sqrt();

        for i in 0..x.len() {
            o[i] = weight[i] * (ss * x[i]);
        }
    }

    fn _rms_norm_self(o: &mut [f32], weight: &[f32]) {
        let mut ss = o.iter().fold(0.0, |acc, &val| acc + (val * val));
        ss /= o.len() as f32;
        ss += 1e-5f32;

        ss = 1f32 / ss.sqrt();

        for i in 0..o.len() {
            o[i] = weight[i] * (ss * o[i]);
        }
    }

    // pub fn softmax(x: &mut [f32]) {
    //     let mut max_val = x[0];
    //     x.iter().for_each(|&e| {
    //         if e > max_val {
    //             max_val = e;
    //         }
    //     });

    //     let mut sum = 0f32;
    //     x.iter_mut().for_each(|e| {
    //         *e = (*e - max_val).exp();
    //         sum += *e;
    //     });

    //     x.iter_mut().for_each(|e| {
    //         *e /= sum;
    //     });
    // }
    pub fn softmax(x: &mut [f32]) {
        // Step 1: Find the maximum value for numerical stability.
        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Step 2: Compute the exponentials of each element (after subtracting max_val).
        let exp_values: Vec<f32> = x.iter().map(|&e| (e - max_val).exp()).collect();

        // Step 3: Compute the sum of exponentials.
        let sum: f32 = exp_values.iter().sum();

        // Step 4: Normalize each element in place.
        x.iter_mut()
            .zip(exp_values.iter())
            .for_each(|(e, &exp_val)| {
                *e = exp_val / sum;
            });
    }
}
