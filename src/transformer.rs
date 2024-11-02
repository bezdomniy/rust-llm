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
    pub k: usize,                // key (dim,)
    pub v: usize,                // value (dim,)
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
            k: 0,
            v: 0,
        })
    }
}

impl TransformerWeights {
    pub fn new(model_file: &mut std::fs::File, config: &Config) -> io::Result<Self> {
        let model_file_size = model_file.metadata()?.len();
        // println!("{:?}", model_file_size);
        let shared_weights = config.vocab_size > 0;

        let head_size = config.dim / config.n_heads;

        let token_embedding_table: Arc<[f32]> = Arc::from(read_variable_length_data::<f32>(
            model_file,
            (config.vocab_size * config.dim) as usize,
        )?);

        let rms_att_weight =
            read_variable_length_data::<f32>(model_file, (config.n_layers * config.dim) as usize)?;

        let wq = read_variable_length_data::<f32>(
            model_file,
            (config.n_layers * config.dim * config.n_heads * head_size) as usize,
        )?;
        let wk = read_variable_length_data::<f32>(
            model_file,
            (config.n_layers * config.dim * config.n_kv_heads * head_size) as usize,
        )?;
        let wv = read_variable_length_data::<f32>(
            model_file,
            (config.n_layers * config.dim * config.n_kv_heads * head_size) as usize,
        )?;
        let wo = read_variable_length_data::<f32>(
            model_file,
            (config.n_layers * config.dim * config.n_heads * head_size) as usize,
        )?;

        let rms_ffn_weight =
            read_variable_length_data::<f32>(model_file, (config.n_layers * config.dim) as usize)?;

        let w1 = read_variable_length_data::<f32>(
            model_file,
            (config.n_layers * config.dim * config.hidden_dim) as usize,
        )?;

        let w2 = read_variable_length_data::<f32>(
            model_file,
            (config.n_layers * config.dim * config.hidden_dim) as usize,
        )?;

        let w3 = read_variable_length_data::<f32>(
            model_file,
            (config.n_layers * config.dim * config.hidden_dim) as usize,
        )?;

        let rms_final_weight = read_variable_length_data::<f32>(model_file, (config.dim) as usize)?;

        model_file.seek_relative(head_size as i64)?; //skip what used to be freq_cis_real and freq_cis_imag (for RoPE)

        let wcls = if shared_weights {
            token_embedding_table.clone()
        } else {
            let stream_position = model_file.stream_position()?;
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

        let config = read_file_to_struct::<Config>(&mut model_file)?;
        println!("{:?}", config);

        let transformer_weights = TransformerWeights::new(&mut model_file, &config)?;
        let state = RunState::new(&config)?;

        Ok(Transformer {
            config,
            transformer_weights,
            state,
        })
    }
}
