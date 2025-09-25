use candle_nn::Activation;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct RopeScalingConfig {
    pub rope_type: String,
    pub long_factor: Vec<f32>,
    pub short_factor: Vec<f32>,
    pub original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct MiniCPM4Config {
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_scaling: RopeScalingConfig,
    pub torch_dtype: String,
    pub vocab_size: usize,
    // pub use_mup: bool,
    pub scale_emb:f32,
    pub dim_model_base: usize,
    pub scale_depth: f32,
    // pub rope_theta: f32,
    // pub kv_channels: i32,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct VoxCPMEncoderConfig {
    hidden_dim: usize,
    ffn_dim: usize,
    num_heads: usize,
    num_layers: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct CfmConfig {
    sigma_min: f32,
    solver: String,
    t_scheduler: String,
    inference_cfg_rate: f32,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct VoxCPMDitConfig {
    hidden_dim: usize,
    ffn_dim: usize,
    num_heads: usize,
    num_layers: usize,
    cfm_config: CfmConfig,
}