use candle_nn::Activation;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_chans: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub spatial_patch_size: usize,
    pub window_size: usize,
    pub fullatt_block_indexes: Vec<usize>,
    pub tokens_per_second: usize,
    pub temporal_patch_size: usize,
}
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct RopeScaling {
    pub r#type: String,
    pub mrope_section: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub attention_dropout: f32,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub vision_start_token_id: usize,
    pub vision_end_token_id: usize,
    pub vision_token_id: usize,
    pub image_token_id: usize,
    pub video_token_id: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub max_window_layers: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub sliding_window: usize,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub use_sliding_window: bool,
    pub vision_config: VisionConfig,
    pub rope_scaling: RopeScaling,
    pub vocab_size: usize,
}

pub struct VisionSetting {
    pub image_factor: u32,
    pub min_pixels: u32,
    pub max_pixels: u32,
    pub max_ratio: u32,
    pub temporal_patch_size: usize,
    pub patch_size: usize,
    pub merge_size: usize,
    pub video_min_pixels: u32,
    pub video_max_pixels: u32,
    pub video_total_pixels: u32,
    pub frame_factor: u32,
    pub fps: f32,
    pub fps_min_frames: u32,
    pub fps_max_frames: u32,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
}

impl VisionSetting {
    pub fn default() -> Self {
        Self {
            image_factor: 28,
            min_pixels: 4 * 28 * 28,
            max_pixels: 16384 * 28 * 28,
            // max_pixels: 1000 * 28 * 28,
            max_ratio: 200,
            temporal_patch_size: 2,
            patch_size: 14,
            merge_size: 2,
            video_min_pixels: 128 * 28 * 28,
            video_max_pixels: 768 * 28 * 28,
            video_total_pixels: 24576 * 28 * 28,
            frame_factor: 2,
            fps: 2.0,
            fps_min_frames: 4,
            fps_max_frames: 768,
            image_mean: vec![0.48145466_f32, 0.4578275, 0.40821073],
            image_std: vec![0.26862954, 0.26130258, 0.27577711],
        }
    }
}
