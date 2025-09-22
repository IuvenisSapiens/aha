use crate::models::qwen2_5vl::config::Config;
use crate::utils::utils::{find_safetensors_files, get_device, get_dtype};
use crate::{
    chat_template::chat_template::ChatTemplate,
    models::{
        GenerateModel,
        qwen2_5vl::{model::Qwen2_5VLModel, processor::Qwen2_5VLProcessor},
    },
    tokenizer::tokenizer::TokenizerModel,
};
use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use openai_dive::v1::resources::chat::ChatCompletionParameters;

pub struct Qwen2_5VLGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    pre_processor: Qwen2_5VLProcessor,
    qwen2_5_vl: Qwen2_5VLModel,
    device: Device,
    dtype: DType,
}

impl<'a> GenerateModel for Qwen2_5VLGenerateModel<'a> {
    fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = &get_device(device);
        let cfg_dtype = cfg.torch_dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let pre_processor = Qwen2_5VLProcessor::new(device, dtype)?;       
        
        let model_list = find_safetensors_files(&path)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)? };
        let qwen2_5_vl = Qwen2_5VLModel::new(cfg, vb)?;
        Ok(Qwen2_5VLGenerateModel {
            chat_template,
            tokenizer,
            pre_processor,
            qwen2_5_vl,
            device: device.clone(),
            dtype: dtype,
        })
    }
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<String> {
        let temperature = match mes.temperature {
            Some(temp) => Some(temp as f64),
            None => None,
        };
        let top_p = match mes.top_p {
            Some(tp) => Some(tp as f64),
            None => None,
        };
        let mut logit_processor = LogitsProcessor::new(34562, temperature, top_p);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let mut input_ids = self
            .tokenizer
            .text_encode(input.replace_text.clone(), &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let end_of_text_id = self.qwen2_5_vl.cfg.bos_token_id as u32;
        let im_end_id = self.qwen2_5_vl.cfg.eos_token_id as u32;
        let mut pixel_values = if input.pixel_values.is_some() {
            Some(&input.pixel_values.unwrap().clone())
        } else {
            None
        };
        let image_grid_thw = if input.image_grid_thw.is_some() {
            Some(&input.image_grid_thw.unwrap().clone())
        } else {
            None
        };
        let mut pixel_values_video = if input.pixel_values_video.is_some() {
            Some(&input.pixel_values_video.unwrap().clone())
        } else {
            None
        };
        let video_grid_thw = if input.video_grid_thw.is_some() {
            Some(&input.video_grid_thw.unwrap().clone())
        } else {
            None
        };
        let second_per_grid_ts = if input.second_per_grid_ts.is_some() {
            Some(input.second_per_grid_ts.unwrap().clone())
        } else {
            None
        };

        let mut mask = Tensor::ones_like(&input_ids)?;
        let mut cache_position = Tensor::ones_like(&input_ids.i(0)?)?
            .to_dtype(candle_core::DType::F64)?
            .cumsum(D::Minus1)?
            .to_dtype(candle_core::DType::U32)?
            .broadcast_sub(&Tensor::new(vec![1_u32], input_ids.device())?)?;

        let mut generate = Vec::new();
        let sample_len = match mes.max_tokens {
            Some(max) => max,
            None => 512,
        };
        for _ in 0..sample_len {
            let logits = self.qwen2_5_vl.forward(
                &input_ids,
                pixel_values,
                image_grid_thw,
                pixel_values_video,
                video_grid_thw,
                &mask,
                Some(&cache_position),
                seqlen_offset,
                second_per_grid_ts.clone(),
            )?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let next_token = logit_processor.sample(&logits)?;
            generate.push(next_token);
            if next_token == end_of_text_id || next_token == im_end_id {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            let appendd_mask = Tensor::ones((1, 1), mask.dtype(), &self.device)?;
            mask = Tensor::cat(&[mask, appendd_mask], 1)?;
            cache_position = Tensor::from_vec(vec![seqlen_offset as u32], 1, &self.device)?;
            pixel_values = None;
            pixel_values_video = None;
        }
        let res = self.tokenizer.token_decode(generate)?;
        self.qwen2_5_vl.clear_kv_cache();
        Ok(res)
    }
}
