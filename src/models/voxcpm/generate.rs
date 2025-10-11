use std::collections::HashMap;

use crate::{
    models::voxcpm::{
        audio_vae::AudioVAE, config::VoxCPMConfig, model::VoxCPMModel,
        tokenizer::SingleChineseTokenizer,
    },
    utils::utils::{find_type_files, get_device, get_dtype},
};
use anyhow::{Ok, Result};
use candle_core::{DType, Device, Tensor, pickle::read_all_with_key};
use candle_nn::VarBuilder;

pub struct VoxCPMGenerate {
    voxcpm: VoxCPMModel,
    prompt_cache: Option<HashMap<String, Tensor>>,
}

impl VoxCPMGenerate {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let device = &get_device(device);
        let config_path = path.to_string() + "/config.json";
        let config: VoxCPMConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let cfg_dtype = config.dtype.as_str();
        let model_list = find_type_files(path, "pth")?;
        // println!(" pth model_list: {:?}", model_list);
        let mut dict_to_hashmap = HashMap::new();
        let mut vae_dtype = candle_core::DType::F32;
        for m in model_list {
            let dict = read_all_with_key(m, Some("state_dict"))?;
            vae_dtype = dict[0].1.dtype();
            for (k, v) in dict {
                // println!("key: {}, tensor shape: {:?}", k, v);
                dict_to_hashmap.insert(k, v);
            }
        }
        let vb_vae = VarBuilder::from_tensors(dict_to_hashmap, vae_dtype, &device);
        let audio_vae = AudioVAE::new(
            vb_vae,
            128,
            vec![2, 5, 8, 8],
            Some(64),
            1536,
            vec![8, 8, 5, 2],
            16000,
        )?;

        let model_list = find_type_files(path, "bin")?;
        // println!(" bin model_list: {:?}", model_list);
        dict_to_hashmap = HashMap::new();
        let mut m_dtype = get_dtype(dtype, cfg_dtype);
        for m in model_list {
            let dict = read_all_with_key(m, Some("state_dict"))?;
            m_dtype = dict[0].1.dtype();
            for (k, v) in dict {
                // println!("key: {}, tensor shape: {:?}", k, v);
                dict_to_hashmap.insert(k, v);
            }
        }
        let vb_voxcpm = VarBuilder::from_tensors(dict_to_hashmap, m_dtype, device);
        let config_path = path.to_string() + "/config.json";
        let config: VoxCPMConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let tokenizer = SingleChineseTokenizer::new(path)?;
        let voxcpm = VoxCPMModel::new(vb_voxcpm, config, tokenizer, audio_vae)?;

        Ok(Self {
            voxcpm,
            prompt_cache: None,
        })
    }

    pub fn build_prompt_cache(
        &mut self,
        prompt_text: String,
        prompt_wav_path: String,
    ) -> Result<()> {
        let cache = self
            .voxcpm
            .build_prompt_cache(prompt_text, prompt_wav_path)?;
        self.prompt_cache = Some(cache);
        Ok(())
    }

    pub fn generate_use_prompt_cache(
        &mut self,
        target_text: String,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
        retry_badcase: bool,
        retry_badcase_ratio_threshold: f64,
    ) -> Result<Tensor> {
        let audio = match &self.prompt_cache {
            Some(cache) => {
                let prompt_cache = cache.clone();
                self.voxcpm.generate_with_prompt_cache(
                    target_text,
                    prompt_cache,
                    min_len,
                    max_len,
                    inference_timesteps,
                    cfg_value,
                    retry_badcase,
                    retry_badcase_ratio_threshold,
                )?
            }
            None => self.generate_simple(target_text)?,
        };
        Ok(audio)
    }

    pub fn generate_with_prompt_simple(
        &mut self,
        target_text: String,
        prompt_text: Option<String>,
        prompt_wav_path: Option<String>,
    ) -> Result<Tensor> {
        let audio = self.generate(
            target_text,
            prompt_text,
            prompt_wav_path,
            2,
            1000,
            10,
            2.0,
            false,
            6.0,
        )?;
        Ok(audio)
    }
    pub fn generate_simple(&mut self, target_text: String) -> Result<Tensor> {
        let audio = self.generate(target_text, None, None, 2, 100, 10, 2.0, false, 6.0)?;
        Ok(audio)
    }
    pub fn generate(
        &mut self,
        target_text: String,
        prompt_text: Option<String>,
        prompt_wav_path: Option<String>,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
        retry_badcase: bool,
        retry_badcase_ratio_threshold: f64,
    ) -> Result<Tensor> {
        let audio = self.voxcpm.generate(
            target_text,
            prompt_text,
            prompt_wav_path,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
            retry_badcase,
            retry_badcase_ratio_threshold,
        )?;
        Ok(audio)
    }
}
