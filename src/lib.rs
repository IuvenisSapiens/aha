use crate::models::{minicpm4::generate::MiniCPMGenerateModel, qwen2_5vl::generate::Qwen2_5VLGenerateModel, GenerateModel};
use anyhow::{Ok, Result};
use candle_core::{DType, Device};
pub mod chat_template;
pub mod models;
pub mod position_embed;
pub mod tokenizer;
pub mod utils;

pub enum ModelType {
    Qwen2_5VL,
    MiniCPM4,
}

impl ModelType {
    pub fn init(
        model_type: ModelType,
        model_path: &str,
        device: Option<&Device>,
        dtype: Option<DType>,
    ) -> Result<Box<dyn GenerateModel>> {
        match model_type {
            ModelType::Qwen2_5VL => {
                let model = Qwen2_5VLGenerateModel::init(model_path, device, dtype)?;
                Ok(Box::new(model) as Box<dyn GenerateModel>)
            },
            ModelType::MiniCPM4 => {
                let model = MiniCPMGenerateModel::init(model_path, device, dtype)?;
                Ok(Box::new(model)as Box<dyn GenerateModel>)
            }
        }
    }
}
