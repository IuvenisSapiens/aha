pub mod qwen2_5vl;
use anyhow::Result;
use candle_core::{DType, Device};
use openai_dive::v1::resources::chat::ChatCompletionParameters;

pub trait GenerateModel {
    fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self>
    where
        Self: Sized;
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<String>;
}
