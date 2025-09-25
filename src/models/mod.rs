pub mod base_modules;
pub mod minicpm4;
pub mod qwen2_5vl;

use anyhow::Result;
use candle_core::{DType, Device};
use openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use rocket::futures::Stream;

pub trait GenerateModel {
    // fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self>
    // where
    //     Self: Sized;
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse>;
    fn generate_stream(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<impl Stream<Item = Result<ChatCompletionChunkResponse, anyhow::Error>>>
    where
        Self: Sized;
}
