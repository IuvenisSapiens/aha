pub mod base_modules;
pub mod minicpm4;
pub mod qwen2_5vl;
pub mod voxcpm;

use anyhow::Result;
use openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use rocket::futures::Stream;

pub trait GenerateModel {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse>;
    fn generate_stream(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<impl Stream<Item = Result<ChatCompletionChunkResponse, anyhow::Error>>>
    where
        Self: Sized;
}
