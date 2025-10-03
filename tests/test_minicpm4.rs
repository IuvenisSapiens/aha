use std::{pin::pin, time::Instant};

use aha::models::{minicpm4::generate::MiniCPMGenerateModel, GenerateModel};
use anyhow::Result;
use candle_core::{DType, Device};
use openai_dive::v1::resources::chat::ChatCompletionParameters;
use rocket::futures::StreamExt;

#[test]
fn minicpm_generate() -> Result<()> {
    // test with cpu :(太慢了, : RUST_BACKTRACE=1 cargo test minicpm_generate -- --nocapture
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda minicpm_generate -- --nocapture
    // test with cuda+flash-attn: RUST_BACKTRACE=1 cargo test -F cuda,flash-attn minicpm_generate -- --nocapture
    
    let model_path = "/home/jhq/huggingface_model/OpenBMB/MiniCPM4-0.5B/";
    let message = r#"
    {
        "temperature": 0.3,
        "top_p": 0.8,
        "model": "minicpm4",
        "messages": [
            {
                "role": "user",
                "content": "贾宝玉和孙悟空有什么关系"
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = MiniCPMGenerateModel::init(model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let result = model.generate(mes)?;
    println!("generate: \n {:?}", result);
    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);

    Ok(())
}

#[tokio::test]
async fn minicpm_stream() -> Result<()> {
    // test with cuda+flash-attn: RUST_BACKTRACE=1 cargo test -F cuda,flash-attn minicpm_stream -- --nocapture
    
    let model_path = "/home/jhq/huggingface_model/OpenBMB/MiniCPM4-0.5B/";

    let message = r#"
    {
        "model": "minicpm4",
        "messages": [
            {
                "role": "user",
                "content": "你是谁"
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = MiniCPMGenerateModel::init(model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let mut stream = pin!(model.generate_stream(mes)?);
    while let Some(item) = stream.next().await {
        println!("generate: \n {:?}", item);
    }

    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);

    Ok(())
}
