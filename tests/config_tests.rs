use aha::models::{minicpm4::config::MiniCPM4Config, qwen2_5vl::config::Qwen2_5VLConfig};
use anyhow::Result;

#[test]
fn qwen2_5_vl_config() -> Result<()> {
    // cargo test -F cuda,flash-attn qwen2_5vl_config -- --nocapture
    let model_path = "/home/jhq/huggingface_model/Qwen/Qwen2.5-VL-3B-Instruct/";
    let config_path = model_path.to_string() + "/config.json";
    let config: Qwen2_5VLConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
    println!("{:?}", config);
    Ok(())
}


#[test]
fn minicpm4_config() -> Result<()> {
    // cargo test -F cuda,flash-attn minicpm4_config -- --nocapture
    let model_path = "/home/jhq/huggingface_model/OpenBMB/MiniCPM4-0.5B/";
    let config_path = model_path.to_string() + "/config.json";
    let config: MiniCPM4Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
    println!("{:?}", config);
    Ok(())
}