use aha::models::qwen2_5vl::config::Config;
use anyhow::Result;

#[test]
fn qwen2_5vl_config() -> Result<()> {
    // cargo test qwen2_5vl_config -- --nocapture
    let model_path = "/home/jhq/huggingface_model/Qwen/Qwen2.5-VL-3B-Instruct/";
    let config_path = model_path.to_string() + "/config.json";
    let config: Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
    println!("{:?}", config);
    Ok(())
}
