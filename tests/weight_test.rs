use aha::utils::utils::find_safetensors_files;
use anyhow::Result;
use candle_core::{safetensors, Device};
#[test]
fn minicpm4_weight() -> Result<()> {
    let model_path = "/home/jhq/huggingface_model/OpenBMB/MiniCPM4-0.5B/";
    let model_list = find_safetensors_files(&model_path)?;
    let device = Device::Cpu;
    for m in model_list {
        let weights = safetensors::load(m, &device)?;
        for (key, tensor) in weights.iter() {
            println!("=== {} ===", key);
            println!("Shape: {:?}", tensor.shape());
            println!("DType: {:?}", tensor.dtype());
        }
    }
    Ok(())
}