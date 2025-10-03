use std::collections::HashMap;

use aha::utils::utils::{find_type_files, get_device};
use anyhow::Result;
use candle_core::{pickle::{read_all_with_key, read_pth_tensor_info, PthTensors}, safetensors, Device, Tensor};
use candle_nn::VarBuilder;

#[test]
fn minicpm4_weight() -> Result<()> {
    let model_path = "/home/jhq/huggingface_model/OpenBMB/MiniCPM4-0.5B/";
    let model_list = find_type_files(&model_path, "safetensors")?;
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

#[test]
fn voxcpm_weight() -> Result<()> {
    let model_path = "/home/jhq/huggingface_model/openbmb/VoxCPM-0.5B/";
    let model_list = find_type_files(&model_path, "pth")?;
    println!("model_list: {:?}", model_list);
    let dev = get_device(None);
    let mut dict_to_hashmap = HashMap::new();
    let mut dtype = candle_core::DType::F16;
    for m in model_list {  
        let dict = read_all_with_key(m, Some("state_dict"))?;        
        dtype = dict[0].1.dtype();
        for (k, v) in dict {
            println!("key: {}, tensor shape: {:?}", k, v);
            dict_to_hashmap.insert(k, v);
        }        
    }
    let vb = VarBuilder::from_tensors(dict_to_hashmap, dtype, &dev);
    let contain_key = vb.contains_tensor("encoder.block.4.block.2.block.3.weight_g");
    println!("contain encoder.block.4.block.2.block.3.weight_g: {}", contain_key);
    Ok(())
}