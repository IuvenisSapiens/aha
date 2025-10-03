use std::collections::HashMap;
use anyhow::{Ok, Result};

use aha::{models::voxcpm::{audio_vae::AudioVAE, config::VoxCPMConfig, model::VoxCPMModel, tokenizer::SingleChineseTokenizer}, utils::utils::{find_type_files, get_device}};
use candle_core::pickle::read_all_with_key;
use candle_nn::VarBuilder;


#[test]
fn voxcpm_generate() -> Result<()> {
    let model_path = "/home/jhq/huggingface_model/openbmb/VoxCPM-0.5B/";
    let model_list = find_type_files(&model_path, "pth")?;
    println!(" pth model_list: {:?}", model_list);
    let dev = get_device(None);
    let mut dict_to_hashmap = HashMap::new();
    let mut dtype = candle_core::DType::F32;
    for m in model_list {  
        let dict = read_all_with_key(m, Some("state_dict"))?;        
        dtype = dict[0].1.dtype();
        for (k, v) in dict {
            // println!("key: {}, tensor shape: {:?}", k, v);
            dict_to_hashmap.insert(k, v);
        }        
    }
    let vb = VarBuilder::from_tensors(dict_to_hashmap, dtype, &dev);
    let audio_vae = AudioVAE::new(vb, 128, vec![2, 5, 8, 8], Some(64), 1536, vec![8, 8, 5, 2], 16000)?;
    println!("audio vae load down");
    let model_list = find_type_files(&model_path, "bin")?;
    println!(" bin model_list: {:?}", model_list);
    dict_to_hashmap = HashMap::new();
    for m in model_list {  
        let dict = read_all_with_key(m, Some("state_dict"))?;        
        dtype = dict[0].1.dtype();
        for (k, v) in dict {
            // println!("key: {}, tensor shape: {:?}", k, v);
            dict_to_hashmap.insert(k, v);
        }        
    }
    let vb_vox = VarBuilder::from_tensors(dict_to_hashmap, dtype, &dev);
    let config_path = model_path.to_string() + "/config.json";
    let config: VoxCPMConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
    let tokenizer = SingleChineseTokenizer::new(model_path)?;
    let mut voxcpm = VoxCPMModel::new(vb_vox, config, tokenizer, audio_vae)?;
    let generate = voxcpm.generate("你好啊，这是初始测试语句".to_string(), None, None, 2, 30, 10, 2.0, false, 3, 6.0)?;
    // let audio_path = "./assets/audio/example.wav";
    
    Ok(())
}

#[test]
fn voxcpm_tokenizer() -> Result<()> {
    let model_path = "/home/jhq/huggingface_model/openbmb/VoxCPM-0.5B/";
    let tokenizer = SingleChineseTokenizer::new(model_path)?;
    let ids = tokenizer.encode("你好啊，你吃饭了吗".to_string())?;
    println!("ids: {:?}", ids);
    Ok(())
}