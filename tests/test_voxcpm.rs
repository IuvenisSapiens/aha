use anyhow::{Ok, Result};
use std::collections::HashMap;

use aha::{
    models::voxcpm::{
        audio_vae::AudioVAE, config::VoxCPMConfig, model::VoxCPMModel,
        tokenizer::SingleChineseTokenizer,
    },
    utils::{
        audio_utils::save_wav,
        utils::{find_type_files, get_device},
    },
};
use candle_core::pickle::read_all_with_key;
use candle_nn::VarBuilder;

#[test]
fn voxcpm_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda,flash-attn voxcpm_generate -- --nocapture
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
            // if k.contains("decoder.model.2.block.1") {
            //     println!("val: {}", v);
            // }
            dict_to_hashmap.insert(k, v);
        }
    }
    let vb = VarBuilder::from_tensors(dict_to_hashmap, dtype, &dev);
    let audio_vae = AudioVAE::new(
        vb,
        128,
        vec![2, 5, 8, 8],
        Some(64),
        1536,
        vec![8, 8, 5, 2],
        16000,
    )?;
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
    let generate = voxcpm.generate(
        "太阳当空照，花儿对我笑，小鸟说早早早".to_string(),
        Some("啥子小师叔，打狗还要看主人，你再要继续，我，就是你的对手".to_string()),
        Some("./assets/audio/voice_01.wav".to_string()),
        // Some("一定被灰太狼给吃了，我已经为他准备好了花圈了".to_string()),
        // Some("./assets/audio/voice_05.wav".to_string()),
        2,
        100,
        10,
        2.0,
        false,
        3,
        6.0,
    )?;
    let _ = save_wav(&generate, "voxcpm_init.wav")?;
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
