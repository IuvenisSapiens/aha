use aha::utils::audio_utils::{load_audio_with_resample};
use anyhow::Result;

#[test]
fn messy_test() -> Result<()> {
    let device = candle_core::Device::Cpu;
    let wav_path = "./assets/audio/example.wav";
    let audio_tensor = load_audio_with_resample(wav_path, device,Some(16000))?;
    
    println!("audio_tensor: {}", audio_tensor);
    // let string = "你好啊".to_string();
    // let vec_str: Vec<String>= string.chars().map(|c| c.to_string()).collect();
    // println!("vec_str: {:?}", vec_str);
    // let t = Tensor::rand(-1.0, 1.0, (2, 2), &device)?;
    // println!("t: {}", t);
    // let re_t = t.recip()?;
    // println!("re_t: {}", re_t);
    Ok(())
}
