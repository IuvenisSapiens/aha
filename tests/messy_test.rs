use aha::utils::tensor_utils::bitor_tensor;
use anyhow::Result;
use candle_core::Tensor;

#[test]
fn messy_test() -> Result<()> {
    let device = &candle_core::Device::Cpu;
    let image_mask = Tensor::new(vec![0u32, 0, 0, 1, 0, 1], device)?;
    let video_mask = Tensor::new(vec![0u32, 1, 0, 1, 0, 1], device)?;
    let visual_mask = bitor_tensor(&image_mask, &video_mask)?;
    println!("visual_mask: {}", visual_mask);
    // let x = Tensor::arange_step(0.0_f32, 5., 0.5, &device)?;
    // let x_int = x.to_dtype(candle_core::DType::U32)?;
    // println!("x: {}", x);
    // println!("x_int: {}", x_int);
    // let x_affine = x_int.affine(1.0, 1.0)?;
    // println!("x_affine: {}", x_affine);
    // let x_clamp = x_affine.clamp(0u32, 3u32)?;
    // println!("x_clamp: {}", x_clamp);
    // let wav_path = "./assets/audio/voice_01.wav";
    // let audio_tensor = load_audio_with_resample(wav_path, device, Some(16000))?;
    // println!("audio_tensor: {}", audio_tensor);
    // let string = "你好啊".to_string();
    // let vec_str: Vec<String>= string.chars().map(|c| c.to_string()).collect();
    // println!("vec_str: {:?}", vec_str);
    // let t = Tensor::rand(-1.0, 1.0, (2, 2), &device)?;
    // println!("t: {}", t);
    // let re_t = t.recip()?;
    // println!("re_t: {}", re_t);
    Ok(())
}
