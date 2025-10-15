use std::io::Cursor;

use anyhow::{Result, anyhow};
use base64::{Engine, engine::general_purpose};
use image::{DynamicImage, ImageReader};

pub fn load_image_from_url(url: &str) -> Result<DynamicImage> {
    let response = reqwest::blocking::get(url)
        .map_err(|e| anyhow!(format!("Failed to fetch image from url: {}", e)))?;
    let bytes = response
        .bytes()
        .map_err(|e| anyhow!(format!("Failed to get image bytes: {}", e)))?;

    let cursor = Cursor::new(bytes);
    let img = ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(|e| anyhow!(format!("Failed to read image format: {}", e)))?
        .decode()
        .map_err(|e| anyhow!(format!("Failed to decode image: {}", e)))?;
    Ok(img)
}

pub fn load_image_from_base64(base64_data: &str) -> Result<DynamicImage> {
    let image_data = general_purpose::STANDARD
        .decode(base64_data)
        .map_err(|e| anyhow!(format!("Failed to decode image: {}", e)))?;
    let cursor = Cursor::new(image_data);
    let img = ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(|e| anyhow!(format!("Failed to read image format: {}", e)))?
        .decode()
        .map_err(|e| anyhow!(format!("Failed to decode image: {}", e)))?;
    Ok(img)
}

pub fn get_image(file: &str) -> Result<DynamicImage> {
    let mut img = None;
    if file.starts_with("http://") || file.starts_with("https://") {
        img = Some(load_image_from_url(file)?);
    }
    if file.starts_with("file://") {
        let mut path = file.to_owned();
        path = path.split_off(7);
        img = Some(
            ImageReader::open(path)
                .map_err(|e| anyhow!(format!("Failed to open file: {}", e)))?
                .decode()
                .map_err(|e| anyhow!(format!("Failed to decode image: {}", e)))?,
        );
    }
    if file.starts_with("data:image") && file.contains("base64,") {
        let data: Vec<&str> = file.split("base64,").collect();
        let data = data[1];
        img = Some(load_image_from_base64(data)?);
    }
    if let Some(img) = img {
        return Ok(img);
    }
    Err(anyhow!("get image from message failed".to_string()))
}
