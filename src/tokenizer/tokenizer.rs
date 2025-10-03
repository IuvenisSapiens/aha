use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

pub struct TokenizerModel {
    tokenizer: Tokenizer,
}

impl TokenizerModel {
    pub fn init(path: &str) -> Result<Self> {
        let path = path.to_string();
        assert!(
            std::path::Path::new(&path).exists(),
            "model path file not exists"
        );
        let tokenizer_file = path.clone() + "/tokenizer.json";
        assert!(
            std::path::Path::new(&tokenizer_file).exists(),
            "tokenizer.json not exists in model path"
        );
        let tokenizer = Tokenizer::from_file(tokenizer_file)
            .map_err(|e| anyhow!(format!("tokenizer from file error{}", e)))?;
        Ok(Self { tokenizer })
    }

    pub fn text_encode(&self, text: String, device: &Device) -> Result<Tensor> {
        let token_id = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!(format!("tokenizer encode error: {}", e)))?
            .get_ids()
            .to_vec();
        let token_tensor = Tensor::from_slice(&token_id, (1, token_id.len()), device)?;
        Ok(token_tensor)
    }

    pub fn token_decode(&self, tokens: Vec<u32>) -> Result<String> {
        let decode = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow!(format!("tokenizer encode error{}", e)))?;
        Ok(decode)
    }

}
