use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::{Result, anyhow};
use minijinja::{Environment, Value as MiniJinjaValue, context};

use crate::utils::string_to_static_str;

pub fn get_template(path: String) -> Result<String> {
    let tokenizer_config_file = path.clone() + "/tokenizer_config.json";
    assert!(
        std::path::Path::new(&tokenizer_config_file).exists(),
        "tokenizer_config.json not exists in model path"
    );
    let tokenizer_config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(tokenizer_config_file)?)
            .map_err(|e| anyhow!(format!("load tokenizer_config file error:{}", e)))?;
    let chat_template = tokenizer_config["chat_template"]
        .as_str()
        .ok_or(anyhow!(format!("chat_template to str error")))?;
    // 修复模板中的问题行
    let fixed_template = chat_template
        .replace(
            "message.content.startswith('<tool_response>')",
            "message.content is startingwith('<tool_response>')", // 使用minijinja中的 is startingwith 替换
        )
        .replace(
            "message.content.endswith('</tool_response>')",
            "message.content is endingwith('</tool_response>')", // 使用minijinja中的 is endingwith 替换
        )
        .replace(
            "content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n')",
            "((content | split('</think>'))[0] | rstrip('\\n') | split('<think>'))[-1] | lstrip('\\n')", // 使用自定义的split, rstrip, lstrip过滤器替换
        )
        .replace(
            "content.split('</think>')[-1].lstrip('\\n')",
            "(content | split('</think>'))[-1] | lstrip('\\n')", // 使用自定义的过滤器替换
        )
        .replace(
            "reasoning_content.strip('\\n')",
            "reasoning_content | strip('\\n')", // 使用自定义的过滤器替换
        )
        .replace(
            "content.lstrip('\\n')",
            "content | lstrip('\\n')", // 使用自定义的过滤器替换
        );
    Ok(fixed_template)
}

pub struct ChatTemplate<'a> {
    env: Environment<'a>,
}

impl<'a> ChatTemplate<'a> {
    pub fn init(path: &str) -> Result<Self> {
        let path = path.to_string();
        assert!(
            std::path::Path::new(&path).exists(),
            "model path file not exists"
        );
        let template = get_template(path)?;
        let template = string_to_static_str(template);
        // 加载jinjaenv处理chat_template
        let mut env = Environment::new();
        // 添加自定义过滤器
        env.add_filter("tojson", |v: MiniJinjaValue| {
            serde_json::to_string(&v).unwrap()
        });

        env.add_filter("split", |s: String, delimiter: String| {
            s.split(&delimiter)
                .map(|s| s.to_string())
                .collect::<Vec<String>>()
        });

        // 添加 lstrip 过滤器
        env.add_filter("lstrip", |s: String, chars: Option<String>| match chars {
            Some(chars_str) => s.trim_start_matches(chars_str.as_str()).to_string(),
            None => s.trim_start().to_string(),
        });

        // 添加 rstrip 过滤器
        env.add_filter("rstrip", |s: String, chars: Option<String>| match chars {
            Some(chars_str) => s.trim_end_matches(chars_str.as_str()).to_string(),
            None => s.trim_end().to_string(),
        });
        // let template = get_template(path.to_string())?;
        let _ = env.add_template("chat", template);

        Ok(Self { env })
    }

    pub fn apply_chat_template(&self, messages: &ChatCompletionParameters) -> Result<String> {
        let context = context! {
            messages => &messages.messages,
            add_generation_prompt => true,
        };
        let template = self
            .env
            .get_template("chat")
            .map_err(|e| anyhow!(format!("render template error {}", e)))?;
        let message_str = template
            .render(context)
            .map_err(|e| anyhow!(format!("render template error {}", e)))?;
        Ok(message_str)
    }
}
