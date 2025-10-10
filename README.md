# aha 
一个基于 Candle 框架的 Rust 模型推理库，提供高效、易用的多模态模型推理能力。

## 特性
* 🚀 高性能推理 - 基于 Candle 框架，提供高效的张量计算和模型推理
* 🎯 多模型支持 - 集成视觉、语言和多模态模型
* 🔧 易于使用 - 简洁的 API 设计，快速上手
* 🛡️ 内存安全 - 得益于 Rust 的所有权系统，确保内存安全
* 📦 轻量级 - 最小化依赖，编译产物小巧
* ⚡ GPU 加速 - 可选 CUDA 支持
* 🧠 注意力优化 - 可选 Flash Attention 支持，优化长序列处理

## 支持的模型
### 当前已实现
* Qwen2.5VL - 阿里通义千问 2.5 多模态大语言模型
* MiniCPM4 - 面壁智能 MiniCPM 系列语言模型
* VoxCPM - 面壁智能语音生成模型

## 计划支持
我们持续扩展支持的模型列表，欢迎贡献！

## 安装
### 作为库使用
* cargo add aha
* 或者在Cargo.toml中添加
```toml
[dependencies]
aha = { git = "https://github.com/jhqxxx/aha.git" }

# 启用 CUDA 支持（可选）
candle-inference = { git = "https://github.com/jhqxxx/aha.git", features = ["cuda"] }

# 启用Flash Attention 支持（可选）
candle-inference = { git = "https://github.com/your-username/your-repo.git", features = ["cuda", "flash-attn"] }
```

### 从源码构建运行测试
```bash
git clone https://github.com/jhqxxx/aha.git
cd aha
# 修改测试用例中模型路径
# 运行 Qwen2.5VL 示例
cargo test -F cuda,flash-attn qwen2_5vl_generate -- --nocapture

# 运行 MiniCPM4 示例  
cargo test -F cuda,flash-attn minicpm_generate -- --nocapture

# 运行 VoxCPM 示例
cargo test -F cuda,flash-attn voxcpm_generate -- --nocapture
```

## 开发
### 项目结构
```text
.
├── Cargo.toml
├── README.md
├── src
│   ├── chat_template
│   ├── models
│   │   ├── common
│   │   ├── minicpm4
│   │   ├── qwen2_5vl
│   │   ├── voxcpm
│   │   └── mod.rs
│   ├── position_embed
│   ├── tokenizer
│   ├── utils
│   └── lib.rs
└── tests
    ├── test_minicpm4.rs
    ├── test_qwen2_5vl.rs
    └── test_voxcpm.rs
```

### 添加新模型
* 在src/models/创建新模型文件
* 在src/models/mod.rs中导出
* 在tests/中添加测试和示例

## 许可证
本项目采用 Apache License, Version 2.0 许可证 - 查看 [LICENSE](./LICENSE) 文件了解详情。

## 致谢
* [Candle](https://github.com/huggingface/candle) - 优秀的 Rust 机器学习框架
* 所有模型的原作者和贡献者

## 支持
#### 如果你遇到问题：
1. 查看 Issues 是否已有解决方案
2. 提交新的 Issue，包含详细描述和复现步骤

## 更新日志
### v0.1.0
* 初始版本发布
* 支持 Qwen2.5VL, MiniCPM4, VoxCPM 模型


⭐ 如果这个项目对你有帮助，请给我们一个 Star！