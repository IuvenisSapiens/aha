use anyhow::{Ok, Result};
use candle_core::{D, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};
use std::{result::Result::Ok as StdOk};

pub struct CausalConv1d {
    conv1d: Conv1d,
    padding: usize,
}

impl CausalConv1d {
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        padding: usize,
        dilation: usize,
        groups: usize,
        stride: usize,
    ) -> Result<Self> {
        let config = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups,
            cudnn_fwd_algo: None,
        };

        let conv1d = Conv1d::new(weight, bias, config);
        Ok(Self { conv1d, padding })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_pad = x.pad_with_zeros(D::Minus1, self.padding * 2, 0)?;
        let x = self.conv1d.forward(&x_pad)?;
        Ok(x)
    }
}

pub struct CausalConvTranspose1d {
    conv_transpose1d: ConvTranspose1d,
    padding: usize,
    output_padding: usize,
}

// 元素间：stride-1
// 两边： k-p-1
// (h-1)*s -2p+k
// (h+1)*s

impl CausalConvTranspose1d {
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        padding: usize,
        dilation: usize,
        output_padding: usize,
        groups: usize,
        stride: usize,
    ) -> Result<Self> {
        let config = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride,
            dilation,
            groups,
        };

        let conv_transpose1d = ConvTranspose1d::new(weight, bias, config.clone());
        Ok(Self {
            conv_transpose1d,
            padding,
            output_padding,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv_transpose1d.forward(x)?;
        let last_dim = x.dim(D::Minus1)?;
        let select_num = last_dim - (self.padding * 2 - self.output_padding);
        let x = x.narrow(D::Minus1, 0, select_num)?;
        Ok(x)
    }
}

pub struct WNCausalConv1d {
    conv: CausalConv1d,
}
impl WNCausalConv1d {
    pub fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        stride: usize,
    ) -> Result<Self> {
        let in_c = in_c / groups;        
        let weight_g = vb.get((out_c, 1, 1), "weight_g")?;
        let weight_v = vb.get((out_c, in_c, kernel_size), "weight_v")?;
        let bias = match vb.get(out_c, "bias") {
            StdOk(b) => Some(b),
            Err(_) => None,
        };
        let weight_norm = weight_v.sqr()?.sum_keepdim(1)?.sum_keepdim(2)?.sqrt()?;
        let normalized_weight = weight_v.broadcast_div(&weight_norm)?;
        let scaled_weight = normalized_weight.broadcast_mul(&weight_g)?;
        let conv = CausalConv1d::new(scaled_weight, bias, padding, dilation, groups, stride)?;
        Ok(Self { conv })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        Ok(x)
    }
}

pub struct WNCausalConvTranspose1d {
    conv_transpose: CausalConvTranspose1d,
}

impl WNCausalConvTranspose1d {
    pub fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        dilation: usize,
        kernel_size: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
        stride: usize,
    ) -> Result<Self> {
        let in_c = in_c / groups;
        let weight_g = vb.get((in_c, 1, 1), "weight_g")?;
        let weight_v = vb.get((in_c, out_c, kernel_size), "weight_v")?;
        let bias = match vb.get(out_c, "bias") {
            StdOk(b) => Some(b),
            Err(_) => None,
        };
        let weight_norm = weight_v.sqr()?.sum_keepdim(1)?.sum_keepdim(2)?.sqrt()?;
        let normalized_weight = weight_v.broadcast_div(&weight_norm)?;
        let scaled_weight = normalized_weight.broadcast_mul(&weight_g)?;
        let conv_transpose = CausalConvTranspose1d::new(
            scaled_weight,
            bias,
            padding,
            dilation,
            output_padding,
            groups,
            stride,
        )?;
        Ok(Self { conv_transpose })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv_transpose.forward(x)?;
        Ok(x)
    }
}

pub struct Snake1d {
    alpha: Tensor,
}
impl Snake1d {
    pub fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        Ok(Self { alpha })
    }

    // x + sin(alpha*x)^2 / alpha
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let x = x.reshape((dims[0], dims[1], ()))?;
        let alpha_ = self.alpha.affine(1.0, 1e-9)?.recip()?;
        let alpha_ = x
            .broadcast_mul(&self.alpha)?
            .sin()?
            .powf(2.0)?
            .broadcast_mul(&alpha_)?;
        let x = x.add(&alpha_)?;
        let x = x.reshape(dims)?;
        Ok(x)
    }
}

pub struct CausalResidualUnit {
    // pad: usize,
    block0: Snake1d,
    block1: WNCausalConv1d,
    block2: Snake1d,
    block3: WNCausalConv1d,
}

impl CausalResidualUnit {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        dilation: usize,
        kernel: usize,
        groups: usize,
    ) -> Result<Self> {
        let pad = ((kernel - 1) * dilation) / 2;
        let block0 = Snake1d::new(vb.pp("block.0"), dim)?;
        let block1 =
            WNCausalConv1d::new(vb.pp("block.1"), dim, dim, kernel, dilation, pad, groups, 1)?;
        let block2 = Snake1d::new(vb.pp("block.2"), dim)?;
        let block3 = WNCausalConv1d::new(vb.pp("block.3"), dim, dim, 1, 1, 0, 1, 1)?;
        Ok(Self {
            // pad,
            block0,
            block1,
            block2,
            block3,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // let orig_dim = x.dims();
        let last_dim_x = x.dim(D::Minus1)?;
        let mut res_x = x.clone();
        let y = self.block0.forward(x)?;
        let y = self.block1.forward(&y)?;
        let y = self.block2.forward(&y)?;
        let y = self.block3.forward(&y)?;
        // let dim = y.dims();
        let last_dim_y = y.dim(D::Minus1)?;
        let pad = (last_dim_x - last_dim_y) / 2;
        if pad > 0 {
            res_x = res_x.narrow(D::Minus1, pad, last_dim_y)?;
        }
        let x = y.add(&res_x)?;
        Ok(x)
    }
}

pub struct CausalEncoderBlock {
    block0: CausalResidualUnit,
    block1: CausalResidualUnit,
    block2: CausalResidualUnit,
    block3: Snake1d,
    block4: WNCausalConv1d,
}

impl CausalEncoderBlock {
    pub fn new(
        vb: VarBuilder,
        in_dim: Option<usize>,
        out_dim: usize,
        stride: usize,
        groups: usize,
    ) -> Result<Self> {
        let in_dim = match in_dim {
            Some(d) => d,
            None => out_dim / 2,
        };
        let block0 = CausalResidualUnit::new(vb.pp("block.0"), in_dim, 1, 7, groups)?;
        let block1 = CausalResidualUnit::new(vb.pp("block.1"), in_dim, 3, 7, groups)?;
        let block2 = CausalResidualUnit::new(vb.pp("block.2"), in_dim, 9, 7, groups)?;
        let block3 = Snake1d::new(vb.pp("block.3"), in_dim)?;
        let padding = (stride as f32 / 2.0).ceil() as usize;
        let block4 = WNCausalConv1d::new(
            vb.pp("block.4"),
            in_dim,
            out_dim,
            2 * stride,
            1,
            padding,
            1,
            stride,
        )?;
        Ok(Self {
            block0,
            block1,
            block2,
            block3,
            block4,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.block0.forward(x)?;
        let x = self.block1.forward(&x)?;
        let x = self.block2.forward(&x)?;
        let x = self.block3.forward(&x)?;
        let x = self.block4.forward(&x)?;
        Ok(x)
    }
}

pub struct CausalEncoder {
    block0: WNCausalConv1d,
    block1_4: Vec<CausalEncoderBlock>,
    fc_mu: WNCausalConv1d,
    fc_logvar: WNCausalConv1d,
}

impl CausalEncoder {
    pub fn new(
        vb: VarBuilder,
        d_model: usize,
        laten_dim: usize,
        strides: Vec<usize>,
        depthwise: bool,
    ) -> Result<Self> {
        let mut d_model = d_model;
        let mut groups = 1;
        let block0 = WNCausalConv1d::new(vb.pp("block.0"), 1, d_model, 7, 1, 3, 1, 1)?;
        let vb_block = vb.pp("block");
        let mut block1_4 = Vec::new();
        for (i, stride) in strides.iter().enumerate() {
            d_model *= 2;
            groups = if depthwise { d_model / 2 } else { 1 };
            let block_i = CausalEncoderBlock::new(vb_block.pp(i+1), None, d_model, *stride, groups)?;
            block1_4.push(block_i);
        }
        let fc_mu = WNCausalConv1d::new(vb.pp("fc_mu"), d_model, laten_dim, 3, 1, 1, 1, 1)?;
        let fc_logvar = WNCausalConv1d::new(vb.pp("fc_logvar"), d_model, laten_dim, 3, 1, 1, 1, 1)?;
        Ok(Self {
            block0,
            block1_4,
            fc_mu,
            fc_logvar,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let mut hidden_state = self.block0.forward(x)?;
        for block_i in &self.block1_4 {
            hidden_state = block_i.forward(&hidden_state)?;
        }
        let mu = self.fc_mu.forward(&hidden_state)?;
        let logvar = self.fc_logvar.forward(&hidden_state)?;
        Ok((hidden_state, mu, logvar))
    }
}

pub struct NoiseBlock {
    linear: WNCausalConv1d,
}

impl NoiseBlock {
    pub fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        let linear = WNCausalConv1d::new(vb.pp("linear"), dim, dim, 1, 1, 0, 1, 1)?;
        Ok(Self { linear })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (bs, _, t) = x.dims3()?;
        let noise = Tensor::randn(0.0_f32, 1.0, (bs, 1, t), x.device())?.to_dtype(x.dtype())?;
        let h = self.linear.forward(x)?;
        let n = h.broadcast_mul(&noise)?;
        let x = x.add(&n)?;
        Ok(x)
    }
}

pub struct CausalDecoderBlock {
    block0: Snake1d,
    block1: WNCausalConvTranspose1d,
    block2: CausalResidualUnit,
    block3: CausalResidualUnit,
    block4: CausalResidualUnit,
}

impl CausalDecoderBlock {
    pub fn new(
        vb: VarBuilder,
        input_dim: usize,
        output_dim: usize,
        stride: usize,
        groups: usize,
    ) -> Result<Self> {
        let block0 = Snake1d::new(vb.pp("block.0"), input_dim)?;
        let padding = (stride as f32 / 2.0).ceil() as usize;
        let block1 = WNCausalConvTranspose1d::new(
            vb.pp("block.1"),
            input_dim,
            output_dim,
            1,
            2 * stride,
            padding,
            stride % 2,
            1,
            stride,
        )?;
        let block2 = CausalResidualUnit::new(vb.pp("block.2"), output_dim, 1, 7, groups)?;
        let block3 = CausalResidualUnit::new(vb.pp("block.3"), output_dim, 3, 7, groups)?;
        let block4 = CausalResidualUnit::new(vb.pp("block.4"), output_dim, 9, 7, groups)?;
        Ok(Self {
            block0,
            block1,
            block2,
            block3,
            block4,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.block0.forward(x)?;
        let x = self.block1.forward(&x)?;
        let x = self.block2.forward(&x)?;
        let x = self.block3.forward(&x)?;
        let x = self.block4.forward(&x)?;
        Ok(x)
    }
}

pub struct CausalDecoder {
    model0: WNCausalConv1d,
    model1: WNCausalConv1d,
    model2_5: Vec<CausalDecoderBlock>,
    model6: Snake1d,
    model7: WNCausalConv1d,
}

impl CausalDecoder {
    pub fn new(
        vb: VarBuilder,
        input_channel: usize,
        channels: usize,
        rates: Vec<usize>,
        d_out: usize,
    ) -> Result<Self> {
        let model0 = WNCausalConv1d::new(
            vb.pp("model.0"),
            input_channel,
            input_channel,
            7,
            1,
            3,
            input_channel,
            1,
        )?;
        let model1 = WNCausalConv1d::new(vb.pp("model.1"), input_channel, channels, 1, 1, 0, 1, 1)?;
        let vb_model = vb.pp("model");
        let mut output_dim = channels;
        let mut model2_5 = Vec::new();
        for (i, stride) in rates.iter().enumerate() {
            let input_dim = channels / 2_usize.pow(i as u32);
            output_dim = channels / 2_usize.pow((i + 1) as u32);
            let groups = output_dim;
            let model_i = CausalDecoderBlock::new(
                vb_model.pp(i + 2),
                input_dim,
                output_dim,
                *stride,
                groups,
            )?;
            model2_5.push(model_i);
        }
        let model6 = Snake1d::new(vb.pp("model.6"), output_dim)?;
        let model7 = WNCausalConv1d::new(vb.pp("model.7"), output_dim, d_out, 7, 1, 3, 1, 1)?;
        Ok(Self {
            model0,
            model1,
            model2_5,
            model6,
            model7,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {   
        let x = self.model0.forward(x)?;  
        let mut x = self.model1.forward(&x)?;
        for model_i in &self.model2_5 {
            x = model_i.forward(&x)?;
        }
        let x = self.model6.forward(&x)?;
        let x = self.model7.forward(&x)?;
        let x = x.tanh()?;
        Ok(x)
    }
}

pub struct AudioVAE {
    // encoder_dim: usize,
    // encoder_rates: Vec<usize>,
    // decoder_dim: usize,
    // decoder_rates: Vec<usize>,
    pub latent_dim: usize,
    hop_length: usize,
    encoder: CausalEncoder,
    decoder: CausalDecoder,
    pub sample_rate: usize,
    pub chunk_size: usize,
}

impl AudioVAE {
    pub fn new(
        vb: VarBuilder,
        encoder_dim: usize,
        encoder_rates: Vec<usize>,
        laten_dim: Option<usize>,
        decoder_dim: usize,
        decoder_rates: Vec<usize>,
        sample_rate: usize,
    ) -> Result<Self> {
        let latent_dim = match laten_dim {
            Some(d) => d,
            None => encoder_dim * (2_usize.pow(encoder_rates.len() as u32)),
        };
        let hop_length = encoder_rates.iter().product();
        let encoder = CausalEncoder::new(
            vb.pp("encoder"),
            encoder_dim,
            latent_dim,
            encoder_rates.clone(),
            true,
        )?;
        let decoder = CausalDecoder::new(
            vb.pp("decoder"),
            latent_dim,
            decoder_dim,
            decoder_rates.clone(),
            1,
        )?;
        let chunk_size = hop_length;
        Ok(Self {
            // encoder_dim,
            // encoder_rates,
            // decoder_dim,
            // decoder_rates,
            latent_dim,
            hop_length,
            encoder,
            decoder,
            sample_rate,
            chunk_size,
        })
    }

    pub fn preprocess(&self, audio_data: &Tensor, sample_rate: Option<usize>) -> Result<Tensor>{
        let sample_rate = match sample_rate {
            Some(r) => r,
            None => self.sample_rate
        };
        assert_eq!(sample_rate, self.sample_rate);
        let pad_to = self.hop_length;
        let length = audio_data.dim(D::Minus1)?;
        let right_pad = (length as f32 / pad_to as f32).ceil() as usize * pad_to - length;
        let audio_data = audio_data.pad_with_zeros(D::Minus1, 0, right_pad)?;
        Ok(audio_data)
    }

    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let x = self.decoder.forward(z)?;
        Ok(x)
    }

    pub fn encode(&self, audio_data: &Tensor, sample_rate: Option<usize>) -> Result<Tensor> {
        let audio_data = match audio_data.rank() {
            2 => audio_data.unsqueeze(1)?,
            _ => audio_data.clone()
        };
        let audio_data = self.preprocess(&audio_data, sample_rate)?;
        let (_, mu, _) = self.encoder.forward(&audio_data)?;
        Ok(mu)
    }
}
