use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor, shape::Dim};

pub fn split(t: &Tensor, splits: &[usize], dim: D) -> Result<Vec<Tensor>> {
    let dim = dim.to_index(t.shape(), "split")?;
    let mut split_res = Vec::new();
    let mut index = 0;
    for split in splits {
        split_res.push(t.narrow(dim, index, *split)?);
        index += *split;
    }
    Ok(split_res)
}

pub fn safe_arg_sort_last_dim(t: &Tensor, ascending: bool) -> Result<Tensor> {
    // tensor在GPU上时，维度超过1024， arg_sort_last_dim方法会报错
    // 所以维度大于1024时，放到CPU上处理
    let last_dim = t.dims()[t.rank() - 1];
    if last_dim <= 1024 {
        let t = t.arg_sort_last_dim(ascending)?;
        Ok(t)
    } else {
        let cpu_tensor = t.to_device(&Device::Cpu)?;
        let sorted_indices = cpu_tensor.arg_sort_last_dim(ascending)?;
        let t = sorted_indices.to_device(t.device())?;
        Ok(t)
    }
}

pub fn nonzero_index_vec(mask: &Tensor) -> Result<Vec<u32>> {
    // 根据mask矩阵选出其中不为0的元素所在索引, 返回vec
    // 只能处理1维数据
    let mut mask = mask.clone();
    if mask.dtype() != DType::U32 {
        mask = mask.to_dtype(DType::U32)?;
    }
    match mask.rank() {
        0 => {
            return Err(anyhow!(format!(
                "input rank must > 0, the input tensor rank: {}",
                mask.rank()
            )));
        }
        1 => {
            let mask_vector = mask.to_vec1::<u32>()?;
            let indices: Vec<u32> = mask_vector
                .iter()
                .enumerate()
                .filter_map(|(idx, &val)| if val != 0 { Some(idx as u32) } else { None })
                .collect();
            Ok(indices)
        }
        _ => {
            return Err(anyhow!(format!(
                "input rank not support, the input tensor rank: {}",
                mask.rank()
            )));
        }
    }
}

pub fn nonzero_index(mask: &Tensor) -> Result<Tensor> {
    // 根据mask矩阵选出其中不为1的元素所在索引, 返回Tensor
    let indices_tensor = match mask.rank() {
        0 => {
            return Err(anyhow!(format!(
                "input rank must > 0, the input tensor rank: {}",
                mask.rank()
            )));
        }
        1 => {
            let index_vec = nonzero_index_vec(mask)?;
            let indices_tensor = Tensor::from_slice(&index_vec, index_vec.len(), mask.device())?;
            indices_tensor
        }
        _ => {
            return Err(anyhow!(format!(
                "input rank must == 1, the input tensor rank: {}",
                mask.rank()
            )));
        }
    };
    Ok(indices_tensor)
}

pub fn zero_index_vec(mask: &Tensor) -> Result<Vec<u32>> {
    // 根据mask矩阵选出其中为0的元素所在索引, 返回vec
    // 只能处理1维数据
    let mut mask = mask.clone();
    if mask.dtype() != DType::U32 {
        mask = mask.to_dtype(DType::U32)?;
    }
    match mask.rank() {
        0 => {
            return Err(anyhow!(format!(
                "input rank must > 0, the input tensor rank: {}",
                mask.rank()
            )));
        }
        1 => {
            let mask_vector = mask.to_vec1::<u32>()?;
            let indices: Vec<u32> = mask_vector
                .iter()
                .enumerate()
                .filter_map(|(idx, &val)| if val == 0 { Some(idx as u32) } else { None })
                .collect();
            Ok(indices)
        }
        _ => {
            return Err(anyhow!(format!(
                "input rank not support, the input tensor rank: {}",
                mask.rank()
            )));
        }
    }
}

pub fn zero_index(mask: &Tensor) -> Result<Tensor> {
    let index_vec = zero_index_vec(mask)?;
    let indices_tensor = Tensor::from_slice(&index_vec, index_vec.len(), mask.device())?;
    Ok(indices_tensor)
}

pub fn nonzero_slice(mask: &Tensor) -> Result<Vec<(usize, usize)>> {
    // 根据mask矩阵选出其中非0的元素所在索引
    // 根据索引获取连续索引间隔
    // 如不为零索引元素为[0, 3, 4, 5, 8, 9]
    // 间隔为: [(0, 1), (3, 6), (8, 10)]
    // 索引前闭后开
    let mut index_vec = nonzero_index_vec(mask)?;
    match index_vec.len() {
        0 => {
            return Ok(vec![]);
        }
        1 => {
            return Ok(vec![(index_vec[0] as usize, (index_vec[0] + 1) as usize)]);
        }
        _ => {
            let mut vec_slice = vec![];
            let mut start = index_vec.remove(0);
            let mut last = start;

            for i in index_vec {
                if i == (last + 1) {
                    last = i;
                    continue;
                } else {
                    vec_slice.push((start as usize, (last + 1) as usize));
                    start = i;
                    last = i;
                }
            }
            vec_slice.push((start as usize, (last + 1) as usize));
            Ok(vec_slice)
        }
    }
}

pub fn masked_scatter_dim0(original: &Tensor, replace: &Tensor, mask: &Tensor) -> Result<Tensor> {
    // 根据mask中非0元素所在索引,使用replace中的数据替换掉original中的数据
    // original: rank = 3: (bs, seq_len, hidden_dim)
    // replace: rank = 2: (seq_len, hidden_dim)
    // mask: rank = 2: (bs, seq_len)
    // 推理时bs=1,为了方便替换,将bs squeeze,替换后再unsqueeze
    // 按行替换
    if original.dim(0)? != 1 || mask.dim(0)? != 1 {
        return Err(anyhow!(format!(
            "masked_scatter_dim0 original bs: {} or mask bs :{} not equal to 1 ",
            original.dim(0)?,
            mask.dim(0)? != 1
        )));
    }
    let mut original = original.squeeze(0)?;
    let mask = mask.squeeze(0)?;
    let slices = nonzero_slice(&mask)?;
    let mut sub_start = 0usize;
    let mut sub_end = 0usize;
    for (start, end) in slices {
        sub_end = sub_start + (end - start);
        let sub_replace = replace.i((sub_start..sub_end, ..))?;
        original = original.slice_assign(&[(start..end), (0..original.dim(1)?)], &sub_replace)?;
        sub_start = sub_end;
    }
    original = original.unsqueeze(0)?;
    Ok(original)
}

pub fn get_equal_mask(input_ids: &Tensor, token_ids: u32) -> Result<Tensor> {
    let image_token_id_tensor = Tensor::new(vec![token_ids], input_ids.device())?;
    let mask = input_ids
        .broadcast_eq(&image_token_id_tensor)?
        .to_dtype(candle_core::DType::U32)?;
    Ok(mask)
}

pub fn get_vision_next_indices(input_ids: &Tensor, token_id: u32) -> Result<Tensor> {
    // input_ids -> shape: (seq_len)
    let mask = get_equal_mask(&input_ids, token_id)?;
    let indices = nonzero_index(&mask)?;
    let indices = indices.broadcast_add(&Tensor::new(vec![1u32], input_ids.device())?)?;
    Ok(indices)
}
