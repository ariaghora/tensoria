use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::Arc;

use wgpu::Device;
use wgpu::util::DeviceExt;

use crate::traits::TensorProps;
use crate::var::{TensorDataType, Variable};

#[derive(Debug)]
pub struct GPUTensorData {
    dtype: TensorDataType,
    buffer: wgpu::Buffer,
    shape: Vec<usize>,
}

impl TensorProps for GPUTensorData {
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn dtype(&self) -> TensorDataType {
        self.dtype.clone()
    }
}

#[derive(Debug)]
pub struct GPUTensor {
    pub data: GPUTensorData,
    pub grad: Option<GPUTensorData>,
    pub requires_grad: bool,
}

pub fn create_storage_buf<'a, T: bytemuck::Pod + Default + Debug>(
    device: &wgpu::Device,
    buf_label: &str,
    values: Option<&'a Vec<T>>,
    shape: &Vec<usize>,
) -> wgpu::Buffer {
    let mut n_items = shape.iter().fold(1, |x, y| x * y) as usize;
    // TODO: proper handling on 0-sized dims or non-zero-length shape but containing 0-length dim
    if n_items == 0 {
        n_items = 1;
    }
    let vals: Cow<'a, Vec<T>> = match values {
        Some(v) => Cow::Borrowed(v),
        None => Cow::Owned(vec![T::default(); n_items]),
    };

    // Some models provides tensors with empty data, i.e., with shape [0]. WGPU does not
    // allow zero buffer binding, so we trick it by using a "dummy" buffer binding with
    // size of 4 (minimum allowed)
    let tensor_has_data = vals.len() > 0;
    let data = if tensor_has_data {
        // We create buffer initialized with tensor's original data
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(format!("{}.storage", buf_label).as_str()),
            contents: bytemuck::cast_slice(&vals),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    } else {
        // The dummy buffer
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("{}.storage", buf_label).as_str()),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    data
}

impl GPUTensor {
    pub fn from_var(var: &Arc<Variable>, device: &Device) -> Self {
        let data_buf = match &var.dtype {
            TensorDataType::F32 => {
                match &var.tensor_data {
                    None => { create_storage_buf::<f32>(device, &var.id.to_string(), None, &var.shape) }
                    Some(data) => { create_storage_buf(device, &var.id.to_string(), Some(data.get_data_f32()), &var.shape) }
                }
            }
            TensorDataType::I32 => {
                match &var.tensor_data {
                    None => { create_storage_buf::<i32>(device, &var.id.to_string(), None, &var.shape) }
                    Some(data) => { create_storage_buf(device, &var.id.to_string(), Some(data.get_data_f32()), &var.shape) }
                }
            }
        };

        GPUTensor {
            data: GPUTensorData {
                dtype: var.dtype.clone(),
                buffer: data_buf,
                shape: var.shape.clone(),
            },
            grad: if var.requires_grad {
                Some(GPUTensorData {
                    dtype: var.dtype.clone(),
                    buffer: match &var.dtype {
                        TensorDataType::F32 => { create_storage_buf::<f32>(device, &var.id.to_string(), None, &var.shape) }
                        TensorDataType::I32 => { create_storage_buf::<i32>(device, &var.id.to_string(), None, &var.shape) }
                    },
                    shape: var.shape.clone(),
                })
            } else { None },
            requires_grad: var.requires_grad,
        }
    }

    pub fn new(data: GPUTensorData, requires_grad: bool, device: &Device) -> GPUTensor {
        let data_shape = data.shape().clone();
        let dtype = data.dtype.clone();
        GPUTensor {
            data,
            grad: if requires_grad {
                Some(GPUTensorData {
                    dtype: TensorDataType::F32,
                    buffer: match dtype {
                        TensorDataType::F32 => { create_storage_buf::<f32>(device, "", None, &data_shape) }
                        TensorDataType::I32 => { create_storage_buf::<i32>(device, "", None, &data_shape) }
                    },
                    shape: data_shape.clone(),
                }
                )
            } else { None },
            requires_grad,
        }
    }
}

impl GPUTensor {
    pub fn add(&self, other: &GPUTensor) -> GPUTensor {
        todo!()
    }
}

