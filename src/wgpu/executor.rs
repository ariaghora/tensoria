use std::borrow::Cow;
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

use ndarray::{ArrayD, IxDyn};
use uuid::Uuid;
use wgpu::BufferUsages;
use wgpu::util::DeviceExt;

use crate::cpu::executor::CPUTensor;
use crate::session::Session;
use crate::traits::{Executor, TensorData};
use crate::var::{TensorDataType, Variable};

#[derive(Debug)]
pub enum GPUTensorData {
    F32(ndarray::ArrayD<f32>),
}

impl TensorData for GPUTensorData {
    fn shape(&self) -> Vec<usize> {
        match self {
            GPUTensorData::F32(val) => { val.shape().to_vec() }
        }
    }

    fn dtype(&self) -> TensorDataType {
        match self {
            GPUTensorData::F32(..) => { TensorDataType::F32 }
        }
    }
}

#[derive(Debug)]
pub struct GPUTensor {
    pub data: GPUTensorData,
    pub grad: Option<GPUTensorData>,
    pub requires_grad: bool,
}

impl GPUTensor {
    pub fn from_var(var: &Arc<Variable>) -> Self {
        match &var.tensor_data {
            Some(data) => {
                match data.dtype() {
                    TensorDataType::F32 => {
                        GPUTensor {
                            data: GPUTensorData::F32(ArrayD::from_shape_vec(IxDyn(var.shape.as_slice()), data.get_data_f32().clone()).unwrap().into_dyn()),
                            grad: None,
                            requires_grad: false,
                        }
                    }
                    _ => panic!("Should be f32 :(")
                }
            }
            None => panic!("Cannot build GPUTensor from None")
        }
    }

    pub fn new(data: GPUTensorData, requires_grad: bool) -> GPUTensor {
        let zero_grad = |shape, requires_grad: bool| {
            if requires_grad {
                Some(GPUTensorData::F32(ArrayD::zeros(shape)))
            } else {
                None
            }
        };

        // Function to handle addition and return a GPUTensor
        let add_tensors = |data: GPUTensorData, requires_grad: bool| {
            let shape = data.shape().clone();
            GPUTensor {
                data,
                grad: zero_grad(shape, requires_grad),
                requires_grad,
            }
        };
        add_tensors(data, requires_grad)
    }
}

impl GPUTensor {
    pub fn add(&self, other: &GPUTensor) -> GPUTensor {
        let requires_grad = self.requires_grad || other.requires_grad;
        match (&self.data, &other.data) {
            (GPUTensorData::F32(a), GPUTensorData::F32(b)) => {
                GPUTensor::new(GPUTensorData::F32(a + b), requires_grad)
            }
            _ => unimplemented!(),
        }
    }
}

pub struct GPUExecutor {
    tensors: HashMap<Uuid, CPUTensor>,
}

impl Executor for GPUExecutor {
    fn execute(&mut self, session: &Session) -> Result<(), Box<dyn Error>> {
        let (device, queue) = pollster::block_on(self.create_device());
        pollster::block_on(self.execute_gpu_inner(&device, &queue));
        Ok(())
    }
}

impl GPUExecutor {
    async fn create_device(&self) -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await.unwrap();

        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = 256 << 20;

        let features = adapter.features();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: features & wgpu::Features::TIMESTAMP_QUERY & wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
                    limits,
                },
                None,
            ).await.unwrap();
        (device, queue)
    }

    async fn execute_gpu_inner(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let shader_module = device.create_shader_module(
            wgpu::ShaderModuleDescriptor {
                label: Some("add_shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("wgsl/add.wgsl"))),
            }
        );

        let input_0: Vec<f32> = vec![1.0, 2.0, 3.0];
        let input_1: Vec<f32> = vec![11.0, 22.0, 33.0];
        let output_0: Vec<f32> = vec![0.0; 3];

        let output_0_staging_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("output_0"),
            contents: bytemuck::cast_slice(&output_0),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        });

        let input_0_storage_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input_0"),
            contents: bytemuck::cast_slice(&input_0),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        let input_1_storage_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input_1"),
            contents: bytemuck::cast_slice(&input_1),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        let output_0_storage_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("output_0"),
            contents: bytemuck::cast_slice(&output_0),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader_module,
            entry_point: "main",
        });

        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_0_storage_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input_1_storage_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_0_storage_buf.as_entire_binding() }
            ],
        });

        // LOOP 1
        // encoder executes multiple pipelines (each op has one pipeline)
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.insert_debug_marker("add_pass");
            cpass.dispatch_workgroups(output_0.len() as u32, 1, 1);
        }
        // END OF LOOP 1

        // LOOP 2
        // Copy buffer (associated to each terminal ops) to its corresponding staging buffer
        encoder.copy_buffer_to_buffer(&output_0_storage_buf, 0, &output_0_staging_buf, 0, output_0_staging_buf.size());
        // END OF LOOP 2

        queue.submit(Some(encoder.finish()));

        // LOOP 3
        let output_0_staging_slice = output_0_staging_buf.slice(..);
        let (sender, receiver) = flume::bounded(1);
        output_0_staging_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        // END OF LOOP 2

        device.poll(wgpu::Maintain::Wait);

        // LOOP 4: for each output staging
        if let Ok(Ok(())) = receiver.recv_async().await {
            let data = output_0_staging_slice.get_mapped_range();
            // TODO: customize output type accordingly
            let output_0_result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

            drop(data);
            output_0_staging_buf.unmap();

            println!("{:?}", output_0_result);

            Some(output_0_result)
        } else {
            panic!("Failed to run on GPU")
        };
        // END OF LOOP 4
    }
}

#[cfg(test)]
mod test {
    use crate::session::Session;
    use crate::traits::Executor;
    use crate::wgpu::executor::GPUExecutor;

    #[test]
    fn add() {
        let mut sess = Session::new();
        let mut executor = GPUExecutor { tensors: Default::default() };

        executor.execute(&mut sess).unwrap();
    }
}
