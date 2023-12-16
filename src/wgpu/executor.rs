use std::borrow::Cow;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::Debug;

use uuid::Uuid;
use wgpu::BufferUsages;
use wgpu::util::DeviceExt;

use crate::cpu::executor::CPUTensor;
use crate::session::Session;
use crate::traits::{Executor, TensorProps};
use crate::wgpu::tensor::GPUTensor;

pub struct GPUExecutor {
    tensors: HashMap<Uuid, CPUTensor>,
}

impl Executor for GPUExecutor {
    fn execute(&mut self, session: &Session) -> Result<(), Box<dyn Error>> {
        let (device, queue) = pollster::block_on(self.create_device());
        pollster::block_on(self.execute_inner(&device, &queue, session));
        Ok(())
    }
}


impl GPUExecutor {
    pub fn new() -> Self {
        Self { tensors: Default::default() }
    }

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

    async fn execute_inner(&self, device: &wgpu::Device, queue: &wgpu::Queue, session: &Session) {
        let sorted_ids = session.sorted_ids();

        // Here we allocate necessary buffers before actually execute the graph
        for id in &sorted_ids {
            let var = &session.tensors.borrow()[id];
            let t = GPUTensor::from_var(var, device);

            // let buf = match var.var_type {
            //     // leaf variable is guaranteed to have tensor data, enforced by initializer API
            //     VarType::Leaf => {
            //         let data = var.tensor_data.as_ref().unwrap();
            //         match data.dtype() {
            //             TensorDataType::F32 => { create_storage_buf(device, &var.id.to_string(), Some(data.get_data_f32()), &var.shape) }
            //             TensorDataType::I32 => { create_storage_buf(device, &var.id.to_string(), Some(data.get_data_i32()), &var.shape) }
            //         }
            //     }
            //     VarType::Add => {
            //         let left = &session.tensors.borrow()[&var.prevs[0]];
            //         let ltype = left.tensor_data.as_ref().unwrap().dtype();
            //         let right = &session.tensors.borrow()[&var.prevs[1]];
            //         let rtype = right.tensor_data.as_ref().unwrap().dtype();
            //         match (&ltype, &rtype) {
            //             (&TensorDataType::F32, &TensorDataType::F32) => { create_storage_buf::<f32>(device, &var.id.to_string(), None, &var.shape) }
            //             (_, _) => { unimplemented!("cannot allocate buffer for addition between {:?} and {:?}", ltype, rtype) }
            //         }
            //     }
            //     VarType::Sub => { todo!() }
            // };
        }
    }

    async fn execute_op(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
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
    use crate::var::TensorData;
    use crate::wgpu::executor::GPUExecutor;

    #[test]
    fn add() {
        let mut sess = Session::new();
        let a = sess.new_tensor_var(TensorData::F32(vec![1., 2.]), vec![2]).unwrap();
        let b = sess.new_tensor_var(TensorData::F32(vec![1., 2.]), vec![2]).unwrap();
        let c = a.add(&b);
        let mut executor = GPUExecutor::new();

        executor.execute(&mut sess).unwrap();
    }
}
