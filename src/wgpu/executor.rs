use std::borrow::Cow;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::Debug;

use uuid::Uuid;
use wgpu::util::DeviceExt;

use crate::session::Session;
use crate::traits::{Executor, TensorProps};
use crate::var::{TensorDataType, VarType};
use crate::wgpu::tensor::{create_staging_buf, GPUTensor};

pub struct GPUExecutor {
    tensors: HashMap<Uuid, GPUTensor>,
    staging_buf: HashMap<Uuid, wgpu::Buffer>,
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
        Self { tensors: Default::default(), staging_buf: Default::default() }
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

    async fn execute_inner(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, session: &Session) {
        let sorted_ids = session.sorted_ids();

        // Here we allocate necessary buffers before actually execute the graph
        for id in &sorted_ids {
            let var = &session.tensors.borrow()[id];
            let t = GPUTensor::from_var(var, device);
            self.tensors.insert(var.id, t);
        }

        // We also prepare staging buffers and ask them to retrieve data from storage buffer.
        // The staging buffers are provided only for terminal variables (variables with no outgoing
        // connection, where `nexts.len() == 0`)
        let terminal_node_ids: Vec<Uuid> = session.tensors.borrow()
            .iter()
            .filter(|(_, v)| v.nexts.borrow().len() == 0)
            .map(|(_, v)| v.id)
            .collect();

        for id in &terminal_node_ids {
            let var = &session.tensors.borrow()[id];
            let staging_buf = match var.dtype {
                TensorDataType::F32 => { create_staging_buf::<f32>(device, id.to_string().as_str(), &None, &var.shape) }
                TensorDataType::I32 => { create_staging_buf::<i32>(device, id.to_string().as_str(), &None, &var.shape) }
            };
            self.staging_buf.insert(*id, staging_buf);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Setup compute pass for each variable in topological order. We will skip leaf variables since
        // the has no operation and only hold values.
        for id in &sorted_ids {
            let var = &session.tensors.borrow()[id];
            if var.var_type == VarType::Leaf {
                continue;
            }

            let input_bufs: Vec<&GPUTensor> = var.prevs.iter().map(|id| &self.tensors[&id]).collect();
            let output_buf = &self.tensors[&var.id];

            // All buffers to bind (all input buffers AND output buffer), required by shaders
            let mut all_bufs = input_bufs;
            all_bufs.push(output_buf);

            let mut bind_idx = 0;
            let bind_group_entries: Vec<wgpu::BindGroupEntry> = all_bufs
                .iter()
                .map(|v| {
                    let entry = wgpu::BindGroupEntry { binding: bind_idx, resource: v.data.buffer.as_entire_binding() };
                    bind_idx += 1;
                    entry
                })
                .collect();

            let shader_module = device.create_shader_module(
                wgpu::ShaderModuleDescriptor {
                    label: Some("add_shader"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("wgsl/add.wgsl"))),
                }
            );
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
                entries: &bind_group_entries,
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&compute_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.insert_debug_marker("add_pass");

                // TODO: use proper workgroups!
                cpass.dispatch_workgroups(1, 1, 1);
            }
        }

        // copy buf of terminal variables to staging buffers
        for id in &terminal_node_ids {
            let out_buf = &self.tensors[id].data.buffer;
            let staging_buf = &self.staging_buf[id];
            encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, staging_buf.size());
        }

        // submit passes
        queue.submit(Some(encoder.finish()));

        let mut receivers = HashMap::new();
        for id in &terminal_node_ids {
            let staging_buf = &self.staging_buf[id];
            let staging_slice = staging_buf.slice(..);

            let (sender, receiver) = flume::bounded(1);
            staging_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

            receivers.insert(*id, receiver);
        }

        device.poll(wgpu::Maintain::Wait);

        for id in &terminal_node_ids {
            let staging_buf = &self.staging_buf[id];
            let staging_slice = staging_buf.slice(..);
            let receiver = &receivers[id];
            if let Ok(Ok(())) = receiver.recv_async().await {
                let data = staging_slice.get_mapped_range();
                // TODO: customize output type accordingly
                let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

                drop(data);
                staging_buf.unmap();

                println!("{:?}", result);

                Some(result)
            } else {
                panic!("Failed to run on GPU")
            };
        }
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
        let a = sess.new_tensor_var(TensorData::F32(vec![1., 2., 3.]), vec![3]).unwrap();
        let b = sess.new_tensor_var(TensorData::F32(vec![1., 2., 3.]), vec![3]).unwrap();
        let c = a.add(&b);
        let mut executor = GPUExecutor::new();

        executor.execute(&mut sess).unwrap();
    }
}
