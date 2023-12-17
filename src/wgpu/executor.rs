use std::borrow::Cow;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::Debug;
use std::sync::Arc;

use flume::Receiver;
use include_dir::{Dir, include_dir};
use uuid::Uuid;
use wgpu::BufferAsyncError;
use wgpu::util::DeviceExt;

use crate::session::Session;
use crate::traits::{Executor, TensorProps};
use crate::var::{TensorData, TensorDataType, Variable, VarType};
use crate::wgpu::tensor::{create_staging_buf, GPUTensor};

static PROJECT_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/src/wgpu/wgsl/");


impl VarType {
    fn op_str<'a>(&self) -> &'a str {
        match self {
            VarType::Add => { "add" }
            VarType::Sub => { "sub" }
            VarType::MatMul => { "matmul" }
            VarType::Leaf => { "leaf" }
        }
    }
}

pub struct GPUExecutor {
    tensors: HashMap<Uuid, GPUTensor>,
    staging_buf: HashMap<Uuid, wgpu::Buffer>,
    receivers: HashMap<Uuid, Receiver<Result<(), BufferAsyncError>>>,
}

impl Executor for GPUExecutor {
    fn execute(&mut self, session: &Session) -> Result<(), Box<dyn Error>> {
        let (device, queue) = pollster::block_on(Self::create_device());
        pollster::block_on(self.alloc_bufs(&device, &queue, session));
        pollster::block_on(self.execute_inner(&device, &queue, session));
        Ok(())
    }
}


impl GPUExecutor {
    pub fn new() -> Self {
        let mut executor = Self {
            tensors: Default::default(),
            staging_buf: Default::default(),
            receivers: Default::default(),
        };
        executor
    }

    pub fn fetch(&self, var: Arc<Variable>) -> TensorData {
        pollster::block_on(self.fetch_async(var))
    }
    async fn fetch_async(&self, var: Arc<Variable>) -> TensorData {
        let staging_buf = &self.staging_buf[&var.id];
        let staging_slice = staging_buf.slice(..);
        let receiver = &self.receivers[&var.id];
        if let Ok(Ok(())) = receiver.recv_async().await {
            let data = staging_slice.get_mapped_range();

            let res = match var.dtype {
                TensorDataType::F32 => { TensorData::F32(bytemuck::cast_slice(&data).to_vec()) }
                TensorDataType::I32 => { unimplemented!() }
            };

            drop(data);
            staging_buf.unmap();

            return res;
        } else {
            panic!("Failed to run on GPU")
        };
    }

    async fn create_device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await.unwrap();

        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = 256 << 22;
        limits.max_buffer_size = 256 << 24;

        let features = adapter.features();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: features &
                        wgpu::Features::TIMESTAMP_QUERY &
                        wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
                    limits,
                },
                None,
            ).await.unwrap();
        (device, queue)
    }

    async fn alloc_bufs(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, session: &Session) {
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
        let terminal_node_ids = session.terminal_ids();
        for id in &terminal_node_ids {
            let var = &session.tensors.borrow()[id];
            let staging_buf = match var.dtype {
                TensorDataType::F32 => { create_staging_buf::<f32>(device, id.to_string().as_str(), &None, &var.shape) }
                TensorDataType::I32 => { create_staging_buf::<i32>(device, id.to_string().as_str(), &None, &var.shape) }
            };
            self.staging_buf.insert(*id, staging_buf);
        }
    }

    async fn execute_inner(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, session: &Session) {
        let sorted_ids = session.sorted_ids();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Setup compute pass for each variable in topological order. We will skip leaf variables since
        // the has no operation and only hold values.
        for id in &sorted_ids {
            let var = &session.tensors.borrow()[id];
            if var.var_type == VarType::Leaf {
                continue;
            }

            let input_tensors: Vec<&GPUTensor> = var.prevs.iter().map(|id| &self.tensors[&id]).collect();
            let current_tensor = &self.tensors[&var.id];

            // All buffers to bind (all input buffers AND output buffer), required by shaders
            let mut all_tensors = input_tensors;
            all_tensors.push(current_tensor);

            let mut bind_idx = 0;
            let bind_group_entries: Vec<wgpu::BindGroupEntry> = all_tensors
                .iter()
                .map(|v| {
                    let entry = wgpu::BindGroupEntry { binding: bind_idx, resource: v.data.buffer.as_entire_binding() };
                    bind_idx += 1;
                    entry
                })
                .collect();

            let op_str = var.var_type.op_str();
            let templ_str = PROJECT_DIR
                .get_file(format!("{}.wgsl", op_str))
                .unwrap()
                .contents_utf8()
                .unwrap();

            let mut templ = tera::Tera::default();
            templ.add_raw_template(op_str, templ_str).unwrap();

            let mut params = tera::Context::new();

            // Ask current tensor to setup shader template parameters
            current_tensor.executable_op.setup_shader(var.id, session, &mut params);

            let shader_src = templ.render(op_str, &params).unwrap();

            let shader_module = device.create_shader_module(
                wgpu::ShaderModuleDescriptor {
                    label: Some(format!("{}_shader", op_str).as_str()),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader_src)),
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

                let [x, y, z] = current_tensor.executable_op.workgroups(var.id, &session);
                cpass.dispatch_workgroups(x, y, z);
            }
        }

        let terminal_node_ids = session.terminal_ids();

        // copy buf of terminal variables to staging buffers
        for id in &terminal_node_ids {
            let out_buf = &self.tensors[id].data.buffer;
            let staging_buf = &self.staging_buf[id];
            encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, staging_buf.size());
        }

        // submit passes
        queue.submit(Some(encoder.finish()));

        // let mut receivers = HashMap::new();
        for id in &terminal_node_ids {
            let staging_buf = &self.staging_buf[id];
            let staging_slice = staging_buf.slice(..);

            let (sender, receiver) = flume::bounded(1);
            staging_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

            self.receivers.insert(*id, receiver);
        }

        device.poll(wgpu::Maintain::Wait);
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

    #[test]
    fn matmul() {
        let mut sess = Session::new();
        let a = sess.new_tensor_var(TensorData::F32(vec![1.0; 100000]), vec![1000, 100]).unwrap();
        let b = sess.new_tensor_var(TensorData::F32(vec![1.0; 100000]), vec![100, 1000]).unwrap();
        let c = a.matmul(&b);
        let mut executor = GPUExecutor::new();

        let tic = std::time::Instant::now();
        executor.execute(&mut sess).unwrap();

        if let TensorData::F32(val) = &executor.fetch(c) {
            assert!(val.iter().all(|v| *v == 100.0));
        } else {
            panic!("Result should be F32")
        }
    }
}
