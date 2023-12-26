use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::error::Error;
use std::rc::Rc;
use std::sync::Arc;

use flume::Receiver;
use include_dir::{include_dir, Dir};
use uuid::Uuid;
use wgpu::{BufferAsyncError, Device, Queue};

use crate::session::Session;
use crate::traits::Executor;
use crate::var::{TensorData, TensorDataType, VarType, Variable};
use crate::wgpu::tensor::{create_staging_buf, create_storage_buf, GPUTensor, GPUTensorData};

static PROJECT_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/src/wgpu/wgsl/");

impl VarType {
    pub fn op_str<'a>(&self) -> &'a str {
        match self {
            VarType::Add => "add",
            VarType::Sub => "sub",
            VarType::MatMul => "matmul",
            VarType::Leaf => "leaf",
            VarType::Mul => "mul",
        }
    }
}

/// Represents a GPU executor that manages tensors, staging buffers, and receivers.
pub struct GPUExecutor {
    pub tensors: Rc<RefCell<HashMap<Uuid, GPUTensor>>>,
    staging_buf: Rc<RefCell<HashMap<Uuid, wgpu::Buffer>>>,
    grad_staging_buf: Rc<RefCell<HashMap<Uuid, wgpu::Buffer>>>,
    receivers: Rc<RefCell<HashMap<Uuid, Receiver<Result<(), BufferAsyncError>>>>>,
    device: Rc<Device>,
    queue: Rc<Queue>,
}

impl Executor for GPUExecutor {
    fn forward(&mut self, session: &Session) -> Result<(), Box<dyn Error>> {
        let (device, queue) = (&self.device, &self.queue);
        pollster::block_on(self.alloc_bufs(&device, session));
        pollster::block_on(self.execute_inner(&device, &queue, session));
        Ok(())
    }

    fn backward(&mut self, var: &Arc<Variable>, session: &Session) -> Result<(), Box<dyn Error>> {
        let (device, queue) = (&self.device, &self.queue);

        let num_elem = var.shape.iter().fold(1, |x, y| x * y);
        let ones_buf = match var.dtype {
            TensorDataType::F32 => create_storage_buf(
                &device,
                var.id.to_string().as_str(),
                Some(&vec![1.0; num_elem]),
                &var.shape,
            ),
            TensorDataType::I32 => {
                todo!()
            }
        };

        // set grad on tensor with var.id to all ones
        let ones = GPUTensorData {
            buffer: ones_buf,
            dtype: var.dtype.clone(),
            shape: var.shape.clone(),
        };
        if let Some(tensor) = self.tensors.borrow_mut().get_mut(&var.id) {
            tensor.grad = Some(ones);
        } else {
            if !var.requires_grad {
                panic!("Calling backward on tensor not requiring grad");
            }
        }

        pollster::block_on(self.backward_inner(&var, &device, &queue, &session));
        Ok(())
    }
}

impl GPUExecutor {
    pub fn new() -> Self {
        let (device, queue) = pollster::block_on(Self::create_device());
        let executor = Self {
            tensors: Default::default(),
            staging_buf: Default::default(),
            grad_staging_buf: Default::default(),
            receivers: Default::default(),
            device: Rc::new(device),
            queue: Rc::new(queue),
        };
        executor
    }

    pub fn fetch(&self, var: Arc<Variable>) -> TensorData {
        pollster::block_on(self.fetch_async(var))
    }
    async fn fetch_async(&self, var: Arc<Variable>) -> TensorData {
        let staging_bufs = self.staging_buf.borrow();
        let receivers = self.receivers.borrow();
        let staging_buf = &staging_bufs[&var.id];
        let staging_slice = staging_buf.slice(..);
        let receiver = &receivers[&var.id];
        if let Ok(Ok(())) = receiver.recv_async().await {
            let data = staging_slice.get_mapped_range();

            let res = match var.dtype {
                TensorDataType::F32 => TensorData::F32(bytemuck::cast_slice(&data).to_vec()),
                TensorDataType::I32 => {
                    unimplemented!()
                }
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
            .await
            .unwrap();

        let mut limits = wgpu::Limits::downlevel_defaults();
        // limits.max_storage_buffer_binding_size = 256 << 22;
        // limits.max_buffer_size = 2147483647;
        limits.max_texture_dimension_1d = 4096;
        limits.max_texture_dimension_2d = 4096;

        let features = adapter.features();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: features, // &
                    // wgpu::Features::TIMESTAMP_QUERY &
                    // wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
                    limits,
                },
                None,
            )
            .await
            .unwrap();
        (device, queue)
    }

    async fn alloc_bufs(&self, device: &wgpu::Device, session: &Session) {
        let sorted_ids = session.sorted_ids();

        let mut tensors = self.tensors.borrow_mut();
        // Here we allocate necessary buffers before actually execute the graph
        for id in &sorted_ids {
            let var = &session.variables.borrow()[id];
            let t = GPUTensor::from_var(var, device);
            tensors.insert(var.id, t);

            // We also prepare staging buffers and ask them to retrieve data from storage buffer.
            let staging_buf = match var.dtype {
                TensorDataType::F32 => {
                    create_staging_buf::<f32>(device, id.to_string().as_str(), &None, &var.shape)
                }
                TensorDataType::I32 => {
                    create_staging_buf::<i32>(device, id.to_string().as_str(), &None, &var.shape)
                }
            };
            self.staging_buf.borrow_mut().insert(*id, staging_buf);

            if var.requires_grad {
                let grad_staging_buf = match var.dtype {
                    TensorDataType::F32 => create_staging_buf::<f32>(
                        device,
                        id.to_string().as_str(),
                        &None,
                        &var.shape,
                    ),
                    TensorDataType::I32 => create_staging_buf::<i32>(
                        device,
                        id.to_string().as_str(),
                        &None,
                        &var.shape,
                    ),
                };
                self.grad_staging_buf
                    .borrow_mut()
                    .insert(*id, grad_staging_buf);
            }
        }
    }

    async fn backward_inner(
        &self,
        var: &Variable,
        _device: &wgpu::Device,
        _: &wgpu::Queue,
        session: &Session,
    ) {
        let tensors = self.tensors.borrow_mut();

        // let sorted_back_ids: Vec<Uuid> = session.sorted_ids().into_iter().rev().collect();

        let id = var.id;
        // for id in &sorted_back_ids {
        let var = &session.variables.borrow()[&id];

        let op_str = var.var_type.op_str();
        let templ_str = PROJECT_DIR
            .get_file(format!("{}_backward.wgsl", op_str))
            .unwrap()
            .contents_utf8()
            .unwrap();

        let mut templ = tera::Tera::default();
        templ.add_raw_template(op_str, templ_str).unwrap();

        let mut params = tera::Context::new();
        // Ask current tensor to setup shader template parameters
        tensors[&id]
            .executable_op
            .setup_shader_backward(var.id, session, &mut params);
        let _ = templ.render(op_str, &params).unwrap();
    }

    async fn execute_inner(&self, device: &wgpu::Device, queue: &wgpu::Queue, session: &Session) {
        let sorted_ids = session.sorted_ids();
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let tensors = self.tensors.borrow_mut();

        // Setup compute pass for each variable in topological order. We will skip leaf variables since
        // the has no operation and only hold values.
        for id in &sorted_ids {
            let var = &session.variables.borrow()[id];
            if var.var_type == VarType::Leaf {
                continue;
            }

            let input_tensors: Vec<&GPUTensor> = var.prevs.iter().map(|id| &tensors[&id]).collect();
            let current_tensor = &tensors[&var.id];

            // All buffers to bind (all input buffers AND output buffer), required by shaders
            let mut all_tensors = input_tensors;
            all_tensors.push(current_tensor);

            let mut bind_idx = 0;
            let bind_group_entries: Vec<wgpu::BindGroupEntry> = all_tensors
                .iter()
                .map(|tensor| {
                    let entry = wgpu::BindGroupEntry {
                        binding: bind_idx,
                        resource: tensor.data.buffer.as_entire_binding().clone(),
                    };
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
            current_tensor
                .executable_op
                .setup_shader_forward(var.id, session, &mut params);

            let shader_src = templ.render(op_str, &params).unwrap();

            let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(format!("{}_shader", op_str).as_str()),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader_src)),
            });
            let compute_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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
                println!("{}, {}, {}", x, y, z);
                cpass.dispatch_workgroups(x, y, z);
            }
        }

        // copy buf of terminal variables to staging buffers
        for id in &sorted_ids {
            let out_buf = &tensors[id].data.buffer;
            let staging_buf = &self.staging_buf.borrow()[id];
            encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, staging_buf.size());
        }

        // submit passes
        queue.submit(Some(encoder.finish()));

        // let mut receivers = HashMap::new();
        for id in &sorted_ids {
            let staging_buf = &self.staging_buf.borrow()[id];
            let staging_slice = staging_buf.slice(..);

            let (sender, receiver) = flume::bounded(1);
            staging_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

            self.receivers.borrow_mut().insert(*id, receiver);
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
        let a = sess
            .init_tensor_var(TensorData::F32(vec![1., 2., 3.]), vec![3])
            .unwrap();
        let b = sess
            .init_tensor_var(TensorData::F32(vec![1., 2., 3.]), vec![3])
            .unwrap();
        let c = a.add(&b);
        let mut executor = GPUExecutor::new();

        executor.forward(&mut sess).unwrap();
        if let TensorData::F32(val) = &executor.fetch(c) {
            assert_eq!(val, &vec![2., 4., 6.])
        } else {
            panic!("Result should be F32")
        }
    }

    #[test]
    fn matmul() {
        let mut sess = Session::new();
        let a = sess
            .init_tensor_var(TensorData::F32(vec![1.0; 100000]), vec![1000, 100])
            .unwrap();
        let b = sess
            .init_tensor_var(TensorData::F32(vec![1.0; 100000]), vec![100, 1000])
            .unwrap();
        let c = a.matmul(&b);

        let x = sess
            .init_tensor_var(TensorData::F32(vec![1., 2., 3., 4.]), vec![2, 2])
            .unwrap();
        let y = sess
            .init_tensor_var(TensorData::F32(vec![2., 2., 2., 2.]), vec![2, 2])
            .unwrap();
        let z = x.matmul(&y);
        let mut executor = GPUExecutor::new();

        executor.forward(&mut sess).unwrap();

        if let TensorData::F32(val) = &executor.fetch(c) {
            assert!(val.iter().all(|v| *v == 100.0));
        } else {
            panic!("Result should be F32")
        }

        if let TensorData::F32(val) = &executor.fetch(z) {
            assert_eq!(val, &vec![6., 6., 14., 14.]);
        } else {
            panic!("Result should be F32")
        }
    }
}
