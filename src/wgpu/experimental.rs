use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;

use include_dir::{Dir, include_dir};
use uuid::Uuid;

use crate::var::{TensorDataType, VarType};
use crate::wgpu::op::add::OpAdd;
use crate::wgpu::op::leaf::OpLeaf;
use crate::wgpu::op::Op;
use crate::wgpu::tensor::{create_staging_buf, create_storage_buf, GPUTensor, GPUTensorData};

static PROJECT_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/src/wgpu/wgsl/");

#[derive(Debug)]
pub struct Session {
    tensors: HashMap<Uuid, Rc<Variable>>,
    staging_buffs: HashMap<Uuid, wgpu::Buffer>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    encoder: wgpu::CommandEncoder,
}

impl Session {
    pub fn new() -> Self {
        let (device, queue) = pollster::block_on(Self::create_device());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        Self {
            tensors: Default::default(),
            device,
            queue,
            encoder,
            staging_buffs: Default::default(),
        }
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
}

pub struct Variable {
    id: Uuid,
    tensor: GPUTensor,
    prevs: Vec<Uuid>,
    session: Rc<RefCell<Session>>,
    var_type: VarType,
}

impl Debug for Variable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{ id: {}, prevs: {:?} }}", self.id, self.prevs)
    }
}

impl Variable {
    pub fn new_tensor_var_f32(session: &Rc<RefCell<Session>>, data: Vec<f32>, shape: Vec<usize>) -> Result<Rc<Self>, Box<dyn Error>> {
        let id = Uuid::new_v4();

        let gpu_tensor = GPUTensor {
            data: GPUTensorData {
                buffer: create_storage_buf(
                    &session.borrow().device, id.to_string().as_str(), Some(&data), &shape,
                ),
                dtype: TensorDataType::F32,
                shape,
            },
            grad: None,
            requires_grad: false,
            executable_op: Box::new(OpLeaf {}),
        };
        let t = Rc::new(Variable {
            id: Uuid::new_v4(),
            tensor: gpu_tensor,
            prevs: vec![],
            session: Rc::clone(session),
            var_type: VarType::Leaf,
        });
        t.session.borrow_mut().tensors.insert(t.id, t.clone());
        Ok(t)
    }

    fn bin_op(&self, other: &Rc<Variable>, var_type: VarType, op: Box<dyn Op>) -> Rc<Variable> {
        let id = Uuid::new_v4();
        let out_dtype = match (&self.tensor.data.dtype, &other.tensor.data.dtype) {
            (TensorDataType::F32, TensorDataType) => TensorDataType::F32,
            (_, _) => { todo!() }
        };
        let requires_grad = self.tensor.requires_grad || other.tensor.requires_grad;
        let out_shape = &self.tensor.data.shape;

        let grad_buf_option = if requires_grad {
            Some(GPUTensorData {
                dtype: out_dtype.clone(),
                buffer: match &out_dtype {
                    TensorDataType::F32 => { create_storage_buf::<f32>(&self.session.borrow().device, &id.to_string(), None, &out_shape) }
                    TensorDataType::I32 => { create_storage_buf::<i32>(&self.session.borrow().device, &id.to_string(), None, &out_shape) }
                },
                shape: out_shape.clone(),
            })
        } else { None };

        let t = Rc::new(Variable {
            id,
            tensor: GPUTensor {
                data: GPUTensorData {
                    buffer: match &out_dtype {
                        TensorDataType::F32 => { create_storage_buf::<f32>(&self.session.borrow().device, &id.to_string(), None, &out_shape) }
                        TensorDataType::I32 => { create_storage_buf::<i32>(&self.session.borrow().device, &id.to_string(), None, &out_shape) }
                    },
                    dtype: TensorDataType::F32,
                    shape: out_shape.clone(),
                },
                grad: grad_buf_option,
                requires_grad,
                executable_op: op,
            },
            prevs: vec![self.id, other.id],
            session: self.session.clone(),
            var_type: var_type.clone(),
        });

        // Register this variable to session
        t.session.borrow_mut().tensors.insert(t.id, t.clone());

        // Register staging buffer for this var
        let staging_buf = match &out_dtype {
            TensorDataType::F32 => { create_staging_buf::<f32>(&t.session.borrow().device, t.id.to_string().as_str(), &None, &out_shape) }
            TensorDataType::I32 => { create_staging_buf::<i32>(&t.session.borrow().device, t.id.to_string().as_str(), &None, &out_shape) }
        };
        t.session.borrow_mut().staging_buffs.insert(t.id, staging_buf);

        // Execute this variable
        let mut sess = &mut self.session.borrow_mut();
        let input_tensors: Vec<&GPUTensor> = t.prevs.iter().map(|id| &sess.tensors[&id].tensor).collect();
        let current_tensor = &self.tensor;
        let mut all_tensors = input_tensors;
        all_tensors.push(current_tensor);

        let mut bind_idx = 0;
        let bind_group_entries: Vec<wgpu::BindGroupEntry> = all_tensors
            .iter()
            .map(|tensor| {
                let entry = wgpu::BindGroupEntry {
                    binding: bind_idx,
                    resource: tensor.data.buffer.as_entire_binding(),
                };
                bind_idx += 1;
                entry
            })
            .collect();

        let op_str = var_type.op_str();
        let templ_str = PROJECT_DIR
            .get_file(format!("{}.wgsl", op_str))
            .unwrap()
            .contents_utf8()
            .unwrap();

        let mut templ = tera::Tera::default();
        templ.add_raw_template(op_str, templ_str).unwrap();

        let mut params = tera::Context::new();

        // Ask current tensor to setup shader template parameters
        // current_tensor.executable_op.setup_shader(id, session, &mut params);
        //
        params.insert("input_0_type", &self.tensor.data.dtype.wgsl_type());
        params.insert("input_1_type", &other.tensor.data.dtype.wgsl_type());
        params.insert("output_0_type", &out_dtype.wgsl_type());

        let shader_src = templ.render(op_str, &params).unwrap();

        let shader_module = sess.device.create_shader_module(
            wgpu::ShaderModuleDescriptor {
                label: Some(format!("{}_shader", op_str).as_str()),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader_src)),
            }
        );
        let compute_pipeline = sess.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader_module,
            entry_point: "main",
        });
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = sess.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &bind_group_entries,
        });

        {
            let mut cpass = &mut sess.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);

            let [x, y, z] = [5, 1, 1];
            cpass.dispatch_workgroups(x, y, z);
        }

        let out_buf = &sess.tensors[&id].tensor.data.buffer;
        let staging_buf = &sess.staging_buffs[&id];
        self.session.borrow_mut().encoder.copy_buffer_to_buffer(out_buf, 0, staging_buf, 0, staging_buf.size());
        // &sess.queue.submit(Some(sess.encoder.finish()));

        return t;
    }

    pub fn add(&self, other: &Rc<Variable>) -> Rc<Variable> {
        self.bin_op(other, VarType::Add, Box::new(OpAdd {}))
    }
}


#[cfg(test)]
mod test {
    use std::cell::RefCell;
    use std::error::Error;
    use std::rc::Rc;

    use crate::wgpu::experimental::{Session, Variable};

    #[test]
    fn simple_session() -> Result<(), Box<dyn Error>> {
        let mut sess = Rc::new(RefCell::new(Session::new()));
        let a = Variable::new_tensor_var_f32(&sess, vec![1., 2.], vec![2])?;
        let b = Variable::new_tensor_var_f32(&sess, vec![2., 3.], vec![2])?;
        let c = a.add(&b);
        println!("{:?}", sess);
        Ok(())
    }
}