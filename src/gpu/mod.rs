use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::rc::Rc;

use lazy_static::lazy_static;
use uuid::Uuid;
use wgpu::BindGroupEntry;

use crate::gpu::context::{Context, Executor};
use crate::wgpu::tensor::{create_staging_buf, create_storage_buf};

mod context;

lazy_static! {
    // static ref GLOBAL_CTX: Rc<Mutex<Context>> = Arc::new(Mutex::new(Context::new()));
}

#[derive(Clone)]
enum ArrayData {
    F32(Vec<f32>),
}

impl ArrayData {
    fn dtype(&self) -> DataType {
        match self {
            ArrayData::F32(_) => DataType::F32,
        }
    }
}

#[derive(Clone)]
enum DataType {
    F32,
}

struct GPUArray {
    id: String,
    initializer: bool,
    context_id: Uuid,
    data_type: DataType,
    shape: Vec<usize>,
    executor: Rc<RefCell<Executor>>,
    main_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
}

impl Debug for GPUArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "GPUArray(id={})", self.id)
    }
}

impl Drop for GPUArray {
    fn drop(&mut self) {
        // we shall sync upon drop
        if !self.executor.borrow().synced {
            self.executor.borrow_mut().sync();
        }
    }
}

impl GPUArray {
    // fn new(data: ArrayData, shape: Vec<usize>) -> Self {
    //     let foo = GLOBAL_CTX.lock().unwrap();
    //     Self::new_with_ctx(&foo, data, shape)
    // }

    fn new_with_ctx(context: &Context, data: ArrayData, shape: Vec<usize>) -> Self {
        Self::new_with_name(context, Uuid::new_v4().to_string().as_str(), data, shape)
    }

    fn new_with_name(context: &Context, id: &str, data: ArrayData, shape: Vec<usize>) -> Self {
        let (storage_buf, staging_buf) = match &data {
            ArrayData::F32(vals) => {
                let storage_buf = create_storage_buf::<f32>(
                    &context.executor.borrow().device,
                    &id,
                    Some(vals),
                    &shape,
                );
                let staging_buf = create_staging_buf::<f32>(
                    &context.executor.borrow().device,
                    &id,
                    &None,
                    &shape,
                );
                (storage_buf, staging_buf)
            }
        };

        Self {
            id: id.to_string(),
            initializer: true,
            context_id: context.id,
            data_type: data.dtype(),
            shape: shape.clone(),
            executor: context.executor.clone(),
            main_buffer: storage_buf,
            staging_buffer: staging_buf,
        }
    }

    pub fn add(&self, other: &GPUArray) -> GPUArray {
        if self.context_id != other.context_id {
            panic!("cannot do operations on GPUArray from different execution context")
        }

        let res_id = Uuid::new_v4().to_string();
        self.executor.borrow_mut().synced = false;

        let (res_storage_buf, staging_buf) = match &self.data_type {
            DataType::F32 => {
                let storage_buf = create_storage_buf::<f32>(
                    &self.executor.borrow().device,
                    &res_id,
                    None,
                    &self.shape,
                );
                let staging_buf = create_staging_buf::<f32>(
                    &self.executor.borrow().device,
                    &res_id,
                    &None,
                    &self.shape,
                );
                (storage_buf, staging_buf)
            }
        };
        let buf_binding_0 = &self.main_buffer;
        let buf_binding_1 = &other.main_buffer;
        let buf_binding_2 = &res_storage_buf;
        let buffers = vec![buf_binding_0, buf_binding_1, buf_binding_2];
        let shader_source = include_str!("wgsl/add.wgsl");

        self.dispatch_compute(buffers, shader_source, (64, 1, 1));

        let mut encoder = &mut self.executor.borrow_mut().encoder;
        encoder.copy_buffer_to_buffer(&res_storage_buf, 0, &staging_buf, 0, staging_buf.size());

        Self {
            id: res_id.clone(),
            initializer: false,
            context_id: self.context_id,
            data_type: self.data_type.clone(),
            shape: self.shape.clone(),
            executor: Rc::clone(&self.executor),
            main_buffer: res_storage_buf,
            staging_buffer: staging_buf,
        }
    }

    fn dispatch_compute(
        &self,
        buffers: Vec<&wgpu::Buffer>,
        shader_source: &str,
        wg_sizes: (u32, u32, u32),
    ) {
        let shader_module = {
            let dev = &self.executor.borrow().device;
            let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
            });
            module
        };

        let compute_pipeline = {
            let dev = &self.executor.borrow().device;
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader_module,
                entry_point: "main",
            })
        };

        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);

        let bind_group_entries: Vec<BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();

        let bind_group = {
            let dev = &self.executor.borrow().device;
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &bind_group_entries,
            })
        };

        {
            let mut encoder = &mut self.executor.borrow_mut().encoder;
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);

            let (x, y, z) = wg_sizes;
            cpass.dispatch_workgroups(x, y, z);
        }
    }

    /// This method copies the actual data from GPU, wrap it as ArrayData then
    /// return it. This method should be used sparingly since frequent GPU <-> CPU data
    /// transfer is costly.
    pub fn data(&self) -> ArrayData {
        pollster::block_on(self.fetch())
    }

    async fn fetch(&self) -> ArrayData {
        // if this array is an initializer, we first copy the data from the main buffer to
        // the staging buffer since we didn't do that by default to conserve GPU memory.
        if self.initializer {
            self.executor.borrow_mut().encoder.copy_buffer_to_buffer(
                &self.main_buffer,
                0,
                &self.staging_buffer,
                0,
                self.staging_buffer.size(),
            );
        }

        // ensure sync
        self.executor.borrow_mut().sync();

        let staging_buf = &self.staging_buffer;
        let staging_slice = staging_buf.slice(..);
        let (sender, receiver) = flume::bounded(1);
        staging_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // TODO: is it proper to call device.poll() again here?
        self.executor.borrow().device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = receiver.recv_async().await {
            let data = staging_slice.get_mapped_range();

            let array_data = match self.data_type {
                DataType::F32 => ArrayData::F32(bytemuck::cast_slice(&data).to_vec()),
            };

            drop(data);
            staging_buf.unmap();
            return array_data;
        } else {
            panic!("Cannot run on GPU")
        }
    }
}

#[cfg(test)]
mod test {
    use crate::gpu::{ArrayData, GPUArray};
    use crate::gpu::context::Context;

    #[test]
    fn test_simple_add() {
        let ctx = Context::new();
        let x = GPUArray::new_with_ctx(&ctx, ArrayData::F32(vec![1., 2., 3.]), vec![3]);
        let y = GPUArray::new_with_ctx(&ctx, ArrayData::F32(vec![2., 3., 4.]), vec![3]);
        let res = x.add(&y);

        if let ArrayData::F32(val) = x.data() {
            assert_eq!(val, vec![1., 2., 3.])
        } else {
            panic!("Should be F32")
        }

        if let ArrayData::F32(val) = res.data() {
            assert_eq!(val, vec![3., 5., 7.])
        } else {
            panic!("Should be F32")
        }
    }
}
