use std::borrow::Cow;
use std::fmt::{Debug, Formatter};
use std::sync::{Arc, RwLock};

use include_dir::{Dir, include_dir};
use lazy_static::lazy_static;
use uuid::Uuid;
use wgpu::BindGroupEntry;

use crate::gpu::context::{Context, Executor};
use crate::gpu::op_type::{Add, MatMul, Shader};
use crate::wgpu::tensor::{create_staging_buf, create_storage_buf};

static PROJECT_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/src/gpu/wgsl/");

lazy_static! {
    static ref GLOBAL_CTX: Context = Context::new();
}

#[derive(Clone)]
pub enum ArrayData {
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
pub enum DataType {
    F32,
}

impl DataType {
    pub fn wgsl_type(&self) -> String {
        match self {
            DataType::F32 => "f32",
        }
            .into()
    }
}

pub struct GPUArray {
    pub id: String,
    pub data_type: DataType,
    pub shape: Vec<usize>,
    initializer: bool,
    context_id: Uuid,
    executor: Arc<RwLock<Executor>>,
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
        let synced = self.executor.read().unwrap().synced;
        if !synced {
            self.executor.write().unwrap().sync();
        }
    }
}

impl GPUArray {
    fn new(data: ArrayData, shape: Vec<usize>) -> Self {
        Self::new_with_ctx(&GLOBAL_CTX, data, shape)
    }

    fn new_with_ctx(context: &Context, data: ArrayData, shape: Vec<usize>) -> Self {
        Self::new_with_name(context, Uuid::new_v4().to_string().as_str(), data, shape)
    }

    fn new_with_name(context: &Context, id: &str, data: ArrayData, shape: Vec<usize>) -> Self {
        let (storage_buf, staging_buf) = match &data {
            ArrayData::F32(vals) => {
                let storage_buf = create_storage_buf::<f32>(
                    &context.executor.read().unwrap().device,
                    &id,
                    Some(vals),
                    &shape,
                );
                let staging_buf = create_staging_buf::<f32>(
                    &context.executor.read().unwrap().device,
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
        self.bin_op_broadcast(other, Add {})
    }

    pub fn matmul(&self, other: &GPUArray) -> GPUArray {
        self.bin_op_broadcast(other, MatMul {})
    }

    pub fn bin_op_broadcast<T: Shader>(&self, other: &GPUArray, op_type: T) -> GPUArray {
        if self.context_id != other.context_id {
            panic!("cannot do operations on GPUArray from different execution context")
        }

        let res_id = Uuid::new_v4().to_string();
        self.executor.write().unwrap().synced = false;

        let (res_storage_buf, staging_buf) = match &self.data_type {
            DataType::F32 => {
                let storage_buf = create_storage_buf::<f32>(
                    &self.executor.read().unwrap().device,
                    &res_id,
                    None,
                    &self.shape,
                );
                let staging_buf = create_staging_buf::<f32>(
                    &self.executor.read().unwrap().device,
                    &res_id,
                    &None,
                    &self.shape,
                );
                (storage_buf, staging_buf)
            }
        };
        let res_gpu = Self {
            id: res_id.clone(),
            initializer: false,
            context_id: self.context_id,
            data_type: self.data_type.clone(),
            shape: self.shape.clone(),
            executor: Arc::clone(&self.executor),
            main_buffer: res_storage_buf,
            staging_buffer: staging_buf,
        };

        let buf_binding_0 = &self.main_buffer;
        let buf_binding_1 = &other.main_buffer;
        let buf_binding_2 = &res_gpu.main_buffer;
        let buffers = vec![buf_binding_0, buf_binding_1, buf_binding_2];

        let shader_template = PROJECT_DIR
            .get_file(op_type.shader_path())
            .unwrap()
            .contents_utf8()
            .unwrap();
        let mut templ = tera::Tera::default();
        let mut params = tera::Context::new();
        templ
            .add_raw_template(&op_type.shader_path(), shader_template)
            .unwrap();

        let operands = vec![self, other];
        let workgroup_sizes = op_type.prepare(operands, &res_gpu, &mut params);

        let shader_source = templ.render(&op_type.shader_path(), &params).unwrap();

        self.dispatch_compute(buffers, &shader_source, workgroup_sizes);

        let encoder = &mut self.executor.write().unwrap().encoder;
        encoder.copy_buffer_to_buffer(
            &res_gpu.main_buffer,
            0,
            &res_gpu.staging_buffer,
            0,
            res_gpu.staging_buffer.size(),
        );

        res_gpu
    }

    fn dispatch_compute(
        &self,
        buffers: Vec<&wgpu::Buffer>,
        shader_source: &str,
        wg_sizes: (u32, u32, u32),
    ) {
        let shader_module = {
            let dev = &self.executor.read().unwrap().device;
            let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
            });
            module
        };

        let compute_pipeline = {
            let dev = &self.executor.read().unwrap().device;
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
            let dev = &self.executor.read().unwrap().device;
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &bind_group_entries,
            })
        };

        {
            let encoder = &mut self.executor.write().unwrap().encoder;
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
            self.executor
                .write()
                .unwrap()
                .encoder
                .copy_buffer_to_buffer(
                    &self.main_buffer,
                    0,
                    &self.staging_buffer,
                    0,
                    self.staging_buffer.size(),
                );
        }

        // ensure sync
        self.executor.write().unwrap().sync();

        let staging_buf = &self.staging_buffer;
        let staging_slice = staging_buf.slice(..);
        let (sender, receiver) = flume::bounded(1);
        staging_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // TODO: is it proper to call device.poll() again here?
        self.executor
            .read()
            .unwrap()
            .device
            .poll(wgpu::Maintain::Wait);

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
#[allow(irrefutable_let_patterns)]
#[allow(unreachable_code)]
mod test {
    use crate::gpu::array::{ArrayData, GPUArray};

    #[test]
    fn test_simple_add() {
        let x = GPUArray::new(ArrayData::F32(vec![1., 2., 3.]), vec![3]);
        let y = GPUArray::new(ArrayData::F32(vec![2., 3., 4.]), vec![3]);
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

    #[test]
    fn test_matmul() {
        let x = GPUArray::new(ArrayData::F32(vec![1., 2., 3., 4.]), vec![2, 2]);
        let y = GPUArray::new(ArrayData::F32(vec![2., 2., 2., 2.]), vec![2, 2]);
        let res = x.matmul(&y);

        if let ArrayData::F32(val) = res.data() {
            assert_eq!(val, vec![6., 6., 14., 14.])
        } else {
            panic!("Should be F32")
        }
    }
}
