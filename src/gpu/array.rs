use std::borrow::Cow;
use std::fmt::{Debug, Formatter};
use std::sync::{Arc, RwLock};

use bytemuck::Pod;
use include_dir::{Dir, include_dir};
use lazy_static::lazy_static;
use uuid::Uuid;
use wgpu::BindGroupEntry;
use wgpu::util::DeviceExt;

use crate::gpu::context::{Executor, GPUContext};
use crate::gpu::op_type::{Add, MatMul, Shader};

static PROJECT_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/src/gpu/wgsl/");

lazy_static! {
    static ref GLOBAL_CTX: GPUContext = GPUContext::new();
}

#[derive(Clone)]
pub enum GPUDataType {
    F32,
    I32,
}

pub trait GetType {
    fn get_type(&self) -> GPUDataType;
}

impl GetType for Vec<f32> {
    fn get_type(&self) -> GPUDataType {
        GPUDataType::F32
    }
}

impl GetType for Vec<i32> {
    fn get_type(&self) -> GPUDataType {
        GPUDataType::I32
    }
}


impl GPUDataType {
    pub fn wgsl_type(&self) -> String {
        let dtype = match self {
            GPUDataType::F32 => "f32",
            GPUDataType::I32 => "i32",
        };
        dtype.into()
    }
}

pub struct GPUArray<T> {
    pub id: String,
    pub data_type: GPUDataType,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    init_data: Option<Vec<T>>,
    initializer: bool,
    context_id: Uuid,
    executor: Arc<RwLock<Executor>>,
    main_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
}

impl<T: Default + Clone + Pod + Default + Debug> Clone for GPUArray<T>
    where Vec<T>: GetType {
    fn clone(&self) -> Self {
        if let Some(init_data) = &self.init_data {
            let mut arr = GPUArray::new(init_data.clone(), self.shape.clone());
            arr.id = self.id.clone();
            arr.data_type = self.data_type.clone();
            arr.initializer = self.initializer;
            arr.context_id = self.context_id;
            arr.executor = self.executor.clone();
            return arr;
        } else {
            let init_data = self.data();
            let mut arr = GPUArray::new(init_data.clone(), self.shape.clone());
            arr.id = self.id.clone();
            arr.data_type = self.data_type.clone();
            arr.initializer = self.initializer;
            arr.context_id = self.context_id;
            arr.executor = self.executor.clone();
            return arr;
        }
    }
}

impl<T> Debug for GPUArray<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "GPUArray(id={})", self.id)
    }
}

impl<T> Drop for GPUArray<T> {
    fn drop(&mut self) {
        // we shall sync upon drop
        let synced = self.executor.read().unwrap().synced;
        if !synced {
            self.executor.write().unwrap().sync();
        }
    }
}

fn shape_to_strides(shape: &Vec<usize>) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1;
    for &dim in shape.iter().rev() {
        strides.push(stride);
        stride *= dim;
    }
    strides.reverse();
    strides
}

pub fn compute_broadcasted_shape_and_strides(
    shape1: &Vec<usize>,
    shape2: &Vec<usize>,
    strides1: &Vec<usize>,
    strides2: &Vec<usize>,
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let length = shape1.len().max(shape2.len());
    let mut adjusted_shape1 = vec![1; length];
    let mut adjusted_shape2 = vec![1; length];
    let mut adjusted_strides1 = vec![0; length];
    let mut adjusted_strides2 = vec![0; length];

    for (i, &dim) in shape1.iter().rev().enumerate() {
        adjusted_shape1[length - 1 - i] = dim;
        adjusted_strides1[length - 1 - i] = if dim == 1 { 0 } else { strides1[shape1.len() - 1 - i] };
    }

    for (i, &dim) in shape2.iter().rev().enumerate() {
        adjusted_shape2[length - 1 - i] = dim;
        adjusted_strides2[length - 1 - i] = if dim == 1 { 0 } else { strides2[shape2.len() - 1 - i] };
    }

    for i in 0..length {
        if adjusted_shape1[i] != adjusted_shape2[i] {
            if adjusted_shape1[i] == 1 {
                adjusted_shape1[i] = adjusted_shape2[i];
                adjusted_strides1[i] = 0;
            } else if adjusted_shape2[i] == 1 {
                adjusted_shape2[i] = adjusted_shape1[i];
                adjusted_strides2[i] = 0;
            } else {
                panic!("Shapes are not broadcastable");
            }
        }
    }

    (adjusted_shape1, adjusted_shape2, adjusted_strides1, adjusted_strides2)
}

impl<T: Clone + Pod + Default + Debug> GPUArray<T> where Vec<T>: GetType {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self::new_with_ctx(&GLOBAL_CTX, data, shape)
    }

    pub fn new_with_ctx(context: &GPUContext, data: Vec<T>, shape: Vec<usize>) -> Self {
        Self::new_with_name(context, Uuid::new_v4().to_string().as_str(), data, shape)
    }

    fn new_with_name(context: &GPUContext, id: &str, data: Vec<T>, shape: Vec<usize>) -> Self {
        let storage_buf = create_storage_buf(
            &context.executor.read().unwrap().device,
            &id,
            Some(&data),
            &shape,
        );
        let staging_buf = create_staging_buf::<T>(
            &context.executor.read().unwrap().device,
            &id,
            &None,
            &shape,
        );

        let dtype = data.get_type();
        Self {
            id: id.to_string(),
            initializer: true,
            init_data: Some(data),
            context_id: context.id,
            data_type: dtype,
            shape: shape.clone(),
            strides: shape_to_strides(&shape),
            executor: context.executor.clone(),
            main_buffer: storage_buf,
            staging_buffer: staging_buf,
        }
    }

    pub fn add(&self, other: &GPUArray<T>) -> GPUArray<T> {
        self.bin_op_broadcast(other, Add {})
    }

    pub fn matmul(&self, other: &GPUArray<T>) -> GPUArray<T> {
        self.bin_op_broadcast(other, MatMul {})
    }

    pub fn bin_op_broadcast<S: Shader>(&self, other: &GPUArray<T>, op_type: S) -> GPUArray<T> {
        if self.context_id != other.context_id {
            panic!("cannot do operations on GPUArray from different execution context")
        }

        let res_id = Uuid::new_v4().to_string();
        self.executor.write().unwrap().synced = false;

        let (res_shape, _, _, _) = compute_broadcasted_shape_and_strides(&self.shape, &other.shape, &self.strides, &other.strides);
        let res_strides = shape_to_strides(&res_shape);
        let (res_storage_buf, staging_buf) = match &self.data_type {
            GPUDataType::F32 => {
                let storage_buf = create_storage_buf::<f32>(&self.executor.read().unwrap().device, &res_id, None, &res_shape);
                let staging_buf = create_staging_buf::<f32>(&self.executor.read().unwrap().device, &res_id, &None, &res_shape);
                (storage_buf, staging_buf)
            }
            GPUDataType::I32 => {
                let storage_buf = create_storage_buf::<i32>(&self.executor.read().unwrap().device, &res_id, None, &res_shape);
                let staging_buf = create_staging_buf::<i32>(&self.executor.read().unwrap().device, &res_id, &None, &res_shape);
                (storage_buf, staging_buf)
            }
        };

        let res_gpu = Self {
            id: res_id.clone(),
            initializer: false,
            init_data: None,
            context_id: self.context_id,
            data_type: self.data_type.clone(),
            shape: res_shape,
            strides: res_strides,
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
    pub fn data(&self) -> Vec<T> {
        pollster::block_on(self.fetch())
    }

    async fn fetch(&self) -> Vec<T> {
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
            let vec_data = bytemuck::cast_slice(&data).to_vec();

            drop(data);
            staging_buf.unmap();
            return vec_data;
        } else {
            panic!("Cannot run on GPU")
        }
    }
}

mod test {
    use crate::gpu::array::GPUArray;
    use crate::gpu::context::GPUContext;

    #[test]
    fn test_simple_add() {
        let ctx = GPUContext::new();
        let x = GPUArray::new_with_ctx(&ctx, vec![1., 2., 3.], vec![1, 3]);
        let y = GPUArray::new_with_ctx(&ctx, vec![2., 3., 4.], vec![1, 3]);
        let res = x.add(&y);

        assert_eq!(x.data(), vec![1., 2., 3.]);
        assert_eq!(res.data(), vec![3., 5., 7.]);
    }

    #[test]
    fn test_add_bcast() {
        let ctx = GPUContext::new();
        let x = GPUArray::new_with_ctx(&ctx, vec![1., 2., 3., 4.], vec![2, 2]);
        let y = GPUArray::new_with_ctx(&ctx, vec![10., 10.], vec![2]);
        let res = x.add(&y);
        assert_eq!(res.data(), vec![11., 12., 13., 14.]);

        let x = GPUArray::new_with_ctx(&ctx, vec![1., 2., 3., 4.], vec![2, 2]);
        let y = GPUArray::new_with_ctx(&ctx, vec![10., 10.], vec![2, 1]);
        let res = x.add(&y);
        assert_eq!(res.data(), vec![11., 12., 13., 14.]);
    }

    #[test]
    fn test_add_bcast_bidirection() {
        let ctx = GPUContext::new();
        let x = GPUArray::new_with_ctx(&ctx, vec![1., 2., 3.], vec![3]);
        let y = GPUArray::new_with_ctx(&ctx, vec![1., 2., 3.], vec![3, 1]);
        let res = x.add(&y);
        assert_eq!(res.data(), vec![2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_matmul() {
        let ctx = GPUContext::new();
        let x = GPUArray::new_with_ctx(&ctx, vec![1., 2., 3., 4.], vec![2, 2]);
        let y = GPUArray::new_with_ctx(&ctx, vec![2., 2., 2., 2.], vec![2, 2]);
        let res = x.matmul(&y);
        assert_eq!(res.data(), vec![6., 6., 14., 14.]);
    }
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
            mapped_at_creation: true,
        })
    };
    data
}

pub fn create_staging_buf<'a, T: bytemuck::Pod + Default + Debug>(
    device: &wgpu::Device,
    buf_label: &str,
    values: &'a Option<Vec<T>>,
    shape: &Vec<usize>,
) -> wgpu::Buffer {
    let n_items = shape.iter().fold(1, |x, y| x * y) as usize;
    let vals: Cow<'a, Vec<T>> = match values {
        Some(v) => Cow::Borrowed(v),
        None => Cow::Owned(vec![T::default(); n_items]),
    };
    let data = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(format!("{}.staging", buf_label).as_str()),
        contents: bytemuck::cast_slice(&vals),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });
    data
}
