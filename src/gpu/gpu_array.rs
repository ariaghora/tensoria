use std::borrow::Cow;
use std::fmt::{Debug, Formatter};
use std::ops::Add;
use std::sync::{Arc, RwLock};

use bytemuck::Pod;
use include_dir::{Dir, include_dir};
use lazy_static::lazy_static;
use num_traits::Num;
use uuid::Uuid;
use wgpu::{BindGroupEntry, ComputePipeline};
use wgpu::util::DeviceExt;

use crate::gpu::context::{Executor, GPUContext};
use crate::gpu::op_type::{MatMul, Shader, Slice};

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
    context: GPUContext,
    executor: Arc<RwLock<Executor>>,
    main_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
}

impl<T: Default + Clone + Pod + Default + Debug + Num> Clone for GPUArray<T>
    where
        Vec<T>: GetType,
{
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
            let init_data = self.to_vec();
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
        adjusted_strides1[length - 1 - i] = if dim == 1 {
            0
        } else {
            strides1[shape1.len() - 1 - i]
        };
    }

    for (i, &dim) in shape2.iter().rev().enumerate() {
        adjusted_shape2[length - 1 - i] = dim;
        adjusted_strides2[length - 1 - i] = if dim == 1 {
            0
        } else {
            strides2[shape2.len() - 1 - i]
        };
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

    (
        adjusted_shape1,
        adjusted_shape2,
        adjusted_strides1,
        adjusted_strides2,
    )
}

impl<T: Clone + Pod + Default + Debug + Num> GPUArray<T>
    where
        Vec<T>: GetType,
{
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
        let staging_buf =
            create_staging_buf::<T>(&context.executor.read().unwrap().device, &id, &None, &shape);

        let dtype = data.get_type();
        Self {
            id: id.to_string(),
            initializer: true,
            init_data: Some(data),
            context_id: context.id,
            context: context.clone(),
            data_type: dtype,
            shape: shape.clone(),
            strides: shape_to_strides(&shape),
            executor: context.executor.clone(),
            main_buffer: storage_buf,
            staging_buffer: staging_buf,
        }
    }

    pub fn matmul(&self, other: &GPUArray<T>) -> GPUArray<T> {
        self.bin_op_broadcast(other, MatMul {})
    }

    pub fn slice_axis<I: AsRef<[i32]>>(&self, axis: i32, indices: I) -> GPUArray<T> {
        let mut out_shape = self.shape.clone();
        out_shape[axis as usize] = indices.as_ref().len();

        // Hackity hack to force indices buffer to be of type i32
        let mut indices_arr = Self::new(vec![T::default()], vec![1]);
        let indices_shape = vec![indices.as_ref().len()];
        let indices_values = indices.as_ref().to_vec();
        let idx_id = &indices_arr.id;
        indices_arr.data_type = GPUDataType::I32;
        indices_arr.executor = self.executor.clone();
        indices_arr.context_id = self.context_id;
        indices_arr.shape = indices_shape;
        indices_arr.strides = shape_to_strides(&indices_arr.shape);
        indices_arr.main_buffer = create_storage_buf::<i32>(
            &indices_arr.executor.read().unwrap().device,
            idx_id,
            Some(&indices_values),
            &indices_arr.shape,
        );
        indices_arr.staging_buffer = create_staging_buf::<i32>(
            &indices_arr.executor.read().unwrap().device,
            idx_id,
            &None,
            &indices_arr.shape,
        );

        self.bin_op(&indices_arr, out_shape, Slice::new(axis))
    }

    pub fn sum_axis(&self, axis: i32, keep_dim: bool) -> GPUArray<T> {
        let axis_len = self.shape[axis as usize];
        if axis_len <= 0 {
            panic!("Sum axis must be positive non-zero");
        }

        let mut res: Option<GPUArray<T>> = None;
        for idx in 0..axis_len {
            let slice = self.slice_axis(axis, [idx as i32]);
            res = Some(match res {
                None => { slice }
                Some(r) => { r.add(&slice) }
            })
        }
        let mut res = res.unwrap();
        let new_shape = if keep_dim {
            res.shape.clone()
        } else {
            res.shape.clone().into_iter().filter(|v| *v != 1).collect::<Vec<usize>>()
        };
        res.shape = new_shape;
        res.strides = shape_to_strides(&res.shape);
        res
    }

    /// General binary operation
    pub fn bin_op<S: Shader>(&self, other: &GPUArray<T>, out_shape: Vec<usize>, op_type: S) -> GPUArray<T> {
        if self.context_id != other.context_id {
            panic!("cannot do operations on GPUArray from different execution context")
        }

        let res_id = Uuid::new_v4().to_string();
        self.executor.write().unwrap().synced = false;
        let out_strides = shape_to_strides(&out_shape);

        let (res_storage_buf, staging_buf) = match &self.data_type {
            GPUDataType::F32 => {
                let storage_buf = create_storage_buf::<f32>(
                    &self.executor.read().unwrap().device,
                    &res_id,
                    None,
                    &out_shape,
                );
                let staging_buf = create_staging_buf::<f32>(
                    &self.executor.read().unwrap().device,
                    &res_id,
                    &None,
                    &out_shape,
                );
                (storage_buf, staging_buf)
            }
            GPUDataType::I32 => {
                let storage_buf = create_storage_buf::<i32>(
                    &self.executor.read().unwrap().device,
                    &res_id,
                    None,
                    &out_shape,
                );
                let staging_buf = create_staging_buf::<i32>(
                    &self.executor.read().unwrap().device,
                    &res_id,
                    &None,
                    &out_shape,
                );
                (storage_buf, staging_buf)
            }
        };

        let res_gpu = Self {
            id: res_id.clone(),
            initializer: false,
            init_data: None,
            context_id: self.context_id,
            context: self.context.clone(),
            data_type: self.data_type.clone(),
            shape: out_shape,
            strides: out_strides,
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

    /// Binary operation involving shape broadcasting
    pub fn bin_op_broadcast<S: Shader>(&self, other: &GPUArray<T>, op_type: S) -> GPUArray<T> {
        let (res_shape, _, _, _) = compute_broadcasted_shape_and_strides(
            &self.shape,
            &other.shape,
            &self.strides,
            &other.strides,
        );

        return self.bin_op(other, res_shape, op_type);
    }

    /// Run compute pipeline from prepared pipeline. This is useful to run pipelines multiple
    /// times while using the same pipeline without recompiling shader modules.
    fn dispatch_compute_shader_pipeline(
        &self,
        buffers: Vec<&wgpu::Buffer>,
        pipeline: &ComputePipeline,
        wg_sizes: (u32, u32, u32),
    ) {
        let bind_group_layout = pipeline.get_bind_group_layout(0);

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
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);

            let (x, y, z) = wg_sizes;
            cpass.dispatch_workgroups(x, y, z);
        }
    }

    /// Directly run compute pipeline from &str shader source
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
        self.dispatch_compute_shader_pipeline(buffers, &compute_pipeline, wg_sizes);
    }

    /// This method copies the actual data from GPU, wrap it as ArrayData then
    /// return it. This method should be used sparingly since frequent GPU <-> CPU data
    /// transfer is costly.
    pub fn to_vec(&self) -> Vec<T> {
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

/// Macro for several binary operators, so it is easier to implement it for both
/// op(Self, &Self) and op(&Self, &Self)
macro_rules! impl_bin_op {
    ($trait:ident, $method:ident, $op:expr) => {
        impl<T: Clone + Pod + Default + Debug + Num> std::ops::$trait<&Self> for GPUArray<T>
        where
            Vec<T>: GetType,
        {
            type Output = GPUArray<T>;

            fn $method(self, rhs: &Self) -> Self::Output {
                self.bin_op_broadcast(rhs, $op)
            }
        }

        impl<T: Clone + Pod + Default + Debug + Num> std::ops::$trait for &GPUArray<T>
        where
            Vec<T>: GetType,
        {
            type Output = GPUArray<T>;

            fn $method(self, rhs: Self) -> Self::Output {
                self.bin_op_broadcast(rhs, $op)
            }
        }
    };
}

impl_bin_op!(Mul, mul, crate::gpu::op_type::Mul {});
impl_bin_op!(Add, add, crate::gpu::op_type::Add {});
impl_bin_op!(Sub, sub, crate::gpu::op_type::Sub {});

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

mod test {
    use std::ops::Add;

    use crate::gpu::context::GPUContext;
    use crate::gpu::gpu_array::GPUArray;

    #[test]
    fn simple_add() {
        let ctx = GPUContext::new();
        let x = GPUArray::new_with_ctx(&ctx, vec![1., 2., 3.], vec![1, 3]);
        let y = GPUArray::new_with_ctx(&ctx, vec![2., 3., 4.], vec![1, 3]);
        let res = &x + &y;

        assert_eq!(x.to_vec(), vec![1., 2., 3.]);
        assert_eq!(res.to_vec(), vec![3., 5., 7.]);
    }

    #[test]
    fn add_bcast() {
        let ctx = GPUContext::new();
        let x = GPUArray::new_with_ctx(&ctx, vec![1., 2., 3., 4.], vec![2, 2]);
        let y = GPUArray::new_with_ctx(&ctx, vec![10., 10.], vec![2]);
        let res = x + &y;
        assert_eq!(res.to_vec(), vec![11., 12., 13., 14.]);

        let x = GPUArray::new_with_ctx(&ctx, vec![1., 2., 3., 4.], vec![2, 2]);
        let y = GPUArray::new_with_ctx(&ctx, vec![10., 10.], vec![2, 1]);
        let res = x.add(&y);
        assert_eq!(res.to_vec(), vec![11., 12., 13., 14.]);
    }

    #[test]
    fn add_bcast_bidirection() {
        let ctx = GPUContext::new();
        let x = GPUArray::new_with_ctx(&ctx, vec![1., 2., 3.], vec![3]);
        let y = GPUArray::new_with_ctx(&ctx, vec![1., 2., 3.], vec![3, 1]);
        let res = x + &y;
        assert_eq!(
            res.to_vec(),
            vec![2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn matmul() {
        let ctx = GPUContext::new();
        let x = GPUArray::new_with_ctx(&ctx, vec![1., 2., 3., 4.], vec![2, 2]);
        let y = GPUArray::new_with_ctx(&ctx, vec![2., 2., 2., 2.], vec![2, 2]);
        let res = x.matmul(&y);
        assert_eq!(res.to_vec(), vec![6., 6., 14., 14.]);
    }

    #[test]
    fn slice() {
        let ctx = GPUContext::new();
        let x = GPUArray::new_with_ctx(&ctx, vec![1., 2., 3., 4., 5., 6., 7., 8., 9.], vec![3, 3]);
        let res = x.slice_axis(0, [1]);
        assert_eq!(res.to_vec(), vec![4., 5., 6.]);

        let ctx = GPUContext::new();
        let x = GPUArray::new_with_ctx(&ctx, vec![1., 2., 3., 4., 5., 6., 7., 8., 9.], vec![3, 3]);
        let res = x.slice_axis(0, [0, 2]);
        assert_eq!(res.to_vec(), vec![1., 2., 3., 7., 8., 9.]);

        let ctx = GPUContext::new();
        let x = GPUArray::new_with_ctx(&ctx, vec![1., 2., 3., 4., 5., 6., 7., 8., 9.], vec![3, 3]);
        let res = x.slice_axis(1, [0, 2]);
        assert_eq!(res.to_vec(), vec![1., 3., 4., 6., 7., 9.]);

        let ctx = GPUContext::new();
        let x = GPUArray::new_with_ctx(&ctx, vec![1., 2., 3., 4., 5., 6.], vec![3, 2]);
        let res = x.slice_axis(1, [1]);
        assert_eq!(res.to_vec(), vec![2., 4., 6.]);

        let ctx = GPUContext::new();
        let x = GPUArray::new_with_ctx(&ctx, vec![2, 2, 2, 2, 1, 1, 1, 1], vec![2, 2, 2]);
        let res = x.slice_axis(0, [1]);
        assert_eq!(res.to_vec(), vec![1, 1, 1, 1]);
        let res = x.slice_axis(1, [1]);
        assert_eq!(res.to_vec(), vec![2, 2, 1, 1]);
    }

    #[test]
    fn sum() {
        let ctx = GPUContext::new();
        let x = GPUArray::new_with_ctx(&ctx, vec![1., 2., 3., 4., 5., 6.], vec![3, 2]);
        let res = x.sum_axis(0, true);
        assert_eq!(res.to_vec(), vec![9., 12.]);
        assert_eq!(res.shape, vec![1, 2]);
        let res = x.sum_axis(1, false);
        assert_eq!(res.to_vec(), vec![3., 7., 11.]);
        assert_eq!(res.shape, vec![3]);
        let x = GPUArray::new_with_ctx(&ctx, vec![2, 2, 2, 2, 1, 1, 1, 1], vec![2, 2, 2]);
        let res = x.sum_axis(0, false);
        assert_eq!(res.to_vec(), vec![3, 3, 3, 3]);
        assert_eq!(res.shape, vec![2, 2]);
        let res = x.sum_axis(1, true);
        assert_eq!(res.to_vec(), vec![4, 4, 2, 2]);
        assert_eq!(res.shape, vec![2, 1, 2]);
    }
}
