<p style="text-align: center">
    <h1>Tensoria</h1>
</p>

---

<p style="text-align: center">
A tensor manipulation library running on CPU and GPU, self-contained, in pure rust
</p>

<p style="text-align: center">
    <img src="assets/img.png" width="200"/>
</p>

Tensoria provides two execution engines:

- GPUExecutor with [wgpu]() backend
- CPUExecutor with [ndarray]() backend

## Example

```rust
fn main() {
    let sess = Session::new();

    let a = sess.new_tensor_var(Some(TensorData::F32(vec![1.0, 2.0])), vec![2]).unwrap();
    let b = sess.new_tensor_var(Some(TensorData::F32(vec![3.0, 4.0])), vec![2]).unwrap();
    let res = a.add(&b);

    let mut executor = GPUExecutor::new();
    executor.execute(&sess).unwrap();

    let res_gpu = executor.tensors.get(&res.id).unwrap();
    if let GPUTensorData::F32(data) = &res_gpu.data {
        assert_eq!(data.as_slice().unwrap(), vec![4.0, 6.0])
    } else {
        panic!("result should be of type F32")
    }
}
```