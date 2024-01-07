<h3 align="center">
    ᕕ(⌐■_■)ᕗ ♪♬
</h3>

<h3 align="center">
    <b>T E N S O R I A</b>
</h3>


<p align="center">
A tensor manipulation library running on GPU, self-contained, in pure rust
</p>

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