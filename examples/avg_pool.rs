use ngraph::*;

fn main() {
    // Build the graph
    const SIZE: usize = 6;
    const STRIDE: usize = SIZE;
    const KERNEL: usize = SIZE;
    const OUTPUT_SIZE: usize = SIZE / STRIDE;
    let input_shape = shape![1, 1, SIZE];
    let input = op::Parameter::new(ElementType::F32, &input_shape);

    let stride = strides![STRIDE];
    let pads_begin = shape![];
    let pads_end = shape![KERNEL - 1];
    let kernel = shape![KERNEL];
    let rounding_mode = op::RoundingType::Floor;
    let avg_pool = op::AvgPool::new(
        &input,
        &stride,
        &pads_begin,
        &pads_end,
        &kernel,
        rounding_mode,
    );
    let output_shape = shape![1, 1, OUTPUT_SIZE];

    // Make the function
    let f = Function::new([&Node::from(&avg_pool)], [&input]);

    let backend = runtime::Backend::create("CPU", true).unwrap();

    // Allocate tensors for arguments a, b, c
    let t_a = backend.create_tensor(ElementType::F32, &input_shape);
    let v = [2., 1., 3., 4., 6., 5.];
    t_a.write::<f32>(&v);

    // Allocate tensor for the result
    let mut t_result = backend.create_tensor(ElementType::F32, &output_shape);

    // Invoke the function
    let exec = backend.compile(&f).unwrap();

    // for _ in 0..100 {
    println!("calling");
    exec.call([&mut t_result], [&t_a]);
    println!("call ok");
    // }

    // Get the result
    let mut r = [0f32; OUTPUT_SIZE];
    t_result.read(&mut r);
    println!("result: {:?}", r);
}
