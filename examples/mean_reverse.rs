use ngraph::*;

// def mean_reverse(series, tau=.5, rand=None):
//     """Mean reverse from https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/"""
//     mu = np.mean(series.rate)
//     sigma = series.rate.std()
//     rand = rand if rand is not None else np.random.randn()
//     sigma_bis = sigma * np.sqrt(2. / tau)
//     return series.tshift(-TIME_DELTA, freq=SECOND).rate + TIME_DELTA * \
//         (-(series.tshift(-TIME_DELTA, freq=SECOND).rate - mu) / tau) + \
//         sigma_bis * SQRTDT * rand

fn main() {
    // Build the graph
    const SIZE: usize = 6;
    const STRIDE: usize = 1;
    const KERNEL: usize = 3;
    const OUTPUT_SIZE: usize = 1;
    let input_shape = shape![SIZE];
    let output_shape = shape![OUTPUT_SIZE];
    let input = op::Parameter::new(ElementType::F32, &input_shape);
    let reduce_axis = op::Constant::new(ElementType::I64, &shape![1], &[0]);
    let mean = op::ReduceMean::new(&input, &reduce_axis, false);
    // Make the function
    let f = Function::new([&Node::from(&mean)], [&input]);

    let devices = runtime::Backend::get_registered_devices();
    println!("registered devices: {:?}", &devices);

    let device = &devices[1];
    println!("using {:?}", device);

    let backend = runtime::Backend::create(device).unwrap();

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
