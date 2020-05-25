fn main() {
    use ngraph::*;

    // Build the graph
    let s = shape![2, 3];
    let a = op::Parameter::new(ElementType::F32, &s);
    let b = op::Parameter::new(ElementType::F32, &s);
    let c = op::Parameter::new(ElementType::F32, &s);

    let t0 = op::Add::new(&a, &b);
    let t1 = op::Multiply::new(&t0, &c);

    // Make the function
    let f = Function::new([&Node::from(&t1)], [&a, &b, &c]);

    let devices = runtime::Backend::get_registered_devices();
    println!("registered devices: {:?}", &devices);

    let device = &devices[0];
    println!("using {:?}", device);

    // Create the backend
    let backend = runtime::Backend::create(device, true).unwrap();

    // Allocate tensors for arguments a, b, c
    let t_a = backend.create_tensor(ElementType::F32, &s);
    let t_b = backend.create_tensor(ElementType::F32, &s);
    let t_c = backend.create_tensor(ElementType::F32, &s);
    // Allocate tensor for the result
    let mut t_result = backend.create_tensor(ElementType::F32, &s);

    // Initialize tensors
    t_a.write::<f32>(&[1., 2., 3., 4., 5., 6.]);
    t_b.write::<f32>(&[7., 8., 9., 10., 11., 12.]);
    t_c.write::<f32>(&[1., 0., -1., -1., 1., 2.]);
    // Invoke the function
    let exec = backend.compile(&f).unwrap();
    exec.call([&mut t_result], [&t_a, &t_b, &t_c]);

    // Get the result
    let mut r = [0f32; 6];
    t_result.read(&mut r);
    println!("[{:?}\n {:?}]", &r[..3], &r[3..]);
}
