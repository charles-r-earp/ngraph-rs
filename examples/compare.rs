use std::time::{Duration, Instant};

use rand::prelude::*;

fn main() {
    use ngraph::*;

    const SIZE: usize = 500_000;
    const CHANNELS: usize = 1;
    const BATCH_SIZE: usize = 1;
    const TOTAL_SIZE: usize = BATCH_SIZE * CHANNELS * SIZE;
    const ITERATIONS: usize = 10;

    // Build the graph
    let s = shape![BATCH_SIZE, CHANNELS, SIZE];
    let a = op::Parameter::new(ElementType::F32, &s);
    let b = op::Parameter::new(ElementType::F32, &s);

    let t0 = op::Less::new(&a, &b);

    // Make the function
    let f = Function::new([&Node::from(&t0)], [&a, &b]);

    let devices = runtime::Backend::get_registered_devices();
    println!("registered devices: {:?}", &devices);

    let device = &devices[0];
    println!("using {:?}", device);

    let now = Instant::now();
    // Create the backend
    let backend = runtime::Backend::create(device).unwrap();
    println!("Backend create duration: {:?}", now.elapsed());

    // Allocate tensors for arguments a, b, c
    let t_a = backend.create_tensor(ElementType::F32, &s);
    let t_b = backend.create_tensor(ElementType::F32, &s);

    // Allocate tensor for the result
    let mut t_result = backend.create_tensor(ElementType::Boolean, &s);

    let mut rng = rand::thread_rng();

    let nums_a: Vec<f32> = (0..TOTAL_SIZE).map(|_| rng.gen::<f32>() * 10.).collect();
    let nums_b: Vec<f32> = (0..TOTAL_SIZE).map(|_| rng.gen::<f32>() * 10.).collect();
    assert_eq!(nums_a.len(), TOTAL_SIZE);

    // Initialize tensors
    t_a.write::<f32>(&nums_a);
    t_b.write::<f32>(&nums_b);

    // Invoke the function
    let _exec = backend.compile(&f).unwrap();
    let now = Instant::now();
    let exec = backend.compile(&f).unwrap();
    println!("Function compile duration: {:?}", now.elapsed());
    println!(
        "Function compile duration: {:?} per million",
        now.elapsed() * 1_000_000u32
    );
    let now = Instant::now();
    // warm up
    for _ in 0..100 {
        exec.call([&mut t_result], [&t_a, &t_b]);
    }
    println!("Warm-up duration: {:?} (100 iterations)", now.elapsed());

    let mut r = [false; TOTAL_SIZE];
    let mut ng_dur = Duration::default();
    for _ in 0..ITERATIONS {
        let nums_a: Vec<f32> = (0..TOTAL_SIZE).map(|_| rng.gen::<f32>() * 10.).collect();
        let nums_b: Vec<f32> = (0..TOTAL_SIZE).map(|_| rng.gen::<f32>() * 10.).collect();
        let now = Instant::now();
        t_a.write::<f32>(&nums_a);
        t_b.write::<f32>(&nums_b);
        exec.call([&mut t_result], [&t_a, &t_b]);
        // Get the result
        t_result.read(&mut r);
        ng_dur += now.elapsed();
    }
    println!(
        "Function execute duration: {:?} ({} iterations)",
        ng_dur, ITERATIONS
    );

    // Get the result
    let mut raw_result = [false; TOTAL_SIZE];
    let mut raw_dur = Duration::default();
    for _ in 0..ITERATIONS {
        let nums_a: Vec<f32> = (0..TOTAL_SIZE).map(|_| rng.gen::<f32>() * 10.).collect();
        let nums_b: Vec<f32> = (0..TOTAL_SIZE).map(|_| rng.gen::<f32>() * 10.).collect();
        let now = Instant::now();
        for i in 0..TOTAL_SIZE {
            raw_result[i] = nums_a[i] < nums_b[i];
        }
        raw_dur += now.elapsed();
    }
    println!(
        "Raw execute duration: {:?} ({} iterations)",
        raw_dur, ITERATIONS
    );

    if raw_dur > ng_dur {
        println!(
            "NGraph faster {:.2} times (ngraph={:?}, raw={:?})",
            raw_dur.as_secs_f64() / ng_dur.as_secs_f64(),
            ng_dur / ITERATIONS as u32 / SIZE as u32,
            raw_dur / ITERATIONS as u32 / SIZE as u32
        );
    } else {
        println!(
            "NGraph slower {:.2} times (ngraph={:?}, raw={:?})",
            ng_dur.as_secs_f64() / raw_dur.as_secs_f64(),
            ng_dur / ITERATIONS as u32 / SIZE as u32,
            raw_dur / ITERATIONS as u32 / SIZE as u32
        );
    }
    println!("Test parameters: iterations={}, SIZE={}", ITERATIONS, SIZE);

    // Get the result
    let mut r = [false; SIZE];
    t_result.read(&mut r);
    // for i in 0..SIZE {
    //     assert_eq!(raw_result[i], r[i]);
    // }
    println!("{:?}", &r[..10]);
}
