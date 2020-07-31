use std::time::Instant;

use rand::prelude::*;

fn main() {
    use ngraph::*;

    // Build the graph
    let mut size: usize = 256000;
    loop {
        let now = Instant::now();
        let devices = runtime::Backend::get_registered_devices();
        println!("registered devices: {:?}", &devices);
        let device = &devices[0];
        println!("using {:?}", device);

        // Create the backend
        let backend = runtime::Backend::create(device, true).unwrap();
        println!("Backend create duration: {:?}", now.elapsed());

        let s = shape![1, size];
        let a = op::Parameter::new(ElementType::F32, &s);
        let b = op::Parameter::new(ElementType::F32, &s);
        let c = op::Parameter::new(ElementType::F32, &s);

        let t0 = op::Add::new(&a, &b);
        let t1 = op::Multiply::new(&t0, &c);

        // Make the function
        let f = Function::new([&Node::from(&t1)], [&a, &b, &c]);

        // Allocate tensors for arguments a, b, c
        let t_a = backend.create_tensor(ElementType::F32, &s);
        let t_b = backend.create_tensor(ElementType::F32, &s);
        let t_c = backend.create_tensor(ElementType::F32, &s);

        // Allocate tensor for the result
        let mut t_result = backend.create_tensor(ElementType::F32, &s);

        let mut rng = rand::thread_rng();

        let mut nums_a: Vec<f32> = (0..size).map(|_| rng.gen::<f32>() * 10.).collect();
        let mut nums_b: Vec<f32> = (0..size).map(|_| rng.gen::<f32>() * 10.).collect();
        let mut nums_c: Vec<f32> = (0..size).map(|_| rng.gen::<f32>() * 10.).collect();
        assert_eq!(nums_a.len(), size);

        // Initialize tensors
        t_a.write::<f32>(&nums_a);
        t_b.write::<f32>(&nums_b);
        t_c.write::<f32>(&nums_c);

        let now = Instant::now();
        // Invoke the function
        let exec = backend.compile(&f).unwrap();
        println!("Function compile duration: {:?}", now.elapsed());
        let now = Instant::now();
        // warm up
        for _ in 0..100 {
            exec.call([&mut t_result], [&t_a, &t_b, &t_c]);
        }
        println!("Warm-up duration: {:?} (100 iterations)", now.elapsed());

        let now = Instant::now();
        let iterations = 1000;
        let mut r = vec_len::<f32>(size);
        for _ in 0..iterations {
            exec.call([&mut t_result], [&t_a, &t_b, &t_c]);
            // Get the result
            t_result.read(&mut r);
        }
        let ng_dur = now.elapsed();
        println!(
            "Function execute duration: {:?} ({} iterations)",
            now.elapsed(),
            iterations
        );

        // Get the result
        let mut raw_result = vec_len::<f32>(size);
        let now = Instant::now();
        for _ in 0..iterations {
            for i in 0..size {
                raw_result[i] = (nums_a[i] + nums_b[i]) * nums_c[i];
            }
        }
        let raw_dur = now.elapsed();
        println!(
            "Raw execute duration: {:?} ({} iterations)",
            now.elapsed(),
            iterations
        );

        if raw_dur > ng_dur {
            println!(
                "NGraph faster {:.2} times (ngraph={:?}, raw={:?}) size = {}",
                raw_dur.as_secs_f64() / ng_dur.as_secs_f64(),
                ng_dur / iterations as u32 / size as u32,
                raw_dur / iterations as u32 / size as u32,
                size
            );
        } else {
            println!(
                "NGraph slower {:.2} times (ngraph={:?}, raw={:?}) size = {}",
                ng_dur.as_secs_f64() / raw_dur.as_secs_f64(),
                ng_dur / iterations as u32 / size as u32,
                raw_dur / iterations as u32 / size as u32,
                size
            );
        }
        println!("Test parameters: iterations={}, size={}", iterations, size);

        // Get the result
        let mut r = vec_len::<f32>(size);
        t_result.read(&mut r);
        println!("{:?}", &r[..10]);
    }
}

fn vec_len<T>(size: usize) -> Vec<T> {
    let mut r = Vec::with_capacity(size);
    unsafe {
        r.set_len(size);
    }
    r
}
