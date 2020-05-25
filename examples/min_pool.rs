use std::time::Instant;

use rand::prelude::*;

use ngraph::*;

fn main() {
    // Build the graph
    const SIZE: usize = 500_000;
    const STRIDE: usize = 1;
    const KERNEL: usize = 160;
    const OUTPUT_SIZE: usize = SIZE / STRIDE;
    let output_shape = shape![1, 1, OUTPUT_SIZE];
    println!("output_shape: {}", output_shape.len());
    return;
    let input_shape = shape![1, 1, SIZE];
    let input = op::Parameter::new(ElementType::F32, &input_shape);
    let negative = op::Negative::new(&input);

    let stride = strides![STRIDE];
    let pads_begin = shape![];
    let pads_end = shape![KERNEL - 1];
    let kernel = shape![KERNEL];
    let rounding_mode = op::RoundingType::Floor;
    let max_pool = op::MaxPool::new(
        &input,
        &stride,
        &pads_begin,
        &pads_end,
        &kernel,
        rounding_mode,
    );
    let min_pool = op::Negative::new(&max_pool);
    let output_shape = shape![1, 1, OUTPUT_SIZE];

    // Make the function
    let f = Function::new([&Node::from(&min_pool)], [&input]);

    let backend = runtime::Backend::create("CPU", true).unwrap();

    // Allocate tensors for arguments a, b, c
    let t_a = backend.create_tensor(ElementType::F32, &input_shape);
    let mut rng = rand::thread_rng();
    let mut nums_a: Vec<f32> = (0..SIZE).map(|_| rng.gen::<f32>() * 10.).collect();
    t_a.write::<f32>(&nums_a);

    // Allocate tensor for the result
    let mut t_result = backend.create_tensor(ElementType::F32, &output_shape);

    // Invoke the function
    let exec = backend.compile(&f).unwrap();
    for _ in 0..10 {
        exec.call([&mut t_result], [&t_a]);
    }

    let now = Instant::now();
    let iterations = 10;
    let mut r = [0f32; OUTPUT_SIZE];
    for _ in 0..iterations {
        exec.call([&mut t_result], [&t_a]);
        // t_result.read(&mut r);
    }
    let ng_dur = now.elapsed();
    println!(
        "Function execute duration: {:?} ({} iterations)",
        now.elapsed(),
        iterations
    );
    println!(
        "Function execute duration: {:?} per iteration",
        now.elapsed() / iterations as u32 / SIZE as u32,
    );

    let now = Instant::now();
    for _ in 0..iterations {
        let mut min = Minimum::new(KERNEL as u32);
        for i in nums_a.iter() {
            #[used]
            let _ = min.calc(*i);
        }
    }
    let ng_dur = now.elapsed();
    println!(
        "Raw execute duration: {:?} ({} iterations)",
        now.elapsed(),
        iterations
    );
    println!(
        "Raw execute duration: {:?} per iteration",
        now.elapsed() / iterations as u32 / SIZE as u32,
    );

    // Get the result
    let mut r = [0f32; OUTPUT_SIZE];
    t_result.read(&mut r);
    println!("result: {:?}", &r[..10]);
}

#[derive(Debug, Clone)]
pub struct Minimum {
    n: usize,
    vec: Vec<f32>,
    min_index: usize,
    cur_index: usize,
}

impl Minimum {
    pub fn new(n: u32) -> Self {
        let n = n as usize;

        let indicator = Self {
            n: n,
            vec: vec![std::f32::INFINITY; n],
            min_index: 0,
            cur_index: 0,
        };

        indicator
    }

    #[inline(always)]
    fn find_min_index(&self) -> usize {
        let mut min = std::f32::INFINITY;
        let mut index: usize = 0;

        for (i, &val) in self.vec.iter().enumerate() {
            if val < min {
                min = val;
                index = i;
            }
        }

        index
    }

    #[inline(always)]
    fn calc(&mut self, input: f32) -> f32 {
        self.cur_index = (self.cur_index + 1) % (self.n as usize);
        self.vec[self.cur_index] = input;

        if input < self.vec[self.min_index] {
            self.min_index = self.cur_index;
        } else if self.min_index == self.cur_index {
            self.min_index = self.find_min_index();
        }

        self.vec[self.min_index]
    }
}
