use std::time::Instant;

use prost::Message;

use ngraph::prelude::*;
use onnx_helpers::prelude::*;
use onnx_pb::{save_model, tensor_proto::DataType, ModelProto};

fn main() {
    let mut graph = builder::Graph::new("reverse");
    let x = graph.input("X").typed(DataType::Float).dim(1).dim(6).node();
    let two = graph.constant("two", 2.0f32);
    let out = -(&x - x.mean(1, true)) * two + x;
    let graph = graph.outputs_typed(out, DataType::Float);
    let model = graph.model().build();
    // let model = optimize_model(model).unwrap();
    // let model = shape_inference(&model).unwrap();
    // let model = shape_inference(&model).unwrap();
    let model = shape_inference(&model).unwrap();
    save_model("model.onnx", &model).unwrap();
    let devices = Backend::get_registered_devices();
    let backend = Backend::create(&devices[0], true).unwrap();
    let func = create_function(&model).unwrap();
    println!("3.");
    let input = backend.create_tensor(ElementType::F32, &shape![1, 6]);
    println!("4.");
    input.write::<f32>(&[10000.0, 11000.0, 10500.0, 11500.0, 12000.0, 16000.0]);
    println!("5.");
    let mut output = backend.create_tensor(ElementType::F32, &shape![1, 6]);
    let exec = backend.compile(&func).unwrap();
    exec.call([&mut output], [&input]);
    let now = Instant::now();
    let iterations = 1000;
    for _ in 0..iterations {
        exec.call([&mut output], [&input]);
    }
    let raw_dur = now.elapsed();
    println!(
        "Raw execute duration: {:?} per iteration",
        now.elapsed() / iterations,
    );
    let mut res = vec![0.0f32; 6];
    output.read(&mut res);
    println!("{:?}", &res);
}
