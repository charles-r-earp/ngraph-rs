use prost::Message;

use ngraph::prelude::*;
use onnx_helpers::prelude::*;
use onnx_pb::{open_model, tensor_proto::DataType, ModelProto};

fn main() {
    let model = open_model("model-optimized.onnx").unwrap();
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
    let mut res = vec![0.0f32; 6];
    output.read(&mut res);
    println!("{:?}", &res);
}
