use cpp::*;
use prost::Message;

use onnx_pb::{Error, ModelProto};

cpp! {{
  #define ONNX_NAMESPACE ngraph_onnx
  #define ONNX_ML TRUE
  #include <stddef.h>
  #include <onnx/proto_utils.h>
  #include <onnx/shape_inference/implementation.h>
}}

/// Infers model shapes.
pub fn shape_inference(model: &ModelProto) -> Result<ModelProto, Error> {
    let mut body = Vec::new();
    model.encode(&mut body).map_err(|e| Error::Encode(e))?;
    let inferred = shape_inference_proto(body.as_slice());
    ModelProto::decode(inferred.as_slice()).map_err(|e| Error::Decode(e))
}

/// Infers model shapes accepting and returning protocol buffers model.
pub fn shape_inference_proto(body: &[u8]) -> Vec<u8> {
    let capacity = body.len() * 12;
    let mut output = Vec::with_capacity(capacity);
    unsafe {
        output.set_len(capacity);
        let buffer = body.as_ptr();
        let size = body.len();
        let out = output.as_mut_ptr();
        let out_size = cpp!(unsafe [buffer as "const char*", size as "size_t", out as "char*"] -> usize as "size_t" {
            using namespace ngraph_onnx;
            ModelProto proto{};
            bool status = ParseProtoFromBytes(&proto, buffer, size);
            shape_inference::InferShapes(proto);
            int written = proto.ByteSizeLong();
            proto.SerializeToArray(out, written);
            return static_cast<size_t>(written);
        });
        output.truncate(out_size);
    }
    output
}
