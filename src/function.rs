use crate::{op, Node, NodeVector, ParameterVector};
use cpp::*;

cpp! {{
  #include <ngraph/function.hpp>
  #include <ngraph/frontend/onnx_import/onnx.hpp>

  struct membuf : std::streambuf
  {
    membuf(const char* begin, size_t size) {
      char* buf(const_cast<char*>(begin));
      this->setg(buf, buf, buf + size);
    }
  };
}}

cpp_class!(pub unsafe struct Function as "std::shared_ptr<ngraph::Function>");

impl Function {
    #[inline]
    pub fn new<'n, 'p, Results, Parameters>(results: Results, parameters: Parameters) -> Self
    where
        Results: AsRef<[&'n Node]>,
        Parameters: AsRef<[&'p op::Parameter]>,
    {
        let results = NodeVector::from(results.as_ref());
        let parameters = ParameterVector::from(parameters.as_ref());
        cpp!(unsafe [results as "ngraph::NodeVector", parameters as "ngraph::ParameterVector"] -> Function as "std::shared_ptr<ngraph::Function>" {
          return std::make_shared<ngraph::Function>(results, parameters);
        })
    }

    pub fn from_onnx_bytes(bytes: &[u8]) -> Self {
        let buffer = bytes.as_ptr();
        let size = bytes.len();
        cpp!(unsafe [buffer as "const char *", size as "size_t"] -> Function as "std::shared_ptr<ngraph::Function>" {
            membuf sbuf(buffer, size);
            std::istream f(&sbuf);
            return ngraph::onnx_import::import_onnx_model(f);
        })
    }
}
