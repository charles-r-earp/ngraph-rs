use crate::{Node, NodeVector, op, ParameterVector};
use cpp::*;

cpp!{{
  #include <ngraph/function.hpp>
}}

cpp_class!(pub unsafe struct Function as "std::shared_ptr<ngraph::Function>");

/*#[inline]
fn new_function(results: NodeVector, parameters: ParameterVector) -> Function {
  cpp!(unsafe [results as "ngraph::NodeVector", parameters as "ngraph::ParameterVector"] -> Function as "std::shared_ptr<ngraph::Function>" {
    return std::make_shared<ngraph::Function>(results, parameters);
  })
}*/

impl Function {
  /*#[inline]
  pub fn new<'p, Results, Parameters>(results: Results, parameters: Parameters) -> Self
    where Results: AsRef<[Node]>,
          Parameters: AsRef<[&'p op::Parameter]> {
    new_function(results.as_ref().into(), parameters.as_ref().into())
  }*/
  #[inline]
  pub fn new<'n, 'p, Results, Parameters>(results: Results, parameters: Parameters) -> Self
    where Results: AsRef<[&'n Node]>,
          Parameters: AsRef<[&'p op::Parameter]> {
    let results = NodeVector::from(results.as_ref());
    let parameters = ParameterVector::from(parameters.as_ref());
    cpp!(unsafe [results as "ngraph::NodeVector", parameters as "ngraph::ParameterVector"] -> Function as "std::shared_ptr<ngraph::Function>" {
      return std::make_shared<ngraph::Function>(results, parameters);
    })
  }
}
