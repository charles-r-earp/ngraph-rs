use cpp::*;

cpp! {{
  #include <ngraph/op/util/attr_types.hpp>
}}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
#[repr(u32)]
pub enum RoundingType {
    Floor = 0,
    Ceil = 1,
}
