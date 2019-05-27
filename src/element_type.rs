use cpp::*;

cpp!{{
  #include <ngraph/type/element_type.hpp>
}}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)] 
pub enum ElementType {
  Undefined,
  Dynamic,
  Boolean,
  BF16,
  F16,
  F32,
  F64,
  I8,
  I16,
  I32,
  I64,
  U8,
  U16,
  U32,
  U64
}
      
