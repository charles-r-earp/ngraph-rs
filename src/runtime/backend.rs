use super::{Executable, Tensor};
use crate::{ElementType, Function, Shape};
use cpp::*;
use std::error::Error;
use std::ffi::{CStr, CString};
use std::fmt;
use std::ops::Index;
use std::os::raw::c_char;

#[derive(Debug)]
pub struct BackendError {
    details: String,
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.details)
    }
}

impl Error for BackendError {
    fn description(&self) -> &str {
        &self.details
    }
}

#[derive(Debug)]
pub struct CompileError {
    details: String,
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.details)
    }
}

impl Error for CompileError {
    fn description(&self) -> &str {
        &self.details
    }
}

cpp! {{
  #include <ngraph/runtime/backend.hpp>
  #include <iostream>
  #include <sstream>
  #include <algorithm>
}}

cpp_class!(unsafe struct StringVector as "std::vector<std::string>");

impl StringVector {
    #[inline]
    fn len(&self) -> usize {
        cpp!(unsafe [self as "std::vector<std::string>*"] -> usize as "std::size_t" {
          return self->size();
        })
    }
}

impl Index<usize> for StringVector {
    type Output = str;
    #[inline]
    fn index(&self, i: usize) -> &str {
        unsafe {
            CStr::from_ptr(cpp!( [self as "std::vector<std::string>*", i as "std::size_t"] -> *const c_char as "const char*" {
        return (*self)[i].c_str();
      })).to_str().unwrap()
        }
    }
}

cpp_class!(pub unsafe struct Backend as "std::shared_ptr<ngraph::runtime::Backend>");

impl Backend {
    #[inline]
    fn is_null(&self) -> bool {
        cpp!(unsafe [self as "std::shared_ptr<ngraph::runtime::Backend>*"] -> bool as "bool" {
          return !bool(*self);
        })
    }
    #[inline]
    pub fn create<S>(name: S) -> Result<Self, BackendError>
    where
        S: AsRef<[u8]>,
    {
        let name = CString::new(name.as_ref()).unwrap();
        let cstr: *const c_char = name.as_ptr();
        let backend = cpp!(unsafe [cstr as "const char*"] -> Backend as "std::shared_ptr<ngraph::runtime::Backend>" {
          return ngraph::runtime::Backend::create(std::string(cstr));
        });
        if !backend.is_null() {
            Ok(backend)
        } else {
            Err(BackendError {
                details: format!("Failed to create backend {:?}, it may not exist!", name),
            })
        }
    }
    #[inline]
    pub fn get_registered_devices() -> Vec<String> {
        let strings = cpp!(unsafe [] -> StringVector as "std::vector<std::string>" {
          return ngraph::runtime::Backend::get_registered_devices();
        });
        let n = strings.len();
        let mut devices = Vec::with_capacity(n);
        for u in 0..n {
            devices.push(strings[u].into());
        }
        devices
    }
    #[inline]
    pub fn create_tensor<'s>(&self, element_type: ElementType, shape: &'s Shape) -> Tensor {
        cpp!(unsafe [self as "std::shared_ptr<ngraph::runtime::Backend>*", element_type as "ngraph::element::Type", shape as "ngraph::Shape*"] -> Tensor as "std::shared_ptr<ngraph::runtime::Tensor>" {
          return (*self)->create_tensor(element_type, *shape);
        })
    }
    #[inline]
    pub fn compile<'a>(&self, func: &'a Function) -> Result<Executable, CompileError> {
        let exec = cpp!(unsafe [self as "std::shared_ptr<ngraph::runtime::Backend>*", func as "std::shared_ptr<ngraph::Function>*"] -> Executable as "std::shared_ptr<ngraph::runtime::Executable>" {
          return (*self)->compile(*func);
        });
        if !exec.is_null() {
            Ok(exec)
        } else {
            Err(CompileError {
                details: format!("Failed to compile function!"),
            })
        }
    }
}
