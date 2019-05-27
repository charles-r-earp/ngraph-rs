use cpp_build;

fn main() {
  println!("cargo:rustc-link-lib=ngraph");
  cpp_build::build("src/lib.rs");
}
