use cpp_build;

fn main() {
  println!("cargo:rustc-link-lib=ngraph");
  println!("cargo:rustc-lib-search=/user/local/lib");
  cpp_build::build("src/lib.rs");
}
