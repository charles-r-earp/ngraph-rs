extern crate cmake;
extern crate cpp_build;

use std::path::PathBuf;

fn main() {
    let travis = std::env::var_os("TRAVIS");
    let gitlab_ci = std::env::var_os("GITLAB_CI");
    // for more: https://github.com/sagiegurari/ci_info/blob/master/src/config.rs
    let is_ci = travis.is_some() || gitlab_ci.is_some();

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ngraph_out_dir = out_dir.join("ngraph");
    if !ngraph_out_dir.join("build").join("CMakeCache.txt").exists() {
        let _ = std::fs::create_dir_all(&ngraph_out_dir.join("build"));
        let _libngraph = cmake::Config::new("third_party/ngraph")
            .profile("Release")
            .define("NGRAPH_NOP_ENABLE", "OFF")
            .define("NGRAPH_CPU_ENABLE", if is_ci { "OFF" } else { "ON" })
            .define("NGRAPH_TOOLS_ENABLE", "OFF")
            .define("NGRAPH_UNIT_TEST_ENABLE", "OFF")
            .define("NGRAPH_USE_PREBUILT_LLVM", "ON")
            .define("NGRAPH_ONNX_IMPORT_ENABLE", "ON")
            .out_dir(&ngraph_out_dir)
            .build();
    } else {
        println!("cargo:root={}", ngraph_out_dir.display());
    }
    println!("cargo:rustc-link-search={}", ngraph_out_dir.display());

    cpp_build::Config::new()
        .include("third_party/ngraph/src")
        .object(ngraph_out_dir.join("lib").join("libngraph.so"))
        .build("src/lib.rs");
    println!(
        "cargo:rustc-link-search={}",
        ngraph_out_dir.join("lib").display()
    );
}
