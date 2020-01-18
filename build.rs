extern crate cmake;
extern crate cpp_build;

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src");
    let travis = std::env::var_os("TRAVIS");
    let gitlab_ci = std::env::var_os("GITLAB_CI");
    // for more: https://github.com/sagiegurari/ci_info/blob/master/src/config.rs
    let is_ci = travis.is_some() || gitlab_ci.is_some();

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ngraph_out_dir = out_dir.join("ngraph");
    if !ngraph_out_dir.join("build").join("CMakeCache.txt").exists() {
        let _ = std::fs::create_dir_all(&ngraph_out_dir.join("build"));
        let enable_cpu = if cfg!(windows) {
            "OFF"
        } else if is_ci {
            "OFF"
        } else {
            "ON"
        };
        let mut ngraph_cmake = cmake::Config::new("third_party/ngraph");
        ngraph_cmake
            .profile("Release")
            .define("NGRAPH_NOP_ENABLE", "OFF")
            .define("NGRAPH_TOOLS_ENABLE", "OFF")
            .define("NGRAPH_UNIT_TEST_ENABLE", "OFF")
            .define(
                "NGRAPH_USE_PREBUILT_LLVM",
                if cfg!(windows) { "OFF" } else { "ON" },
            )
            .define("NGRAPH_CPU_ENABLE", enable_cpu)
            .define("NGRAPH_INTERPRETER_ENABLE", "ON")
            .define("NGRAPH_JSON_ENABLE", "OFF")
            .define("NGRAPH_PLAIDML_ENABLE", "OFF")
            .define("NGRAPH_ONNX_IMPORT_ENABLE", "ON")
            .out_dir(&ngraph_out_dir);
        if cfg!(windows) {
            ngraph_cmake.define("CMAKE_CXX_FLAGS", "/w");
        }
        let _libngraph = ngraph_cmake.build();
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
