extern crate cmake;
extern crate cpp_build;

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src");
    // for more: https://github.com/sagiegurari/ci_info/blob/master/src/config.rs
    let is_ci = std::env::var_os("TRAVIS").is_some() || std::env::var_os("GITLAB_CI").is_some();
    let ngraph_debug = std::env::var("PROFILE")
        .map(|v| v.to_lowercase() != "release")
        .unwrap_or(false);
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ngraph_out_dir = out_dir.join("ngraph");
    if is_ci || !ngraph_out_dir.join("build").join("CMakeCache.txt").exists() {
        let _ = std::fs::create_dir_all(&ngraph_out_dir.join("build"));
        let enable_cpu = std::env::var("NGRAPH_CPU_ENABLE").unwrap_or("ON".to_owned());
        let mut ngraph_cmake = cmake::Config::new("third_party/ngraph");
        ngraph_cmake
            .profile(if ngraph_debug { "Debug" } else { "Release" })
            .define("NGRAPH_NOP_ENABLE", "OFF")
            .define("NGRAPH_TOOLS_ENABLE", "OFF")
            .define("NGRAPH_UNIT_TEST_ENABLE", "OFF")
            .define("NGRAPH_TEST_UTIL_ENABLE", "OFF")
            .define("NGRAPH_ENABLE_CPU_CONV_AUTO", "OFF")
            .define(
                "NGRAPH_USE_PREBUILT_LLVM",
                if cfg!(windows) { "OFF" } else { "ON" },
            )
            .define("NGRAPH_CPU_ENABLE", &enable_cpu)
            .define(
                "NGRAPH_DEBUG_ENABLE",
                if ngraph_debug { "ON" } else { "OFF" },
            )
            .define("NGRAPH_INTERPRETER_ENABLE", "ON")
            .define("NGRAPH_JSON_ENABLE", "OFF")
            .define("NGRAPH_ONNX_IMPORT_ENABLE", "ON")
            .out_dir(&ngraph_out_dir);
        if cfg!(windows) {
            ngraph_cmake.define("CMAKE_CXX_FLAGS", "/w");
        }
        let _libngraph = ngraph_cmake.build();
    } else {
        println!("cargo:root={}", ngraph_out_dir.display());
    }

    let build_dir = ngraph_out_dir.join("build");
    let build_deps_dir = ngraph_out_dir.join("build").join("_deps");
    let onnx_build_dir = build_deps_dir.join("ext_onnx-build");
    println!("cargo:rustc-link-search={}", ngraph_out_dir.display());
    println!(
        "cargo:rustc-link-search={}",
        ngraph_out_dir.join("lib").display()
    );

    println!("cargo:rustc-link-search={}", onnx_build_dir.display());
    println!("cargo:rustc-link-lib=onnx");
    println!("cargo:rustc-link-lib=onnx_proto");

    let mut build = cpp_build::Config::new();
    build
        .include("third_party/ngraph/src")
        .include(build_deps_dir.join("ext_onnx-src"))
        .include(onnx_build_dir.clone())
        .include(build_dir.join("protobuf").join("include"))
        .object(onnx_build_dir.join("libonnx.a"))
        .object(onnx_build_dir.join("libonnx_proto.a"))
        .object(ngraph_out_dir.join("lib").join("libngraph.so"))
        .object(ngraph_out_dir.join("lib").join("libonnx_importer.so"));
    if !is_ci {
        build.object(ngraph_out_dir.join("lib").join("libcpu_backend.so"));
    }
    build.build("src/lib.rs");
}
