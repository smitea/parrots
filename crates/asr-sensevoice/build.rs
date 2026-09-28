use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let lib_dir = manifest.join("../../vendor/sherpa-onnx/lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=sherpa-onnx-c-api");
    // dylib 依赖(@rpath/libonnxruntime.dylib 等)由可执行文件的 rpath 解析:
    // - 测试/示例二进制位于 target/{debug,release}/deps/,向上三级 = 仓库根
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../../../vendor/sherpa-onnx/lib");
    // - 直接二进制(apps/*)位于 target/{debug,release}/,向上两级 = 仓库根
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../../vendor/sherpa-onnx/lib");
    println!("cargo:rerun-if-changed={}", lib_dir.display());
}
