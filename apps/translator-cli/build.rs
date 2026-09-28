use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let lib_dir = manifest.join("../../vendor/sherpa-onnx/lib");
    // 与 crates/tts-zipvoice/build.rs 相同的 rpath 约定:link-arg 不会跨包传播,
    // 故 app 层需自行声明,供本包的 bin 与 test 二进制解析 @rpath sherpa/onnxruntime
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../../../vendor/sherpa-onnx/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../../vendor/sherpa-onnx/lib");
    println!("cargo:rerun-if-changed={}", lib_dir.display());
}
