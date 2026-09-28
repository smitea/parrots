//! Link directives for the vendored sherpa-onnx dylib (same rationale as
//! crates/asr-sensevoice: the `links` key is already taken by tts-zipvoice,
//! so this build script emits its own link-search/link-lib/rpath args —
//! duplicated directives are harmless).

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let lib_dir = format!("{manifest}/../../vendor/sherpa-onnx/lib");
    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=dylib=sherpa-onnx-c-api");
    // Runtime search paths for test/binary layouts under target/
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../../../vendor/sherpa-onnx/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../../vendor/sherpa-onnx/lib");
}
