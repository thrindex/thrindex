// crates/thrindex-backends/akida/build.rs
//
// Build script for thrindex-backend-akida.
//
// Without `--features hardware`: emits only rerun-if directives.
//
// With `--features hardware`:
//   1.  Reads THRINDEX_AKIDA_ENGINE_PATH (required; build fails loudly if absent).
//   2.  cmake configure + build of the Engine Library to produce
//       libakida_engine.a (static).
//   3.  cxx_build: compiles bridge.cpp + pcie_driver.cpp together with the
//       cxx-generated C++ for src/ffi.rs, links the result as
//       libthrindex_akida_bridge.a.
//   4.  Emits cargo:rustc-link-lib directives for both libraries + stdc++.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Always re-run when these change, regardless of feature flag.
    println!("cargo:rerun-if-env-changed=THRINDEX_AKIDA_ENGINE_PATH");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/bridge.h");
    println!("cargo:rerun-if-changed=src/bridge.cpp");
    println!("cargo:rerun-if-changed=runtime/src/pcie_driver.h");
    println!("cargo:rerun-if-changed=runtime/src/pcie_driver.cpp");
    println!("cargo:rerun-if-changed=runtime/tests/mock_driver.h");
    println!("cargo:rerun-if-changed=src/ffi.rs");

    // Only compile FFI when the `hardware` feature is active.
    if env::var("CARGO_FEATURE_HARDWARE").is_err() {
        // No hardware feature — nothing to build.
        return;
    }

    // ── 1.  Locate the Engine Library source drop ─────────────────────────

    let engine_path = match env::var("THRINDEX_AKIDA_ENGINE_PATH") {
        Ok(p) => {
            println!("cargo:rustc-cfg=akida_engine_available");
            PathBuf::from(p)
        }
        Err(_) => {
            println!(
                "cargo:warning=THRINDEX_AKIDA_ENGINE_PATH is not set — \
                 skipping Engine Library build. Hardware inference will not be available. \
                 Set THRINDEX_AKIDA_ENGINE_PATH to build with hardware support."
            );
            return;
        }
    };

    // Validate — must contain CMakeLists.txt
    if !engine_path.join("CMakeLists.txt").exists() {
        panic!("THRINDEX_AKIDA_ENGINE_PATH={engine_path:?} does not contain CMakeLists.txt");
    }

    // ── 2.  cmake configure + build ───────────────────────────────────────

    let out_dir = env::var("OUT_DIR").unwrap();
    let engine_build_dir = Path::new(&out_dir).join("engine_build");
    std::fs::create_dir_all(&engine_build_dir).expect("failed to create engine_build dir");

    // cmake configure
    let status = Command::new("cmake")
        .arg("-S")
        .arg(&engine_path)
        .arg("-B")
        .arg(&engine_build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        // Build a static library (the CMakeLists.txt may support this via option).
        // VERIFY: the exact cmake option name — check engine/engine/CMakeLists.txt.
        .arg("-DBUILD_SHARED_LIBS=OFF")
        .status()
        .expect("cmake configure failed — is cmake installed?");
    assert!(status.success(), "cmake configure failed; see output above");

    // cmake build
    let status = Command::new("cmake")
        .arg("--build")
        .arg(&engine_build_dir)
        .arg("--config")
        .arg("Release")
        // Limit parallelism to avoid OOM on Pi (4-core, 8 GB RAM)
        .arg("--")
        .arg("-j4")
        .status()
        .expect("cmake --build failed");
    assert!(status.success(), "cmake --build failed; see output above");

    // ── 3.  Locate the built static library ──────────────────────────────

    // The engine CMakeLists.txt names its target "akida" → libakida.a.
    // (Deployment package confirms: libakida.a at the engine root after build.)
    // Older builds may produce libakida_engine.a; we check both.
    let lib_file_primary = "libakida.a";
    let lib_file_secondary = "libakida_engine.a";

    let candidates: Vec<_> = ["", "lib", "src", "akida_engine", "akida"]
        .iter()
        .flat_map(|sub| {
            let base = if sub.is_empty() {
                engine_build_dir.clone()
            } else {
                engine_build_dir.join(sub)
            };
            vec![base.join(lib_file_primary), base.join(lib_file_secondary)]
        })
        .collect();

    let found = candidates.iter().find(|p| p.exists()).unwrap_or_else(|| {
        panic!(
            "Could not find libakida.a or libakida_engine.a under {engine_build_dir:?}\n\
                 Checked: {candidates:?}\n\
                 Inspect the engine CMakeLists.txt for the actual output path."
        )
    });

    let lib_name = if found
        .file_name()
        .map(|n| n == "libakida.a")
        .unwrap_or(false)
    {
        "akida"
    } else {
        "akida_engine"
    };

    let lib_dir = found.parent().unwrap().to_path_buf();

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static={lib_name}");
    println!("cargo:rustc-link-lib=stdc++");

    // ── 4.  cxx_build: bridge shim + PCIe driver + cxx glue ──────────────

    // Include paths exposed to bridge.cpp and pcie_driver.cpp:
    //   - {engine_path}/api/     →  akida/hardware_device.h, input_conversion.h, etc.
    //   - {engine_path}/inc/     →  internal Engine Library headers (if any)
    //   - {engine_path}/api/infra/ exposed indirectly via api/ search
    //   - src/                   →  bridge.h (found as "bridge.h" in include!())
    //   - runtime/src/           →  pcie_driver.h (found as "runtime/src/pcie_driver.h")
    //   - runtime/tests/         →  mock_driver.h

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let manifest_dir = Path::new(&manifest_dir);

    cxx_build::bridge("src/ffi.rs")
        .file(manifest_dir.join("src/bridge.cpp"))
        .file(manifest_dir.join("src/system_linux.cpp")) // Engine Library runtime hooks
        .file(manifest_dir.join("runtime/src/pcie_driver.cpp"))
        .include(engine_path.join("api"))
        .include(engine_path.join("inc"))
        .include(manifest_dir) // src/ and runtime/ are under manifest_dir
        .include(manifest_dir.join("src")) // so include!("bridge.h") resolves
        .include(manifest_dir.join("runtime/src"))
        .include(manifest_dir.join("runtime/tests"))
        .flag("-std=c++17")
        .flag("-Wall")
        // Silence warnings from Engine Library headers (we don't control them)
        .flag("-Wno-unused-parameter")
        .compile("thrindex_akida_bridge");

    // The bridge library is linked automatically by cxx_build via .compile().
}
