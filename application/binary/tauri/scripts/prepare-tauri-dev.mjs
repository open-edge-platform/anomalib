import fs from "node:fs";
import path from "node:path";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";

/**
 * Prepares the PyInstaller sidecar output for Tauri (dev and build).
 *
 * - Copies the sidecar binary (with Rust target-triple suffix) into src-tauri/sidecar/.
 * - Places _internal next to the binary so PyInstaller can find its runtime.
 *   By default a junction/symlink is used (fast, no disk duplication).
 *   Pass --copy to deep-copy instead (used by `tauri build` to avoid
 *   transient file-lock errors on Windows).
 *
 * After this script runs, tauri.conf.json references:
 *   externalBin  → "sidecar/anomalib-studio-backend"
 *
 * NOTE: _internal is NOT listed in tauri.conf.json "resources" by default.
 * Tauri's build script walks every resource file to register cargo:rerun-if-changed
 * directives. The _internal directory contains thousands of Python runtime files
 * (.pyd, .dll, .so), and scanning them all causes two problems:
 *   1. Significant compile-time overhead on every rebuild.
 *   2. On Windows, freshly-built .pyd files may still be locked by antivirus
 *      or the OS, causing "os error 32" (file in use) build failures.
 *
 * For dev mode the link created here is sufficient — the sidecar binary finds
 * _internal next to itself. For production builds, pass the resources config
 * via --config flag to `tauri build`:
 *   npx tauri build --config '{"bundle":{"resources":{"sidecar/_internal/":"_internal/"}}}'
 */

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const tauriRoot = path.resolve(scriptDir, "..");
const srcTauriDir = path.join(tauriRoot, "src-tauri");
const sidecarDistDir = path.resolve(
  tauriRoot,
  "..",
  "sidecar",
  "dist",
  "anomalib-studio-backend",
);
const sidecarTargetDir = path.join(srcTauriDir, "sidecar");

// Detect the Rust host triple (e.g. x86_64-pc-windows-msvc)
const rustcOutput = execSync("rustc -Vv", { encoding: "utf-8" });
const hostMatch = rustcOutput.match(/host:\s+(\S+)/);
if (!hostMatch) {
  console.error("Could not determine Rust host triple from `rustc -Vv`");
  process.exit(1);
}
const triple = hostMatch[1];
const ext = process.platform === "win32" ? ".exe" : "";

const srcBinary = path.join(sidecarDistDir, `anomalib-studio-backend${ext}`);
const srcInternal = path.join(sidecarDistDir, "_internal");

if (!fs.existsSync(srcBinary)) {
  console.error(
    "Sidecar not built. Build it first:\n" +
      "  cd application/binary/sidecar\n" +
      "  uv run --active --with pyinstaller pyinstaller anomalib_studio.spec",
  );
  process.exit(1);
}

// Ensure target directory exists
fs.mkdirSync(sidecarTargetDir, { recursive: true });

// Copy binary with the target-triple suffix that Tauri expects
const dstBinary = path.join(
  sidecarTargetDir,
  `anomalib-studio-backend-${triple}${ext}`,
);
fs.copyFileSync(srcBinary, dstBinary);
console.log(`Copied sidecar binary -> ${path.relative(srcTauriDir, dstBinary)}`);

// Place _internal next to the sidecar binary so PyInstaller can find its
// runtime files.  Two strategies:
//
//   Default (dev):  Create a junction/symlink — fast, no disk duplication.
//   --copy  (build): Deep-copy the directory.  Avoids "os error 32" failures
//                     on Windows where Tauri's build script walks every
//                     resource file and races with antivirus/indexer locks
//                     on the junction target.
const copyMode = process.argv.includes("--copy");
const dstInternal = path.join(sidecarTargetDir, "_internal");
fs.rmSync(dstInternal, { recursive: true, force: true });

if (fs.existsSync(srcInternal)) {
  if (copyMode) {
    fs.cpSync(srcInternal, dstInternal, { recursive: true });
    console.log(
      `Copied _internal -> ${path.relative(srcTauriDir, dstInternal)}`,
    );
  } else {
    const linkType = process.platform === "win32" ? "junction" : "dir";
    fs.symlinkSync(srcInternal, dstInternal, linkType);
    console.log(
      `Linked _internal -> ${path.relative(srcTauriDir, dstInternal)}`,
    );
  }
} else {
  console.warn("_internal directory not found in sidecar dist");
}

console.log("Sidecar prepared successfully.");
