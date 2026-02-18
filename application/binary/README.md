# Anomalib Studio Application

## Architecture

The backend is converted to a single binary using pyinstaller. This is added to the tauri application as a sidecar.

## Folder Structure

```bash
binary/
├── sidecar/ # The sidecar is built here
├── tauri/ # The tauri application is built here
```

## Building the sidecar

### Linux/MacOS

```bash
cd application/backend
uv sync --no-dev --extra cpu
source .venv/bin/activate
cd ../binary/sidecar
uv run --active --with pyinstaller pyinstaller anomalib_studio.spec
```

### Windows (PowerShell)

```powershell
cd application\backend
uv sync --no-dev --extra cpu
.\.venv\Scripts\Activate.ps1
cd ..\binary\sidecar
uv run --active --with pyinstaller pyinstaller anomalib_studio.spec
```

## Building the Tauri application

The `prepare-tauri-dev.mjs` script automatically copies the sidecar binary (with the correct Rust target-triple
suffix) and links `_internal` into `src-tauri/sidecar/`. It runs as part of both `tauri dev` and `tauri build` via
`beforeDevCommand` / `beforeBuildCommand` in `tauri.conf.json`.

Config is passed during runtime as he PyInstaller `_internal` directory contains thousands of Python runtime
files (.pyd, .dll, .so). If listed in `tauri.conf.json` `resources`, Tauri's Cargo build script walks every file to
register `cargo:rerun-if-changed`, which causes slow recompilation and "file in use" errors on Windows when antivirus
or the OS still holds locks on freshly-built files. To avoid this, `_internal` is omitted from the default config and
only injected for production builds via `--config`.

### Development (all platforms)

Run the app locally with hot-reload for both the UI and the Rust shell:

```bash
cd application/binary/tauri
npm ci
npx tauri dev
```

The UI dev server starts at `http://localhost:3000` and the sidecar backend is spawned automatically.
DevTools open by default in debug builds.

### Production builds

#### MacOS

```bash
cd application/binary/tauri
npm ci
npx tauri build --bundles dmg --no-sign -v --config '{"bundle":{"resources":{"sidecar/_internal/":"_internal/"}}}'
# Patch the .dmg to create _internal symlink (PyInstaller needs _internal next to the sidecar binary)
bash src-tauri/install_scripts/macos_patch_dmg.sh src-tauri/target/release/bundle/dmg/*.dmg
```

#### Linux

```bash
cd application/binary/tauri
npm ci
npx tauri build --bundles deb -v --config '{"bundle":{"resources":{"sidecar/_internal/":"_internal/"}}}'
```

#### Windows (PowerShell)

```powershell
cd application\binary\tauri
npm ci
npx tauri build --bundles msi -v --config '{\"bundle\":{\"resources\":{\"sidecar/_internal/\":\"_internal/\"}}}'
```

### Debug builds

Same as production but adds `--debug` — the resulting bundle includes debug symbols and
is not optimised, which makes it easier to diagnose runtime issues:

```bash
# macOS
npx tauri build --bundles dmg --debug --no-sign -v --config '{"bundle":{"resources":{"sidecar/_internal/":"_internal/"}}}'
bash src-tauri/install_scripts/macos_patch_dmg.sh src-tauri/target/debug/bundle/dmg/*.dmg

# Linux
npx tauri build --bundles deb --debug -v --config '{"bundle":{"resources":{"sidecar/_internal/":"_internal/"}}}'
```

```powershell
# Windows (PowerShell)
npx tauri build --bundles msi --debug -v --config '{\"bundle\":{\"resources\":{\"sidecar/_internal/\":\"_internal/\"}}}'
```
