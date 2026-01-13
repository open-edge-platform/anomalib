# Geti Inspect Application

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
uv sync --no-dev
source .venv/bin/activate
cd ../binary/sidecar
uv run --active --with pyinstaller pyinstaller geti_inspect_cpu.spec # since env can clash with root anomalib env
mv dist/geti-inspect-backend/geti-inspect-backend dist/geti-inspect-backend/geti-inspect-backend-$(rustc -Vv | grep host | cut -f2 -d' ')
cd ../tauri
npx tauri build
```
