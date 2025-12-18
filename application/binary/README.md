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

```bash
uv run