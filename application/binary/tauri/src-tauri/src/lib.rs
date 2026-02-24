use std::sync::{Arc, Mutex};
// Manager is used by get_webview_window() in cfg(target_os = "linux") and
// cfg(debug_assertions) blocks, is unused in Windows release builds.
#[allow(unused_imports)]
use tauri::{Manager, RunEvent};
use tauri_plugin_shell::process::CommandChild;
use tauri_plugin_shell::ShellExt;

/// Sidecar name - matches the binary name in externalBin (tauri.conf.json).
const SIDECAR_NAME: &str = "anomalib-studio-backend";

/// Spawns the sidecar backend (externalBin).
fn spawn_backend<R: tauri::Runtime>(app: &tauri::App<R>) -> std::io::Result<CommandChild> {
    let command = app
        .shell()
        .sidecar(SIDECAR_NAME)
        .map_err(|e| std::io::Error::other(format!("Failed to get sidecar command: {e}")))?;

    let mut cors_origins = "http://tauri.localhost,tauri://localhost".to_string();
    #[cfg(debug_assertions)]
    {
        cors_origins = format!("{},http://localhost:3000", cors_origins);
    }
    let command = command.env("CORS_ORIGINS", cors_origins);

    let (_rx, child) = command
        .spawn()
        .map_err(|e| std::io::Error::other(format!("Failed to spawn backend sidecar: {e}")))?;

    log::info!("Spawned backend sidecar: {}", SIDECAR_NAME);
    Ok(child)
}

/// Kill a process and all of its descendants.
///
/// On Windows this shells out to `taskkill /F /T /PID` which recursively
/// terminates the entire process tree.  On Unix we first kill all children
/// via `pkill`, then kill the root process itself.
fn kill_process_tree(pid: u32) {
    let pid_str = pid.to_string();
    let null = std::process::Stdio::null;

    #[cfg(windows)]
    {
        let status = std::process::Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid_str])
            .stdout(null())
            .stderr(null())
            .status();
        match status {
            Ok(s) if s.success() => log::info!("Killed backend process tree (pid {pid})"),
            Ok(s) => log::warn!("taskkill exited with {s} for pid {pid}"),
            Err(e) => log::error!("Failed to run taskkill for pid {pid}: {e}"),
        }
    }

    #[cfg(unix)]
    {
        // Kill child processes first, then the root process.
        let _ = std::process::Command::new("pkill")
            .args(["-KILL", "-P", &pid_str])
            .stdout(null())
            .stderr(null())
            .status();
        let _ = std::process::Command::new("kill")
            .args(["-9", &pid_str])
            .stdout(null())
            .stderr(null())
            .status();
        log::info!("Killed backend process tree (pid {pid})");
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Shared handle so we can kill it on exit
    let child_handle: Arc<Mutex<Option<CommandChild>>> = Arc::new(Mutex::new(None));

    // Build the app
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_log::Builder::default().build())
        .setup({
            let child_handle = child_handle.clone();
            move |app| {
                let child = spawn_backend(app).expect("Failed to spawn backend");
                *child_handle.lock().unwrap() = Some(child);

                // Enable WebRTC and media stream on Linux (WebKitGTK)
                #[cfg(target_os = "linux")]
                {
                    let webview = app.get_webview_window("main").unwrap();
                    webview
                        .with_webview(|webview| {
                            use webkit2gtk::{SettingsExt, WebViewExt};
                            if let Some(settings) = webview.inner().settings() {
                                settings.set_enable_webrtc(true);
                                settings.set_enable_media_stream(true);
                                log::info!("WebRTC and MediaStream enabled for WebKitGTK");
                            }
                        })
                        .expect("Failed to configure WebRTC settings");
                }

                #[cfg(debug_assertions)]
                {
                    let window = app.get_webview_window("main").unwrap();
                    window.open_devtools();
                }

                Ok(())
            }
        })
        .invoke_handler(tauri::generate_handler![])
        .build(tauri::generate_context!())
        .expect("error building Tauri");

    // Run and on Exit kill the backend **and all its child processes**.
    // CommandChild::kill() only terminates the root process; on Windows the
    // worker subprocesses (Training, Inference, StreamLoader, …) survive.
    let exit_handle = child_handle.clone();
    app.run(move |_app_handle, event| {
        if let RunEvent::Exit = event {
            if let Some(child) = exit_handle.lock().unwrap().take() {
                let pid = child.pid();
                kill_process_tree(pid);
            }
        }
    });
}
