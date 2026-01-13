// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    env,
    process::{Child, Command},
    sync::{Arc, Mutex},
};
use tauri::RunEvent;

#[cfg(debug_assertions)]
use tauri::Manager; // for open_devtools

/// "geti-inspect-backend.exe" on Windows, "geti-inspect-backend" elsewhere.
fn backend_filename() -> &'static str {
    if cfg!(windows) {
        "geti-inspect-backend.exe"
    } else {
        "geti-inspect-backend"
    }
}

/// Spawns the side-car in the same folder as this executable.
fn spawn_backend() -> std::io::Result<Child> {
    // Locate the Tauri executable, then its parent folder
    let exe_path = env::current_exe().expect("failed to get current exe path");
    let exe_dir = exe_path
        .parent()
        .expect("failed to get parent directory of exe");

    // Build the full path to geti-inspect-backend
    // Tauri build will have renamed the suffixed file to plain name next to the exe.
    let backend_path = exe_dir.join(backend_filename());

    log::info!("▶ Looking for backend side-car at {:?}", backend_path);
    let mut command = Command::new(&backend_path);
    command.env("CORS_ORIGINS", "http://tauri.localhost");

    #[cfg(all(windows, not(debug_assertions)))]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }

    let child = command.spawn()?;

    log::info!("▶ Spawned backend: {:?}", backend_path);
    Ok(child)
}

fn main() {
    // Shared handle so we can kill it on exit
    let child_handle: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(None));

    // Build the app
    let app = tauri::Builder::default()
        .setup({
            let child_handle = child_handle.clone();
            move |app| {
                let child = spawn_backend().expect("Failed to spawn backend");
                *child_handle.lock().unwrap() = Some(child);

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

    // Run and on Exit make sure to kill the backend
    let exit_handle = child_handle.clone();
    app.run(move |_app_handle, event| {
        if let RunEvent::Exit = event {
            if let Some(mut child) = exit_handle.lock().unwrap().take() {
                let _ = child.kill();
                log::info!("Backend terminated");
            }
        }
    });
}
