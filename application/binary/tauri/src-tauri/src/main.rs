use std::process::Command as StdCommand;
use std::sync::Mutex;
use tauri::{AppHandle,  Manager, State, WindowEvent};
use tauri_plugin_shell::process::{CommandChild};
use tauri_plugin_shell::ShellExt;

const BACKEND_SIDECAR: &str = "geti-inspect-backend";

struct APIManagerState {
    child: Mutex<Option<CommandChild>>,
}

#[tauri::command]
async fn start_server(app: AppHandle, state: State<'_, APIManagerState>) -> Result<String, String> {
    let mut child_lock = state
        .child
        .lock()
        .map_err(|e| format!("Failed to lock mutex: {}", e))?;

    if child_lock.is_some() {
        println!("API server is already running");
        return Ok("API server is already running".into());
    }

    println!("Attempting to start geti_inspect backend...");

    let (_, child) = {
        app.shell()
            .sidecar(BACKEND_SIDECAR)
            .map_err(|e| format!("failed to create `{}` sidecar command: {}", BACKEND_SIDECAR, e))?
            .spawn()
            .map_err(|e| format!("Failed to spawn backend: {}", e))?
    };

    println!("geti_inspect process spawned successfully");


    *child_lock = Some(child);
    println!("geti_inspect backend started successfully");
    Ok("geti_inspect backend started successfully".into())
}

#[tauri::command]
async fn stop_server(state: State<'_, APIManagerState>) -> Result<String, String> {
    let mut child_lock = state
        .child
        .lock()
        .map_err(|e| format!("Failed to lock mutex: {}", e))?;

    if let Some(child) = child_lock.take() {
        println!("Attempting to stop geti_inspect backend");

        // Get the process ID
        let pid = child.pid();

        // On Unix-like systems (macOS, Linux)
        #[cfg(unix)]
        {
            // Use pkill to terminate all child processes
            if let Err(e) = StdCommand::new("pkill")
                .args(&["-P", &pid.to_string()])
                .output()
            {
                eprintln!("Failed to terminate child processes: {}", e);
            }
        }

        // On Windows
        #[cfg(windows)]
        {
            // Use taskkill to terminate the process tree
            if let Err(e) = StdCommand::new("taskkill")
                .args(&["/F", "/T", "/PID", &pid.to_string()])
                .output()
            {
                eprintln!("Failed to terminate process tree: {}", e);
            }
        }

        // Now kill the main process
        match child.kill() {
            Ok(_) => {
                println!("geti_inspect backend stopped successfully");
                Ok("geti_inspect backend stopped successfully".into())
            }
            Err(e) => {
                eprintln!("Failed to stop geti_inspect backend: {}", e);
                Err(format!("Failed to stop geti_inspect backend: {}", e))
            }
        }
    } else {
        println!("geti_inspect backend is not running");
        Ok("geti_inspect backend is not running".into())
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(APIManagerState {
            child: Mutex::new(None),
        })
        .setup(|app| {
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let state = app_handle.state::<APIManagerState>();
                match start_server(app_handle.clone(), state).await {
                    Ok(msg) => println!("{}", msg),
                    Err(e) => eprintln!("Failed to start API server: {}", e),
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::Destroyed = event {
                let state = window.state::<APIManagerState>();
                tauri::async_runtime::block_on(async {
                    match stop_server(state).await {
                        Ok(msg) => println!("{}", msg),
                        Err(e) => eprintln!("Error stopping API server: {}", e),
                    }
                });
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
