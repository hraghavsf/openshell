// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "e2e-vm")]

//! VM-specific E2E coverage for starting sandboxes after a standalone gateway
//! restart, including a custom workspace on the gateway-managed VM driver.
//!
//! This test is gated behind the `e2e-vm` feature because it requires the VM
//! driver runtime prepared by `e2e/rust/e2e-vm.sh`.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use openshell_e2e::harness::cli::{
    run_cli, sandbox_names, wait_for_healthy, wait_for_sandbox_exec_contains,
    wait_for_sandbox_phase,
};
use openshell_e2e::harness::gateway::ManagedGateway;
use openshell_e2e::harness::sandbox::SandboxGuard;
use prost::Message;
use tokio::time::sleep;

const READY_MARKER: &str = "vm-gateway-start-ready";
const STOPPED_READY_MARKER: &str = "vm-gateway-start-stopped-ready";
const STATE_FILE: &str = "vm-gateway-start-state";
const VM_STATE_DIR_ENV: &str = "OPENSHELL_E2E_VM_STATE_DIR";
const CUSTOM_WORKSPACE_ROOT: &str = "/home/openshell-e2e/workspace";

const CHECK_WORKSPACE: &str = r#"set -eu
workdir=$(pwd -P)
if [ "$workdir" != "$1" ]; then
    printf 'working directory: expected %s, got %s\n' "$1" "$workdir" >&2
    exit 1
fi
if [ "$1" != /sandbox ]; then
    home=$(getent passwd sandbox | cut -d: -f6)
    if [ "$home" != "$1" ]; then
        printf 'sandbox account home: expected %s, got %s\n' "$1" "$home" >&2
        exit 1
    fi
fi
"#;

#[derive(Clone, PartialEq, Message)]
struct PersistedDriverSandbox {
    #[prost(string, tag = "2")]
    name: String,
}

fn vm_sandbox_stopped(sandbox_name: &str) -> Result<bool, String> {
    let state_dir = std::env::var_os(VM_STATE_DIR_ENV)
        .map(PathBuf::from)
        .ok_or_else(|| format!("{VM_STATE_DIR_ENV} must be set"))?;
    let sandboxes_dir = state_dir.join("sandboxes");
    for entry in fs::read_dir(&sandboxes_dir)
        .map_err(|err| format!("read '{}': {err}", sandboxes_dir.display()))?
    {
        let path = entry
            .map_err(|err| format!("read VM sandbox entry: {err}"))?
            .path();
        let request_path = path.join("sandbox.pb");
        let bytes = match fs::read(&request_path) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => return Err(format!("read '{}': {err}", request_path.display())),
        };
        let sandbox = PersistedDriverSandbox::decode(bytes.as_slice())
            .map_err(|err| format!("decode '{}': {err}", request_path.display()))?;
        if sandbox.name == sandbox_name {
            return Ok(path.join("stopped").exists());
        }
    }
    Err(format!(
        "VM state for sandbox '{sandbox_name}' was not found"
    ))
}

async fn wait_for_vm_stopped_marker(
    sandbox_name: &str,
    expected: bool,
    timeout: Duration,
) -> Result<(), String> {
    let start = std::time::Instant::now();
    let mut last_state;
    loop {
        match vm_sandbox_stopped(sandbox_name) {
            Ok(stopped) if stopped == expected => return Ok(()),
            Ok(stopped) => last_state = format!("stopped={stopped}"),
            Err(err) => last_state = err,
        }
        if start.elapsed() > timeout {
            return Err(format!(
                "VM '{sandbox_name}' did not reach stopped={expected}: {last_state}"
            ));
        }
        sleep(Duration::from_secs(1)).await;
    }
}

#[tokio::test]
async fn vm_gateway_restart_preserves_running_and_stopped_intent() {
    if std::env::var("OPENSHELL_E2E_DRIVER").as_deref() != Ok("vm") {
        eprintln!("Skipping VM gateway start test: e2e driver is not vm");
        return;
    }
    let Some(gateway) = ManagedGateway::from_env().expect("load managed e2e gateway metadata")
    else {
        eprintln!("Skipping VM gateway start test: e2e gateway is not managed by this test run");
        return;
    };

    let workspace_root =
        if std::env::var("OPENSHELL_E2E_EXTERNAL_COMPUTE_DRIVER").as_deref() == Ok("1") {
            // The shell wrapper starts the external driver, which keeps its default workspace.
            "/sandbox"
        } else {
            let config = std::env::var_os("OPENSHELL_E2E_GATEWAY_CONFIG")
                .expect("load the VM e2e gateway config path");
            let mut config = OpenOptions::new()
                .append(true)
                .open(config)
                .expect("open the VM e2e gateway config");
            writeln!(config, "workspace_root = \"{CUSTOM_WORKSPACE_ROOT}\"")
                .expect("configure the custom workspace in the last TOML table");
            gateway.stop().expect("stop the default e2e gateway");
            gateway
                .start()
                .expect("start the e2e gateway with a custom workspace");
            CUSTOM_WORKSPACE_ROOT
        };

    wait_for_healthy(Duration::from_secs(30))
        .await
        .expect("gateway should start healthy");

    let main_script = format!("{CHECK_WORKSPACE}echo {READY_MARKER}; exec sleep infinity");
    let mut sandbox = SandboxGuard::create_keep(
        &["sh", "-c", &main_script, "sh", workspace_root],
        READY_MARKER,
    )
    .await
    .expect("start the VM main process in the expected workspace");

    // Write from the default exec directory and flush the overlay before
    // shutdown. The restarted main process cannot recreate this marker.
    let write_script = format!("{CHECK_WORKSPACE}printf 'before-restart\n' > {STATE_FILE}; sync");
    sandbox
        .exec(&["sh", "-c", &write_script, "sh", workspace_root])
        .await
        .expect("write VM state in the workspace");
    let read_script = format!("{CHECK_WORKSPACE}cat \"$1/{STATE_FILE}\"");
    let read_state = ["sh", "-c", &read_script, "sh", workspace_root];
    let before_restart = sandbox
        .exec(&read_state)
        .await
        .expect("read VM sandbox state before restart");
    assert!(
        before_restart.contains("before-restart"),
        "VM sandbox state was not written before restart:\n{before_restart}"
    );

    let stopped_script = format!("echo {STOPPED_READY_MARKER}; while true; do sleep 1; done");
    let mut stopped_sandbox =
        SandboxGuard::create_keep(&["sh", "-lc", &stopped_script], STOPPED_READY_MARKER)
            .await
            .expect("create VM sandbox that will remain stopped");
    let (stop_output, stop_code) = run_cli(&["sandbox", "stop", &stopped_sandbox.name]).await;
    assert_eq!(stop_code, 0, "sandbox stop should succeed:\n{stop_output}");
    wait_for_sandbox_phase(&stopped_sandbox.name, "Stopped", Duration::from_mins(2))
        .await
        .expect("VM sandbox should be stopped before gateway restart");

    gateway.stop().expect("stop e2e gateway");
    wait_for_vm_stopped_marker(&sandbox.name, true, Duration::from_mins(1))
        .await
        .expect("gateway shutdown should stop the running-intent VM through its driver");
    gateway.start().expect("restart e2e gateway");
    wait_for_healthy(Duration::from_mins(2))
        .await
        .expect("gateway should become healthy after restart");
    wait_for_vm_stopped_marker(&sandbox.name, false, Duration::from_mins(2))
        .await
        .expect("gateway startup should restart the running-intent VM");

    let names = sandbox_names().await.expect("list sandboxes after restart");
    assert!(
        names.contains(&sandbox.name),
        "sandbox '{}' should still be listed after gateway restart. Names: {names:?}",
        sandbox.name
    );
    wait_for_sandbox_phase(&stopped_sandbox.name, "Stopped", Duration::from_mins(2))
        .await
        .expect("explicitly stopped VM sandbox should remain stopped after restart");

    wait_for_sandbox_exec_contains(
        &sandbox.name,
        &read_state,
        "before-restart",
        Duration::from_mins(4),
    )
    .await
    .expect("VM sandbox should become ready again with its overlay state preserved");

    sandbox.cleanup().await;
    stopped_sandbox.cleanup().await;
}
