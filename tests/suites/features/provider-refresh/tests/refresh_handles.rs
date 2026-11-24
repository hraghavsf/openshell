// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Refresh-managed workload credential handles.
//!
//! The test serves a token issuer and a resource from one host server. The
//! gateway calls the issuer on `127.0.0.1`, so the gateway must run on the
//! machine that runs the tests; the sandbox reaches the resource through
//! `host.openshell.internal`.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use host_server_support::{HttpRecorder, Request, Response, TEST_HOST};
use serde_json::json;
use testkit::policy::{self, Access, Endpoint as _};
use testkit::sdk::raw::proto;
use testkit::{Context, Credential, PYTHON_BINARIES, Poll, Profile, Result, Sandbox, poll_until};

fn main() -> std::process::ExitCode {
    testkit::main()
}

const TOKEN_ENV: &str = "REFRESH_E2E_ACCESS_TOKEN";
const BOOTSTRAP_TOKEN: &str = "bootstrap-token";
const ROTATIONS: usize = 12;
const PROBE_TIMEOUT: Duration = Duration::from_secs(45);

/// The main process keeps the environment it started with. It checks that
/// its credential is a stable handle, then runs the probe in [`probe`]
/// whenever `/sandbox/probe-trigger` appears and writes `ok` or `failed`
/// to `/sandbox/probe-result`.
fn main_process(resource_url: &str) -> String {
    format!(
        r#"case "${TOKEN_ENV}" in
  openshell:resolve:env:s*_{TOKEN_ENV}) ;;
  *) exit 64 ;;
esac
while true; do
  if [ -f /sandbox/probe-trigger ]; then
    rm -f /sandbox/probe-trigger
    if {probe}; then
      echo ok > /sandbox/probe-result
    else
      echo failed > /sandbox/probe-result
    fi
  fi
  sleep 0.1
done"#,
        probe = probe(resource_url)
    )
}

/// A shell command that requests `resource_url` twice with the credential
/// from the environment, as a bearer token and as the password of HTTP Basic
/// `x-access-token`, and fails unless the resource accepts both.
fn probe(resource_url: &str) -> String {
    format!(
        r#"python3 -c 'import os, base64, urllib.request; token = os.environ["{TOKEN_ENV}"]; auths = ["Bearer " + token, "Basic " + base64.b64encode(("x-access-token:" + token).encode()).decode()]; [urllib.request.urlopen(urllib.request.Request("{resource_url}", headers={{"Authorization": auth}}), timeout=5).read() for auth in auths]'"#
    )
}

/// One long-running process keeps its original environment while the
/// gateway rotates the credential 12 times, each rotation invalidating the
/// last token. The process still reaches the resource with the newest token.
/// Reconfiguring the refresh revokes the process's handle, and a new process
/// gets a working one.
#[testkit::test]
async fn long_running_process_survives_rotations_and_reconfigure_revokes(
    cx: &mut Context,
) -> testkit::Result {
    let issuer = Issuer::start().await?;
    let provider = cx.name("refresh");
    let profile = profile(&provider, issuer.port()).to_yaml()?;
    cx.install_provider(&provider, &profile, &[(TOKEN_ENV, BOOTSTRAP_TOKEN)])
        .await?;
    let refresh = cx.provider_refresh(&provider, TOKEN_ENV);
    let configure = async || {
        refresh
            .configure(
                "oauth2-client-credentials",
                &[("client_id", "e2e-client")],
                &[("client_secret", "e2e-client-secret")],
            )
            .await?
            .success()
    };
    configure().await?;

    let resource_url = format!("http://{TEST_HOST}:{}/probe", issuer.port());
    let sandbox = cx
        .sandbox("rfh")
        .image(testkit::python_image())
        .providers([provider.as_str()])
        .policy(probe_policy(issuer.port()))
        .command(["sh", "-c", &main_process(&resource_url)])
        .create_ready()
        .await?;

    let first = trigger_probe(&sandbox).await?;
    testkit::ensure!(
        first == "ok",
        "the first probe from the long-running process returned {first:?}; issuer saw {}",
        issuer.summary()
    );

    for _ in 0..ROTATIONS {
        refresh.rotate().await?.success()?;
        wait_for_probe(
            &sandbox,
            "ok",
            "the long-running process to use the latest rotated token",
        )
        .await
        .map_err(|error| error.context(format!("issuer saw {}", issuer.summary())))?;
    }

    configure().await?;
    wait_for_probe(
        &sandbox,
        "failed",
        "reconfiguring the refresh to revoke the long-running process's handle",
    )
    .await?;

    refresh.rotate().await?.success()?;
    poll_until(
        "a new process to reach the resource after reconfiguring",
        PROBE_TIMEOUT,
        Duration::from_millis(500),
        async || {
            let fresh = sandbox.exec(["sh", "-c", &probe(&resource_url)]).await?;
            Ok(if fresh.exit_code == 0 {
                Poll::Ready(())
            } else {
                Poll::Pending(format!(
                    "the probe exited {}: {}",
                    fresh.exit_code,
                    fresh.stderr.trim()
                ))
            })
        },
    )
    .await
    .map_err(|error| error.context(format!("issuer saw {}", issuer.summary())))?;
    let old = trigger_probe(&sandbox).await?;
    testkit::ensure!(
        old == "failed",
        "the long-running process's revoked handle worked again after a new token was issued: {old:?}"
    );
    Ok(())
}

/// Ask the main process to run its probe, and return what it wrote.
async fn trigger_probe(sandbox: &Sandbox) -> Result<String> {
    let output = sandbox
        .exec([
            "sh",
            "-c",
            "rm -f /sandbox/probe-result; touch /sandbox/probe-trigger; \
             for _ in $(seq 100); do \
               if [ -f /sandbox/probe-result ]; then cat /sandbox/probe-result; exit 0; fi; \
               sleep 0.1; \
             done; exit 1",
        ])
        .await?
        .success()?;
    Ok(output.stdout.trim().to_string())
}

async fn wait_for_probe(sandbox: &Sandbox, expected: &str, what: &str) -> Result {
    poll_until(
        what,
        PROBE_TIMEOUT,
        Duration::from_millis(500),
        async || {
            let result = trigger_probe(sandbox).await?;
            Ok(if result == expected {
                Poll::Ready(())
            } else {
                Poll::Pending(format!("the probe returned {result:?}"))
            })
        },
    )
    .await
}

/// An OAuth issuer and the resource it guards. Each `POST /token` issues a
/// new token and invalidates the last. `GET /probe` answers `204` to the
/// current token, as a bearer token or as the password of HTTP Basic
/// `x-access-token`, and `401` to anything else.
struct Issuer {
    server: HttpRecorder,
}

impl Issuer {
    async fn start() -> Result<Self> {
        let state = Arc::new(Mutex::new((0_u32, BOOTSTRAP_TOKEN.to_string())));
        let server = HttpRecorder::serve(move |request: &Request| {
            let mut state = state.lock().unwrap_or_else(PoisonError::into_inner);
            match (request.method(), request.path()) {
                ("POST", "/token") => {
                    state.0 += 1;
                    state.1 = format!("access-token-{}", state.0);
                    Response::json(&json!({
                        "access_token": state.1,
                        "expires_in": 300,
                        "token_type": "Bearer",
                    }))
                }
                ("GET", "/probe")
                    if request.header("authorization").is_some_and(|auth| {
                        auth == format!("Bearer {}", state.1)
                            || auth
                                == format!(
                                    "Basic {}",
                                    BASE64.encode(format!("x-access-token:{}", state.1))
                                )
                    }) =>
                {
                    Response::new(204)
                }
                _ => Response::new(401),
            }
        })
        .await?;
        Ok(Self { server })
    }

    fn port(&self) -> u16 {
        self.server.port
    }

    /// The requests so far, without their headers, for a failure report.
    fn summary(&self) -> String {
        let requests: Vec<String> = self
            .server
            .requests()
            .iter()
            .map(|request| format!("{} {}", request.method(), request.path()))
            .collect();
        format!("{requests:?}")
    }
}

/// A profile whose credential refreshes with OAuth client credentials from
/// the issuer on `port`, and is bound to the resource on the same port.
fn profile(id: &str, port: u16) -> Profile {
    let material = |name: &str, secret: bool| proto::ProviderCredentialRefreshMaterial {
        name: name.to_string(),
        required: true,
        secret,
        ..proto::ProviderCredentialRefreshMaterial::default()
    };
    let seconds = |seconds| Some(prost_types::Duration { seconds, nanos: 0 });
    Profile::new(id)
        .display_name("Provider refresh handle test")
        .credential(
            Credential::new("access_token")
                .env_var(TOKEN_ENV)
                .required()
                .auth_style("bearer")
                .header_name("authorization")
                .refresh(proto::ProviderCredentialRefresh {
                    strategy: proto::ProviderCredentialRefreshStrategy::Oauth2ClientCredentials
                        as i32,
                    token_url: format!("http://127.0.0.1:{port}/token"),
                    refresh_before: seconds(30),
                    max_lifetime: seconds(300),
                    material: vec![
                        material("client_id", false),
                        material("client_secret", true),
                    ],
                    ..proto::ProviderCredentialRefresh::default()
                }),
        )
        .endpoint(TEST_HOST, port, |ep| {
            ep.rest(Access::Full).enforce().path("/probe");
        })
        .binaries(PYTHON_BINARIES.iter().copied())
}

fn probe_policy(port: u16) -> proto::SandboxPolicy {
    let mut policy = policy::base();
    policy::allow(
        &mut policy,
        "refresh_probe",
        TEST_HOST,
        port,
        PYTHON_BINARIES,
    )
    .rest(Access::Full)
    .enforce()
    .path("/probe");
    policy
}
