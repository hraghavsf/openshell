// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Confinement for the libkrun worker: the host process that becomes each VM.
//!
//! A guest that escapes through libkrun's device emulation lands in this
//! process, which otherwise runs with all of the user's access: their files,
//! credentials and network. The worker needs much less. It uses its own disks,
//! console log and control socket and the hypervisor, and no network at all:
//! VM sandboxes have no virtual NIC, and implicit vsock (TSI) is off. So on
//! Linux, right before `krun_start_enter`, the worker:
//! - restricts its file access with Landlock to those paths plus read-only
//!   system paths, and refuses every TCP bind and connect;
//! - installs a seccomp filter that refuses sockets other than Unix sockets,
//!   `execve`, and kernel-facing calls it never makes (ptrace, mount, module
//!   loading, bpf and the like).
//!
//! It deliberately gets no access to the sandbox's state folder, which also
//! holds the host supervisor's credentials.
//!
//! Set `OPENSHELL_VM_WORKER_CONFINEMENT=off` on the driver to turn it off, for
//! debugging.

use std::path::PathBuf;

/// Driver environment variable that turns confinement off when set to `off`.
pub const DISABLE_ENV: &str = "OPENSHELL_VM_WORKER_CONFINEMENT";

/// Paths the worker may use once it becomes the VM.
#[derive(Debug, Default, Clone)]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub struct WorkerPaths {
    /// The writable overlay disk, the console log, writable shares, and the
    /// folder holding the control socket.
    pub read_write: Vec<PathBuf>,
    /// The cached base and image disks, the runtime libraries, and read-only
    /// shares.
    pub read_only: Vec<PathBuf>,
}

/// Whether the operator turned confinement off.
pub fn disabled() -> bool {
    std::env::var(DISABLE_ENV).is_ok_and(|value| value.eq_ignore_ascii_case("off"))
}

/// Confine the calling process, which must be the single-threaded worker right
/// after `fork`. Returns a short status for the log.
#[cfg(target_os = "linux")]
pub fn confine(paths: &WorkerPaths) -> Result<String, String> {
    let landlock = linux::restrict_filesystem_and_network(paths)?;
    linux::install_seccomp_filter()?;
    Ok(format!("Landlock {landlock}, seccomp filter installed"))
}

/// Not implemented outside Linux yet. On macOS, a Seatbelt profile would play
/// the same role.
#[cfg(not(target_os = "linux"))]
#[allow(clippy::unnecessary_wraps)] // Same signature as the Linux version.
pub fn confine(_paths: &WorkerPaths) -> Result<String, String> {
    Ok("not implemented on this OS".to_string())
}

#[cfg(target_os = "linux")]
mod linux {
    use super::WorkerPaths;
    use landlock::{
        ABI, Access, AccessFs, AccessNet, BitFlags, CompatLevel, Compatible, PathBeneath, PathFd,
        Ruleset, RulesetAttr, RulesetCreated, RulesetCreatedAttr, RulesetStatus,
    };
    use seccompiler::{
        SeccompAction, SeccompCmpArgLen, SeccompCmpOp, SeccompCondition, SeccompFilter, SeccompRule,
    };
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    /// System paths the worker may read: shared libraries (already loaded, but
    /// unwinding can load `libgcc_s`), `/proc` and `/sys` for CPU and memory
    /// topology, and the random and null devices.
    const READ_ONLY_SYSTEM: &[&str] = &[
        "/usr",
        "/lib",
        "/lib64",
        "/proc",
        "/sys",
        "/dev/null",
        "/dev/urandom",
        "/dev/random",
    ];

    /// The hypervisor.
    const READ_WRITE_SYSTEM: &[&str] = &["/dev/kvm"];

    /// Calls the worker never makes once it is the VM. Each fails with EPERM.
    const DENIED_SYSCALLS: &[libc::c_long] = &[
        libc::SYS_execve,
        libc::SYS_execveat,
        libc::SYS_ptrace,
        libc::SYS_process_vm_readv,
        libc::SYS_process_vm_writev,
        libc::SYS_mount,
        libc::SYS_umount2,
        libc::SYS_pivot_root,
        libc::SYS_chroot,
        libc::SYS_setns,
        libc::SYS_unshare,
        libc::SYS_init_module,
        libc::SYS_finit_module,
        libc::SYS_delete_module,
        libc::SYS_kexec_load,
        libc::SYS_bpf,
        libc::SYS_perf_event_open,
        libc::SYS_keyctl,
        libc::SYS_add_key,
        libc::SYS_request_key,
        libc::SYS_open_by_handle_at,
        libc::SYS_name_to_handle_at,
    ];

    pub(super) fn restrict_filesystem_and_network(paths: &WorkerPaths) -> Result<String, String> {
        let abi = ABI::V4;
        let mut ruleset = Ruleset::default()
            .set_compatibility(CompatLevel::BestEffort)
            .handle_access(AccessFs::from_all(abi))
            .map_err(|err| format!("Landlock filesystem rights: {err}"))?
            .handle_access(AccessNet::from_all(abi))
            .map_err(|err| format!("Landlock network rights: {err}"))?
            .create()
            .map_err(|err| format!("create Landlock ruleset: {err}"))?;

        let read_only = READ_ONLY_SYSTEM
            .iter()
            .map(Path::new)
            .chain(paths.read_only.iter().map(PathBuf::as_path));
        for path in read_only {
            ruleset = allow(ruleset, path, AccessFs::from_read(abi), abi)?;
        }
        let read_write = READ_WRITE_SYSTEM
            .iter()
            .map(Path::new)
            .chain(paths.read_write.iter().map(PathBuf::as_path));
        for path in read_write {
            ruleset = allow(ruleset, path, AccessFs::from_all(abi), abi)?;
        }

        // No network rules: every TCP bind and connect is refused.
        let status = ruleset
            .restrict_self()
            .map_err(|err| format!("enforce Landlock ruleset: {err}"))?;
        Ok(match status.ruleset {
            RulesetStatus::FullyEnforced => "fully enforced",
            RulesetStatus::PartiallyEnforced => "partially enforced (older kernel)",
            RulesetStatus::NotEnforced => "not enforced (kernel without Landlock)",
        }
        .to_string())
    }

    fn allow(
        ruleset: RulesetCreated,
        path: &Path,
        access: BitFlags<AccessFs>,
        abi: ABI,
    ) -> Result<RulesetCreated, String> {
        // Optional system paths (/lib64, for one) may not exist.
        if !path.exists() {
            return Ok(ruleset);
        }
        let access = if path.is_dir() {
            access
        } else {
            access & AccessFs::from_file(abi)
        };
        let fd = PathFd::new(path)
            .map_err(|err| format!("open {} for Landlock: {err}", path.display()))?;
        ruleset
            .add_rule(PathBeneath::new(fd, access))
            .map_err(|err| format!("Landlock rule for {}: {err}", path.display()))
    }

    #[allow(unsafe_code)]
    pub(super) fn install_seccomp_filter() -> Result<(), String> {
        let mut rules: BTreeMap<i64, Vec<SeccompRule>> = BTreeMap::new();
        // Sockets other than AF_UNIX: the worker's only channels are Unix
        // sockets, and it has no network.
        let not_unix = SeccompCondition::new(
            0,
            SeccompCmpArgLen::Dword,
            SeccompCmpOp::Ne,
            libc::AF_UNIX as u64,
        )
        .map_err(|err| format!("seccomp socket condition: {err}"))?;
        rules.insert(
            libc::SYS_socket,
            vec![SeccompRule::new(vec![not_unix]).map_err(|err| format!("seccomp rule: {err}"))?],
        );
        // An empty rule list matches every call.
        for syscall in DENIED_SYSCALLS {
            rules.insert(*syscall, Vec::new());
        }

        let architecture = std::env::consts::ARCH
            .try_into()
            .map_err(|err| format!("seccomp architecture: {err:?}"))?;
        let program: seccompiler::BpfProgram = SeccompFilter::new(
            rules,
            SeccompAction::Allow,
            SeccompAction::Errno(libc::EPERM.cast_unsigned()),
            architecture,
        )
        .map_err(|err| format!("build seccomp filter: {err}"))?
        .try_into()
        .map_err(|err| format!("compile seccomp filter: {err}"))?;

        // SAFETY: PR_SET_NO_NEW_PRIVS is a one-way flag that an unprivileged
        // seccomp filter requires. Landlock has already set it; setting it again
        // is harmless.
        if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } != 0 {
            return Err(format!(
                "PR_SET_NO_NEW_PRIVS: {}",
                std::io::Error::last_os_error()
            ));
        }
        seccompiler::apply_filter(&program).map_err(|err| format!("apply seccomp filter: {err}"))
    }
}

#[cfg(all(test, target_os = "linux"))]
#[allow(unsafe_code)]
mod tests {
    use super::{WorkerPaths, confine};

    // Bits the confined child sets for each check that came out wrong.
    const WRITE_OWN_PATH_FAILED: i32 = 1;
    const READ_OTHER_FILE_ALLOWED: i32 = 2;
    const INET_SOCKET_ALLOWED: i32 = 4;
    const UNIX_SOCKET_FAILED: i32 = 8;
    const EXEC_ALLOWED: i32 = 16;
    const CONFINE_FAILED: i32 = 32;

    /// Forks a child, confines it as a libkrun worker allowed one scratch folder,
    /// and checks what it can still do. Filesystem checks are skipped where the
    /// kernel has no Landlock.
    #[test]
    fn confined_worker_keeps_its_own_paths_and_loses_network_and_exec() {
        let allowed = tempfile::tempdir().expect("allowed dir");
        let other = tempfile::tempdir().expect("other dir");
        let secret = other.path().join("secret");
        std::fs::write(&secret, b"not for the worker").expect("write secret");
        let paths = WorkerPaths {
            read_write: vec![allowed.path().to_path_buf()],
            read_only: Vec::new(),
        };
        let own_file = allowed.path().join("console.log");

        // SAFETY: the child only runs the checks below and then calls _exit.
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0, "fork failed");
        if pid == 0 {
            let mut failures = 0;
            let Ok(status) = confine(&paths) else {
                unsafe { libc::_exit(CONFINE_FAILED) }
            };
            let landlock_enforced = status.contains("fully enforced");
            if std::fs::write(&own_file, b"ok").is_err() {
                failures |= WRITE_OWN_PATH_FAILED;
            }
            if landlock_enforced && std::fs::read(&secret).is_ok() {
                failures |= READ_OTHER_FILE_ALLOWED;
            }
            let inet = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) };
            if inet >= 0 {
                failures |= INET_SOCKET_ALLOWED;
            }
            let unix = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM, 0) };
            if unix < 0 {
                failures |= UNIX_SOCKET_FAILED;
            }
            if std::process::Command::new("/bin/true").status().is_ok() {
                failures |= EXEC_ALLOWED;
            }
            unsafe { libc::_exit(failures) };
        }

        let mut status = 0;
        // SAFETY: waits for the child forked above.
        unsafe { libc::waitpid(pid, &raw mut status, 0) };
        assert!(
            libc::WIFEXITED(status),
            "child didn't exit normally: {status}"
        );
        assert_eq!(libc::WEXITSTATUS(status), 0, "failed checks (bit flags)");
    }
}
