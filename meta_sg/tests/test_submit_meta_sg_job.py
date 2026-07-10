import os
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "meta_sg" / "scripts" / "submit_meta_sg_job.sh"


def run_launcher(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        cwd=REPO_ROOT,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def make_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def wait_for_file(path: Path) -> str:
    for _ in range(50):
        if path.exists():
            return path.read_text()
        time.sleep(0.02)
    raise AssertionError(f"{path} was not created")


def test_dry_run_prints_systemd_run_command(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    make_executable(fake_bin / "systemd-run", "#!/usr/bin/env bash\nexit 0\n")

    result = run_launcher(
        "--unit",
        "meta-sg-test",
        "--log",
        "runs/test/job_logs/train.log",
        "--dry-run",
        "--",
        "meta_sg/scripts/run_4d_both_global_backdoor_mixed_h200_job.sh",
        env={"PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )

    assert result.returncode == 0, result.stderr
    assert "systemd-run --user" in result.stdout
    assert "--unit=meta-sg-test" in result.stdout
    assert "--same-dir" in result.stdout
    assert "runs/test/job_logs/train.log" in result.stdout
    assert "run_4d_both_global_backdoor_mixed_h200_job.sh" in result.stdout


def test_submit_invokes_systemd_run_with_redirected_job(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured = tmp_path / "systemd-args.txt"
    make_executable(
        fake_bin / "systemd-run",
        f"#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > {captured}\n",
    )

    result = run_launcher(
        "--unit",
        "meta-sg-submit",
        "--log",
        "runs/test/job_logs/train.log",
        "--",
        "meta_sg/scripts/run_4d_both_global_backdoor_mixed_h200_job.sh",
        env={"PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )

    assert result.returncode == 0, result.stderr
    captured_args = captured.read_text()
    assert "--user" in captured_args
    assert "--unit=meta-sg-submit" in captured_args
    assert "--same-dir" in captured_args
    assert "--collect" in captured_args
    assert "exec meta_sg/scripts/run_4d_both_global_backdoor_mixed_h200_job.sh" in captured_args
    assert ">> runs/test/job_logs/train.log 2>&1" in captured_args


def test_systemd_environment_is_limited_to_job_variables(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    make_executable(fake_bin / "systemd-run", "#!/usr/bin/env bash\nexit 0\n")

    result = run_launcher(
        "--unit",
        "meta-sg-env",
        "--dry-run",
        "--",
        "meta_sg/scripts/run_4d_both_global_backdoor_mixed_h200_job.sh",
        env={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "RUN_ROOT": "runs/example",
            "TOTAL_ITERATIONS": "50",
            "SHOULD_NOT_LEAK": "private-value",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "--setenv=RUN_ROOT=runs/example" in result.stdout
    assert "--setenv=TOTAL_ITERATIONS=50" in result.stdout
    assert "SHOULD_NOT_LEAK" not in result.stdout
    assert "private-value" not in result.stdout


def test_submit_can_fallback_to_setsid_when_systemd_disabled(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured = tmp_path / "setsid-args.txt"
    make_executable(
        fake_bin / "setsid",
        f"#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > {captured}\n",
    )

    pid_file = tmp_path / "job.pid"
    result = run_launcher(
        "--pid-file",
        str(pid_file),
        "--log",
        "runs/test/job_logs/train.log",
        "--",
        "meta_sg/scripts/run_4d_both_global_backdoor_mixed_h200_job.sh",
        env={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "META_SG_SUBMIT_DISABLE_SYSTEMD": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "falling back to setsid" in result.stderr
    captured_args = wait_for_file(captured)
    assert "bash" in captured_args
    assert "exec meta_sg/scripts/run_4d_both_global_backdoor_mixed_h200_job.sh" in captured_args
    assert pid_file.read_text().strip()
