import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib import request as urlrequest


DEFAULT_LOCAL_MODEL = "/data1/dg123_data/Qwen-32B"
DEFAULT_SERVED_MODEL = "local-qwen-32b"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Launch a local OpenAI-compatible H100 model server.")
    parser.add_argument("--backend", choices=("sglang", "vllm"), default="sglang")
    parser.add_argument("--model-path", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--served-model-name", default=DEFAULT_SERVED_MODEL)
    parser.add_argument("--fallback-model-path", default=None, help="Optional explicit fallback model path. Never used unless this flag is set.")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reasoning-parser")
    parser.add_argument("--log-dir", default="agent_results/local_h100_long/server_logs")
    parser.add_argument("--status-output", default="agent_results/local_h100_long/server_status.json")
    parser.add_argument("--failure-output", default="agent_results/local_h100_long/server_failure_report.json")
    parser.add_argument("--python", default=None, help="Python executable for SGLang. Defaults to .venv-sglang/bin/python if present.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--conservative", action="store_true", help="Use stable but slower SGLang backends for VLM/local tests.")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path.cwd()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = Path(args.status_output)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = base_status(args)
    status["gpu_info"] = gpu_info()

    try:
        cmd, env = build_command(args, root)
        status["command"] = cmd
        status["environment"] = {key: env.get(key) for key in ("CUDA_VISIBLE_DEVICES", "USE_HUB_KERNELS", "PATH") if env.get(key)}
        if args.dry_run:
            status["status"] = "dry_run"
            write_status(status_path, status)
            print(json.dumps(status, indent=2))
            return status
        proc, log_path = launch(cmd, env, log_dir, args.backend)
        status.update({"pid": proc.pid, "log_path": str(log_path), "status": "starting"})
        write_status(status_path, status)
        ok, health = wait_for_health(args, proc, timeout=args.timeout)
        if not ok and args.backend == "sglang" and args.reasoning_parser:
            status["fallback_reason"] = "health_check_failed_with_reasoning_parser"
            terminate(proc.pid)
            args_no_reasoning = argparse.Namespace(**vars(args))
            args_no_reasoning.reasoning_parser = None
            cmd, env = build_command(args_no_reasoning, root)
            proc, log_path = launch(cmd, env, log_dir, f"{args.backend}_fallback")
            status.update({"pid": proc.pid, "log_path": str(log_path), "command": cmd, "status": "fallback_starting"})
            write_status(status_path, status)
            ok, health = wait_for_health(args_no_reasoning, proc, timeout=args.timeout)
        if not ok and args.fallback_model_path:
            status["fallback_reason"] = "health_check_failed_with_primary_model"
            status["primary_model_path"] = args.model_path
            terminate(proc.pid)
            args_fallback = argparse.Namespace(**vars(args))
            args_fallback.model_path = args.fallback_model_path
            cmd, env = build_command(args_fallback, root)
            proc, log_path = launch(cmd, env, log_dir, f"{args.backend}_explicit_model_fallback")
            status.update({"pid": proc.pid, "log_path": str(log_path), "command": cmd, "model_path": args_fallback.model_path, "status": "fallback_model_starting"})
            write_status(status_path, status)
            ok, health = wait_for_health(args_fallback, proc, timeout=args.timeout)
        status["health_check"] = health
        status["status"] = "running" if ok else "failed"
        if proc.poll() is not None:
            status["returncode"] = proc.returncode
            status["status"] = "failed"
        write_status(status_path, status)
        if status["status"] != "running":
            write_failure_report(Path(args.failure_output), status)
        print(f"{status['status']} pid={status.get('pid')} status={status_path}")
        return status
    except Exception as exc:
        status.update({"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)}})
        write_status(status_path, status)
        write_failure_report(Path(args.failure_output), status)
        print(json.dumps(status, indent=2))
        return status


def base_status(args) -> dict:
    return {
        "backend": args.backend,
        "model_path": args.model_path,
        "served_model_name": args.served_model_name,
        "host": args.host or ("127.0.0.1" if args.backend == "sglang" else "127.0.0.1"),
        "port": args.port or (30000 if args.backend == "sglang" else 8000),
        "tp": args.tp,
        "timestamp": time.time(),
        "internal_kv_metrics": "unavailable from launcher; collect backend metrics separately if exposed",
    }


def build_command(args, root: Path):
    env = os.environ.copy()
    env.setdefault("USE_HUB_KERNELS", "0")
    if args.backend == "sglang":
        py = args.python or detect_sglang_python(root)
        help_text = command_help([py, "-m", "sglang.launch_server"])
        model_flag = "--model-path" if "--model-path" in help_text else "--model"
        tp_flag = "--tp" if "--tp " in help_text or "--tp," in help_text else "--tensor-parallel-size"
        host = args.host or "127.0.0.1"
        port = args.port or 30000
        cmd = [
            py,
            "-m",
            "sglang.launch_server",
            model_flag,
            args.model_path,
            "--host",
            host,
            "--port",
            str(port),
            tp_flag,
            str(args.tp),
            "--served-model-name",
            args.served_model_name,
            "--trust-remote-code",
        ]
        if args.reasoning_parser and "--reasoning-parser" in help_text:
            cmd.extend(["--reasoning-parser", args.reasoning_parser])
        if args.conservative:
            append_supported(cmd, help_text, "--enable-multimodal")
            append_supported(cmd, help_text, "--dtype", "bfloat16")
            append_supported(cmd, help_text, "--context-length", "8192")
            append_supported(cmd, help_text, "--max-total-tokens", "8192")
            append_supported(cmd, help_text, "--mem-fraction-static", "0.80")
            append_supported(cmd, help_text, "--attention-backend", "torch_native")
            append_supported(cmd, help_text, "--mm-attention-backend", "sdpa")
            append_supported(cmd, help_text, "--sampling-backend", "pytorch")
            append_supported(cmd, help_text, "--grammar-backend", "none")
            append_supported(cmd, help_text, "--disable-cuda-graph")
            append_supported(cmd, help_text, "--disable-custom-all-reduce")
        return cmd, env

    host = args.host or "127.0.0.1"
    port = args.port or 8000
    exe = "vllm"
    cmd = [
        exe,
        "serve",
        args.model_path,
        "--tensor-parallel-size",
        str(args.tp),
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        args.served_model_name,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    return cmd, env


def detect_sglang_python(root: Path) -> str:
    candidate = root / ".venv-sglang" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def command_help(base_cmd: list) -> str:
    try:
        out = subprocess.run(base_cmd + ["--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        return out.stdout
    except Exception:
        return ""


def append_supported(cmd: list, help_text: str, flag: str, value: str = None):
    if flag not in help_text:
        return
    cmd.append(flag)
    if value is not None:
        cmd.append(value)


def launch(cmd: list, env: dict, log_dir: Path, tag: str):
    log_path = log_dir / f"{tag}_{int(time.time())}.log"
    handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    return proc, log_path


def wait_for_health(args, proc, timeout: float):
    host = args.host or "127.0.0.1"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = args.port or (30000 if args.backend == "sglang" else 8000)
    base = f"http://{host}:{port}"
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, {"status": "process_exited", "returncode": proc.returncode, "last": last}
        for path in ("/v1/models", "/model_info"):
            try:
                with urlrequest.urlopen(base + path, timeout=3) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                last = {"url": base + path, "status_code": resp.status, "body": body[:1000]}
                if resp.status == 200:
                    chat = chat_health(base, args.served_model_name)
                    if chat.get("ok"):
                        return True, {"models": last, "chat": chat}
                    last["chat"] = chat
            except Exception as exc:
                last = {"url": base + path, "error": str(exc)}
        time.sleep(2)
    return False, {"status": "timeout", "last": last}


def chat_health(base: str, model: str):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "回答两个字：成功"}]}],
        "max_tokens": 4,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(base + "/v1/chat/completions", data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
        return {"ok": resp.status == 200, "status_code": resp.status, "body": parsed}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        )
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception as exc:
        return [{"error": str(exc)}]


def terminate(pid: int):
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(3)
    except Exception:
        pass
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception:
        pass


def write_status(path: Path, status: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")


def write_failure_report(path: Path, status: dict):
    report = {
        "status": "server_launch_failed",
        "model_path": status.get("model_path"),
        "served_model_name": status.get("served_model_name"),
        "backend": status.get("backend"),
        "command": status.get("command"),
        "environment": status.get("environment"),
        "gpu_info": status.get("gpu_info"),
        "health_check": status.get("health_check"),
        "error": status.get("error"),
        "log_path": status.get("log_path"),
        "message": "Primary model failed to launch; no implicit fallback model was used.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
