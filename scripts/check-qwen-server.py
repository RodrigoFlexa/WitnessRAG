"""Check the served model and CUDA before starting a long benchmark run."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path


def server_info(base_url: str, model: str) -> dict:
    base_url = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'local-pilot')}"}
    request = urllib.request.Request(f"{base_url}/models", headers=headers)
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.load(response)
    models = {row["id"]: row for row in payload.get("data", [])}
    if model not in models:
        raise ValueError(f"Requested model {model!r} is not served; available={sorted(models)}")
    info = {"base_url": base_url, "model": model, "model_card": models[model],
            "model_revision": os.environ.get("MODEL_REVISION", "")}
    try:
        version_url = base_url.removesuffix("/v1") + "/version"
        with urllib.request.urlopen(urllib.request.Request(version_url, headers=headers), timeout=15) as response:
            info["server_version"] = json.load(response)
    except OSError:
        info["server_version"] = "unavailable"
    return info


def cuda_info() -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the benchmark environment. Install a CUDA PyTorch wheel.")
    properties = torch.cuda.get_device_properties(0)
    return {"visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "name": properties.name, "total_memory_bytes": properties.total_memory,
            "torch_version": torch.__version__, "torch_cuda": torch.version.cuda}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8095/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--check-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    info = server_info(args.base_url, args.model)
    if args.check_cuda:
        info["embedding_gpu"] = cuda_info()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Server preflight failed: {error}", file=sys.stderr)
        raise SystemExit(1)
