"""Measured tensor execution, not a training or recognition-accuracy experiment."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(20261019)
    result = dict(schema="apphear.continuous_asl.gpu_probe.v1", python=sys.version,
                  executable=sys.executable, torch=torch.__version__,
                  cuda_runtime=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                  cuda_available=torch.cuda.is_available(), training_started=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        result["nvidia_smi_before"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
             "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        result["nvidia_smi_before"] = "UNAVAILABLE"
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        result["gpu_name"] = torch.cuda.get_device_name(0)
    x = torch.randn((512, 512), device=device, requires_grad=True)
    y = torch.randn((512, 512), device=device)
    durations = []
    for step in range(8):
        x.grad = None
        synchronize()
        start = time.perf_counter()
        loss = (x @ y).square().mean()
        loss.backward()
        synchronize()
        if step >= 2:
            durations.append((time.perf_counter() - start) * 1000)
    result.update(tensor_device=str(x.device), output_device=str(loss.device),
                  gradient_device=str(x.grad.device), finite_gradient=bool(torch.isfinite(x.grad).all()),
                  operation="float32 512x512 matmul + backward",
                  warmups=2, measured_iterations=len(durations),
                  median_ms=statistics.median(durations), maximum_ms=max(durations),
                  gpu_execution_verified=device.type == "cuda")
    if device.type == "cuda":
        result.update(peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
                      allocated_vram_bytes=torch.cuda.memory_allocated(),
                      reserved_vram_bytes=torch.cuda.memory_reserved())
    result["status"] = "GPU_PASS" if device.type == "cuda" else "CPU_ONLY"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return int(args.require_cuda and device.type != "cuda")
if __name__ == "__main__":
    raise SystemExit(main())

