from pathlib import Path
import ctypes,importlib.metadata,json,os,subprocess,time
RUN=Path(__file__).resolve().parent
def save(n,v):
 with (RUN/n).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.write('\n')
# Existing installation capability only. No model object or tensor created.
hs=time.perf_counter();libs={}
for name in ['numpy','torch','cupy','cupy-cuda12x','numba','tensorflow','nvidia-cuda-runtime-cu12','psutil']:
 try:libs[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:libs[name]=None
import torch
ca=torch.cuda.is_available()
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader'],capture_output=True,text=True)
class MEM(ctypes.Structure):
 _fields_=[('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ['ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual']]
mem=MEM();mem.dwLength=ctypes.sizeof(mem);ok=ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
hardware=dict(gpu_query=gpu.stdout.strip(),gpu_query_error=gpu.stderr.strip(),torch_version=torch.__version__,torch_compiled_cuda=torch.version.cuda,torch_cuda_available=ca,installed_numerical_libraries=libs,cpu_logical_threads=os.cpu_count(),torch_cpu_threads=torch.get_num_threads(),total_physical_memory_bytes=mem.ullTotalPhys if ok else None,available_physical_memory_bytes=mem.ullAvailPhys if ok else None,memory_method='Windows GlobalMemoryStatusEx fallback; PowerShell CIM access denied',cached_arrays='On-disk NPZ at entry; indices/timestamps/masks retained in RAM for review; OS page-cache residency unknown',cached_array_ram_bytes_at_entry=0,gpu_acceleration_used=False,gpu_numerical_work_seconds=None,cpu_cached_array_review_seconds=None,capability_inspection_seconds=time.perf_counter()-hs,reason='Small per-utterance arrays (8-246 observations), 12 final events; CPU vectorization avoids device initialization/transfers. No substantial GPU operation justified; no GPU benchmark or speedup claimed.',independent_cpu_membership_checks_passed=None)
save('hardware_capability.json',hardware)
print(json.dumps(hardware,indent=2))

