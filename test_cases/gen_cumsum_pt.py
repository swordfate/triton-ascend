# coding=utf-8
# 手动生成 _cumsum_v2.pt（(4, 2048) float32，dim=-1 → grid (4,1,1), TILE_SIZE=2048）
# 在 NPU 或 CPU 机器上运行：python gen_cumsum_pt.py
import torch

N = 2048
M = 4
x = torch.randn(M, N, dtype=torch.float32)
out = torch.empty_like(x, dtype=torch.float32)

data = {
    "grid": (M, 1, 1),
    "input_data": {
        "in_ptr": x,
        "out_ptr": out,
        "N": N,
        "TILE_SIZE": 2048,  # triton.next_power_of_2(N)
    },
    "gpu_output": {
        "out_ptr": torch.cumsum(x, dim=1).to(torch.float32),
    },
}
torch.save(data, "_cumsum_v2.pt")
print("saved _cumsum_v2.pt")
print("grid:", data["grid"])
print("input keys:", list(data["input_data"].keys()))
print("in_ptr shape:", x.shape, "out_ptr shape:", out.shape)
