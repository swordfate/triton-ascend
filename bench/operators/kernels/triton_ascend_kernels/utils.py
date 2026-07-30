# Source: triton-ascend-kernels/src/triton_ascend_kernels/utils.py
# (minimal copy — only the functions needed by matmul.py)

from functools import lru_cache

import torch
import triton.runtime.driver as driver


@lru_cache(maxsize=1)
def get_npu_aicore_num():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)["num_aicore"]
