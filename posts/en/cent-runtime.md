## Context and Motivation

During the *decoding* phase of Large Language Model (LLM) inference such as Llama-2 7B (`batch_size = 1`), throughput is constrained by memory bandwidth (**Memory Wall**) rather than GPU compute capacity. For each generated token, gigabytes of stationary model weights must be transferred from memory to compute units solely to execute vector-matrix multiplications (`GEMV`).

The **CENT** architecture (*PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference*, Gu et al., **ASPLOS 2025**) mitigates this bottleneck by offloading `GEMV` operations to **Near-Bank PIM** units integrated into **GDDR6-AiM** memory banks coupled over **CXL 3.0**.

However, the reference simulator for the architecture (`aim_simulator`, built on **Ramulator 2.0**) operates strictly as a standalone C++ binary consuming static trace files (`.trace`), with no programmatic interface for machine learning frameworks.

This technical report documents the architecture and implementation of [`cent_runtime`](https://github.com/josedercysilva/cent-pim-runtime): a **C++17** runtime and hardware dispatcher with a native **C ABI** boundary, developed at UFAM to connect **PyTorch** and **NumPy** linear layers directly to the CENT ISA dispatch model and Ramulator 2.0 trace generation.

---

## Architecture Overview

`cent_runtime` adopts the same binary integration pattern used by libraries such as `libcudart` and `libcublas`: the Python layer has no direct dependency on hardware registers or memory alignment details and communicates exclusively through a stable C boundary (`extern "C"`).

```text
+-----------------------------------------------------------------------+
|  PYTHON LAYER (python/cent_runtime/)                                  |
|  cent_runtime.matmul(x, W, layer_id=0, simulate=False)                |
|  - Validates GEMV scope (batch = 1) for PyTorch Tensor / NumPy ndarray|
+-----------------------------------+-----------------------------------+
                                    | ctypes (contiguous pointer, zero copy)
                                    v
+-----------------------------------------------------------------------+
|  C ABI BOUNDARY (include/cent/runtime_api.h -> libcent_runtime.so)    |
|  cent_dispatch_gemv(x, M, K, W, y, layer_id, trace_path, &telemetry)  |
+-----------------------------------+-----------------------------------+
                                    |
          +-------------------------+-------------------------+
          v                                                   v
+-----------------------------+               +-----------------------------+
|  C++17 CORE (src/)          |               |  NUMERICAL EXECUTION (FP64) |
|  - topology: DVid & CHmask  |               |  y = x · W^T                |
|  - sram_buffer: 2 MB SRAM   |               |  Preserves tensor integrity |
|  - isa_emitter: PIM blocks  |               |  for PyTorch layers         |
+-------------+---------------+               +-----------------------------+
              |
              +---> Binary Buffer (14-byte packets in RAM/SRAM)
              +---> .trace File (CENT ISA) ---> Ramulator 2.0 (optional)
```

---

## Execution Partitioning: PIM vs. Host CPU

In the CENT architecture, AiM-equipped DRAM banks include only multiply-accumulate (`MAC`) units, without support for general-purpose arithmetic operations. `cent_runtime` enforces this strict separation between **Near-Bank PIM** and **PNM / Host CPU**:

| Model Operation | Execution Target | Architectural Rationale |
| :--- | :--- | :--- |
| Dense linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with `batch_size = 1` | **`cent_runtime` (Near-Bank PIM)** | `GEMV` operations dominate weight memory traffic (>99%) during *decoding* |
| `Softmax`, `RMSNorm`, `SiLU`, `RoPE` | **Host CPU (native PyTorch)** | Non-linear operations, reductions, and transcendental functions remain on host / PNM |
| *Prefill* phase (`batch_size > 1`) or 3D+ tensors | **Host CPU (rejected by PIM ABI)** | The dispatcher strictly enforces the `GEMV` contract and raises `ValueError` for `GEMM` |

---

## C ABI Specification, 14-Byte Packet, and SRAM Buffer

To keep the cross-language boundary predictable, the shared library `libcent_runtime.so` is compiled with `-std=c++17 -O3 -Wall -Wextra -Werror`, **without C++ exceptions (`throw`/`try-catch`)** in the control flow. Errors are returned explicitly through `enum CentStatus` (`0` for success, negative codes such as `-1` and `-2` for null pointers or invalid dimensions), with functions marked `noexcept`.

### 1. The CENT ISA Binary Packet (`CentInstruction`)

Each command sent to the CXL controller is packed into exactly **14 bytes** (`include/cent/isa.h`), enforcing tight binary layout at compile time via `static_assert`:

```c
enum CentOpcode : uint8_t {
    CENT_OP_WR_GB   = 0x02, // Writes activation vector to active channels' Global Buffer
    CENT_OP_WR_BIAS = 0x03, // Initializes accumulators across DRAM banks
    CENT_OP_RD_MAC  = 0x05, // Reads accumulated result from the banks
    CENT_OP_MAC_ABK = 0x0F, // Parallel Multiply-Accumulate across all 16 banks (All-Bank)
    CENT_OP_EOC     = 0xFF  // End of Computation (terminates the simulator trace)
};

#pragma pack(push, 1)
typedef struct CentInstruction {
    uint8_t  opcode;       // 1 byte  : CENT ISA opcode
    uint8_t  dvid;         // 1 byte  : Target CXL device (0 to 31)
    uint32_t channel_mask; // 4 bytes : Bitmask of the 32 GDDR6-AiM channels
    uint16_t bank_index;   // 2 bytes : DRAM bank index (0 to 15)
    uint32_t row_addr;     // 4 bytes : Physical row address (Row Buffer)
    uint16_t opsize;       // 2 bytes : Operation size in bursts (64 by default)
} CentInstruction;
#pragma pack(pop)

static_assert(sizeof(CentInstruction) == 14,
              "ABI Error: CentInstruction must be exactly 14 bytes packed");
```

### 2. Topology, Decomposition, and 2 MB SRAM Buffer

Upon receiving a `cent_dispatch_gemv(x, M, K, W, y, layer_id, trace_path, &telem)` call:

1. **Device routing (`DVid`):** selects one of the 32 CXL 3.0 boards using `((layer_id % 32) + 32) % 32`.
2. **Channel mask (`CHmask`):** enables `32` channels (`0xffffffff`) for `K >= 1024`, `16` channels (`0x0000ffff`) for `K >= 512`, or `8` channels (`0x000000ff`) below that.
3. **16-bank block decomposition:** splits `M` into `ceil(M / 16)` blocks. For each block `row_addr`, writes the sequence `WR_GB`, `WR_BIAS`, `MAC_ABK`, and `RD_MAC` into a contiguous **2 MB** static buffer (`std::array<uint8_t, 2 * 1024 * 1024>`) — with zero dynamic allocations on the critical path. At the end, it emits `EOC`.

When `trace_path` is provided, the exact same pass emits the textual `.trace` file consumed by Ramulator 2.0:

```text
AiM WR_GB 64 0 0xffffffff
AiM WR_BIAS 0 0xffffffff
AiM MAC_ABK 64 0xffffffff 0
AiM RD_MAC 0 0xffffffff
...
AiM EOC
```

Additionally, `cent_get_instruction_buffer_hex` exposes the *hex dump* of the first 64 bytes written into the emulated SRAM.

---

## Build, Validation, and Integration

### Build and Local Test Suite

The project is self-contained. To create the virtual environment, compile the shared library `build/libcent_runtime.so`, and run the test suites and benchmark demo:

```bash
git clone https://github.com/josedercysilva/cent-pim-runtime.git
cd cent-pim-runtime

./setup_env.sh
source .venv/bin/activate

make test        # Python suite (unittest): ABI, numerical precision, shape errors, and .trace
make test-core   # Native C++ testbench
make demo        # Demonstration with real Llama-2 7B layers
```

> **About Ramulator 2.0:** The cycle-accurate hardware simulator is **100% optional**. The C++ core independently computes numerical outputs, the SRAM *hex dump*, instruction counts, and traffic telemetry. Ramulator is only invoked when `simulate=True` is passed or `make simulate` is executed.

### PyTorch Integration Example

In Python, a `GEMV` linear projection is dispatched via `cent_runtime.matmul`, while non-linear activations remain on the CPU:

```python
import torch
import torch.nn.functional as F
import cent_runtime

# Single-token decoding activation and stationary weight matrix in DRAM
x = torch.randn(1, 4096)
W = torch.randn(4096, 4096)

# Dispatch to the C++ core (returns PyTorch tensor + telemetry dictionary)
y, telemetry = cent_runtime.matmul(x, W, layer_id=0)

# Non-linear operation stays transparent in PyTorch
out = F.silu(y)

print(f"Output: {out.shape}")
print(f"CXL Board (DVid): {telemetry['dvid']} | Channels: {telemetry['channel_mask_hex']}")
print(f"Instructions emitted: {telemetry['total_instructions']}")
print(f"External traffic saved: {telemetry['memory_traffic_saved_pct']:.2f}%")
```

If `torch` is not installed in the environment, the package automatically falls back to `numpy.ndarray`, preserving the input's 1D `(M,)` or 2D `(1, M)` shape.

---

## Experimental Validation on Llama-2 7B

The `examples/llama2_7b_demo.py` script exercises two real Llama-2 7B layers (assuming BF16 precision at 2 bytes/element when estimating external bus traffic against a GPU baseline streaming weights out of HBM):

| Layer (Llama-2 7B) | Dimensions (`M × K`) | `DVid` | Instructions (`EOC` included) | SRAM Used | GPU vs. CENT CXL Traffic | `WR_GB` Cycles (Ramulator 2.0) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Attention** (`q/k/v/o proj`) | `4096 × 4096` | `0` | `1,025` (`256` blocks × 4 + `1`) | `14,350 B` | `~32.01 MB` vs. `16.00 KB` (**99.95%** less) | `16,416` |
| **MLP** (`gate/up/down proj`) | `11008 × 4096` | `1` | `2,753` (`688` blocks × 4 + `1`) | `38,542 B` | Activation + output only on the bus | `44,064` (`2.68x`) |

Proportional scaling is observed across layers: since the MLP layer has `11,008 / 4,096 = 2.6875x` more output rows than the Attention layer, both total instructions emitted into SRAM and the `CH31_AiM_ISR_WR_GB_cycles` counter reported by the simulator scale by the same factor (`~2.68x`).

---

## References

* **Project repository:** [github.com/josedercysilva/cent-pim-runtime](https://github.com/josedercysilva/cent-pim-runtime)
* **Base paper:** Gu et al., *PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference*, ASPLOS 2025.
