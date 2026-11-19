The **CENT** architecture (*PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference*, Gu et al., **ASPLOS 2025**) proposes running *Large Language Model* (LLM) inference without GPUs, replacing expensive HBM memory with **GDDR6-AiM** modules connected over **CXL 3.0**.

While the analytical and simulation gains presented in the paper are substantial, a practical gap separates academic figures from real execution: there is no system layer capable of accepting **PyTorch** tensors and dispatching them onto the architecture's execution model.

This technical report documents the architectural context, the physical bottlenecks motivating the design, the role of a **hardware dispatcher**, how the official simulator (**`aim_simulator` / Ramulator 2.0**) works, and the implementation of [`cent_runtime`](https://github.com/josedercysilva/cent-pim-runtime) — a **C++17 + C ABI + Python** bridge connecting linear layers of models such as **Llama-2 7B** to the CENT instruction set architecture.

---

## Table of Contents

1. **Fundamentals and the *Memory Wall* Bottleneck** — Classical memory hierarchy, DRAM bottleneck anatomy, and the *Prefill* (`GEMM`) vs. *Decoding* (`GEMV`) duality.
2. **The CENT Architecture (ASPLOS '25)** — *GPU-Free* CXL 3.0 topology, PIM/PNM microarchitecture, measured gains, and critical limitations.
3. **What `aim_simulator` (Ramulator 2.0) Is** — How the paper's cycle-accurate simulator works and why it cannot talk to PyTorch on its own.
4. **What a *Hardware Dispatcher* Is** — The role of the intermediary layer between AI frameworks and the memory controller/hardware.
5. **Our Implementation: Anatomy of `cent_runtime`** — Python layer (`ops.py`, `_bindings.py`, `simulator.py`), C ABI boundary (`extern "C"`), 14-byte binary packet (`CentInstruction`), 2 MB SRAM buffer, and C++17 trace emitter.
6. **Validation on Llama-2 7B** — Exact metrics for Attention (`4K×4K`) and MLP (`11K×4K`), 99.95% bus traffic savings, SRAM *hex dump*, and practical usage.
7. **Conclusion and References** — What the implementation solves today, its boundaries, and references to the code and foundational work.

---

## 1. Fundamentals and the *Memory Wall* Bottleneck

In classical computer architecture (Patterson & Hennessy, Ch. 5), the memory hierarchy balances speed, capacity, and cost through a well-known tiering:

```text
CPU / GPU  <--->  L1 Cache  <--->  L2 Cache  <--->  L3 Cache  <--->  Main Memory (DRAM)  <--->  SSD / Disk
 (fastest, smallest capacity)                                                        (slowest, largest capacity)
```

This arrangement works well when **temporal and spatial locality** hold — that is, when data fetched from DRAM is reused dozens or hundreds of times inside caches before eviction. LLM inference breaks this assumption during its dominant execution phase.

### Physical Anatomy of the DRAM Bottleneck

Inside a DRAM chip, data is organized into 2D cell arrays. When a row is activated, an entire block of **8,192 bits** is loaded at once into the internal **Row Buffer** in roughly 1 internal access cycle — meaning **massive bandwidth exists inside the memory die itself**.

The bottleneck arises at the chip boundary. To move those 8,192 bits to the processor (CPU or GPU), everything must pass through the **external bus** and physical I/O pins operating at roughly **64 bits per cycle** (or 32 bits in specific pin configurations). It takes roughly **128 bus cycles just to transfer a single row** that was already sitting ready in the *Row Buffer*. While data streams over the bus, GPU compute cores enter a **stall** state waiting on memory.

### The *Prefill* (`GEMM`) vs. *Decoding* (`GEMV`) Duality

LLM inference runs in two phases with opposite physical execution profiles:

| Characteristic | Phase 1: *Prefill* (Initial Prompt) | Phase 2: *Decoding* (Token-by-Token Generation) |
| :--- | :--- | :--- |
| **Share of total runtime** | ~2% of inference time | **~98% of inference time** |
| **Mathematical operation** | Matrix × Matrix (`GEMM`, e.g., `4K×4K` × `4K×4K`) | Vector × Matrix (`GEMV`, `batch_size = 1`) |
| **Arithmetic intensity** | High (~2,000 Ops/Byte) | Very low (**1 to 16 Ops/Byte**) |
| **Execution regime** | **Compute-Bound** (saturates *Tensor Cores*) | **Memory-Bound** (pinned to memory bandwidth) |

During *decoding*, generating each token requires streaming entire weight matrices (tens of megabytes per layer, totaling tens of gigabytes across the full model) only to perform a single vector-matrix product and immediately discard the weights. On a **4× NVIDIA A100** node, this lack of reuse limits *Tensor Core* utilization to roughly **21%** (**79% of cycles spent stalled** waiting on HBM).

---

## 2. The CENT Architecture (ASPLOS '25)

The core idea of **Processing-in-Memory (PIM)** — specifically the **GDDR6-AiM** (*Accelerator-in-Memory*) design used in CENT — is to place small Arithmetic Logic Units (**ALUs / PUs** with multiply-accumulate units) directly next to each DRAM bank, before the narrow I/O pin bottleneck.

Each GDDR6-AiM chip therefore consumes weights directly at the output of the *Row Buffer* without wasting energy and cycles crossing the external bus:
* An **NVIDIA A100** delivers roughly **2 TB/s** of peak HBM2e bandwidth.
* A **GDDR6-PIM** device reaches up to **16 TB/s** of effective internal bandwidth.
* Across a full **32-device CENT** system, aggregate internal bandwidth reaches **512 TB/s**.

### Global Topology and Internal Microarchitecture

Instead of relying on dedicated GPUs and costly HBM interposers, CENT connects a **Host CPU** to **32 CXL 3.0 devices** through a 48-port **CXL Switch**:

```text
                       +------------+
                       |  Host CPU  |
                       +-----+------+
                             |
                             v
                   +--------------------+
                   |   CXL 3.0 Switch   |
                   +---------+----------+
                             |
        +--------------------+--------------------+
        |                    |                    |
        v                    v                    v
+---------------+    +---------------+    +---------------+
| CXL Device #0 |    | CXL Device #1 | .. | CXL Device #31|
+-------+-------+    +---------------+    +---------------+
        |
        +---> CXL Port + Inter-device Communication Controller
        +---> Instruction Buffer (2 MB SRAM) -> PC -> Decoder
        +---> PNM Units + Shared Buffer (64 KB)
        +---> 16x PIM Ctrl ---> 16x GDDR6-PIM (32 channels per board)
```

Inside each of the 32 CXL devices, computation is partitioned into two complementary tiers:
1. **PIM Units (inside DRAM banks):** execute >99% of *decoding* math — dense linear projections (`GEMV`) — reading the activation vector from a *Global Buffer* and accumulating dot products in parallel across all 16 banks of each channel.
2. **PNM (*Processing-Near-Memory*) Units + Host CPU:** handle lightweight operations (<1% of FLOPs) requiring non-linear logic or cross-channel global reductions such as `Softmax`, `RMSNorm`, `SiLU`, and `RoPE`, transferring only compact partial vectors (<16 KB) instead of entire weight matrices.

### What the Paper's Results Show (Gains and Limitations)

In benchmarks against a **4× NVIDIA A100** server (80 GB each):
* ***Decoding* performance:** CENT achieves up to **4.13× higher tokens/s** on Llama-2 13B (**2.97×** on 7B) and **2.32×** geometric mean overall (*end-to-end*), since *decoding* dominates 98% of total latency.
* **Energy efficiency and cost:** each CXL board draws ~32.4 W (vs. ~300 W per GPU), delivering **2.88× more tokens per Joule** (up to **4.26×** during *decoding*) at roughly **\$14,873** (CPU + 512 GB GDDR6-PIM + controllers + CXL switch) compared to **\$42,128** for the 4× A100 cluster — around **3× lower hardware cost**.

On the other hand, a critical technical assessment also highlights **three real limitations** of the architecture:
1. **Slow Prefill (0.33× to 0.46×):** small DRAM-bank ALUs lack the dense matrix throughput of *Tensor Cores*. Long prompts (such as 32k-token RAG workloads) suffer severe prefill slowdown.
2. **Scaling bottleneck above 70B:** *decoding* speedup drops from **4.13×** (13B) to **1.25×** (70B) as inter-board communication across the CXL switch grows.
3. **Software and ISA barrier:** CENT introduces a custom ISA (CENT ISA) with no native PyTorch, vLLM, or mature ecosystem comparable to 15+ years of CUDA tooling.

This third barrier — the **missing link between high-level ML software and the CENT simulator/hardware** — is what our work addresses.

---

## 3. What `aim_simulator` (Ramulator 2.0) Is

To validate the microarchitecture prior to tape-out, the CENT authors released **`aim_simulator`**, built as an extension of **Ramulator 2.0** (one of the most widely adopted cycle-accurate DRAM simulators in computer architecture research).

In practice, `aim_simulator` compiles into a `ramulator2` binary (backed by `libramulator.so`). It is **not an AI runtime** and does not evaluate neural networks; it is a **discrete-event hardware simulator** structured as follows:

```text
  [ configs/gddr6_aim.yaml ] ──┐
  (Physical parameters:        │
   32 channels, 16 banks,      ├──► [ ramulator2 (aim_simulator) ] ──► Silicon Counters
   GDDR6 timings, FRFCFS)      │                                       (e.g. CH31_AiM_ISR_WR_GB_cycles,
                               │                                             memory_system_cycles)
  [ file.trace ] ──────────────┘
  (Static list of textual
   CENT ISA commands)
```

When invoked via CLI (`ramulator2 -f configs/gddr6_aim.yaml -t file.trace`), it simulates every *Row Buffer* access, bank conflict, command scheduler decision (`FRFCFS`), refresh (`AllBank`), and AiM instruction cycle-by-cycle across **32 GDDR6-AiM channels**, outputting exact cycle counters per instruction class.

### Why `aim_simulator` Alone Cannot Serve Software Pipelines

1. **Consumes only static `.trace` files:** users must manually generate text files containing `AiM WR_GB`, `AiM MAC_ABK`, etc., already decomposed into rows and physical addresses.
2. **Computes no numerical outputs:** the simulator models *timing and cycles*, not floating-point weights, so it cannot return an output tensor `y` to feed subsequent Transformer layers in PyTorch.
3. **Research-grade build dependencies:** as an academic artifact, compiling it relies on pinned `gcc`/`cmake` toolchains, making direct embedding inside standard Python ML pipelines impractical.

---

## 4. What a *Hardware Dispatcher* Is

In systems architecture, a **dispatcher** is the low-level software layer sitting between a high-level application framework and an accelerator device. When you run `torch.matmul(x, W)` on an NVIDIA GPU, PyTorch does not touch GPU registers directly: it routes through the *PyTorch Dispatcher* into native C/C++ libraries (`libcudart.so`, `libcublas.so`) acting as hardware intermediaries.

For CENT, a **hardware dispatcher** must fulfill four well-defined responsibilities:

1. **Interception and Scope Validation:** receive Python tensors and verify that the operation respects the accelerator's physical contract (for CENT PIM, vector-matrix multiplication `GEMV` with `batch_size = 1`). Any *prefill* `GEMM` (`batch > 1`) is rejected from PIM offload so the host handles it.
2. **Topology Routing:** select **which of the 32 CXL boards** (`DVid`) hosts the layer and **which of the 32 GDDR6 channels** (`CHmask`) are active for the hidden dimension `K`.
3. **Lowering to the Hardware ISA:** decompose the `[M × K]` weight matrix into blocks matching the **16 DRAM banks** per channel, encode each step into binary **CENT ISA** packets inside the 2 MB instruction buffer (SRAM), and optionally emit a `.trace` file.
4. **Telemetry Collection and Numerical Integrity:** compute the output vector `y` directly into PyTorch's destination memory (keeping the neural network executable end-to-end) and return auditable counts of emitted instructions and saved bus traffic.

---

## 5. Our Implementation: Anatomy of `cent_runtime`

**`cent_runtime`** implements this dispatcher as a clean two-layer architecture — **Python (PyTorch/NumPy)** calling **C++17** across a **stable C ABI (`libcent_runtime.so`)** — plus an optional automated bridge to `aim_simulator` (`ramulator2`).

```text
+-----------------------------------------------------------------------------+
| 1. PYTHON LAYER (python/cent_runtime/)                                      |
|    cent_runtime.matmul(x, W, layer_id=0, trace_path=None, simulate=False)   |
|    - Validates GEMV contract (batch=1) in ops.py                            |
|    - Normalizes to contiguous float32 and extracts pointer via ctypes       |
+--------------------------------------+--------------------------------------+
                                       | ctypes (_bindings.py)
                                       v
======================= [ C ABI BOUNDARY (extern "C") ] =======================
| 2. PUBLIC CONTRACT (include/cent/runtime_api.h -> build/libcent_runtime.so) |
|    cent_dispatch_gemv(x, M, K, W, y, layer_id, trace_path, &telemetry)      |
+--------------------------------------+--------------------------------------+
                                       |
          +----------------------------+----------------------------+
          v                                                         v
+-----------------------------------+     +-----------------------------------+
| 3. C++17 ENGINE (src/)            |     | 4. NUMERICAL COMPUTATION          |
|  - topology.cpp   -> DVid/CHmask  |     |  - Row-major loop over W [M x K]  |
|  - sram_buffer.cpp -> 2 MB SRAM   |     |  - FP64 accumulation -> FP32 out  |
|  - isa_emitter.cpp -> PIM blocks  |     |  - Writes directly to pointer 'y' |
+-----------------+-----------------+     +-----------------------------------+
                  |
                  +---> Binary packets (14 B) in SRAM + Hex Dump
                  +---> .trace file (CENT ISA) ---> simulator.py -> ramulator2
```

Below is a breakdown of each component.

### 5.1. Python Layer (`python/cent_runtime/`)

The Python package exports `cent_runtime.matmul`, `simulate_on_ramulator`, and `get_instruction_buffer_hex`:

* **`ops.py` (Strict validation and direct `ctypes` pointer handoff):** accepts both `torch.Tensor` and `numpy.ndarray` (automatically falling back to NumPy when PyTorch is absent). Before crossing into C, it strictly validates the `GEMV` contract: `x` must have shape `(K,)` or `(1, K)`, `W` must have shape `(M, K)`, and the inner dimension `K` must match. 3D+ tensors or `batch_size > 1` immediately raise `ValueError` explaining that Near-Bank PIM banks strictly execute `GEMV` (*decoding*). It then ensures contiguous `float32` memory (avoiding extra copies when the tensor already resides contiguously in `float32` on CPU), allocates the output buffer `y` matching `(M,)` or `(1, M)`, and passes raw pointers directly through `ctypes` without serialization or IPC.
* **`_bindings.py` (Native loading and telemetry):** locates `build/libcent_runtime.so` (automatically running `make shared` if missing) and maps the `CentDispatchTelemetry` struct (56 bytes under x86_64 natural alignment) into a clean Python dictionary.
* **`simulator.py` (Optional Ramulator 2.0 bridge):** when `simulate=True` (creating a temporary `.trace` via `tempfile` if `trace_path=None` so disk stays clean), locates `ramulator2` in portable order (explicit `ramulator_bin` → `CENT_RAMULATOR_BIN` → `external/ramulator2` → `../CENT/aim_simulator/build/ramulator2` → `PATH`), runs `[ramulator_bin, "-f", config_path, "-t", trace_path]` via `subprocess`, and parses `wr_gb_cycles`, `mac_abk_cycles`, and `memory_system_cycles`.

### 5.2. Why a Separate C ABI Gate (`.h`) in Front of the C++17 Engine (`.hpp`)?

In C++, namespaced or overloaded functions undergo **Name Mangling**. A symbol like `cent::compute_dvid(int)` turns into `_ZN4cent12compute_dvidEi` in the object file, and C++ has no universal binary ABI across compilers (GCC, Clang, and MSVC lay out *vtables*, exceptions, and symbols differently).

By contrast, **C** is the lingua franca of operating systems and FFI (*Foreign Function Interface*) layers. Therefore, `cent_runtime` cleanly separates:
* **`.h` headers (`isa.h`, `telemetry.h`, `runtime_api.h`):** public boundary wrapped in `#ifdef __cplusplus extern "C" { #endif`. Exported symbols in `build/libcent_runtime.so` remain unmangled (`cent_init_runtime`, `cent_dispatch_gemv`, `cent_get_instruction_buffer_hex`), directly callable from Python `ctypes`.
* **`.hpp` headers (`topology.hpp`, `sram_buffer.hpp`, `isa_emitter.hpp`):** internal **modern C++17** code using `namespace cent`, classes, RAII, and `std::array`.

In addition, the entire core is compiled with `-std=c++17 -O3 -fPIC -Wall -Wextra -Werror` and **zero C++ exceptions (`throw` / `try-catch`)** on the control path. Boundary functions are marked `noexcept`, and errors return explicit `enum CentStatus` codes (`CENT_OK = 0`, `CENT_ERR_NULL_PTR = -1`, `CENT_ERR_INVALID_SHAPE = -2`, `CENT_ERR_SRAM_OVERFLOW = -3`, `CENT_ERR_TRACE_IO = -4`) with early returns.

### 5.3. The 14-Byte Binary Packet (`CentInstruction`)

In `include/cent/isa.h`, we define the controller constants (`CENT_SRAM_BUFFER_BYTES = 2 MB`, `32` CXL devices, `32` channels per device, `16` banks per channel, `64` bursts default operation size), the 5 CENT ISA opcodes, and the packed binary `struct`:

```c
enum CentOpcode : uint8_t {
    CENT_OP_WR_GB   = 0x02, // Writes activation vector to active channels' Global Buffer
    CENT_OP_WR_BIAS = 0x03, // Initializes accumulator registers across DRAM banks
    CENT_OP_RD_MAC  = 0x05, // Reads accumulated result from banks to channel buffer
    CENT_OP_MAC_ABK = 0x0F, // Parallel Multiply-Accumulate across all 16 banks (All-Bank)
    CENT_OP_EOC     = 0xFF  // End of Computation (terminates the aim_simulator trace)
};

#pragma pack(push, 1)
typedef struct CentInstruction {
    uint8_t  opcode;       //  8 bits : CENT ISA opcode
    uint8_t  dvid;         //  8 bits : Target CXL device ID (0 to 31)
    uint32_t channel_mask; // 32 bits : Bitmask of the 32 GDDR6 channels
    uint16_t bank_index;   // 16 bits : DRAM bank index (0 to 15)
    uint32_t row_addr;     // 32 bits : Physical row address (Row Buffer)
    uint16_t opsize;       // 16 bits : Operation size in bursts
} CentInstruction;
#pragma pack(pop)

static_assert(sizeof(CentInstruction) == 14,
              "ABI Error: CentInstruction must be exactly 14 bytes packed");
```

Combining `#pragma pack(push, 1)` with `static_assert` guarantees at compile time that no hidden alignment *padding* is inserted between fields: each instruction occupies exactly `1 + 1 + 4 + 2 + 4 + 2 = 14` contiguous bytes in SRAM.

### 5.4. Topology, 2 MB SRAM Buffer, and Near-Bank PIM Decomposition

Once `cent_dispatch_gemv` enters the C++ core (`src/runtime_api.cpp`), four modules execute in sequence:

1. **Routing (`src/topology.cpp`):**
   * `DVid = ((layer_id % 32) + 32) % 32` — distributes Transformer layers across the 32 CXL 3.0 devices.
   * `CHmask`: for `K >= 1024`, enables all 32 channels (`0xFFFFFFFFu`); for `K >= 512`, enables 16 channels (`0x0000FFFFu`); below that, enables 8 channels (`0x000000FFu`).
2. **2 MB SRAM Emulator (`src/sram_buffer.cpp`):**
   * `SramBuffer` holds a preallocated static **2 MB** `std::array<uint8_t, CENT_SRAM_BUFFER_BYTES>`, eliminating `new`/`delete` on the critical path. Each instruction is copied via `std::memcpy` if space permits, returning `CENT_ERR_SRAM_OVERFLOW` otherwise.
   * Also implements `format_hex_dump`, formatting up to the first 64 bytes as uppercase hex byte pairs (16 bytes per line) for binary inspection.
3. **PIM Block Decomposition and `.trace` Emission (`src/isa_emitter.cpp`):**
   * Because each GDDR6-AiM channel drives **16 banks in parallel** (*All-Bank PIM*), output dimension `M` is split into `num_blocks = (M + 15) / 16` blocks.
   * For each block `blk` (`row_addr = blk`), the emitter pushes 4 binary packets into SRAM (`WR_GB` with `opsize=64`, `WR_BIAS` with `opsize=0`, `MAC_ABK` with `opsize=64` and `row_addr=blk`, and `RD_MAC` with `opsize=0`) and, when `trace_path` is provided, writes via `std::ofstream` (RAII) the 4 corresponding `.trace` lines:
     ```text
     AiM WR_GB 64 0 0xffffffff
     AiM WR_BIAS 0 0xffffffff
     AiM MAC_ABK 64 0xffffffff 0
     AiM RD_MAC 0 0xffffffff
     ```
   * After completing all blocks, emits **1 final `CENT_OP_EOC` instruction** (`AiM EOC` in `.trace`), required for `aim_simulator` to terminate simulation.
4. **Numerical Computation and Bus Telemetry (`src/runtime_api.cpp` & `src/isa_emitter.cpp`):**
   * Computes `y = x · W^T` iterating `W` in row-major order with `double` (`FP64`) accumulation cast to `float` (`FP32`), writing directly into the caller-supplied `y` buffer.
   * Computes external bus traffic savings assuming **BF16** precision (2 bytes/element):
     * **GPU Baseline (HBM):** `(M × K + K) × 2` bytes (transfers full weight matrix `W` + input `x`).
     * **CENT CXL 3.0 (Near-Bank PIM):** `(K + M) × 2` bytes (weights remain stationary in DRAM; only input `x` and output `y` cross the bus).
     * **Savings (%):** `100 × (1 - pcie_traffic_bytes / gpu_traffic_bytes)`.

---

## 6. Validation on Llama-2 7B

The repository includes a Python unit test suite (`make test`), a native C++ testbench (`make test-core`), and a dedicated demonstration exercising real **Llama-2 7B** layer dimensions (`examples/llama2_7b_demo.py`).

### Running Locally

```bash
git clone https://github.com/josedercysilva/cent-pim-runtime.git
cd cent-pim-runtime

./setup_env.sh
source .venv/bin/activate

make test        # Tests C ABI, numerical accuracy, GEMM rejection, SRAM, and .trace
make test-core   # Native C++ testbench
make demo        # Runs Llama-2 7B Attention and MLP layers (Ramulator optional)
make simulate    # Optional: runs Ramulator 2.0 on the generated .trace files
```

### PyTorch Usage Example

```python
import torch
import torch.nn.functional as F
import cent_runtime

# Decoding phase (batch_size = 1)
x = torch.randn(1, 4096)        # Input activation
W = torch.randn(4096, 4096)     # Stationary weights in GDDR6-AiM memory

# Dispatch to the C++ runtime (returns PyTorch tensor + telemetry)
y, telemetry = cent_runtime.matmul(x, W, layer_id=0)

# Non-linear operations continue normally on CPU via PyTorch
out = F.silu(y)

print(f"Output: {out.shape}")
print(f"CXL Board: #{telemetry['dvid']} | Mask: {telemetry['channel_mask_hex']}")
print(f"Instructions: {telemetry['total_instructions']} ({telemetry['instruction_buffer_bytes']} B SRAM)")
print(f"Bus traffic saved: {telemetry['memory_traffic_saved_pct']:.2f}%")
```

### Audited Metrics (Attention vs. MLP in Llama-2 7B)

When executing the **Attention** projection (`q/k/v/o_proj`, `M = 4096, K = 4096, layer_id = 0`) and **MLP** projection (`gate/up/down_proj`, `M = 11008, K = 4096, layer_id = 1`), the metrics match both the analytical model and the cycle-accurate simulator:

| Metric | Attention Layer (`4096 × 4096`) | MLP Layer (`11008 × 4096`) |
| :--- | :--- | :--- |
| **CXL Device (`DVid`)** | Board `#0` | Board `#1` |
| **Active Channels (`CHmask`)** | `0xffffffff` (`32` channels) | `0xffffffff` (`32` channels) |
| **16-Bank Blocks (`ceil(M/16)`)** | `256` blocks | `688` blocks |
| **Total Instructions (`4/block + EOC`)** | **`1,025`** (`256×4 + 1`) | **`2,753`** (`688×4 + 1`) |
| **SRAM Footprint (`14 B/inst`)** | **`14,350` bytes** (`14.01 KB`) | **`38,542` bytes** (`37.64 KB`) |
| **GPU HBM vs. CENT CXL Traffic (BF16)** | `~32.01 MB` vs. `16.00 KB` (**99.95%** less) | Activation + output only on the bus |
| **Ramulator 2.0 Cycles (`CH31_AiM_ISR_WR_GB_cycles`)** | **`16,416` cycles** | **`44,064` cycles** |

Notice the physical consistency of the scaling ratio between the two layers: the MLP output dimension is `11,008 / 4,096 = 2.6875×` larger than Attention, and both the number of generated SRAM instructions and the silicon cycles measured by `aim_simulator` scale by **`2.68×`** (`44,064 / 16,416`).

Inspecting the first SRAM bytes via `cent_get_instruction_buffer_hex` exposes the 14-byte *little-endian* binary packet directly in memory (`opcode = 0x02` for `WR_GB`, `dvid = 0x00`, `channel_mask = FF FF FF FF`, `bank = 00 00`, `row = 00 00 00 00`, `opsize = 64 = 0x0040 -> 40 00`, followed by the start of the second instruction `03 00` = `WR_BIAS`):

```text
[SRAM] 02 00 FF FF FF FF 00 00 00 00 00 00 40 00 03 00
```

---

## 7. Conclusion and References

`cent_runtime` bridges computer architecture research and systems software engineering:

* **Without requiring `ramulator2`**, developers can install the Python package, dispatch PyTorch/NumPy projections through a C++17 C ABI via direct pointers (no serialization, and *zero-copy* for contiguous `float32` tensors on CPU), inspect the 14-byte SRAM packet *hex dump*, and verify the 99.95% reduction in external bus traffic.
* **When coupled with `aim_simulator`**, the emitted `.trace` files enable cycle-accurate validation across 32 GDDR6-AiM channels without writing a single trace line by hand.

### References

1. **Project Code:** [`josedercysilva/cent-pim-runtime`](https://github.com/josedercysilva/cent-pim-runtime)
2. **Original Paper (CENT):** Gu, Y., Khadem, A., et al. (2025). *PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference*. **ASPLOS '25**.
3. **Memory Hierarchy Fundamentals:** Patterson, D. A., & Hennessy, J. L. (2014). *Computer Organization and Design: The Hardware/Software Interface* (5th Ed. — MIPS Edition), Ch. 5.
4. **CXL Specification:** Compute Express Link Consortium (2022). *CXL Specification Revision 3.0*.
