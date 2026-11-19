The CENT architecture (*PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference*, Gu et al., ASPLOS 2025) proposes running Large Language Model (LLM) inference without GPUs, replacing HBM memory with GDDR6-AiM modules connected over CXL 3.0.

While the analytical and simulation gains reported in the paper are substantial, a practical gap separates academic results from software execution: there is no system layer capable of taking PyTorch tensors and dispatching them onto the architecture's execution model.

This technical report documents the architectural context, the physical bottlenecks motivating the project, the role of a hardware dispatcher, how the official simulator (`aim_simulator` / Ramulator 2.0) operates, and the implementation of [`cent_runtime`](https://github.com/josedercysilva/cent-pim-runtime) — a C++17, C ABI, and Python library connecting linear layers of models such as Llama-2 7B to the CENT instruction set architecture. The seminar slide deck accompanying this work is available at the repository root in [`apresentacao_cent.pdf`](https://github.com/josedercysilva/cent-pim-runtime/blob/main/apresentacao_cent.pdf).

---

## Table of Contents

1. Fundamentals and the Memory Wall Bottleneck
2. The CENT Architecture (ASPLOS '25)
3. The Official Simulator: `aim_simulator` (Ramulator 2.0)
4. Role of the Hardware Dispatcher
5. Implementation of `cent_runtime`
6. Validation on Llama-2 7B
7. Conclusion and References

---

## 1. Fundamentals and the Memory Wall Bottleneck

In classical computer architecture (Patterson & Hennessy, Ch. 5), the memory hierarchy balances speed, capacity, and cost across successive tiers:

```text
CPU / GPU  <--->  L1 Cache  <--->  L2 Cache  <--->  L3 Cache  <--->  Main Memory (DRAM)  <--->  SSD / Disk
 (fastest, smallest capacity)                                                        (slowest, largest capacity)
```

This organization relies on temporal and spatial locality: a block fetched from DRAM must be reused dozens or hundreds of times in caches before eviction. LLM inference breaks this assumption during its longest execution phase.

### Physical Bottleneck at the DRAM Boundary

Inside a DRAM chip, data is organized into two-dimensional cell arrays. Activating a row loads an entire 8,192-bit block at once into the internal Row Buffer in roughly 1 internal access cycle — meaning high internal bandwidth exists within the memory die itself.

The bottleneck sits at the chip's output interface. Moving those 8,192 bits to the processor (CPU or GPU) requires crossing the external bus and physical I/O pins operating at roughly 64 bits per cycle (or 32 bits in specific pin configurations). Transferring a single row that was already sitting in the Row Buffer takes roughly 128 bus cycles. While data crosses the bus, GPU compute units remain in a *stall* state waiting on memory.

### Prefill (`GEMM`) vs. Decoding (`GEMV`)

LLM inference splits into two phases with opposite physical profiles:

| Characteristic | Phase 1: Prefill (Initial Prompt) | Phase 2: Decoding (Token-by-Token Generation) |
| :--- | :--- | :--- |
| Share of total runtime | ~2% of inference time | ~98% of inference time |
| Mathematical operation | Matrix × Matrix (`GEMM`, e.g., `4K×4K` × `4K×4K`) | Vector × Matrix (`GEMV`, `batch_size = 1`) |
| Arithmetic intensity | High (~2,000 Ops/Byte) | Low (1 to 16 Ops/Byte) |
| Execution regime | Compute-Bound (saturates Tensor Cores) | Memory-Bound (limited by memory bandwidth) |

During decoding, generating each token requires reading entire weight matrices (tens of megabytes per layer, totaling tens of gigabytes across the model) only to compute a single vector-matrix product and discard the weights immediately after. On a 4× NVIDIA A100 node, this lack of reuse drops effective Tensor Core utilization to roughly 21% (with 79% of execution time spent stalled on HBM reads).

---

## 2. The CENT Architecture (ASPLOS '25)

Processing-in-Memory (PIM) — specifically the GDDR6-AiM (*Accelerator-in-Memory*) design adopted in CENT — places multiply-accumulate execution units (PUs/ALUs) directly beside each DRAM bank, ahead of the narrow I/O pin interface.

Each GDDR6-AiM chip therefore consumes weights directly at the Row Buffer output without streaming weight matrices over the external bus:

- **NVIDIA A100:** delivers roughly 2 TB/s of peak HBM2e bandwidth.
- **GDDR6-PIM device:** reaches up to 16 TB/s of aggregate internal bank bandwidth.
- **CENT system (32 devices):** aggregates up to 512 TB/s of total internal bandwidth.

### CXL 3.0 Topology and Internal Microarchitecture

Instead of dedicated GPUs and HBM interposers, CENT connects a Host CPU to 32 CXL 3.0 devices through a 48-port CXL Switch:

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

Inside each CXL device, execution is divided across two tiers:

1. **PIM Units (DRAM banks):** execute over 99% of decoding arithmetic — dense linear projections (`GEMV`) — reading the input vector from a Global Buffer and accumulating dot products in parallel across the 16 banks of each channel.
2. **PNM (Processing-Near-Memory) Units and Host CPU:** run the remaining operations (<1% of arithmetic volume) requiring non-linear logic or cross-channel reductions such as `Softmax`, `RMSNorm`, `SiLU`, and `RoPE`, transferring only compact partial vectors (<16 KB).

### Reported Gains and Architectural Limitations

In the paper's evaluation against a 4× NVIDIA A100 server (80 GB per GPU):

- **Decoding throughput:** CENT reaches up to 4.13× higher tokens/s on Llama-2 13B (2.97× on 7B) and 2.32× geometric mean overall (*end-to-end*), since decoding accounts for 98% of total runtime.
- **Energy efficiency and cost:** each CXL board draws ~32.4 W (vs. ~300 W for a GPU), delivering 2.88× more tokens per Joule (up to 4.26× during decoding) at an estimated hardware cost of \$14,873 (CPU + 512 GB GDDR6-PIM + controllers + CXL switch) compared to \$42,128 for the 4× A100 node.

At the same time, the architecture has three clear technical limitations:

1. **Slow prefill (0.33× to 0.46×):** in-bank DRAM arithmetic units lack the dense matrix throughput of GPU Tensor Cores. On long prompts (such as 32k-token RAG workloads), prefill latency degrades overall response time.
2. **Scaling above 70B:** decoding speedup drops from 4.13× (13B) to 1.25× (70B) as inter-board communication across the CXL switch increases.
3. **Missing software stack:** CENT defines a custom instruction set (CENT ISA) without native integration into PyTorch or mature libraries comparable to the CUDA ecosystem.

This third gap is the focus of our work: building the system layer that connects high-level tensors to the architecture's instruction format and cycle-accurate simulator.

---

## 3. The Official Simulator: `aim_simulator` (Ramulator 2.0)

To validate the microarchitecture, the CENT authors released `aim_simulator`, built on top of Ramulator 2.0 (a cycle-accurate academic DRAM simulator).

`aim_simulator` compiles into a `ramulator2` executable (paired with `libramulator.so`). It is not a neural network runtime, but a discrete-event hardware simulator:

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

When invoked from the command line (`ramulator2 -f configs/gddr6_aim.yaml -t file.trace`), it models Row Buffer accesses, bank conflicts, `FRFCFS` command scheduling, `AllBank` refresh, and AiM instructions cycle by cycle across 32 GDDR6-AiM channels, printing per-instruction cycle counters on completion.

### Limitations for Software Integration

1. **Consumes only static `.trace` files:** requires pre-generating text files containing `AiM WR_GB`, `AiM MAC_ABK`, etc., already decomposed into physical rows and channel masks.
2. **No numerical computation:** the simulator tracks timing and cycle counts without multiplying floating-point weights or producing an output tensor `y` for the next model layer.
3. **Toolchain coupling:** as a research artifact, building it depends on specific C++ and CMake toolchains, making direct embedding inside standard Python environments impractical.

---

## 4. Role of the Hardware Dispatcher

In systems software, a dispatcher is the intermediary layer sitting between a high-level framework and the accelerator controller. When PyTorch runs `torch.matmul(x, W)` on a GPU, the call passes through the internal dispatcher into native C/C++ libraries (`libcudart.so`, `libcublas.so`) that configure and submit work to the device.

For CENT, the hardware dispatcher fulfills four responsibilities:

1. **Physical contract validation:** inspects incoming Python tensor shapes and enforces vector-matrix multiplication (`GEMV`, `batch_size = 1`). Prefill `GEMM` calls (`batch_size > 1`) are rejected so they remain on the host.
2. **Topology routing:** determines which of the 32 CXL devices (`DVid`) hosts the layer and which of the 32 GDDR6 channels (`CHmask`) are enabled for dimension `K`.
3. **Lowering to the CENT ISA:** partitions the `[M × K]` weight matrix into blocks aligned to the 16 DRAM banks per channel, writes the binary descriptors into the 2 MB SRAM buffer, and emits the corresponding `.trace` file.
4. **Numerical integrity and telemetry:** computes the resulting vector `y` directly into PyTorch's output buffer and returns instruction counts, SRAM byte usage, and saved bus traffic.

---

## 5. Implementation of `cent_runtime`

`cent_runtime` structures this dispatcher into two main layers — Python (PyTorch/NumPy) and C++17 connected across a stable C ABI (`libcent_runtime.so`) — plus an optional integration module for `ramulator2`.

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

### 5.1. Python Layer (`python/cent_runtime/`)

The Python package exposes `cent_runtime.matmul`, `simulate_on_ramulator`, and `get_instruction_buffer_hex`:

- **`ops.py`:** accepts `torch.Tensor` or `numpy.ndarray` (falling back to NumPy when PyTorch is not installed). Validates `x` as `(K,)` or `(1, K)` and `W` as `(M, K)`. 3D+ tensors or `batch_size > 1` raise `ValueError`. It then ensures a contiguous `float32` layout on CPU (performing no extra copy when the tensor already satisfies this layout), allocates the output tensor `y`, and passes raw memory pointers directly via `ctypes`.
- **`_bindings.py`:** loads `build/libcent_runtime.so` (running `make shared` automatically if the library has not been built yet) and maps the C struct `CentDispatchTelemetry` (56 bytes on x86_64) into a Python dictionary.
- **`simulator.py`:** when `simulate=True` (creating a temporary `.trace` file via `tempfile` if `trace_path=None`), resolves the `ramulator2` binary in configurable order (explicit `ramulator_bin` → `CENT_RAMULATOR_BIN` environment variable → `external/ramulator2` → `../CENT/aim_simulator/build/ramulator2` → `PATH`), invokes the simulator via `subprocess`, and extracts `wr_gb_cycles`, `mac_abk_cycles`, and `memory_system_cycles`.

### 5.2. C ABI Boundary (`.h`) and C++17 Core (`.hpp`)

In C++, namespaced or overloaded functions undergo *name mangling* (for instance, `cent::compute_dvid(int)` becomes `_ZN4cent12compute_dvidEi` in the symbol table), and the C++ ABI differs across compilers. To guarantee direct FFI interoperability with `ctypes`, the codebase separates:

- **`.h` headers (`isa.h`, `telemetry.h`, `runtime_api.h`):** public interface declared under `extern "C"`, exporting unmangled symbols (`cent_init_runtime`, `cent_dispatch_gemv`, `cent_get_instruction_buffer_hex`).
- **`.hpp` headers (`topology.hpp`, `sram_buffer.hpp`, `isa_emitter.hpp`):** internal C++17 implementation using `namespace cent`, classes, RAII, and `std::array`.

The native library compiles with `-std=c++17 -O3 -fPIC -Wall -Wextra -Werror` and uses no C++ exceptions (`throw` / `try-catch`) across the boundary. All exported functions are marked `noexcept` and return explicit status codes via `enum CentStatus` (`CENT_OK = 0`, `CENT_ERR_NULL_PTR = -1`, `CENT_ERR_INVALID_SHAPE = -2`, `CENT_ERR_SRAM_OVERFLOW = -3`, `CENT_ERR_TRACE_IO = -4`).

### 5.3. 14-Byte Binary Descriptor (`CentInstruction`)

`include/cent/isa.h` defines the controller parameters (`CENT_SRAM_BUFFER_BYTES = 2 MB`, 32 CXL devices, 32 channels per device, 16 banks per channel, and `opsize = 64` bursts), the 5 CENT ISA opcodes, and the packed binary instruction layout:

```cpp
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

The `#pragma pack(push, 1)` directive verified by `static_assert` eliminates alignment padding between fields, fixing each instruction at 14 contiguous bytes (`1 + 1 + 4 + 2 + 4 + 2`).

### 5.4. Topology, 2 MB SRAM Buffer, and Instruction Emission

During `cent_dispatch_gemv` (`src/runtime_api.cpp`), four stages run in sequence:

1. **Routing (`src/topology.cpp`):** computes `DVid = ((layer_id % 32) + 32) % 32` to distribute layers across the 32 CXL devices. The channel mask `CHmask` enables 32 channels (`0xFFFFFFFFu`) when `K >= 1024`, 16 channels (`0x0000FFFFu`) when `K >= 512`, and 8 channels (`0x000000FFu`) for smaller dimensions.
2. **2 MB SRAM buffer (`src/sram_buffer.cpp`):** `SramBuffer` manages a preallocated static 2 MB `std::array<uint8_t, CENT_SRAM_BUFFER_BYTES>`, avoiding dynamic allocation (`new`/`delete`) during dispatch. Instructions are copied via `std::memcpy` with bounds checking (`CENT_ERR_SRAM_OVERFLOW`) and can be inspected in hexadecimal through `format_hex_dump`.
3. **Near-Bank PIM decomposition and `.trace` emission (`src/isa_emitter.cpp`):** since each channel drives 16 banks in parallel, output dimension `M` is partitioned into `num_blocks = (M + 15) / 16` blocks. For each block `blk` (`row_addr = blk`), the emitter writes 4 instructions into SRAM (`WR_GB`, `WR_BIAS`, `MAC_ABK`, and `RD_MAC`) and, when `trace_path` is provided, writes the corresponding lines to the `.trace` file:
   ```text
   AiM WR_GB 64 0 0xffffffff
   AiM WR_BIAS 0 0xffffffff
   AiM MAC_ABK 64 0xffffffff 0
   AiM RD_MAC 0 0xffffffff
   ```
   After all blocks complete, it emits a `CENT_OP_EOC` instruction (`AiM EOC`), required to terminate simulation in `aim_simulator`.
4. **Numerical computation and bus telemetry (`src/runtime_api.cpp` and `src/isa_emitter.cpp`):** computes `y = x · W^T` in row-major order using a `double` (`FP64`) accumulator cast to `float` (`FP32`) in the output pointer `y`. Alongside computation, it estimates external bus traffic in BF16 precision (2 bytes per element), comparing the GPU baseline (`(M × K + K) × 2` bytes, transferring both `W` and `x`) against the CENT model (`(K + M) × 2` bytes, where only `x` and `y` cross the bus).

---

## 6. Validation on Llama-2 7B

The repository includes automated Python unit tests (`make test`), a native C++ testbench (`make test-core`), and a validation script exercising real Llama-2 7B layer dimensions (`examples/llama2_7b_demo.py`).

### Local Execution

```bash
git clone https://github.com/josedercysilva/cent-pim-runtime.git
cd cent-pim-runtime

./setup_env.sh
source .venv/bin/activate

make test        # Tests C ABI, numerical accuracy, GEMM rejection, SRAM, and .trace
make test-core   # Native C++ testbench
make demo        # Runs Llama-2 7B Attention and MLP layers
make simulate    # Optional: runs Ramulator 2.0 on the generated .trace files
```

### Usage with PyTorch

```python
import torch
import torch.nn.functional as F
import cent_runtime

# Decoding phase (batch_size = 1)
x = torch.randn(1, 4096)        # Input activation
W = torch.randn(4096, 4096)     # Stationary weights in GDDR6-AiM memory

# Dispatch to the C++ runtime (returns PyTorch tensor + telemetry)
y, telemetry = cent_runtime.matmul(x, W, layer_id=0)

# Non-linear operations remain on CPU via PyTorch
out = F.silu(y)

print(f"Output: {out.shape}")
print(f"CXL Board: #{telemetry['dvid']} | Mask: {telemetry['channel_mask_hex']}")
print(f"Instructions: {telemetry['total_instructions']} ({telemetry['instruction_buffer_bytes']} B SRAM)")
print(f"Bus traffic saved: {telemetry['memory_traffic_saved_pct']:.2f}%")
```

### Audited Metrics: Attention vs. MLP

The table below summarizes the results obtained for the Attention projection (`q/k/v/o_proj`, `M = 4096, K = 4096, layer_id = 0`) and the MLP projection (`gate/up/down_proj`, `M = 11008, K = 4096, layer_id = 1`):

| Metric | Attention Layer (`4096 × 4096`) | MLP Layer (`11008 × 4096`) |
| :--- | :--- | :--- |
| CXL device (`DVid`) | Board `#0` | Board `#1` |
| Active channels (`CHmask`) | `0xffffffff` (32 channels) | `0xffffffff` (32 channels) |
| 16-bank blocks (`ceil(M/16)`) | 256 blocks | 688 blocks |
| Total instructions (`4/block + EOC`) | 1,025 (`256×4 + 1`) | 2,753 (`688×4 + 1`) |
| SRAM footprint (`14 B/inst`) | 14,350 bytes (14.01 KB) | 38,542 bytes (37.64 KB) |
| GPU HBM vs. CENT CXL traffic (BF16) | ~32.01 MB vs. 16.00 KB (99.95% lower) | ~86.01 MB vs. 29.50 KB (99.97% lower) |
| Ramulator 2.0 cycles (`CH31_AiM_ISR_WR_GB_cycles`) | 16,416 cycles | 44,064 cycles |

The ratio between the two layers reflects the matrix dimensions directly: the MLP output dimension is `11,008 / 4,096 = 2.6875×` larger than Attention, and both the total SRAM instruction count and the silicon cycle count reported by `aim_simulator` scale by the same `2.68×` factor (`44,064 / 16,416`).

Inspecting the first 16 SRAM bytes via `cent_get_instruction_buffer_hex` shows the first 14-byte packet serialized in *little-endian* order (`opcode = 0x02` for `WR_GB`, `dvid = 0x00`, `channel_mask = FF FF FF FF`, `bank = 00 00`, `row = 00 00 00 00`, `opsize = 64 = 0x0040 -> 40 00`), followed by the first two bytes of the second instruction (`03 00`, `WR_BIAS`):

```text
[SRAM] 02 00 FF FF FF FF 00 00 00 00 00 00 40 00 03 00
```

---

## 7. Conclusion and References

`cent_runtime` connects the CENT architectural specification to an executable Python and C++17 workflow:

- **Standalone execution (without `ramulator2`):** dispatches PyTorch/NumPy linear projections across a C ABI via direct memory pointers (with no serialization and no extra copy for contiguous `float32` CPU tensors), exposes the 14-byte binary packets in SRAM, and measures external bus traffic reduction.
- **Simulator-coupled execution (with `aim_simulator`):** automatically emits CENT ISA `.trace` files and invokes Ramulator 2.0 to extract exact cycle counts across 32 GDDR6-AiM channels.

### References

1. **Project Code:** [`josedercysilva/cent-pim-runtime`](https://github.com/josedercysilva/cent-pim-runtime)
2. **Presentation Slides (UFAM Seminar):** [`apresentacao_cent.pdf`](https://github.com/josedercysilva/cent-pim-runtime/blob/main/apresentacao_cent.pdf)
3. **Foundational Paper (CENT):** Gu, Y., Khadem, A., et al. (2025). *PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference*. ASPLOS '25.
4. **Memory Hierarchy:** Patterson, D. A., & Hennessy, J. L. (2014). *Computer Organization and Design: The Hardware/Software Interface* (5th Ed. — MIPS Edition), Ch. 5.
5. **CXL Specification:** Compute Express Link Consortium (2022). *CXL Specification Revision 3.0*.
