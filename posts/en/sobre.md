I'm **José Dercy**, a Computer Engineering undergraduate at the Federal University of Amazonas (UFAM) in Manaus, Brazil. My work sits at the intersection of low-level systems (C, Linux Kernel, real-time concurrency, and Verilog hardware architecture) and fullstack software development (Java, TypeScript, Python, SQL, and web applications).

I've always liked opening the "black box" of the tools I use every day: beyond building web applications, APIs, and databases, I enjoy dropping down the abstraction stack to understand how the operating system schedules threads, how memory is managed in C, and how code actually executes on hardware. I built this space to document my studies, engineering projects, and technical notes along the way.

---

## Experience & Academic Roles

- **R&D Fellow — SUPER Project** (Samsung / UFAM): 240 hours at CETELI working on engineering, rapid prototyping, and technical modeling within the Samsung–UFAM technology cooperation program.
- **Object-Oriented Programming Teaching Assistant** (IComp / UFAM): Hands-on mentoring for undergraduate Java classes covering class modeling, encapsulation, inheritance, polymorphism, interfaces, collections (`java.util`), and code debugging.
- **Tech Lead** (Computer Architecture): Led an engineering team designing a 32-bit MIPS processor in Verilog HDL, defining bus contracts, GitHub Pull Request workflows, and unit testbench validation.

---

## Systems, Kernel & Concurrency Projects

- **Upstream Linux Kernel & PREEMPT_RT** (`7.2.y` / `6.6 LTS`): Cloned and built the official stable kernel tree maintained by Greg Kroah-Hartman, authored a custom `Kconfig` (retaining kernel Rust support and BTF metadata while stripping redundant DWARF symbols), packaged natively via `pacman-pkg`, and validated bare-metal boot (`7.2.7-jdercy-greg+`). Includes root-cause investigation of Intel `i915` driver issues under real-time kernels.
- **Real-Time Concurrent Simulation** (C / POSIX Threads): Implemented deterministic periodic tasks in C using drift-free absolute timing (`clock_nanosleep` with `CLOCK_MONOTONIC`), priority inversion mitigation via `PTHREAD_PRIO_INHERIT`, and strict coding compliance (`MISRA C` and `BARR-C`).
- **Processing-in-Memory Runtime — CENT/PIM** (C): Conceptual runtime prototype in C built from UFAM architecture research on CENT/PIM, modeling command descriptors and a ring buffer to explore near-memory command dispatch and the von Neumann *memory wall*.
- **Parallel Image Processing** (Go / Concurrency): Concurrent image filters (*Box Blur* and *Grayscale*) built with *goroutines* and matrix partitioning, paired with empirical benchmarks measuring speedup, CPU saturation, and context-switch overhead against Amdahl's Law.

---

## Hardware, Architecture & Low-Level Algorithms

- **32-bit Single-Cycle MIPS Processor** (Verilog HDL / FPGA): Designed and integrated the full microarchitecture (datapath, ALU, register file, control unit, and `jal`, `jr`, `j` branch/jump logic), verified running a Bubble Sort benchmark in MIPS Assembly (built with GNU Binutils) and synthesized for an Intel DE10-Lite (MAX 10) FPGA in Quartus Prime.
- **8-bit SAP-1 CPU & FPGA Digital Systems** (Verilog HDL): Synthesizable Verilog implementation of an 8-bit SAP-1 CPU (W-bus, PC, MAR, IR, ALU, microinstruction sequencer) and a synchronous hardware finite-state machine, simulated with Icarus Verilog / GTKWave and synthesized with Yosys / Quartus. Complemented by the ChampionCHIP program (133+ hours in microelectronics and RISC-V architecture).
- **O(1) Syntax Validator** (C): Modular C engine for balanced-delimiter validation (`()`, `[]`, `{}`, `$?`) using a direct 256-byte ASCII lookup table and an input-sized dynamic stack to prevent buffer overflows.
- **Self-Balancing AVL Tree** (ANSI C89): Complete binary search tree implementation (LL, RR, LR, RL rotations) in strict ANSI C89 with `Makefile` automation and Doxygen docs.
- **B-Tree & Contiguous Memory Allocation** (C): Contiguous memory structure experiments in C featuring a custom *Arena Allocator* and efficient `memmove`-based block shifting.
- **Graph Algorithms & M/M/m Queue Simulator** (C++17): Modular C++17 graph library (BFS, DFS, Dijkstra) using move semantics (`std::move`), plus a discrete-event simulator in C++/CMake for Markovian queue analysis.

---

## Software, Web & Data

- **Urban Routing System** (Java / Clean Architecture): Routing system modeled with Domain-Driven Design (DDD), separating domain entities (`RoadNetwork`, `RoadSegment`, `Intersection`), interface contracts, and *Value Objects*.
- **Portfolio & Technical Writing** (josedercysilva.github.io): This site, built without heavy frameworks using HTML5, CSS3, and modular vanilla JavaScript, featuring `PT`/`EN` internationalization, theme switching, and dynamic Markdown rendering.
- **Fullstack & Data Engineering:** Building RESTful APIs and relational schemas with PostgreSQL, TypeScript, Java, and Python, alongside Bash and Python automation tooling.

---

## Stack & Contact

- **Languages:** C (C89/C99), C++17, Java, Verilog HDL, Go, TypeScript/JavaScript, Python, SQL, Bash.
- **Systems & Hardware:** Linux Kernel (`Kconfig`, `PREEMPT_RT`), POSIX Threads, QEMU/KVM, Intel MAX 10 FPGA, Icarus Verilog, GTKWave, Quartus Prime, Yosys.
- **Tooling:** GCC/Clang, GNU Binutils, Make, CMake, Git (`git bisect`, Conventional Commits), PostgreSQL, Linux (`Manjaro` / `Fedora`).

Direct links for code, background, and contact:

- **GitHub:** [github.com/josedercysilva](https://github.com/josedercysilva)
- **LinkedIn:** [linkedin.com/in/josedercysilva](https://linkedin.com/in/josedercysilva)
- **Email:** [josedercysilva@gmail.com](mailto:josedercysilva@gmail.com)
