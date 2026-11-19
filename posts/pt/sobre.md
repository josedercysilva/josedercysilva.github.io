Sou **José Dercy**, graduando em Engenharia de Computação na Universidade Federal do Amazonas (UFAM), em Manaus. Meu trabalho se concentra na interseção entre sistemas de baixo nível (C, Kernel Linux, tempo real e arquitetura de hardware em Verilog) e desenvolvimento fullstack (Java, TypeScript, Python, SQL e aplicações web).

Sempre tive curiosidade de abrir a "caixa-preta" das ferramentas que uso no dia a dia: além de desenvolver aplicações web, APIs e bancos de dados, gosto de descer o nível de abstração para entender como o sistema operacional agenda *threads*, como a memória é gerenciada em C e como o código executa no hardware. Criei este espaço para documentar estudos, projetos e notas técnicas ao longo dessa trajetória.

---

## Experiência & Atuação Acadêmica

- **Bolsista de P&D — Projeto SUPER** (Samsung / UFAM): Atuação no CETELI (240 horas) em atividades de engenharia, prototipagem rápida e modelagem técnica dentro do programa de cooperação tecnológica entre a Samsung e a UFAM.
- **Monitor Acadêmico de POO** (IComp / UFAM): Suporte direto a turmas de graduação em Programação Orientada a Objetos em Java, auxiliando na modelagem de classes, encapsulamento, herança, polimorfismo, interfaces, coleções (`java.util`) e depuração de código.
- **Liderança Técnica** (Arquitetura de Computadores): Coordenação de equipe no projeto dos processadores MIPS de 32 bits (Monociclo e Pipeline de 5 estágios) em Verilog HDL, definindo contratos de barramento, fluxo de Pull Requests no GitHub e validação por testbenches unitários.

---

## Projetos de Sistemas, Kernel & Concorrência

- **Kernel Linux Upstream & PREEMPT_RT** (`7.2.y` / `6.6 LTS`): Clonagem e compilação da árvore estável oficial mantida por Greg Kroah-Hartman, customização completa via `Kconfig` (mantendo suporte a Rust no kernel e metadados BTF, com remoção de símbolos DWARF redundantes), empacotamento nativo via `pacman-pkg` e validação de boot em bare-metal (`7.2.7-jdercy-greg+`). Inclui investigação prática de falhas de driver (`i915`) sob kernel de tempo real.
- **Simulação Concorrente em Tempo Real** (C / POSIX Threads): Implementação de tarefas periódicas determinísticas em C usando temporização absoluta sem *drift* (`clock_nanosleep` com `CLOCK_MONOTONIC`), mitigação de inversão de prioridade com `PTHREAD_PRIO_INHERIT` e código escrito em conformidade com diretrizes de sistemas críticos (`MISRA C` e `BARR-C`).
- **Runtime Processing-in-Memory — `cent_runtime`** (C++17 / C ABI / PyTorch): Hardware dispatcher em C++17 com C ABI estável (`libcent_runtime.so`) e camada Python/PyTorch para a arquitetura Near-Bank PIM CENT (ASPLOS 2025, GDDR6-AiM via CXL 3.0), traduzindo projeções lineares (`GEMV`) do Llama-2 7B em pacotes binários de 14 bytes na SRAM de 2 MB, com redução de >99,95% no tráfego externo de barramento e integração direta com o simulador de ciclos Ramulator 2.0 (`aim_simulator`).
- **Processamento Paralelo de Imagens** (Go / Concorrência): Aplicação concorrente para filtros de imagem (*Box Blur* e *Escala de Cinza*) usando *goroutines* e particionamento matricial, com suíte de benchmarks comparando *speedup* real, saturação de CPU e custo de *context switch* frente à Lei de Amdahl.

---

## Hardware, Arquitetura & Algoritmos em Baixo Nível

- **Processador MIPS 32-bits Pipeline de 5 Estágios & Monociclo** (Verilog HDL / FPGA): Projeto completo em Verilog parametrizado das microarquiteturas MIPS Monociclo e Pipeline de 5 estágios (`IF/ID/EX/MEM/WB`), com *Forwarding Unit* (mitigação de hazards de dados RAW), *Hazard Detection Unit* (*Load-Use stall*) e *flushes* síncronos de controle. Validado com 14 testbenches unitários/top-level, Bubble Sort e recursão dupla (`fib(7)`), e sintetizado para FPGA Intel DE10-Lite (MAX 10) no Quartus Prime atingindo 58,25 MHz (1,29× de ganho sobre os 45,31 MHz do baseline Monociclo).
- **CPU 8-bits SAP-1 & Sistemas Digitais** (Verilog HDL / FPGA): Implementação em Verilog sintetizável de uma CPU 8-bits baseada na arquitetura SAP-1 (barramento W-bus, PC, MAR, IR, ALU e sequenciador de microinstruções) e de uma máquina de estados síncrona em hardware, com simulação no Icarus Verilog / GTKWave e síntese via Yosys / Quartus. Complementado pela formação ChampionCHIP (133h+ em microeletrônica e arquitetura RISC-V).
- **Validador Sintático O(1)** (C): Motor modular em C puro para verificação de delimitadores balanceados (`()`, `[]`, `{}`, `$?`), utilizando tabela de busca direta indexada em ASCII (256 bytes) e pilha dinâmica dimensionada pela entrada para evitar *buffer overflow*.
- **Árvore AVL Balanceada** (ANSI C89): Implementação de árvore binária de busca auto-balanceada (rotações LL, RR, LR, RL) escrita em ANSI C89 estrito, com `Makefile` e documentação Doxygen.
- **Árvore B & Alocação Contígua de Memória** (C): Experimentos de estruturação e alocação contígua em C com *Arena Allocator* customizado e movimentação eficiente de blocos via `memmove`.
- **Algoritmos em Grafos & Simulador de Filas M/M/m** (C++17): Biblioteca modular de grafos em C++17 (BFS, DFS, Dijkstra) com semântica de movimento (`std::move`), além de simulador de eventos discretos em C++/CMake para análise estocástica de filas markovianas.

---

## Software, Web & Dados

- **Sistema de Roteamento Urbano** (Java / Clean Architecture): Modelagem de sistema de roteamento aplicando Domain-Driven Design (DDD), com separação entre entidades de domínio (`RoadNetwork`, `RoadSegment`, `Intersection`), contratos de interface e *Value Objects*.
- **Portfólio & Publicações** (josedercysilva.github.io): Este site, desenvolvido sem frameworks pesados em HTML5, CSS3 e JavaScript modular, com sistema de internacionalização (`PT`/`EN`), alternância de tema e renderização dinâmica de artigos em Markdown.
- **Desenvolvimento Fullstack & Dados:** Construção de APIs RESTful e modelagem de bancos relacionais com PostgreSQL, TypeScript, Java e Python, além de scripts de automação em Bash e Python.

---

## Stack & Contato

- **Linguagens:** C (C89/C99), C++17, Java, Verilog HDL, Go, TypeScript/JavaScript, Python, SQL, Bash.
- **Sistemas & Hardware:** Kernel Linux (`Kconfig`, `PREEMPT_RT`), POSIX Threads, QEMU/KVM, FPGA Intel MAX 10, Icarus Verilog, GTKWave, Quartus Prime, Yosys.
- **Ferramentas:** GCC/Clang, GNU Binutils, Make, CMake, Git (`git bisect`, Conventional Commits), PostgreSQL, Linux (`Manjaro` / `Fedora`).

Links diretos para código, trajetória profissional e contato:

- **GitHub:** [github.com/josedercysilva](https://github.com/josedercysilva)
- **LinkedIn:** [linkedin.com/in/josedercysilva](https://linkedin.com/in/josedercysilva)
- **E-mail:** [josedercysilva@gmail.com](mailto:josedercysilva@gmail.com)
