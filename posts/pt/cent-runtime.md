A arquitetura **CENT** (*PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference*, Gu et al., **ASPLOS 2025**) propõe executar a inferência de *Large Language Models* (LLMs) sem GPUs, substituindo memórias HBM por módulos **GDDR6-AiM** conectados via **CXL 3.0**.

Embora os ganhos analíticos e de simulação sejam expressivos no artigo, existe uma lacuna prática entre os resultados acadêmicos e a execução real: falta uma camada de sistema capaz de receber tensores do **PyTorch** e despachá-los para o modelo de execução da arquitetura.

Este relatório técnico documenta o contexto arquitetural, os gargalos físicos que motivaram o projeto, o papel de um **hardware dispatcher**, o funcionamento do simulador oficial (**`aim_simulator` / Ramulator 2.0**) e a implementação do [`cent_runtime`](https://github.com/josedercysilva/cent-pim-runtime) — uma ponte em **C++17 + C ABI + Python** que conecta camadas lineares de modelos como o **Llama-2 7B** ao conjunto de instruções da arquitetura CENT. Os slides do seminário que acompanham este trabalho também estão disponíveis na raiz do repositório em [`apresentacao_cent.pdf`](https://github.com/josedercysilva/cent-pim-runtime/blob/main/apresentacao_cent.pdf).

---

## Sumário

1. **Fundamentos e o Gargalo do *Memory Wall*** — Hierarquia clássica de memória, anatomia do funil DRAM e a dualidade *Prefill* (`GEMM`) vs. *Decoding* (`GEMV`).
2. **A Arquitetura CENT (ASPLOS '25)** — Topologia *GPU-Free* com CXL 3.0, microarquitetura PIM/PNM, ganhos reais e limitações críticas.
3. **O que é o `aim_simulator` (Ramulator 2.0)** — Como funciona o simulador de ciclo exato do artigo e por que ele não conversa com o PyTorch sozinho.
4. **O que é um *Hardware Dispatcher*** — O papel da camada intermediária entre frameworks de IA e o controlador de memória/hardware.
5. **A Nossa Implementação: Anatomia do `cent_runtime`** — Camada Python (`ops.py`, `_bindings.py`, `simulator.py`), fronteira C ABI (`extern "C"`), pacote binário de 14 bytes (`CentInstruction`), buffer SRAM de 2 MB e emissor de traços em C++17.
6. **Validação no Llama-2 7B** — Métricas exatas de Atenção (`4K×4K`) e MLP (`11K×4K`), economia de 99,95% no barramento, *hex dump* da SRAM e código prático.
7. **Conclusão e Referências** — O que a implementação resolve hoje, seus limites e onde consultar o código e os trabalhos base.

---

## 1. Fundamentos e o Gargalo do *Memory Wall*

Na arquitetura clássica de computadores (Patterson & Hennessy, Cap. 5), a hierarquia de memória equilibra velocidade, capacidade e custo por meio de uma escada bem conhecida:

```text
CPU / GPU  <--->  Cache L1  <--->  Cache L2  <--->  Cache L3  <--->  Memória Principal (DRAM)  <--->  SSD / Disco
 (mais rápida, menor capacidade)                                                   (mais lenta, maior capacidade)
```

Esse arranjo funciona muito bem quando há **localidade temporal e espacial** — ou seja, quando um dado buscado da DRAM é reutilizado dezenas ou centenas de vezes dentro das caches antes de ser descartado. O problema é que a inferência de LLMs quebra esse pressuposto na sua etapa mais demorada.

### A Anatomia Física do Funil na DRAM

Dentro de um chip DRAM, os dados são organizados em matrizes de células 2D. Quando uma linha é ativada, um bloco inteiro de **8.192 bits** é carregado de uma só vez para o **Row Buffer** interno em cerca de 1 ciclo de acesso interno — ou seja, **dentro da pastilha de memória existe uma largura de banda massiva**.

O gargalo nasce na saída do chip. Para levar esses 8.192 bits até o processador (CPU ou GPU), tudo precisa atravessar o **barramento externo** e os pinos físicos de I/O, que operam a cerca de **64 bits por ciclo** (ou 32 bits em configurações específicas de pino). São necessários aproximadamente **128 ciclos de barramento apenas para transferir uma única linha** que já estava pronta no *Row Buffer*. Enquanto os dados viajam pelo barramento, os núcleos de cálculo da GPU entram em estado de **stall** (ociosos esperando dados).

### A Dualidade *Prefill* (`GEMM`) vs. *Decoding* (`GEMV`)

A inferência de um LLM acontece em duas fases com perfis físicos completamente opostos:

| Característica | Fase 1: *Prefill* (Prompt Inicial) | Fase 2: *Decoding* (Geração Token a Token) |
| :--- | :--- | :--- |
| **Fração do tempo total** | ~2% do tempo de inferência | **~98% do tempo de inferência** |
| **Operação matemática** | Matriz × Matriz (`GEMM`, ex.: `4K×4K` × `4K×4K`) | Vetor × Matriz (`GEMV`, `batch_size = 1`) |
| **Intensidade aritmética** | Alta (~2.000 Ops/Byte) | Muito baixa (**1 a 16 Ops/Byte**) |
| **Regime de execução** | **Compute-Bound** (satura os *Tensor Cores*) | **Memory-Bound** (preso na banda da memória) |

Durante o *decoding*, cada token gerado exige ler matrizes inteiras de pesos (dezenas de megabytes por camada, somando dezenas de gigabytes no modelo completo) para realizar apenas uma multiplicação por vetor e descartar os pesos logo em seguida. Em um nó com **4× GPUs NVIDIA A100**, essa falta de reuso faz com que os *Tensor Cores* operem a apenas **21% de utilização** (passando **79% do tempo em *stall*** aguardando a memória HBM).

---

## 2. A Arquitetura CENT (ASPLOS '25)

A ideia central do **Processing-in-Memory (PIM)** — especificamente do padrão **GDDR6-AiM** (*Accelerator-in-Memory*) usado no CENT — é colocar pequenas Unidades Lógicas e Aritméticas (**ULAs / PUs** de multiplicação-acumulação) diretamente ao lado de cada banco DRAM, antes do funil dos pinos de saída.

Com isso, cada chip GDDR6-AiM consome os dados diretamente na saída do *Row Buffer* sem desperdiçar energia e ciclos atravessando o barramento externo:
* Uma **NVIDIA A100** entrega cerca de **2 TB/s** de banda de pico na HBM2e.
* Um dispositivo **GDDR6-PIM** atinge até **16 TB/s** de banda interna efetiva.
* Em um sistema **CENT** completo com **32 dispositivos**, a banda interna agregada chega a **512 TB/s**.

### Topologia Global e Microarquitetura Interna

Em vez de depender de GPUs dedicadas e interposers caros de HBM, o sistema CENT conecta uma **CPU Host** a **32 dispositivos CXL 3.0** por meio de um **Switch CXL** de 48 portas:

```text
                       +------------+
                       |  Host CPU  |
                       +-----+------+
                             |
                             v
                   +--------------------+
                   |   Switch CXL 3.0   |
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
        +---> Instruction Buffer (SRAM de 2 MB) -> PC -> Decoder
        +---> PNM Units + Shared Buffer (64 KB)
        +---> 16x PIM Ctrl ---> 16x GDDR6-PIM (32 canais por placa)
```

Dentro de cada um dos 32 dispositivos CXL, o trabalho é dividido em duas hierarquias complementares:
1. **Unidades PIM (nos bancos DRAM):** executam >99% das operações matemáticas do *decoding* — as projeções lineares densas (`GEMV`) — lendo o vetor de entrada de um *Global Buffer* e acumulando os produtos escalares em paralelo nos 16 bancos de cada canal.
2. **Unidades PNM (*Processing-Near-Memory*) + Host CPU:** cuidam das operações leves (<1% das contas) que exigem lógica não-linear ou redução global entre canais, como `Softmax`, `RMSNorm`, `SiLU` e `RoPE`, recebendo apenas vetores parciais compactos (<16 KB) em vez de matrizes inteiras.

### O Que os Resultados do Artigo Mostram (Ganhos e Limitações)

Nos experimentos reproduzidos e comparados contra um servidor com **4× NVIDIA A100** (80 GB cada):
* **Desempenho no *Decoding*:** o CENT alcança até **4,13× mais tokens/s** no Llama-2 13B (**2,97×** no 7B) e **2,32×** na média geométrica geral (*end-to-end*), porque o *decoding* domina 98% do tempo total.
* **Eficiência energética e custo:** cada placa CXL consome ~32,4 W (contra ~300 W de uma GPU), entregando **2,88× mais tokens por Joule** (até **4,26×** no *decoding*) e custando cerca de **\$14.873** (CPU + 512 GB GDDR6-PIM + controladores + switch CXL) contra **\$42.128** do cluster de 4× A100 — cerca de **3× mais barato**.

Por outro lado, uma análise técnica crítica também expõe **três limitações reais** da proposta:
1. **Prefill lento (0,33× a 0,46×):** as mini-ULAs nos bancos DRAM não têm o poder de computação matricial dos *Tensor Cores*. Em prompts longos (como cenários RAG de 32k tokens), o *prefill* sofre queda severa.
2. **Gargalo de escala acima de 70B:** o speedup no *decoding* cai de **4,13×** (13B) para **1,25×** (70B) à medida que a comunicação entre placas através do switch CXL começa a pesar.
3. **Barreira de Software e ISA:** o CENT exige uma ISA própria (CENT ISA) e não possui suporte nativo a frameworks como PyTorch, vLLM ou ecossistema equivalente aos mais de 15 anos de maturidade do CUDA.

É exatamente nessa terceira barreira — a **falta de integração entre software de alto nível e o simulador/hardware CENT** — que entra o nosso trabalho.

---

## 3. O que é o `aim_simulator` (Ramulator 2.0)

Para validar a microarquitetura antes de fabricar silício físico, os autores do CENT disponibilizaram o **`aim_simulator`**, construído como uma extensão do **Ramulator 2.0** (um dos simuladores de memória DRAM de ciclo exato mais respeitados da academia).

Na prática, o `aim_simulator` é compilado em um executável chamado `ramulator2` (apoiado pela biblioteca `libramulator.so`). Ele **não é um runtime de IA** nem executa redes neurais reais; ele é um **simulador discreto de eventos de hardware** que funciona assim:

```text
  [ configs/gddr6_aim.yaml ] ──┐
  (Parâmetros físicos:         │
   32 canais, 16 bancos,       ├──► [ ramulator2 (aim_simulator) ] ──► Contadores de Silício
   timings GDDR6, FRFCFS)      │                                       (ex: CH31_AiM_ISR_WR_GB_cycles,
                               │                                        memory_system_cycles)
  [ arquivo.trace ] ───────────┘
  (Lista estática de comandos
   textuais da CENT ISA)
```

Quando executado via linha de comando (`ramulator2 -f configs/gddr6_aim.yaml -t arquivo.trace`), ele simula ciclo por ciclo cada acesso ao *Row Buffer*, conflitos de banco, escalonamento de comandos (`FRFCFS`), refresh (`AllBank`) e instruções AiM nos **32 canais GDDR6-AiM**, imprimindo ao final um relatório detalhado com a contagem exata de ciclos gastos em cada tipo de instrução.

### Por que o `aim_simulator` sozinho não resolve a ponta do software?

1. **Só consome traços estáticos (`.trace`):** você precisa gerar manualmente arquivos de texto com comandos `AiM WR_GB`, `AiM MAC_ABK`, etc., já decompondo linhas e endereços físicos.
2. **Não calcula valores numéricos:** o simulador mede *tempo e ciclos*, não multiplica os pesos reais em ponto flutuante para devolver um tensor `y` que possa alimentar a próxima camada Transformer no PyTorch.
3. **Código acadêmico frágil para integração direta:** como artefato de pesquisa, sua compilação depende de versões específicas de `gcc`/`cmake`, tornando inviável acoplá-lo diretamente dentro de pipelines Python comuns.

---

## 4. O que é um *Hardware Dispatcher*

Em arquitetura de sistemas, um **dispatcher** (despachante) é a camada de software de baixo nível posicionada entre a aplicação/framework de alto nível e o dispositivo acelerador. Quando você executa `torch.matmul(x, W)` em uma GPU NVIDIA, o PyTorch não "fala" diretamente com os registradores da placa de vídeo: ele passa pelo *PyTorch Dispatcher* e aciona bibliotecas nativas em C/C++ (`libcudart.so`, `libcublas.so`) que fazem o papel de intermediárias.

No contexto do nosso projeto, um **hardware dispatcher para CENT** precisa cumprir quatro funções bem definidas:

1. **Interceptação e Validação de Escopo:** receber os tensores do Python e verificar se a operação respeita o contrato físico do acelerador (no caso do CENT PIM, multiplicação vetor-matriz `GEMV` com `batch_size = 1`). Se o usuário enviar uma `GEMM` de *prefill* (`batch > 1`), o dispatcher rejeita o envio ao PIM para que o host trate.
2. **Roteamento de Topologia:** decidir **em qual das 32 placas CXL** (`DVid`) a camada reside e **quais dos 32 canais GDDR6** (`CHmask`) devem ser ativados em paralelo para aquela dimensão oculta `K`.
3. **Tradução (*Lowering*) para a ISA do Hardware:** decompor a matriz de pesos `[M × K]` em blocos compatíveis com os **16 bancos DRAM** de cada canal, codificar cada passo em pacotes binários da **CENT ISA** dentro do buffer de instruções (SRAM de 2 MB) e, se desejado, materializar o arquivo `.trace`.
4. **Coleta de Telemetria e Integridade Numérica:** calcular o vetor resultante `y` diretamente na memória de destino do PyTorch (garantindo que a rede neural continue executando normalmente) e retornar métricas auditáveis de instruções geradas e tráfego de barramento economizado.

---

## 5. A Nossa Implementação: Anatomia do `cent_runtime`

O **`cent_runtime`** materializa esse *dispatcher* em uma arquitetura limpa de duas camadas — **Python (PyTorch/NumPy)** falando com **C++17** por meio de uma **C ABI estável (`libcent_runtime.so`)** — e uma ponte opcional para disparar o `aim_simulator` (`ramulator2`) de forma automatizada.

```text
+-----------------------------------------------------------------------------+
| 1. CAMADA PYTHON (python/cent_runtime/)                                     |
|    cent_runtime.matmul(x, W, layer_id=0, trace_path=None, simulate=False)   |
|    - Valida GEMV (batch=1) em ops.py                                        |
|    - Normaliza para float32 contíguo e extrai ponteiro via ctypes           |
+--------------------------------------+--------------------------------------+
                                       | ctypes (_bindings.py)
                                       v
======================= [ FRONTEIRA C ABI (extern "C") ] ======================
| 2. CONTRATO PÚBLICO (include/cent/runtime_api.h -> build/libcent_runtime.so)|
|    cent_dispatch_gemv(x, M, K, W, y, layer_id, trace_path, &telemetry)      |
+--------------------------------------+--------------------------------------+
                                       |
          +----------------------------+----------------------------+
          v                                                         v
+-----------------------------------+     +-----------------------------------+
| 3. MOTOR C++17 (src/)             |     | 4. COMPUTAÇÃO NUMÉRICA            |
|  - topology.cpp   -> DVid/CHmask  |     |  - Laço row-major sobre W [M x K] |
|  - sram_buffer.cpp -> SRAM 2 MB   |     |  - Acumulação FP64 -> saída FP32  |
|  - isa_emitter.cpp -> Blocos PIM  |     |  - Grava direto no ponteiro 'y'   |
+-----------------+-----------------+     +-----------------------------------+
                  |
                  +---> Pacotes binários (14 B) na SRAM + Hex Dump
                  +---> Arquivo .trace (CENT ISA) ---> simulator.py -> ramulator2
```

A seguir, abrimos cada peça dessa engrenagem.

### 5.1. Camada Python (`python/cent_runtime/`)

O pacote Python expõe a API principal `cent_runtime.matmul`, além de `simulate_on_ramulator` e `get_instruction_buffer_hex`:

* **`ops.py` (Validação estrita e passagem direta via `ctypes`):** aceita tanto `torch.Tensor` quanto `numpy.ndarray` (fazendo *fallback* automático para NumPy caso o PyTorch não esteja instalado). Antes de tocar na biblioteca C, valida estritamente o contrato `GEMV`: `x` deve ter formato `(K,)` ou `(1, K)`, `W` deve ter formato `(M, K)` e a dimensão interna `K` deve coincidir. Tensores 3D+ ou `batch_size > 1` disparam `ValueError` imediato explicando que bancos Near-Bank PIM operam exclusivamente em `GEMV` (*decoding*). Em seguida, garante contiguidade em memória (`float32`, sem cópia adicional quando o array já reside contíguo nesse formato na CPU), aloca a saída `y` no mesmo formato (`(M,)` ou `(1, M)`) e passa os ponteiros brutos diretamente via `ctypes`, sem serialização nem IPC.
* **`_bindings.py` (Carga nativa e telemetria):** localiza `build/libcent_runtime.so` (disparando `make shared` automaticamente se o `.so` ainda não existir) e mapeia a struct `CentDispatchTelemetry` (56 bytes com alinhamento natural x86_64) para converter os contadores de hardware em um dicionário Python legível.
* **`simulator.py` (Ponte opcional para o Ramulator 2.0):** se o usuário passar `simulate=True` (criando um arquivo `.trace` via `tempfile` caso `trace_path=None`, para não sujar o disco), resolve o executável `ramulator2` em ordem portátil (`ramulator_bin` explícito → `CENT_RAMULATOR_BIN` → `external/ramulator2` → `../CENT/aim_simulator/build/ramulator2` → `PATH`), executa `[ramulator_bin, "-f", config_path, "-t", trace_path]` via `subprocess` e extrai métricas como `wr_gb_cycles`, `mac_abk_cycles` e `memory_system_cycles`.

### 5.2. Por que uma "Porta C ABI" (`.h`) separada do Motor C++17 (`.hpp`)?

Em C++, funções dentro de `namespace` ou com sobrecarga sofrem **Name Mangling** (*decoração de nomes*). Uma função `cent::compute_dvid(int)` vira algo como `_ZN4cent12compute_dvidEi` no binário objeto, e o C++ não possui uma ABI binária universal entre compiladores (GCC, Clang e MSVC organizam *vtables*, exceções e nomes de formas distintas).

Já a linguagem **C** é o "esperanto" dos sistemas operacionais e das interfaces FFI (*Foreign Function Interface*). Por isso, o `cent_runtime` separa rigorosamente:
* **Cabeçalhos `.h` (`isa.h`, `telemetry.h`, `runtime_api.h`):** fronteira pública envolta em `#ifdef __cplusplus extern "C" { #endif`. Os símbolos exportados em `build/libcent_runtime.so` permanecem puros (`cent_init_runtime`, `cent_dispatch_gemv`, `cent_get_instruction_buffer_hex`), legíveis diretamente pelo `ctypes` do Python.
* **Cabeçalhos `.hpp` (`topology.hpp`, `sram_buffer.hpp`, `isa_emitter.hpp`):** código interno em **C++17 moderno** com `namespace cent`, classes, RAII e `std::array`.

Além disso, todo o núcleo é compilado com `-std=c++17 -O3 -fPIC -Wall -Wextra -Werror` e **zero exceções C++ (`throw` / `try-catch`)** no fluxo de controle. Todas as funções da fronteira são marcadas com `noexcept` e erros são tratados via retorno explícito de `enum CentStatus` (`CENT_OK = 0`, `CENT_ERR_NULL_PTR = -1`, `CENT_ERR_INVALID_SHAPE = -2`, `CENT_ERR_SRAM_OVERFLOW = -3`, `CENT_ERR_TRACE_IO = -4`) com *early return*.

### 5.3. O Pacote Binário de 14 Bytes (`CentInstruction`)

Em `include/cent/isa.h`, definimos as constantes microarquiteturais do controlador (`CENT_SRAM_BUFFER_BYTES = 2 MB`, `32` dispositivos CXL, `32` canais por dispositivo, `16` bancos por canal, `64` bursts por comando padrão), os 5 opcodes da CENT ISA e a `struct` binária empacotada:

```c
enum CentOpcode : uint8_t {
    CENT_OP_WR_GB   = 0x02, // Escreve vetor de ativação no Global Buffer dos canais ativos
    CENT_OP_WR_BIAS = 0x03, // Inicializa registradores acumuladores nos bancos DRAM
    CENT_OP_RD_MAC  = 0x05, // Lê o resultado acumulado dos bancos para o buffer do canal
    CENT_OP_MAC_ABK = 0x0F, // Multiply-Accumulate paralelo em todos os 16 bancos (All-Bank)
    CENT_OP_EOC     = 0xFF  // End of Computation (encerra o fluxo no aim_simulator)
};

#pragma pack(push, 1)
typedef struct CentInstruction {
    uint8_t  opcode;       //  8 bits : Opcode da CENT ISA
    uint8_t  dvid;         //  8 bits : ID do dispositivo CXL alvo (0 a 31)
    uint32_t channel_mask; // 32 bits : Máscara binária dos 32 canais GDDR6
    uint16_t bank_index;   // 16 bits : Índice do banco DRAM (0 a 15)
    uint32_t row_addr;     // 32 bits : Endereço da linha física (Row Buffer)
    uint16_t opsize;       // 16 bits : Tamanho da operação em bursts
} CentInstruction;
#pragma pack(pop)

static_assert(sizeof(CentInstruction) == 14,
              "ABI Error: CentInstruction must be exactly 14 bytes packed");
```

O uso de `#pragma pack(push, 1)` combinado com `static_assert` garante em tempo de compilação que o compilador não inserirá *padding* oculto entre os campos: cada instrução ocupa exatamente `1 + 1 + 4 + 2 + 4 + 2 = 14 bytes` contíguos na SRAM.

### 5.4. Topologia, Buffer SRAM de 2 MB e Decomposição Near-Bank PIM

Quando `cent_dispatch_gemv` entra no núcleo C++ (`src/runtime_api.cpp`), quatro módulos atuam em sequência:

1. **Roteamento (`src/topology.cpp`):**
   * `DVid = ((layer_id % 32) + 32) % 32` — distribui as camadas Transformer pelas 32 placas CXL 3.0.
   * `CHmask`: se `K >= 1024`, ativa todos os 32 canais (`0xFFFFFFFFu`); se `K >= 512`, ativa 16 canais (`0x0000FFFFu`); abaixo disso, ativa 8 canais (`0x000000FFu`).
2. **Emulador da SRAM de 2 MB (`src/sram_buffer.cpp`):**
   * A classe `SramBuffer` mantém um `std::array<uint8_t, CENT_SRAM_BUFFER_BYTES>` estático de **2 MB** pré-alocado, eliminando `new`/`delete` no caminho crítico. Cada instrução é copiada via `std::memcpy` se couber no limite, caso contrário retorna `CENT_ERR_SRAM_OVERFLOW`.
   * Também implementa `format_hex_dump`, formatando até os primeiros 64 bytes em pares hexadecimais maiúsculos (16 bytes por linha) para inspeção binária.
3. **Decomposição em Blocos PIM e Escrita `.trace` (`src/isa_emitter.cpp`):**
   * Como cada canal GDDR6-AiM opera **16 bancos em paralelo** (*All-Bank PIM*), a dimensão de saída `M` é dividida em `num_blocks = (M + 15) / 16` blocos.
   * Para cada bloco `blk` (`row_addr = blk`), o emissor grava 4 pacotes binários na SRAM (`WR_GB` com `opsize=64`, `WR_BIAS` com `opsize=0`, `MAC_ABK` com `opsize=64` e `row_addr=blk`, e `RD_MAC` com `opsize=0`) e, quando `trace_path` está ativo, escreve via `std::ofstream` (RAII) as 4 linhas correspondentes no `.trace`:
     ```text
     AiM WR_GB 64 0 0xffffffff
     AiM WR_BIAS 0 0xffffffff
     AiM MAC_ABK 64 0xffffffff 0
     AiM RD_MAC 0 0xffffffff
     ```
   * Ao terminar o loop de blocos, emite **1 instrução final `CENT_OP_EOC`** (`AiM EOC` no `.trace`), obrigatória para o `aim_simulator` encerrar a simulação.
4. **Cálculo Numérico e Telemetria de Barramento (`src/runtime_api.cpp` & `src/isa_emitter.cpp`):**
   * Calcula `y = x · W^T` percorrendo `W` em ordem *row-major* com acumulação em `double` (`FP64`) convertida para `float` (`FP32`), gravando diretamente no ponteiro `y` recebido do PyTorch.
   * Calcula o comparativo de tráfego externo considerando precisão **BF16** (2 bytes/elemento):
     * **GPU Baseline (HBM):** `(M × K + K) × 2` bytes (move toda a matriz de pesos `W` + entrada `x`).
     * **CENT CXL 3.0 (Near-Bank PIM):** `(K + M) × 2` bytes (pesos ficam estacionários na DRAM; apenas entrada `x` e saída `y` cruzam o barramento).
     * **Economia (%):** `100 × (1 - pcie_traffic_bytes / gpu_traffic_bytes)`.

---

## 6. Validação no Llama-2 7B

O repositório já inclui suíte de testes unitários em Python (`make test`), testbench nativo em C++ (`make test-core`) e uma demonstração dedicada às dimensões reais do **Llama-2 7B** (`examples/llama2_7b_demo.py`).

### Executando Localmente

```bash
git clone https://github.com/josedercysilva/cent-pim-runtime.git
cd cent-pim-runtime

./setup_env.sh
source .venv/bin/activate

make test        # Testa C ABI, precisão numérica, rejeição de GEMM, SRAM e .trace
make test-core   # Testbench nativo em C++
make demo        # Roda camadas de Atenção e MLP do Llama-2 7B (sem exigir Ramulator)
make simulate    # Opcional: dispara o Ramulator 2.0 sobre os .trace gerados
```

### Exemplo de Uso no PyTorch

```python
import torch
import torch.nn.functional as F
import cent_runtime

# Fase de decoding (batch_size = 1)
x = torch.randn(1, 4096)        # Ativação de entrada
W = torch.randn(4096, 4096)     # Pesos estacionários na memória GDDR6-AiM

# Despacho para o runtime C++ (retorna tensor PyTorch + telemetria)
y, telemetry = cent_runtime.matmul(x, W, layer_id=0)

# Operações não-lineares continuam normalmente na CPU pelo PyTorch
out = F.silu(y)

print(f"Saída: {out.shape}")
print(f"Placa CXL: #{telemetry['dvid']} | Máscara: {telemetry['channel_mask_hex']}")
print(f"Instruções: {telemetry['total_instructions']} ({telemetry['instruction_buffer_bytes']} B SRAM)")
print(f"Economia de barramento: {telemetry['memory_traffic_saved_pct']:.2f}%")
```

### Métricas Auditadas (Atenção vs. MLP no Llama-2 7B)

Ao executar a camada de **Atenção** (`q/k/v/o_proj`, `M = 4096, K = 4096, layer_id = 0`) e a camada **MLP** (`gate/up/down_proj`, `M = 11008, K = 4096, layer_id = 1`), os resultados batem exatamente com o modelo analítico e com o simulador de ciclo exato:

| Métrica | Camada de Atenção (`4096 × 4096`) | Camada MLP (`11008 × 4096`) |
| :--- | :--- | :--- |
| **Dispositivo CXL (`DVid`)** | Placa `#0` | Placa `#1` |
| **Canais Ativos (`CHmask`)** | `0xffffffff` (`32` canais) | `0xffffffff` (`32` canais) |
| **Blocos de 16 Bancos (`ceil(M/16)`)** | `256` blocos | `688` blocos |
| **Total de Instruções (`4/bloco + EOC`)** | **`1.025`** (`256×4 + 1`) | **`2.753`** (`688×4 + 1`) |
| **Ocupação na SRAM (`14 B/inst`)** | **`14.350` bytes** (`14,01 KB`) | **`38.542` bytes** (`37,64 KB`) |
| **Tráfego GPU HBM vs. CENT CXL (BF16)** | `~32,01 MB` vs. `16,00 KB` (**99,95%** menos) | Apenas ativação + saída no barramento |
| **Ciclos no Ramulator 2.0 (`CH31_AiM_ISR_WR_GB_cycles`)** | **`16.416` ciclos** | **`44.064` ciclos** |

Observe a consistência física da razão entre as duas camadas: a dimensão de saída da MLP é `11.008 / 4.096 = 2,6875×` maior que a da Atenção, e tanto o volume de instruções geradas na SRAM quanto os ciclos medidos no silício pelo `aim_simulator` escalam exatamente em **`2,68×`** (`44.064 / 16.416`).

Inspecionando os primeiros bytes da SRAM via `cent_get_instruction_buffer_hex`, conseguimos inclusive ver o pacote binário de 14 bytes em *little-endian* na prática (com `opcode = 0x02` para `WR_GB`, `dvid = 0x00`, `channel_mask = FF FF FF FF`, `bank = 00 00`, `row = 00 00 00 00`, `opsize = 64 = 0x0040 -> 40 00`, seguido pelo início da segunda instrução `03 00` = `WR_BIAS`):

```text
[SRAM] 02 00 FF FF FF FF 00 00 00 00 00 00 40 00 03 00
```

---

## 7. Conclusão e Referências

O `cent_runtime` fecha o ciclo entre a pesquisa em arquitetura de computadores e a engenharia de software de sistemas:

* **Sem depender obrigatoriamente do `ramulator2`**, é possível instalar o pacote Python, despachar projeções do PyTorch/NumPy através de uma C ABI em C++17 via ponteiro direto (sem serialização, e *zero-copy* para tensores `float32` contíguos na CPU), auditar o *hex dump* de pacotes de 14 bytes na SRAM e verificar a redução de 99,95% no tráfego externo.
* **Quando conectado ao `aim_simulator`**, os traços `.trace` gerados automaticamente permitem validar os ciclos exatos nos 32 canais GDDR6-AiM sem escrever uma única linha de trace à mão.

### Referências

1. **Código do Projeto:** [`josedercysilva/cent-pim-runtime`](https://github.com/josedercysilva/cent-pim-runtime)
2. **Slides da Apresentação (Seminário UFAM):** [`apresentacao_cent.pdf`](https://github.com/josedercysilva/cent-pim-runtime/blob/main/apresentacao_cent.pdf) *(raiz do repositório)*
3. **Artigo Original (CENT):** Gu, Y., Khadem, A., et al. (2025). *PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference*. **ASPLOS '25**.
4. **Fundamentos de Hierarquia de Memória:** Patterson, D. A., & Hennessy, J. L. (2014). *Computer Organization and Design: The Hardware/Software Interface* (5th Ed. — MIPS Edition), Cap. 5.
5. **Especificação CXL:** Compute Express Link Consortium (2022). *CXL Specification Revision 3.0*.
