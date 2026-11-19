A arquitetura CENT (*PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference*, Gu et al., ASPLOS 2025) propõe executar a inferência de Large Language Models (LLMs) sem GPUs, substituindo memórias HBM por módulos GDDR6-AiM conectados via CXL 3.0.

Embora os ganhos analíticos e de simulação apresentados no artigo sejam expressivos, existe uma lacuna prática entre os resultados acadêmicos e a execução em software: falta uma camada de sistema capaz de receber tensores do PyTorch e despachá-los para o modelo de execução da arquitetura.

Este relatório técnico documenta o contexto arquitetural, os gargalos físicos que motivaram o projeto, o papel de um hardware dispatcher, o funcionamento do simulador oficial (`aim_simulator` / Ramulator 2.0) e a implementação do [`cent_runtime`](https://github.com/josedercysilva/cent-pim-runtime) — uma biblioteca em C++17, C ABI e Python que conecta camadas lineares de modelos como o Llama-2 7B ao conjunto de instruções da arquitetura CENT. Os slides do seminário que acompanham este trabalho estão disponíveis na raiz do repositório em [`apresentacao_cent.pdf`](https://github.com/josedercysilva/cent-pim-runtime/blob/main/apresentacao_cent.pdf).

---

## Sumário

1. Fundamentos e o Gargalo do Memory Wall
2. A Arquitetura CENT (ASPLOS '25)
3. O Simulador Oficial: `aim_simulator` (Ramulator 2.0)
4. Papel do Hardware Dispatcher
5. Implementação do `cent_runtime`
6. Validação no Llama-2 7B
7. Conclusão e Referências

---

## 1. Fundamentos e o Gargalo do Memory Wall

Na arquitetura clássica de computadores (Patterson & Hennessy, Cap. 5), a hierarquia de memória equilibra velocidade, capacidade e custo em níveis sucessivos:

```text
CPU / GPU  <--->  Cache L1  <--->  Cache L2  <--->  Cache L3  <--->  Memória Principal (DRAM)  <--->  SSD / Disco
 (mais rápida, menor capacidade)                                                   (mais lenta, maior capacidade)
```

Esse arranjo depende de localidade temporal e espacial: um bloco carregado da DRAM precisa ser reutilizado dezenas ou centenas de vezes nas caches antes de ser descartado. A inferência de LLMs rompe essa premissa na sua etapa mais longa.

### O Funil Físico na Saída da DRAM

Dentro de um chip DRAM, os dados são organizados em matrizes bidimensionais de células. Quando uma linha é ativada, um bloco de 8.192 bits é carregado de uma só vez para o Row Buffer interno em cerca de 1 ciclo de acesso — ou seja, no interior da pastilha de memória há largura de banda elevada.

O gargalo está na interface de saída do chip. Para transportar esses 8.192 bits até o processador (CPU ou GPU), os dados precisam atravessar o barramento externo e os pinos físicos de I/O, que operam a cerca de 64 bits por ciclo (ou 32 bits em configurações específicas de pino). São necessários aproximadamente 128 ciclos de barramento apenas para transferir uma única linha que já estava disponível no Row Buffer. Enquanto essa transferência ocorre, as unidades de cálculo da GPU permanecem em estado de *stall*, ociosas aguardando dados.

### Prefill (`GEMM`) vs. Decoding (`GEMV`)

A inferência de um LLM divide-se em duas fases com comportamentos físicos distintos:

| Característica | Fase 1: Prefill (Prompt Inicial) | Fase 2: Decoding (Geração Token a Token) |
| :--- | :--- | :--- |
| Fração do tempo total | ~2% do tempo de inferência | ~98% do tempo de inferência |
| Operação matemática | Matriz × Matriz (`GEMM`, ex.: `4K×4K` × `4K×4K`) | Vetor × Matriz (`GEMV`, `batch_size = 1`) |
| Intensidade aritmética | Alta (~2.000 Ops/Byte) | Baixa (1 a 16 Ops/Byte) |
| Regime de execução | Compute-Bound (satura os Tensor Cores) | Memory-Bound (limitado pela banda de memória) |

Durante o decoding, cada token gerado exige ler matrizes inteiras de pesos (dezenas de megabytes por camada, somando dezenas de gigabytes no modelo completo) para realizar apenas uma multiplicação vetor-matriz e descartar os pesos em seguida. Em um nó com 4× GPUs NVIDIA A100, essa ausência de reuso reduz a utilização efetiva dos Tensor Cores para cerca de 21% (com 79% do tempo em *stall* aguardando leitura da HBM).

---

## 2. A Arquitetura CENT (ASPLOS '25)

A proposta do Processing-in-Memory (PIM) — especificamente do padrão GDDR6-AiM (*Accelerator-in-Memory*) adotado no CENT — é posicionar unidades de multiplicação e acumulação (PUs/ULAs) diretamente ao lado de cada banco DRAM, antes do funil dos pinos de I/O.

Com isso, cada chip GDDR6-AiM consome os pesos diretamente na saída do Row Buffer, sem trafegar as matrizes pelo barramento externo:

- **NVIDIA A100:** entrega cerca de 2 TB/s de banda de pico na HBM2e.
- **Dispositivo GDDR6-PIM:** atinge até 16 TB/s de banda interna agregada nos bancos.
- **Sistema CENT (32 dispositivos):** soma até 512 TB/s de banda interna total.

### Topologia CXL 3.0 e Microarquitetura Interna

Em vez de utilizar GPUs dedicadas e interposers de HBM, o sistema CENT conecta uma CPU Host a 32 dispositivos CXL 3.0 por meio de um Switch CXL de 48 portas:

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

Dentro de cada dispositivo CXL, o processamento é dividido em dois níveis:

1. **Unidades PIM (bancos DRAM):** executam mais de 99% das operações aritméticas do decoding — as projeções lineares densas (`GEMV`) — lendo o vetor de entrada de um Global Buffer e acumulando os produtos escalares em paralelo nos 16 bancos de cada canal.
2. **Unidades PNM (Processing-Near-Memory) e Host CPU:** processam as operações restantes (<1% do volume aritmético) que exigem funções não-lineares ou redução entre canais, como `Softmax`, `RMSNorm`, `SiLU` e `RoPE`, trafegando apenas vetores parciais compactos (<16 KB).

### Resultados e Limitações da Proposta

Nos experimentos do artigo comparados contra um servidor com 4× NVIDIA A100 (80 GB cada):

- **Throughput no Decoding:** o CENT atinge até 4,13× mais tokens/s no Llama-2 13B (2,97× no 7B) e 2,32× na média geométrica geral (*end-to-end*), dado o peso de 98% do decoding no tempo total.
- **Eficiência energética e custo:** cada placa CXL consome ~32,4 W (frente a ~300 W de uma GPU), entregando 2,88× mais tokens por Joule (até 4,26× no decoding) e custo estimado em \$14.873 (CPU + 512 GB GDDR6-PIM + controladores + switch CXL) contra \$42.128 do nó com 4× A100.

Em contrapartida, a arquitetura apresenta três limitações técnicas claras:

1. **Prefill lento (0,33× a 0,46×):** as unidades aritméticas nos bancos DRAM não possuem a densidade computacional matricial dos Tensor Cores. Em prompts longos (como cargas RAG de 32k tokens), a fase de prefill degrada o tempo de resposta.
2. **Escalabilidade acima de 70B:** o ganho no decoding cai de 4,13× (13B) para 1,25× (70B) devido ao aumento de comunicação entre placas através do switch CXL.
3. **Ausência de stack de software:** o CENT define uma ISA própria (CENT ISA) sem integração nativa com PyTorch ou bibliotecas equivalentes ao ecossistema CUDA.

O foco deste projeto está nessa terceira limitação: construir a camada de software que liga tensores de alto nível ao modelo de instruções e ao simulador da arquitetura.

---

## 3. O Simulador Oficial: `aim_simulator` (Ramulator 2.0)

Para validar a microarquitetura, os autores do CENT disponibilizaram o `aim_simulator`, construído sobre o Ramulator 2.0 (simulador acadêmico de memória DRAM de ciclo exato).

O `aim_simulator` compila um binário `ramulator2` (junto à biblioteca `libramulator.so`). Ele não é um runtime de redes neurais, mas um simulador discreto de eventos de hardware:

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

Ao ser executado via linha de comando (`ramulator2 -f configs/gddr6_aim.yaml -t arquivo.trace`), ele modela ciclo a ciclo acessos ao Row Buffer, conflitos de banco, escalonamento `FRFCFS`, refresh `AllBank` e instruções AiM nos 32 canais GDDR6-AiM, emitindo ao final os contadores de ciclos gastos por operação.

### Limitações para Integração com Software

1. **Entrada restrita a arquivos `.trace` estáticos:** exige gerar previamente arquivos de texto com instruções `AiM WR_GB`, `AiM MAC_ABK`, etc., já decompostas em linhas e máscaras de canais.
2. **Ausência de cálculo numérico:** o simulador contabiliza ciclos e temporização, mas não calcula o produto matricial em ponto flutuante nem produz um tensor de saída `y` para a camada seguinte do modelo.
3. **Acoplamento de compilação:** como artefato de pesquisa, depende de toolchains específicos de C++ e CMake, dificultando seu uso direto dentro de aplicações Python.

---

## 4. Papel do Hardware Dispatcher

Em software de sistemas, um dispatcher é a camada intermediária posicionada entre o framework de alto nível e o controlador do dispositivo acelerador. Quando o PyTorch executa `torch.matmul(x, W)` em uma GPU, a chamada atravessa o dispatcher interno e aciona bibliotecas nativas em C/C++ (`libcudart.so`, `libcublas.so`) responsáveis por configurar e submeter o trabalho ao hardware.

Para a arquitetura CENT, o hardware dispatcher implementado cumpre quatro funções:

1. **Validação de contrato físico:** inspeciona as dimensões dos tensores vindos do Python e garante que a operação seja uma multiplicação vetor-matriz (`GEMV`, `batch_size = 1`). Operações `GEMM` de prefill (`batch_size > 1`) são recusadas para execução no host.
2. **Roteamento de topologia:** seleciona em qual dos 32 dispositivos CXL (`DVid`) a camada será executada e quais dos 32 canais GDDR6 (`CHmask`) serão habilitados para a dimensão `K`.
3. **Tradução (*lowering*) para a CENT ISA:** divide a matriz de pesos `[M × K]` em blocos alinhados aos 16 bancos DRAM de cada canal, grava os descritores binários na SRAM de 2 MB e emite o arquivo `.trace` correspondente.
4. **Integridade numérica e telemetria:** computa o vetor resultante `y` diretamente no buffer de saída do PyTorch e retorna contadores de instruções, bytes ocupados na SRAM e tráfego de barramento economizado.

---

## 5. Implementação do `cent_runtime`

O `cent_runtime` estrutura esse dispatcher em duas camadas principais — Python (PyTorch/NumPy) e C++17 conectadas por uma C ABI estável (`libcent_runtime.so`) — além de um módulo opcional de integração com o `ramulator2`.

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

### 5.1. Camada Python (`python/cent_runtime/`)

O pacote Python expõe as funções `cent_runtime.matmul`, `simulate_on_ramulator` e `get_instruction_buffer_hex`:

- **`ops.py`:** aceita `torch.Tensor` ou `numpy.ndarray` (com fallback automático para NumPy caso o PyTorch não esteja instalado). Valida o formato de `x` como `(K,)` ou `(1, K)` e de `W` como `(M, K)`. Tensores com 3 ou mais dimensões ou `batch_size > 1` levantam `ValueError`. Em seguida, garante layout contíguo em `float32` na CPU (sem cópia adicional quando o tensor já atende a esse formato), aloca o tensor de saída `y` e transfere os ponteiros de memória diretamente via `ctypes`.
- **`_bindings.py`:** carrega `build/libcent_runtime.so` (acionando `make shared` automaticamente se a biblioteca ainda não estiver compilada) e mapeia a struct C `CentDispatchTelemetry` (56 bytes em x86_64) para um dicionário Python.
- **`simulator.py`:** quando `simulate=True` (criando um arquivo `.trace` temporário via `tempfile` caso `trace_path=None`), localiza o binário `ramulator2` em ordem configurável (`ramulator_bin` explícito → variável `CENT_RAMULATOR_BIN` → `external/ramulator2` → `../CENT/aim_simulator/build/ramulator2` → `PATH`), executa o simulador via `subprocess` e extrai `wr_gb_cycles`, `mac_abk_cycles` e `memory_system_cycles`.

### 5.2. Fronteira C ABI (`.h`) e Núcleo C++17 (`.hpp`)

Em C++, funções dentro de `namespace` ou com sobrecarga passam por *name mangling* (por exemplo, `cent::compute_dvid(int)` torna-se `_ZN4cent12compute_dvidEi` na tabela de símbolos), e a ABI de C++ varia entre compiladores. Para garantir interoperabilidade direta com FFI (`ctypes`), o projeto separa:

- **Cabeçalhos `.h` (`isa.h`, `telemetry.h`, `runtime_api.h`):** interface pública declarada sob `extern "C"`, exportando símbolos planos (`cent_init_runtime`, `cent_dispatch_gemv`, `cent_get_instruction_buffer_hex`).
- **Cabeçalhos `.hpp` (`topology.hpp`, `sram_buffer.hpp`, `isa_emitter.hpp`):** implementação interna em C++17 utilizando `namespace cent`, classes, RAII e `std::array`.

O código nativo é compilado com `-std=c++17 -O3 -fPIC -Wall -Wextra -Werror` e não utiliza exceções C++ (`throw` / `try-catch`) na fronteira. Todas as funções exportadas são marcadas com `noexcept` e retornam códigos explícitos via `enum CentStatus` (`CENT_OK = 0`, `CENT_ERR_NULL_PTR = -1`, `CENT_ERR_INVALID_SHAPE = -2`, `CENT_ERR_SRAM_OVERFLOW = -3`, `CENT_ERR_TRACE_IO = -4`).

### 5.3. Descritor Binário de 14 Bytes (`CentInstruction`)

O arquivo `include/cent/isa.h` define os parâmetros do controlador (`CENT_SRAM_BUFFER_BYTES = 2 MB`, 32 dispositivos CXL, 32 canais por dispositivo, 16 bancos por canal e `opsize = 64` bursts), os 5 opcodes da CENT ISA e o formato binário da instrução:

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

A diretiva `#pragma pack(push, 1)` verificada por `static_assert` remove qualquer *padding* de alinhamento entre campos, fixando cada instrução em 14 bytes contíguos (`1 + 1 + 4 + 2 + 4 + 2`).

### 5.4. Topologia, Buffer SRAM de 2 MB e Emissão de Instruções

Na execução de `cent_dispatch_gemv` (`src/runtime_api.cpp`), quatro etapas são processadas em sequência:

1. **Roteamento (`src/topology.cpp`):** calcula `DVid = ((layer_id % 32) + 32) % 32` para distribuir as camadas entre os 32 dispositivos CXL. A máscara de canais `CHmask` ativa 32 canais (`0xFFFFFFFFu`) quando `K >= 1024`, 16 canais (`0x0000FFFFu`) quando `K >= 512` e 8 canais (`0x000000FFu`) para dimensões menores.
2. **Buffer SRAM de 2 MB (`src/sram_buffer.cpp`):** a classe `SramBuffer` gerencia um `std::array<uint8_t, CENT_SRAM_BUFFER_BYTES>` estático de 2 MB pré-alocado, evitando alocação dinâmica (`new`/`delete`) durante o despacho. As instruções são gravadas via `std::memcpy` com verificação de limite (`CENT_ERR_SRAM_OVERFLOW`) e podem ser inspecionadas em hexadecimal por `format_hex_dump`.
3. **Decomposição Near-Bank PIM e emissão `.trace` (`src/isa_emitter.cpp`):** como cada canal opera 16 bancos em paralelo, a dimensão de saída `M` é particionada em `num_blocks = (M + 15) / 16` blocos. Para cada bloco `blk` (`row_addr = blk`), o emissor grava 4 instruções na SRAM (`WR_GB`, `WR_BIAS`, `MAC_ABK` e `RD_MAC`) e, quando `trace_path` é informado, escreve os comandos correspondentes no arquivo `.trace`:
   ```text
   AiM WR_GB 64 0 0xffffffff
   AiM WR_BIAS 0 0xffffffff
   AiM MAC_ABK 64 0xffffffff 0
   AiM RD_MAC 0 0xffffffff
   ```
   Ao final dos blocos, emite uma instrução `CENT_OP_EOC` (`AiM EOC`), necessária para encerrar a simulação no `aim_simulator`.
4. **Cálculo numérico e telemetria de barramento (`src/runtime_api.cpp` e `src/isa_emitter.cpp`):** computa `y = x · W^T` em ordem *row-major* com acumulador `double` (`FP64`) convertido para `float` (`FP32`) no ponteiro de saída `y`. Em paralelo, calcula o tráfego externo em precisão BF16 (2 bytes por elemento), comparando o baseline de GPU (`(M × K + K) × 2` bytes, que transfere `W` e `x`) contra o modelo CENT (`(K + M) × 2` bytes, onde apenas `x` e `y` trafegam pelo barramento).

---

## 6. Validação no Llama-2 7B

O repositório inclui testes automatizados em Python (`make test`), testbench nativo em C++ (`make test-core`) e um script de validação com as dimensões reais das camadas do Llama-2 7B (`examples/llama2_7b_demo.py`).

### Execução Local

```bash
git clone https://github.com/josedercysilva/cent-pim-runtime.git
cd cent-pim-runtime

./setup_env.sh
source .venv/bin/activate

make test        # Testa C ABI, precisão numérica, rejeição de GEMM, SRAM e .trace
make test-core   # Testbench nativo em C++
make demo        # Executa camadas de Atenção e MLP do Llama-2 7B
make simulate    # Opcional: executa o Ramulator 2.0 sobre os arquivos .trace gerados
```

### Uso com PyTorch

```python
import torch
import torch.nn.functional as F
import cent_runtime

# Fase de decoding (batch_size = 1)
x = torch.randn(1, 4096)        # Ativação de entrada
W = torch.randn(4096, 4096)     # Pesos estacionários na memória GDDR6-AiM

# Despacho para o runtime C++ (retorna tensor PyTorch + telemetria)
y, telemetry = cent_runtime.matmul(x, W, layer_id=0)

# Operações não-lineares permanecem na CPU via PyTorch
out = F.silu(y)

print(f"Saída: {out.shape}")
print(f"Placa CXL: #{telemetry['dvid']} | Máscara: {telemetry['channel_mask_hex']}")
print(f"Instruções: {telemetry['total_instructions']} ({telemetry['instruction_buffer_bytes']} B SRAM)")
print(f"Economia de barramento: {telemetry['memory_traffic_saved_pct']:.2f}%")
```

### Métricas Auditadas: Atenção vs. MLP

A tabela abaixo resume os valores obtidos para a projeção de Atenção (`q/k/v/o_proj`, `M = 4096, K = 4096, layer_id = 0`) e para a projeção MLP (`gate/up/down_proj`, `M = 11008, K = 4096, layer_id = 1`):

| Métrica | Camada de Atenção (`4096 × 4096`) | Camada MLP (`11008 × 4096`) |
| :--- | :--- | :--- |
| Dispositivo CXL (`DVid`) | Placa `#0` | Placa `#1` |
| Canais ativos (`CHmask`) | `0xffffffff` (32 canais) | `0xffffffff` (32 canais) |
| Blocos de 16 bancos (`ceil(M/16)`) | 256 blocos | 688 blocos |
| Total de instruções (`4/bloco + EOC`) | 1.025 (`256×4 + 1`) | 2.753 (`688×4 + 1`) |
| Ocupação na SRAM (`14 B/inst`) | 14.350 bytes (14,01 KB) | 38.542 bytes (37,64 KB) |
| Tráfego GPU HBM vs. CENT CXL (BF16) | ~32,01 MB vs. 16,00 KB (99,95% menor) | ~86,01 MB vs. 29,50 KB (99,97% menor) |
| Ciclos no Ramulator 2.0 (`CH31_AiM_ISR_WR_GB_cycles`) | 16.416 ciclos | 44.064 ciclos |

A proporção entre as duas camadas reflete diretamente a geometria das matrizes: a dimensão de saída da MLP é `11.008 / 4.096 = 2,6875×` maior que a da Atenção, e tanto o total de instruções na SRAM quanto a contagem de ciclos reportada pelo `aim_simulator` escalam na mesma razão de `2,68×` (`44.064 / 16.416`).

A inspeção dos primeiros 16 bytes da SRAM via `cent_get_instruction_buffer_hex` mostra o primeiro pacote de 14 bytes serializado em *little-endian* (`opcode = 0x02` para `WR_GB`, `dvid = 0x00`, `channel_mask = FF FF FF FF`, `bank = 00 00`, `row = 00 00 00 00`, `opsize = 64 = 0x0040 -> 40 00`), seguido pelos dois primeiros bytes da segunda instrução (`03 00`, `WR_BIAS`):

```text
[SRAM] 02 00 FF FF FF FF 00 00 00 00 00 00 40 00 03 00
```

---

## 7. Conclusão e Referências

O `cent_runtime` conecta a especificação da arquitetura CENT a um fluxo executável em Python e C++17:

- **Execução standalone (sem `ramulator2`):** permite despachar projeções lineares do PyTorch/NumPy através de uma C ABI via ponteiros diretos (sem serialização e sem cópia extra para tensores `float32` contíguos na CPU), inspecionar os pacotes binários de 14 bytes na SRAM e medir a redução de tráfego externo.
- **Execução integrada ao `aim_simulator`:** gera automaticamente os arquivos `.trace` da CENT ISA e invoca o Ramulator 2.0 para extrair a contagem exata de ciclos nos 32 canais GDDR6-AiM.

### Referências

1. **Código do Projeto:** [`josedercysilva/cent-pim-runtime`](https://github.com/josedercysilva/cent-pim-runtime)
2. **Slides da Apresentação (Seminário UFAM):** [`apresentacao_cent.pdf`](https://github.com/josedercysilva/cent-pim-runtime/blob/main/apresentacao_cent.pdf)
3. **Artigo Base (CENT):** Gu, Y., Khadem, A., et al. (2025). *PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference*. ASPLOS '25.
4. **Hierarquia de Memória:** Patterson, D. A., & Hennessy, J. L. (2014). *Computer Organization and Design: The Hardware/Software Interface* (5th Ed. — MIPS Edition), Cap. 5.
5. **Especificação CXL:** Compute Express Link Consortium (2022). *CXL Specification Revision 3.0*.
