Imagine tentar acelerar a inferência de um Large Language Model (LLM) como o Llama-2 7B durante a geração de texto token a token. Nessa etapa — conhecida como *decoding* — o processamento não é limitado pela capacidade de cálculo da GPU, mas pelo **Memory Wall**: a cada novo token, gigabytes de pesos precisam trafegar da memória até as unidades de cálculo apenas para realizar multiplicações vetor-matriz (`GEMV`, `batch_size = 1`).

A arquitetura **CENT** (*PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference*, Gu et al., **ASPLOS 2025**) ataca esse gargalo invertendo a lógica tradicional: em vez de mover os pesos até o processador, ela executa as operações próximo aos bancos de memória (**Near-Bank PIM**) dentro de módulos **GDDR6-AiM** conectados via **CXL 3.0**.

Ao estudar essa arquitetura na UFAM, esbarrei em uma lacuna prática: o simulador oficial do CENT (`aim_simulator`, baseado no **Ramulator 2.0**) é um binário C++ fechado que consome apenas arquivos estáticos de traço (`.trace`). Não existe uma camada para ligar frameworks reais de aprendizado de máquina ao modelo de execução da arquitetura.

Foi para resolver essa ponte que desenvolvi o [`cent_runtime`](https://github.com/josedercysilva/cent-pim-runtime): um *runtime* e *hardware dispatcher* em **C++17** com fronteira **C ABI nativa**, que permite executar camadas lineares do **PyTorch** e do **NumPy** diretamente sobre o modelo de despacho da CENT ISA.

---

## Visão Geral da Arquitetura

O `cent_runtime` segue a mesma filosofia de integração usada por bibliotecas como `libcudart` e `libcublas`: o código Python não conhece detalhes de registradores ou alinhamento de memória, apenas invoca uma fronteira C estável (`extern "C"`).

```text
+-----------------------------------------------------------------------+
|  CAMADA PYTHON (python/cent_runtime/)                                 |
|  cent_runtime.matmul(x, W, layer_id=0, simulate=False)                |
|  - Valida escopo GEMV (batch = 1) para PyTorch Tensor / NumPy ndarray |
+-----------------------------------+-----------------------------------+
                                    | ctypes (ponteiro contíguo, zero cópia)
                                    v
+-----------------------------------------------------------------------+
|  FRONTEIRA C ABI (include/cent/runtime_api.h -> libcent_runtime.so)   |
|  cent_dispatch_gemv(x, M, K, W, y, layer_id, trace_path, &telemetry)  |
+-----------------------------------+-----------------------------------+
                                    |
          +-------------------------+-------------------------+
          v                                                   v
+-----------------------------+               +-----------------------------+
|  NÚCLEO C++17 (src/)        |               |  EXECUÇÃO NUMÉRICA (FP64)   |
|  - topology: DVid e CHmask  |               |  y = x · W^T                |
|  - sram_buffer: SRAM 2 MB   |               |  Garante integridade para   |
|  - isa_emitter: blocos PIM  |               |  os tensores do PyTorch     |
+-------------+---------------+               +-----------------------------+
              |
              +---> Buffer Binário (pacotes de 14 bytes em RAM/SRAM)
              +---> Arquivo .trace (CENT ISA) ---> Ramulator 2.0 (opcional)
```

---

## O que vai para o PIM e o que fica no PyTorch

Um ponto importante da arquitetura CENT é que bancos de memória DRAM equipados com unidades AiM possuem apenas multiplicadores-acumuladores (`MAC`), e não unidades completas de propósito geral. O `cent_runtime` reflete exatamente essa divisão entre **PIM** (bancos DRAM) e **PNM / Host CPU**:

| Operação no Modelo | Onde Executa | Motivo Arquitetural |
| :--- | :--- | :--- |
| Projeções lineares densas (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) com `batch_size = 1` | **`cent_runtime` (Near-Bank PIM)** | Operações `GEMV` dominam o tráfego de pesos (>99%) na fase de *decoding* |
| `Softmax`, `RMSNorm`, `SiLU`, `RoPE` | **Host CPU (PyTorch nativo)** | Operações não-lineares, reduções e funções transcendentais ficam no host / PNM |
| Fase de *Prefill* (`batch_size > 1`) ou tensores 3D+ | **Host CPU (rejeitado pela ABI PIM)** | O dispatcher valida estritamente o contrato `GEMV` e levanta `ValueError` para `GEMM` |

---

## Por Baixo do Capô: C ABI, Pacote de 14 Bytes e SRAM

Para manter a fronteira previsível entre linguagens, a biblioteca compartilhada `libcent_runtime.so` é compilada com `-std=c++17 -O3 -Wall -Wextra -Werror`, **sem uso de exceções C++ (`throw`/`try-catch`)** no fluxo de controle. Erros são retornados explicitamente via `enum CentStatus` (`0` para sucesso, códigos negativos como `-1` e `-2` para ponteiros nulos ou dimensões inválidas), com funções marcadas como `noexcept`.

### 1. O Pacote Binário da CENT ISA (`CentInstruction`)

Cada comando enviado ao controlador CXL é empacotado em exatamente **14 bytes** (`include/cent/isa.h`), garantindo alinhamento binário fechado em tempo de compilação com `static_assert`:

```c
enum CentOpcode : uint8_t {
    CENT_OP_WR_GB   = 0x02, // Escreve ativação no Global Buffer dos canais ativos
    CENT_OP_WR_BIAS = 0x03, // Inicializa acumuladores nos bancos DRAM
    CENT_OP_RD_MAC  = 0x05, // Lê o resultado acumulado dos bancos
    CENT_OP_MAC_ABK = 0x0F, // Multiply-Accumulate paralelo nos 16 bancos (All-Bank)
    CENT_OP_EOC     = 0xFF  // End of Computation (encerra o traço no simulador)
};

#pragma pack(push, 1)
typedef struct CentInstruction {
    uint8_t  opcode;       // 1 byte  : Opcode da CENT ISA
    uint8_t  dvid;         // 1 byte  : Dispositivo CXL alvo (0 a 31)
    uint32_t channel_mask; // 4 bytes : Máscara dos 32 canais GDDR6-AiM
    uint16_t bank_index;   // 2 bytes : Índice do banco DRAM (0 a 15)
    uint32_t row_addr;     // 4 bytes : Endereço da linha física (Row Buffer)
    uint16_t opsize;       // 2 bytes : Tamanho da operação em bursts (64 por padrão)
} CentInstruction;
#pragma pack(pop)

static_assert(sizeof(CentInstruction) == 14,
              "ABI Error: CentInstruction must be exactly 14 bytes packed");
```

### 2. Topologia, Decomposição e Buffer SRAM de 2 MB

Ao receber uma chamada `cent_dispatch_gemv(x, M, K, W, y, layer_id, trace_path, &telem)`:

1. **Roteamento de dispositivo (`DVid`):** seleciona uma das 32 placas CXL 3.0 usando `((layer_id % 32) + 32) % 32`.
2. **Máscara de canais (`CHmask`):** ativa `32` canais (`0xffffffff`) para `K >= 1024`, `16` canais (`0x0000ffff`) para `K >= 512`, ou `8` canais (`0x000000ff`) abaixo disso.
3. **Decomposição em blocos de 16 bancos:** divide `M` em `ceil(M / 16)` blocos. Para cada bloco `row_addr`, grava a sequência `WR_GB`, `WR_BIAS`, `MAC_ABK` e `RD_MAC` dentro de um buffer estático contíguo de **2 MB** (`std::array<uint8_t, 2 * 1024 * 1024>`) — sem alocações dinâmicas no caminho crítico. Ao final, emite `EOC`.

Quando `trace_path` é informado, o mesmo ciclo gera o arquivo `.trace` textual consumido pelo Ramulator 2.0:

```text
AiM WR_GB 64 0 0xffffffff
AiM WR_BIAS 0 0xffffffff
AiM MAC_ABK 64 0xffffffff 0
AiM RD_MAC 0 0xffffffff
...
AiM EOC
```

Além disso, a função `cent_get_instruction_buffer_hex` permite inspecionar diretamente o *hex dump* dos primeiros 64 bytes gravados na SRAM emulada.

---

## Como Usar na Prática

### Instalação e Validação Local

O projeto é autocontido. Para criar o ambiente virtual, compilar a biblioteca compartilhada `build/libcent_runtime.so` e rodar os testes e a demonstração:

```bash
git clone https://github.com/josedercysilva/cent-pim-runtime.git
cd cent-pim-runtime

./setup_env.sh
source .venv/bin/activate

make test        # Suíte Python (unittest): ABI, precisão, erros de shape e .trace
make test-core   # Testbench nativo em C++
make demo        # Demonstração com camadas reais do Llama-2 7B
```

> **Sobre o Ramulator 2.0:** O simulador físico de ciclos é **100% opcional**. O núcleo C++ já calcula os resultados numéricos, o *hex dump* da SRAM, a contagem de instruções e a telemetria de tráfego de forma independente. O Ramulator só entra em cena quando você passa `simulate=True` ou roda `make simulate`.

### Exemplo com PyTorch

Na camada Python, basta substituir uma projeção linear `GEMV` por `cent_runtime.matmul` e seguir com as ativações normalmente na CPU:

```python
import torch
import torch.nn.functional as F
import cent_runtime

# Ativação de 1 token (decoding) e matriz de pesos estacionária na DRAM
x = torch.randn(1, 4096)
W = torch.randn(4096, 4096)

# Despacho para o núcleo C++ (retorna tensor PyTorch + dicionário de telemetria)
y, telemetry = cent_runtime.matmul(x, W, layer_id=0)

# Operação não-linear continua transparente no PyTorch
out = F.silu(y)

print(f"Saída: {out.shape}")
print(f"Placa CXL (DVid): {telemetry['dvid']} | Canais: {telemetry['channel_mask_hex']}")
print(f"Instruções emitidas: {telemetry['total_instructions']}")
print(f"Economia de tráfego externo: {telemetry['memory_traffic_saved_pct']:.2f}%")
```

Caso `torch` não esteja instalado no ambiente, o pacote faz *fallback* automático para `numpy.ndarray`, preservando o formato 1D `(M,)` ou 2D `(1, M)` da entrada.

---

## Números Validados no Llama-2 7B

O script `examples/llama2_7b_demo.py` exercita duas camadas reais do Llama-2 7B (considerando precisão BF16 de 2 bytes/elemento na estimativa de tráfego externo frente ao baseline de GPU movendo pesos da HBM):

| Camada (Llama-2 7B) | Dimensões (`M × K`) | `DVid` | Instruções (`EOC` incluso) | SRAM Usada | Tráfego GPU vs. CENT CXL | Ciclos `WR_GB` (Ramulator 2.0) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Atenção** (`q/k/v/o proj`) | `4096 × 4096` | `0` | `1.025` (`256` blocos × 4 + `1`) | `14.350 B` | `~32,01 MB` vs. `16,00 KB` (**99,95%** menos) | `16.416` |
| **MLP** (`gate/up/down proj`) | `11008 × 4096` | `1` | `2.753` (`688` blocos × 4 + `1`) | `38.542 B` | Apenas ativação + saída no barramento | `44.064` (`2,68x`) |

Repare na consistência física entre as camadas: como a camada MLP tem `11.008 / 4.096 = 2,6875x` mais linhas de saída do que a camada de Atenção, tanto o total de instruções geradas na SRAM quanto os ciclos `CH31_AiM_ISR_WR_GB_cycles` medidos no simulador escalam exatamente na mesma proporção (`2,68x`).

---

## Referências

* **Repositório do projeto:** [github.com/josedercysilva/cent-pim-runtime](https://github.com/josedercysilva/cent-pim-runtime)
* **Artigo base:** Gu et al., *PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference*, ASPLOS 2025.
