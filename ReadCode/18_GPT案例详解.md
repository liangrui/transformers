# GPT-2 案例详解：Decoder-only 自回归模型的完整生命周期

> 本文档以 GPT-2 模型为例，将 Transformers 框架的所有模块串联起来，重点展示 Decoder-only 自回归模型从配置加载、分词编码、前向传播、注意力机制、自回归生成到训练推理的完整生命周期。
>
> 源码文件：
> - [configuration_gpt2.py](src/transformers/models/gpt2/configuration_gpt2.py)
> - [modeling_gpt2.py](src/transformers/models/gpt2/modeling_gpt2.py)
> - [tokenization_gpt2.py](src/transformers/models/gpt2/tokenization_gpt2.py)
> - [\_\_init\_\_.py](src/transformers/models/gpt2/__init__.py)

---

## 1. GPT-2 在 Transformers 中的定位

GPT-2 是 OpenAI 于 2019 年发布的 Decoder-only 因果语言模型，采用**自回归生成**范式：每次只预测下一个 token，已生成的内容作为上下文参与后续预测。与 BERT 等 Encoder 模型不同，GPT-2 使用**因果掩码（Causal Mask）**确保每个位置只能看到自身及之前的 token。

GPT-2 的一个标志性设计是使用 `Conv1D` 线性层（而非标准 `nn.Linear`），这是 OpenAI 原始实现的遗留设计——权重矩阵的形状为 `(in_features, out_features)`，与 `nn.Linear` 的 `(out_features, in_features)` 互为转置。

### 架构定位图

```mermaid
graph TD
    subgraph "Decoder-only 因果语言模型家族"
        GPT2["GPT-2<br/>2019 · OpenAI<br/>Conv1D + Post-Norm<br/>768/1024/1280/1600"]
        GPTNeo["GPT-Neo / GPT-J<br/>2021 · EleutherAI<br/>nn.Linear + Parallel Attention"]
        LLaMA["LLaMA<br/>2023 · Meta<br/>RMSNorm + SwiGLU + RoPE"]
        Qwen["Qwen 系列<br/>2023- · 阿里<br/>RMSNorm + SwiGLU + RoPE"]
    end

    GPT2 -->|"Conv1D→nn.Linear<br/>Post-Norm→Pre-Norm"| GPTNeo
    GPTNeo -->|"绝对位置→RoPE<br/>GELU→SwiGLU"| LLaMA
    LLaMA -->|"架构继承<br/>规模扩展"| Qwen

    style GPT2 fill:#e74c3c,color:#fff,stroke:#c0392b
    style GPTNeo fill:#e67e22,color:#fff,stroke:#d35400
    style LLaMA fill:#3498db,color:#fff,stroke:#2980b9
    style Qwen fill:#2ecc71,color:#fff,stroke:#27ae60
```

### 核心特征总结

| 特征 | GPT-2 | LLaMA（对比） |
|------|-------|--------------|
| 线性层 | `Conv1D`（权重转置） | `nn.Linear` |
| 归一化 | `LayerNorm` + Post-Norm | `RMSNorm` + Pre-Norm |
| 激活函数 | `gelu_new`（近似 GELU） | `silu`（SwiGLU） |
| 位置编码 | 可学习绝对位置 `wpe` | 旋转位置编码 RoPE |
| 注意力缩放 | `scale_attn_weights` + 可选层逆缩放 | 标准 `head_dim^-0.5` |
| 权重绑定 | `lm_head.weight ↔ wte.weight` | 同样绑定 |

---

## 2. Config 定义与特殊设计

`GPT2Config` 继承自 `PreTrainedConfig`，定义于 [configuration_gpt2.py](src/transformers/models/gpt2/configuration_gpt2.py)。它使用 `@strict` 装饰器（来自 `huggingface_hub`）确保数据类字段的严格类型检查，并通过 `attribute_map` 将 GPT-2 原始命名映射到 Transformers 统一命名。

### 关键参数解析

```python
# 源自 configuration_gpt2.py L78-L103
vocab_size: int = 50257              # 词表大小
n_positions: int = 1024              # 最大位置编码长度
n_embd: int = 768                    # 隐藏层维度
n_layer: int = 12                    # Transformer 层数
n_head: int = 12                     # 注意力头数
n_inner: int | None = None           # MLP 中间层维度，默认 4 * n_embd
activation_function: str = "gelu_new" # 近似 GELU 激活
scale_attn_weights: bool = True      # 是否缩放注意力权重（1/√d_k）
reorder_and_upcast_attn: bool = False # 混合精度下重排序并上溯注意力计算
add_cross_attention: bool = False    # 是否添加交叉注意力（用于编码器-解码器场景）
tie_word_embeddings: bool = True     # 权重绑定：lm_head ↔ wte
```

### Config 类图

```mermaid
classDiagram
    class PreTrainedConfig {
        +model_type: str
        +is_decoder: bool
        +from_pretrained()
        +to_dict()
    }

    class GPT2Config {
        +vocab_size: int = 50257
        +n_positions: int = 1024
        +n_embd: int = 768
        +n_layer: int = 12
        +n_head: int = 12
        +n_inner: int | None
        +activation_function: str
        +scale_attn_weights: bool
        +reorder_and_upcast_attn: bool
        +add_cross_attention: bool
        +tie_word_embeddings: bool
        +attribute_map: dict
    }

    PreTrainedConfig <|-- GPT2Config

    note for GPT2Config "attribute_map 将 GPT-2 原始命名\n映射到 Transformers 统一命名:\nhidden_size → n_embd\nnum_attention_heads → n_head\nnum_hidden_layers → n_layer\nmax_position_embeddings → n_positions"
```

### Conv1D vs nn.Linear 对比图

```mermaid
graph LR
    subgraph "nn.Linear"
        direction TB
        L_in["输入 x<br/>(batch, seq, in_features)"]
        L_w["权重 W<br/>shape: (out_features, in_features)"]
        L_out["输出 y = xW^T + b<br/>(batch, seq, out_features)"]
        L_in --> L_out
        L_w -.-> L_out
    end

    subgraph "Conv1D（GPT-2 使用）"
        direction TB
        C_in["输入 x<br/>(batch, seq, nx)"]
        C_w["权重 W<br/>shape: (nx, nf) ← 转置！"]
        C_out["输出 y = xW + b<br/>(batch, seq, nf)"]
        C_in --> C_out
        C_w -.-> C_out
    end

    L_out ==|"数学等价<br/>xW^T ≡ x·W_transposed"| C_out

    style C_w fill:#e74c3c,color:#fff
    style L_w fill:#3498db,color:#fff
```

Conv1D 的核心差异（定义于 [pytorch_utils.py](src/transformers/pytorch_utils.py) L97-L123）：

```python
class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.nx = nx
        # 关键：权重形状为 (nx, nf)，即 (in_features, out_features)
        # 而 nn.Linear 的权重形状为 (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        # torch.addmm(bias, input, weight) 计算 bias + input @ weight
        # 等价于 nn.Linear 的 F.linear(x, W.T, b) = x @ W.T + b
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x
```

---

## 3. from_pretrained 完整时序

从 `GPT2LMHeadModel.from_pretrained('gpt2')` 到模型就绪，涉及多个关键步骤。

### 时序图

```mermaid
sequenceDiagram
    participant User as 用户代码
    participant Auto as AutoModelForCausalLM
    participant Config as GPT2Config
    participant Model as GPT2LMHeadModel
    participant PTM as PreTrainedModel
    participant Hub as HuggingFace Hub

    User->>Auto: from_pretrained('gpt2')
    Auto->>Hub: 下载 config.json
    Hub-->>Auto: config.json
    Auto->>Config: GPT2Config.from_pretrained()
    Config-->>Auto: config 实例

    Auto->>Auto: 根据 model_type='gpt2'<br/>分发到 GPT2LMHeadModel
    Auto->>Model: GPT2LMHeadModel.from_pretrained('gpt2')

    Model->>PTM: PreTrainedModel.from_pretrained()
    PTM->>PTM: 1. 解析模型架构<br/>_load_pretrained_model()
    PTM->>Hub: 2. 下载权重文件<br/>model.safetensors
    Hub-->>PTM: 权重 state_dict

    PTM->>PTM: 3. 加载权重到模型<br/>model.load_state_dict()
    Note over PTM: Conv1D 权重无需转置<br/>因为 checkpoint 已是 (nx, nf) 格式

    PTM->>PTM: 4. 权重绑定处理<br/>tie_weights()
    Note over PTM: lm_head.weight ← transformer.wte.weight<br/>_tied_weights_keys 指定绑定关系

    PTM->>PTM: 5. 设备放置与 dtype 转换
    PTM-->>Model: 模型就绪
    Model-->>User: 可用模型实例
```

### 权重绑定机制

`GPT2LMHeadModel` 通过 `_tied_weights_keys` 声明权重绑定关系（[modeling_gpt2.py](src/transformers/models/gpt2/modeling_gpt2.py) L646）：

```python
class GPT2LMHeadModel(GPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        # lm_head 使用 nn.Linear，权重形状 (vocab_size, n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()  # 在此触发 tie_weights()
```

绑定过程：`lm_head.weight`（形状 `(50257, 768)`）与 `transformer.wte.weight`（形状 `(50257, 768)`）共享同一存储，修改一个另一个同步变化。

### Conv1D 权重加载的特殊处理

由于 GPT-2 的 checkpoint 中 Conv1D 权重已经以 `(nx, nf)` 格式存储，加载时无需额外转置。但需注意：
- `c_attn.weight`：形状 `(768, 2304)`，一次投影出 Q/K/V
- `c_proj.weight`：形状 `(768, 768)`，注意力输出投影
- `c_fc.weight`：形状 `(768, 3072)`，MLP 上投影
- `c_proj.weight`（MLP）：形状 `(3072, 768)`，MLP 下投影

---

## 4. Tokenizer 编码流程

`GPT2Tokenizer` 定义于 [tokenization_gpt2.py](src/transformers/models/gpt2/tokenization_gpt2.py)，继承自 `TokenizersBackend`，使用 **Byte-Level BPE** 分词算法。

### 核心设计特点

1. **ByteLevel 预处理**：将所有字符先转为 UTF-8 字节，再映射到 Unicode 字符，确保任何文本都可编码（无 `<unk>` 问题）
2. **无 [CLS]/[SEP]**：GPT-2 没有 BERT 风格的特殊分隔 token，只有 `<|endoftext|>` 作为 BOS/EOS
3. **空格敏感**：词首有无空格会产生不同 token（如 `"Hello"` vs `" Hello"`）

```python
# 源自 tokenization_gpt2.py L94-L129
class GPT2Tokenizer(TokenizersBackend):
    vocab_files_names = VOCAB_FILES_NAMES  # {"vocab_file": "vocab.json", "merges_file": "merges.txt"}
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE  # 使用 BPE 模型

    def __init__(self, vocab, merges, errors="replace",
                 unk_token="<|endoftext|>", bos_token="<|endoftext|>",
                 eos_token="<|endoftext|>", pad_token=None,
                 add_prefix_space=False, **kwargs):
        self.add_prefix_space = add_prefix_space
        self._vocab = vocab if vocab is not None else {}
        self._merges = merges or []
        # 构建 tokenizers 库的 BPE 模型
        self._tokenizer = Tokenizer(BPE(
            vocab=self._vocab, merges=self._merges,
            dropout=None, continuing_subword_prefix="",
            end_of_word_suffix="", fuse_unk=False,
        ))
        # ByteLevel 预分词器：将文本转为字节级表示
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=add_prefix_space
        )
        # ByteLevel 解码器：将字节级表示还原为文本
        self._tokenizer.decoder = decoders.ByteLevel()
```

### 编码流程图

```mermaid
flowchart TD
    A["原始文本<br/>'Hello world'"] --> B["ByteLevel 预处理<br/>pre_tokenizers.ByteLevel"]
    B --> B1["1. 按 Unicode 分割<br/>识别词边界（空格→Ġ前缀）"]
    B1 --> B2["2. 字节映射<br/>每个字符→UTF-8字节→Unicode映射<br/>'H'→'H', 'e'→'e', ' '→'Ġ'"]
    B2 --> C["BPE 分词<br/>Tokenizer(BPE)"]
    C --> C1["1. 初始化：每个字节映射为子词"]
    C1 --> C2["2. 迭代合并<br/>按 merges.txt 中的优先级<br/>合并最高频的相邻子词对"]
    C2 --> C3["3. 输出子词序列<br/>['Hello', 'Ġworld']"]
    C3 --> D["词表查找<br/>vocab.json"]
    D --> D1["input_ids: [15496, 995]"]
    D1 --> E["attention_mask 生成<br/>默认全1（无padding时）"]
    E --> E1["attention_mask: [1, 1]"]
    E1 --> F["模型输入<br/>input_ids + attention_mask"]

    style A fill:#3498db,color:#fff
    style B fill:#e67e22,color:#fff
    style C fill:#e74c3c,color:#fff
    style D fill:#9b59b6,color:#fff
    style F fill:#2ecc71,color:#fff
```

### 空格敏感性示例

```python
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tokenizer("Hello world")["input_ids"]     # [15496, 995]    → "Hello" + "Ġworld"
tokenizer(" Hello world")["input_ids"]    # [18435, 995]    → "ĠHello" + "Ġworld"
# 注意："Hello" 和 " Hello" 编码为不同的 token！
```

---

## 5. 模型前向传播全链路

从 `input_ids` 到 `logits` 的完整数据流，定义于 [modeling_gpt2.py](src/transformers/models/gpt2/modeling_gpt2.py) 的 `GPT2LMHeadModel.forward()` 和 `GPT2Model.forward()`。

### 数据流图

```mermaid
flowchart TD
    INPUT["input_ids<br/>(batch, seq_len)"] --> WTE["wte: nn.Embedding<br/>词嵌入<br/>(batch, seq_len, 768)"]
    INPUT --> POS["position_ids<br/>自动生成<br/>(1, seq_len)"]
    POS --> WPE["wpe: nn.Embedding<br/>位置嵌入<br/>(1, seq_len, 768)"]

    WTE --> ADD["⊕ 相加<br/>hidden = wte + wpe"]
    WPE --> ADD
    ADD --> DROP["drop: Dropout<br/>embd_pdrop=0.1"]

    DROP --> BLOCK1["GPT2Block #0<br/>ln_1 → Attn → + → ln_2 → MLP → +"]
    BLOCK1 --> BLOCK2["GPT2Block #1<br/>ln_1 → Attn → + → ln_2 → MLP → +"]
    BLOCK2 --> DOT["..."]
    DOT --> BLOCKN["GPT2Block #11<br/>ln_1 → Attn → + → ln_2 → MLP → +"]

    BLOCKN --> LNF["ln_f: LayerNorm<br/>最终层归一化"]
    LNF --> LMHEAD["lm_head: nn.Linear<br/>(768, 50257, bias=False)"]
    LMHEAD --> LOGITS["logits<br/>(batch, seq_len, 50257)"]

    style INPUT fill:#3498db,color:#fff
    style WTE fill:#e74c3c,color:#fff
    style WPE fill:#e67e22,color:#fff
    style LOGITS fill:#2ecc71,color:#fff
```

### 单层 GPT2Block 内部结构图

```mermaid
flowchart TD
    IN["hidden_states<br/>(batch, seq, 768)"] --> LN1["ln_1: LayerNorm"]
    LN1 --> ATTN["GPT2Attention<br/>c_attn → Q/K/V split →<br/>Causal Attention → c_proj"]
    ATTN --> RES1["+ 残差连接<br/>hidden = attn_out + residual"]

    RES1 --> LN2["ln_2: LayerNorm"]
    LN2 --> MLP["GPT2MLP<br/>c_fc → gelu_new → c_proj"]
    MLP --> RES2["+ 残差连接<br/>hidden = mlp_out + residual"]
    RES2 --> OUT["输出 hidden_states<br/>(batch, seq, 768)"]

    IN -.->|"保存 residual"| RES1
    RES1 -.->|"保存 residual"| RES2

    style LN1 fill:#f39c12,color:#fff
    style LN2 fill:#f39c12,color:#fff
    style ATTN fill:#e74c3c,color:#fff
    style MLP fill:#3498db,color:#fff
```

对应源码（[modeling_gpt2.py](src/transformers/models/gpt2/modeling_gpt2.py) L262-L309）：

```python
class GPT2Block(GradientCheckpointingLayer):
    def forward(self, hidden_states, past_key_values=None,
                attention_mask=None, ...):
        # Post-Norm：先归一化，再注意力，再残差
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, _ = self.attn(hidden_states, ...)
        hidden_states = attn_output + residual  # 第一个残差连接

        # Post-Norm：先归一化，再MLP，再残差
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states  # 第二个残差连接

        return hidden_states
```

### Post-Norm 架构对比图

```mermaid
flowchart LR
    subgraph "GPT-2: Post-Norm"
        direction TB
        P_IN["x"] --> P_ATTN["Attention(x)"]
        P_ATTN --> P_ADD1["x + Attention(x)"]
        P_ADD1 --> P_LN1["LayerNorm(x + Attention(x))"]
        P_LN1 --> P_MLP["MLP(LN_out)"]
        P_MLP --> P_ADD2["LN_out + MLP(LN_out)"]
        P_ADD2 --> P_LN2["LayerNorm(LN_out + MLP(LN_out))"]
        P_LN2 --> P_OUT["输出"]
    end

    subgraph "LLaMA: Pre-Norm"
        direction TB
        Q_IN["x"] --> Q_LN1["RMSNorm(x)"]
        Q_LN1 --> Q_ATTN["Attention(RMSNorm(x))"]
        Q_ATTN --> Q_ADD1["x + Attention(RMSNorm(x))"]
        Q_ADD1 --> Q_LN2["RMSNorm(x + Attn_out)"]
        Q_LN2 --> Q_MLP["MLP(RMSNorm(x + Attn_out))"]
        Q_MLP --> Q_ADD2["x + Attn_out + MLP(RMSNorm(...))"]
        Q_ADD2 --> Q_OUT["输出"]
    end

    style P_LN1 fill:#e74c3c,color:#fff
    style P_LN2 fill:#e74c3c,color:#fff
    style Q_LN1 fill:#3498db,color:#fff
    style Q_LN2 fill:#3498db,color:#fff
```

> **Post-Norm vs Pre-Norm**：GPT-2 采用 Post-Norm（归一化在残差之后），训练时梯度可能不稳定；LLaMA 采用 Pre-Norm（归一化在子层之前），训练更稳定，是现代 LLM 的主流选择。

---

## 6. 注意力系统运作

`GPT2Attention` 定义于 [modeling_gpt2.py](src/transformers/models/gpt2/modeling_gpt2.py) L75-L226，是 GPT-2 的核心计算模块。

### 注意力前向流程图

```mermaid
flowchart TD
    H["hidden_states<br/>(batch, seq, 768)"] --> CATTN["c_attn: Conv1D(2304, 768)<br/>一次投影 Q/K/V"]
    CATTN --> SPLIT["split(768, dim=2)<br/>→ Q, K, V 各 (batch, seq, 768)"]

    SPLIT --> RESHAPE_Q["Q.view + transpose<br/>(batch, num_heads, seq, head_dim)"]
    SPLIT --> RESHAPE_K["K.view + transpose<br/>(batch, num_heads, seq, head_dim)"]
    SPLIT --> RESHAPE_V["V.view + transpose<br/>(batch, num_heads, seq, head_dim)"]

    RESHAPE_K --> CACHE["KV Cache 更新<br/>past_key_values.update(K, V)"]
    RESHAPE_V --> CACHE
    CACHE --> K_CACHED["K_full (含历史)"]
    CACHE --> V_CACHED["V_full (含历史)"]

    RESHAPE_Q --> SCORE["attn_weights = Q @ K_full^T × scaling<br/>scaling = 1/√head_dim"]
    K_CACHED --> SCORE
    SCORE --> MASK["+ 因果掩码<br/>create_causal_mask()"]
    MASK --> SOFTMAX["Softmax(dim=-1)"]
    SOFTMAX --> DROP1["attn_dropout"]
    DROP1 --> OUTPUT["attn_output = weights @ V_full<br/>(batch, num_heads, seq, head_dim)"]
    V_CACHED --> OUTPUT
    OUTPUT --> RESHAPE_O["reshape → (batch, seq, 768)"]
    RESHAPE_O --> CPROJ["c_proj: Conv1D(768, 768)<br/>输出投影"]
    CPROJ --> DROP2["resid_dropout"]
    DROP2 --> OUT["attn_output<br/>(batch, seq, 768)"]

    style CATTN fill:#e74c3c,color:#fff
    style CACHE fill:#9b59b6,color:#fff
    style MASK fill:#f39c12,color:#fff
```

### 因果掩码生成图

因果掩码由 [masking_utils.py](src/transformers/masking_utils.py) 中的 `create_causal_mask()` 生成，确保每个位置只能关注自身及之前的 token：

```mermaid
flowchart TD
    A["create_causal_mask()<br/>masking_utils.py L894"] --> B{"config.is_causal?"}
    B -->|"True（默认）"| C["causal_mask_function<br/>kv_idx <= q_idx"]
    B -->|"False"| D["create_bidirectional_mask<br/>双向掩码"]

    C --> E{"注意力实现类型?"}
    E -->|"eager"| F["eager_mask()<br/>生成 4D float 掩码<br/>0（可见）/-inf（屏蔽）"]
    E -->|"sdpa"| G["sdpa_mask()<br/>生成 4D bool 掩码<br/>True（可见）/False（屏蔽）"]
    E -->|"flash_attention_2"| H["flash_attention_mask()<br/>返回 2D 掩码或 None"]

    F --> I["4D 掩码<br/>(batch, 1, q_len, kv_len)"]
    G --> I
    H --> I

    style A fill:#3498db,color:#fff
    style C fill:#e74c3c,color:#fff
    style I fill:#2ecc71,color:#fff
```

因果掩码矩阵示意（5×5）：

```
位置  0  1  2  3  4
 0 [ ■  ⬚  ⬚  ⬚  ⬚ ]    ■ = 可见（0）
 1 [ ■  ■  ⬚  ⬚  ⬚ ]    ⬚ = 屏蔽（-inf）
 2 [ ■  ■  ■  ⬚  ⬚ ]
 3 [ ■  ■  ■  ■  ⬚ ]
 4 [ ■  ■  ■  ■  ■ ]
```

### KV Cache 增量更新时序图

```mermaid
sequenceDiagram
    participant Model as GPT2Model
    participant Attn as GPT2Attention
    participant Cache as DynamicCache

    Note over Model,Cache: === 首次前向（Prefill） ===
    Model->>Attn: hidden_states (batch, 5, 768)
    Attn->>Attn: c_attn → Q, K, V<br/>K: (batch, 12, 5, 64)<br/>V: (batch, 12, 5, 64)
    Attn->>Cache: update(K, V, layer_idx=0)
    Cache-->>Attn: K_full=(batch,12,5,64), V_full=(batch,12,5,64)
    Attn->>Attn: Q @ K_full^T → softmax → @ V_full
    Attn-->>Model: attn_output (batch, 5, 768)

    Note over Model,Cache: === 生成第1个token ===
    Model->>Attn: hidden_states (batch, 1, 768)
    Attn->>Attn: c_attn → Q, K, V<br/>K_new: (batch, 12, 1, 64)<br/>V_new: (batch, 12, 1, 64)
    Attn->>Cache: update(K_new, V_new, layer_idx=0)
    Note over Cache: 拼接历史：<br/>K_full = cat(K_old, K_new)<br/>→ (batch, 12, 6, 64)
    Cache-->>Attn: K_full=(batch,12,6,64), V_full=(batch,12,6,64)
    Attn->>Attn: Q @ K_full^T → softmax → @ V_full
    Attn-->>Model: attn_output (batch, 1, 768)

    Note over Model,Cache: === 生成第2个token ===
    Model->>Attn: hidden_states (batch, 1, 768)
    Attn->>Cache: update(K_new, V_new, layer_idx=0)
    Note over Cache: K_full → (batch, 12, 7, 64)
    Cache-->>Attn: K_full=(batch,12,7,64), V_full=(batch,12,7,64)
    Attn-->>Model: attn_output (batch, 1, 768)
```

KV Cache 的关键源码（[modeling_gpt2.py](src/transformers/models/gpt2/modeling_gpt2.py) L193-L199）：

```python
# 在 GPT2Attention.forward() 中
if (past_key_values is not None and not is_cross_attention) or (
    past_key_values is not None and is_cross_attention and not is_updated
):
    # 将新的 K/V 与缓存中的历史 K/V 拼接
    key_states, value_states = curr_past_key_values.update(
        key_states, value_states, self.layer_idx
    )
```

---

## 7. generate() 生成全流程

`GPT2LMHeadModel` 继承了 `GenerationMixin`（[modeling_gpt2.py](src/transformers/models/gpt2/modeling_gpt2.py) L645），其 `generate()` 方法定义于 [generation/utils.py](src/transformers/generation/utils.py)。

### 生成循环时序图

```mermaid
sequenceDiagram
    participant User as 用户代码
    participant Gen as GenerationMixin.generate()
    participant Prep as 准备阶段
    participant Loop as 生成循环
    participant Model as GPT2LMHeadModel.forward()
    participant Sample as 采样策略
    participant Cache as KV Cache
    participant Stream as Streamer（可选）

    User->>Gen: generate(input_ids, max_new_tokens=50)
    Gen->>Prep: 1. 准备生成配置<br/>GenerationConfig 合并
    Prep->>Prep: 2. 准备 KV Cache<br/>DynamicCache()
    Prep->>Prep: 3. 准备注意力掩码<br/>create_causal_mask()

    Note over Loop,Cache: === Prefill 阶段 ===
    Prep->>Model: 首次前向（全部 input_ids）
    Model-->>Loop: logits (batch, seq_len, vocab_size)<br/>past_key_values（已缓存）

    Note over Loop,Cache: === Decode 循环 ===
    loop 每个生成步骤
        Loop->>Loop: 提取最后一个 token 的 logits<br/>logits[:, -1, :]
        Loop->>Sample: Logits 处理
        Note over Sample: 1. Repetition Penalty<br/>2. Temperature 缩放<br/>3. Top-K 过滤<br/>4. Top-P (Nucleus) 过滤
        Sample->>Sample: 采样/贪心选择<br/>next_token
        Loop->>Cache: KV Cache 已在 forward 中更新
        Loop->>Loop: 停止条件检查
        Note over Loop: • EOS token？<br/>• max_new_tokens 达到？<br/>• 停止词匹配？
        Loop->>Stream: streamer.put(next_token)<br/>（流式输出，可选）
        alt 未停止
            Loop->>Model: 下一步前向<br/>input_ids=next_token<br/>past_key_values=cache
            Model-->>Loop: logits (batch, 1, vocab_size)
        else 停止
            Loop-->>Gen: 生成序列
        end
    end

    Gen-->>User: generated_ids (batch, total_len)
```

### 辅助解码流程图（Speculative Decoding）

当使用辅助模型进行推测解码时，流程如下：

```mermaid
flowchart TD
    A["主模型: GPT2LMHeadModel"] --> B["辅助模型: assistant_model"]
    B --> C["辅助模型生成 K 个候选 token"]
    C --> D["主模型验证 K 个候选 token<br/>一次前向传播"]
    D --> E{"验证结果"}
    E -->|"全部接受"| F["接受 K 个 token<br/>+ 继续生成"]
    E -->|"第 j 个被拒绝"| G["接受前 j-1 个 token<br/>+ 从主模型采样第 j 个"]
    G --> H["继续下一轮推测"]
    F --> H

    style A fill:#e74c3c,color:#fff
    style B fill:#3498db,color:#fff
    style D fill:#f39c12,color:#fff
```

---

## 8. 训练流程

### 训练循环时序图

```mermaid
sequenceDiagram
    participant Data as 数据加载器
    participant Tok as GPT2Tokenizer
    participant Model as GPT2LMHeadModel
    participant Loss as Loss Function
    participant Opt as Optimizer

    Data->>Tok: 原始文本 batch
    Tok-->>Model: input_ids, attention_mask

    Note over Model: === 前向传播 ===
    Model->>Model: GPT2Model.forward()<br/>wte + wpe → Blocks → ln_f
    Model->>Model: lm_head(hidden_states)<br/>→ logits (batch, seq, 50257)

    Note over Model,Loss: === 损失计算 ===
    Model->>Loss: logits + labels
    Note over Loss: labels = input_ids（自回归标签）<br/>内部自动 shift：<br/>shift_logits = logits[:, :-1, :]<br/>shift_labels = labels[:, 1:]
    Loss->>Loss: CrossEntropyLoss<br/>shift_logits vs shift_labels
    Loss-->>Opt: loss 标量

    Note over Opt: === 反向传播 ===
    Opt->>Model: loss.backward()
    Note over Model: 梯度回传<br/>通过 lm_head → wte<br/>（权重绑定：梯度自动累加）
    Opt->>Opt: optimizer.step()
    Opt->>Model: 更新参数
```

### 权重绑定梯度流图

```mermaid
flowchart TD
    subgraph "前向传播"
        WTE_FWD["wte: nn.Embedding<br/>weight: (50257, 768)"]
        LM_FWD["lm_head: nn.Linear<br/>weight: (50257, 768)"]
        WTE_FWD -.->|"共享存储"| LM_FWD
    end

    subgraph "反向传播"
        LM_GRAD["lm_head.weight.grad<br/>来自 logits 的梯度"]
        WTE_GRAD["wte.weight.grad<br/>来自嵌入层的梯度"]
    end

    LM_GRAD -->|"梯度累加<br/>同一参数"| SHARED["共享权重<br/>梯度 = lm_head梯度 + wte梯度"]
    WTE_GRAD --> SHARED
    SHARED --> UPDATE["optimizer.step()<br/>一次性更新共享权重"]

    style WTE_FWD fill:#e74c3c,color:#fff
    style LM_FWD fill:#3498db,color:#fff
    style SHARED fill:#f39c12,color:#fff
```

损失计算的关键源码（[modeling_gpt2.py](src/transformers/models/gpt2/modeling_gpt2.py) L708-L716）：

```python
# GPT2LMHeadModel.forward()
loss = None
if labels is not None:
    # labels 自动 shift：logits 取前 n-1 位，labels 取后 n-1 位
    # 即：用位置 i 的输出预测位置 i+1 的 token
    loss = self.loss_function(
        logits,
        labels,
        vocab_size=self.config.vocab_size,
        **kwargs,
    )
```

### GPT-2 特殊的残差缩放初始化

GPT-2 采用了特殊的残差路径初始化策略（[modeling_gpt2.py](src/transformers/models/gpt2/modeling_gpt2.py) L448-L458）：

```python
# GPT2PreTrainedModel._init_weights()
if isinstance(module, PreTrainedModel):
    for name, p in module.named_parameters():
        if name == "c_proj.weight":
            # 残差投影层的权重缩小 1/√(2N)
            # N 为残差层数，2 是因为每个 Block 有 2 个残差连接
            init.normal_(p, mean=0.0,
                std=self.config.initializer_range / math.sqrt(2 * self.config.n_layer))
```

这一策略来自 GPT-2 论文：随着模型深度增加，残差路径上的方差会累积，通过缩小残差层权重来抵消这种累积效应。

---

## 9. Pipeline 推理

`pipeline("text-generation", model="gpt2")` 使用 `TextGenerationPipeline`（[text_generation.py](src/transformers/pipelines/text_generation.py)），封装了分词、生成、后处理的完整流程。

### Pipeline 时序图

```mermaid
sequenceDiagram
    participant User as 用户代码
    participant Pipe as TextGenerationPipeline
    participant Tok as GPT2Tokenizer
    participant Model as GPT2LMHeadModel
    participant Post as 后处理

    User->>Pipe: pipeline("text-generation",<br/>model="gpt2")

    Note over Pipe: === 初始化 ===
    Pipe->>Pipe: 1. 加载 Tokenizer
    Pipe->>Pipe: 2. 加载 Model
    Pipe->>Pipe: 3. 设置 padding_side="left"<br/>（Decoder-only 批量生成需要左填充）
    Pipe->>Pipe: 4. 默认 GenerationConfig<br/>max_new_tokens=256<br/>do_sample=True, temperature=0.7

    User->>Pipe: ("Hello, I'm a language model",<br/>max_new_tokens=50)

    Note over Pipe,Tok: === preprocess ===
    Pipe->>Tok: tokenizer(prefix + prompt_text,<br/>return_tensors="pt")
    Tok-->>Pipe: input_ids, attention_mask

    Note over Pipe,Model: === _forward ===
    Pipe->>Model: model.generate(<br/>input_ids=input_ids,<br/>attention_mask=attention_mask,<br/>max_new_tokens=50)
    Model-->>Pipe: generated_sequence<br/>(batch, num_return, total_len)

    Note over Pipe,Post: === postprocess ===
    Pipe->>Post: 截取新生成的 token<br/>sequence[prompt_len:]
    Post->>Tok: tokenizer.decode(<br/>new_tokens,<br/>skip_special_tokens=True)
    Tok-->>Post: 生成文本
    Post-->>User: [{"generated_text": "Hello, I'm a language model,<br/>and I'm here to help..."}]
```

Pipeline 的关键初始化逻辑（[text_generation.py](src/transformers/pipelines/text_generation.py) L99-L106）：

```python
class TextGenerationPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_model_type(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
        # Decoder-only 模型需要左填充以确保批量生成正确
        if self.tokenizer is not None and self.tokenizer.padding_side == "right":
            self.tokenizer.padding_side = "left"
```

Pipeline 默认生成配置（[text_generation.py](src/transformers/pipelines/text_generation.py) L93-L97）：

```python
_default_generation_config = GenerationConfig(
    max_new_tokens=256,
    do_sample=True,       # 自由文本生成通常使用采样
    temperature=0.7,
)
```

---

## 10. 状态与生命周期总结

GPT-2 模型在 Transformers 中的完整生命周期，从配置创建到推理输出，经历以下状态转换：

### 状态机图

```mermaid
stateDiagram-v2
    [*] --> ConfigCreated: GPT2Config()

    ConfigCreated --> ModelInstantiated: GPT2LMHeadModel(config)<br/>post_init() → _init_weights()
    ModelInstantiated --> WeightsLoaded: from_pretrained('gpt2')<br/>下载权重 + 加载 + 权重绑定

    WeightsLoaded --> Ready: 模型就绪<br/>.eval() 模式

    state Ready {
        [*] --> EvalMode
        EvalMode --> TrainMode: model.train()
        TrainMode --> EvalMode: model.eval()
    }

    Ready --> Tokenized: tokenizer(text)<br/>ByteLevel BPE 编码
    Tokenized --> Prefilling: model.forward(input_ids)<br/>首次前向 + KV Cache 填充
    Prefilling --> Decoding: 生成循环<br/>逐 token 解码
    Decoding --> Decoding: next_token → forward → sample<br/>KV Cache 增量更新
    Decoding --> Generated: EOS / max_length<br/>生成完成
    Generated --> PostProcessed: tokenizer.decode()<br/>还原文本

    Ready --> Training: model.train()<br/>labels=input_ids
    Training --> Training: forward → loss → backward → step<br/>权重绑定梯度累加
    Training --> Ready: model.eval()

    PostProcessed --> [*]: 输出结果
```

### 生命周期关键节点总结

| 阶段 | 关键函数/类 | 源码位置 |
|------|------------|---------|
| 配置创建 | `GPT2Config` | [configuration_gpt2.py](src/transformers/models/gpt2/configuration_gpt2.py) L25 |
| 模型实例化 | `GPT2LMHeadModel.__init__` | [modeling_gpt2.py](src/transformers/models/gpt2/modeling_gpt2.py) L648 |
| 权重初始化 | `GPT2PreTrainedModel._init_weights` | [modeling_gpt2.py](src/transformers/models/gpt2/modeling_gpt2.py) L433 |
| 预训练加载 | `PreTrainedModel.from_pretrained` | [modeling_utils.py](src/transformers/modeling_utils.py) |
| 权重绑定 | `_tied_weights_keys` + `tie_weights()` | [modeling_gpt2.py](src/transformers/models/gpt2/modeling_gpt2.py) L646 |
| 分词编码 | `GPT2Tokenizer.__call__` | [tokenization_gpt2.py](src/transformers/models/gpt2/tokenization_gpt2.py) L94 |
| 前向传播 | `GPT2Model.forward` | [modeling_gpt2.py](src/transformers/models/gpt2/modeling_gpt2.py) L522 |
| 注意力计算 | `GPT2Attention.forward` | [modeling_gpt2.py](src/transformers/models/gpt2/modeling_gpt2.py) L144 |
| 因果掩码 | `create_causal_mask` | [masking_utils.py](src/transformers/masking_utils.py) L894 |
| KV Cache | `DynamicCache.update` | [cache_utils.py](src/transformers/cache_utils.py) L1229 |
| 自回归生成 | `GenerationMixin.generate` | [generation/utils.py](src/transformers/generation/utils.py) L339 |
| 损失计算 | `GPT2LMHeadModel.loss_function` | [modeling_gpt2.py](src/transformers/models/gpt2/modeling_gpt2.py) L711 |
| Pipeline | `TextGenerationPipeline` | [text_generation.py](src/transformers/pipelines/text_generation.py) L23 |
| Conv1D 线性层 | `Conv1D` | [pytorch_utils.py](src/transformers/pytorch_utils.py) L97 |

### 模型家族一览

GPT-2 在 Transformers 中提供了多种任务头：

| 类名 | 任务 | 头部 |
|------|------|------|
| `GPT2Model` | 基础模型（提取隐藏状态） | 无 |
| `GPT2LMHeadModel` | 因果语言建模 | `lm_head` (Linear, 权重绑定) |
| `GPT2DoubleHeadsModel` | 语言建模 + 多项选择 | `lm_head` + `multiple_choice_head` |
| `GPT2ForSequenceClassification` | 序列分类 | `score` (Linear) |
| `GPT2ForTokenClassification` | Token 分类 | `classifier` (Linear) |
| `GPT2ForQuestionAnswering` | 问答 | `qa_outputs` (Linear → 2) |

---

> **总结**：GPT-2 作为 Decoder-only 因果语言模型的开山之作，其设计深刻影响了后续所有 LLM。从 Conv1D 到 nn.Linear、从 Post-Norm 到 Pre-Norm、从绝对位置编码到 RoPE，每一代演进都在 GPT-2 的基础上优化。理解 GPT-2 的完整生命周期，就是理解现代大语言模型的基石。
