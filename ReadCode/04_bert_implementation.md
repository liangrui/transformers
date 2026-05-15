
# Transformers 核心模型实现分析 (以 BERT 为例)

## 概述图

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                          BERT 模型架构概览                                                  │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                           │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐   │
│  │  Input Layer (输入层)                                                              │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │   │
│  │  │ Input Embedding │  │ Segment Emb.    │  │ Positional Emb. │                  │   │
│  │  │ Word Embeddings │  │ Token Type IDs  │  │ Position IDs    │                  │   │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘                  │   │
│  │           │                    │                    │                           │   │
│  │           └────────────────────┼────────────────────┘                           │   │
│  │                                ↓                                                    │   │
│  │                    ┌─────────────────────┐                                         │   │
│  │                    │ LayerNorm + Dropout │                                         │   │
│  │                    └──────────┬──────────┘                                         │   │
│  └───────────────────────────────┼────────────────────────────────────────────────────┘   │
│                                  ↓                                                        │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐   │
│  │  Transformer Encoder Layers (N 层)                                                │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────┐   │   │
│  │  │ Encoder Layer [1..N]                                                       │   │   │
│  │  │ ┌───────────────────────────────────────────────────────────────────────┐ │   │   │
│  │  │ │ Multi-Head Self-Attention                                              │ │   │   │
│  │  │ │ - Q/K/V Projections                                                    │ │   │   │
│  │  │ │ - Attention Scores Calculation                                         │ │   │   │
│  │  │ │ - Attention Mask (Padding Mask)                                        │ │   │   │
│  │  │ │ - Output Projection                                                    │ │   │   │
│  │  │ └─────────────────────┬─────────────────────────────────────────────────┘ │   │   │
│  │  │                       ↓                                                     │   │   │
│  │  │        ┌───────────────────────────┐                                      │   │   │
│  │  │        │   Residual + LayerNorm    │                                      │   │   │
│  │  │        └────────────┬──────────────┘                                      │   │   │
│  │  │                     ↓                                                     │   │   │
│  │  │  ┌─────────────────────────────────────────────────────────────────────┐  │   │   │
│  │  │  │ Feed-Forward Network (FFN)                                          │  │   │   │
│  │  │  │ - Linear Layer 1 (hidden_size → intermediate_size)                   │  │   │   │
│  │  │  │ - Activation Function (GELU)                                        │  │   │   │
│  │  │  │ - Linear Layer 2 (intermediate_size → hidden_size)                   │  │   │   │
│  │  │  │ - Dropout                                                           │  │   │   │
│  │  │  └─────────────────────────────┬───────────────────────────────────────┘  │   │   │
│  │  │                                ↓                                          │   │   │
│  │  │               ┌───────────────────────────┐                               │   │   │
│  │  │               │   Residual + LayerNorm    │                               │   │   │
│  │  │               └───────────────────────────┘                               │   │   │
│  │  └───────────────────────────────────────────────────────────────────────────┘   │   │
│  └───────────────────────────────────────────────────────────────────────────────────┘   │
│                                  ↓                                                        │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐   │
│  │  Output Layer (输出层)                                                             │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────┐   │   │
│  │  │ <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> Token Output (Pooled Output)                                           │   │   │
│  │  │  - Linear Layer + Tanh Activation                                          │   │   │
│  │  │  - For sequence classification, QA, etc.                                   │   │   │
│  │  └───────────────────────────────────────────────────────────────────────────┘   │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────┐   │   │
│  │  │  Token-Level Output (Sequence Output)                                     │   │   │
│  │  │  - For token classification (NER), masked language modeling, etc.         │   │   │
│  │  └───────────────────────────────────────────────────────────────────────────┘   │   │
│  └───────────────────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

## 1. 整体结构

### 1.1 文件组织

在 `models/bert/` 目录下：
```
models/bert/
├── __init__.py              # 导出类
├── configuration_bert.py    # 配置类
├── modeling_bert.py         # 模型实现
├── tokenization_bert.py     # 分词器
└── tokenization_bert_fast.py # 快速分词器
```

### 1.2 核心类层次

```python
PreTrainedModel (基类)
    ↓
BertPreTrainedModel (BERT 基类，初始化权重)
    ├── BertModel (基础模型，输出隐藏状态)
    ├── BertForPreTraining (预训练任务：MLM + NSP)
    ├── BertForMaskedLM (掩码语言建模)
    ├── BertForNextSentencePrediction (下一句预测)
    ├── BertForSequenceClassification (序列分类)
    ├── BertForTokenClassification (Token 分类)
    ├── BertForQuestionAnswering (问答)
    ├── BertForMultipleChoice (多项选择)
    └── BertLMHeadModel (因果语言建模)
```

## 2. BERT 配置类详解

### 2.1 BertConfig 完整定义

```python
@auto_docstring(checkpoint="google-bert/bert-base-uncased")
@strict
class BertConfig(PreTrainedConfig):
    """
    BERT 配置类，定义模型架构的所有参数
    """
    
    # 模型类型标识
    model_type = "bert"
    
    # ==================== 基础架构参数 ====================
    vocab_size: int = 30522                  # 词汇表大小
    hidden_size: int = 768                   # 隐藏层维度 (base:768, large:1024)
    num_hidden_layers: int = 12              # 编码器层数 (base:12, large:24)
    num_attention_heads: int = 12            # 注意力头数 (base:12, large:16)
    intermediate_size: int = 3072            # FFN 中间层维度 (base:3072, large:4096)
    
    # ==================== 激活与正则化 ====================
    hidden_act: str = "gelu"                 # 隐藏层激活函数
    hidden_dropout_prob: float = 0.1         # 隐藏层 dropout
    attention_probs_dropout_prob: float = 0.1 # 注意力 dropout
    
    # ==================== 位置编码 ====================
    max_position_embeddings: int = 512       # 最大序列长度
    type_vocab_size: int = 2                 # Token Type 词汇表 (Segment IDs)
    initializer_range: float = 0.02          # 初始化范围
    
    # ==================== 归一化 ====================
    layer_norm_eps: float = 1e-12            # LayerNorm epsilon
    
    # ==================== 特殊 Token ID ====================
    pad_token_id: int = 0
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # ==================== 其他 ====================
    use_cache: bool = True                   # 是否使用 KV 缓存
    classifier_dropout: Optional[float] = None # 分类器 dropout
    is_decoder: bool = False                 # 是否作为解码器
    add_cross_attention: bool = False        # 是否添加交叉注意力
    tie_word_embeddings: bool = True         # 输入输出 embedding 是否共享
    
    # ==================== 注意力实现 ====================
    _attn_implementation: Optional[str] = None # "eager", "sdpa", "flash_attention_2"
```

## 3. 核心组件详解

### 3.1 BertEmbeddings - 嵌入层

```python
class BertEmbeddings(nn.Module):
    """
    构建输入 embedding，包含三个部分：
    1. Word Embeddings - 词嵌入
    2. Position Embeddings - 位置编码
    3. Token Type Embeddings - 段嵌入 (用于区分句子A/B)
    """
    
    def __init__(self, config):
        super().__init__()
        # 词嵌入层
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            padding_idx=config.pad_token_id
        )
        # 位置编码层 (可学习的位置 embedding)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, 
            config.hidden_size
        )
        # Token Type 嵌入层 (Segment ID)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, 
            config.hidden_size
        )
        
        # LayerNorm 和 Dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 注册 position_ids 和 token_type_ids buffer
        self.register_buffer(
            "position_ids", 
            torch.arange(config.max_position_embeddings).expand((1, -1)), 
            persistent=False
        )
        self.register_buffer(
            "token_type_ids", 
            torch.zeros(self.position_ids.size(), dtype=torch.long), 
            persistent=False
        )
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -&gt; torch.Tensor:
        """
        前向传播
        
        计算方式：
        embedding = word_emb + position_emb + token_type_emb
        embedding = LayerNorm(embedding)
        embedding = Dropout(embedding)
        """
        # 确定输入形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        
        batch_size, seq_length = input_shape
        
        # 计算 position_ids
        if position_ids is None:
            position_ids = self.position_ids[
                :, 
                past_key_values_length : seq_length + past_key_values_length
            ]
        
        # 计算 token_type_ids
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
                buffered_token_type_ids = torch.gather(
                    buffered_token_type_ids, 
                    dim=1, 
                    index=position_ids
                )
                token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
            else:
                token_type_ids = torch.zeros(
                    input_shape, 
                    dtype=torch.long, 
                    device=self.position_ids.device
                )
        
        # 获取 word embeddings
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # 获取 token_type embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # 相加
        embeddings = inputs_embeds + token_type_embeddings
        
        # 获取 position embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings
        
        # LayerNorm 和 Dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
```

### 3.2 BertSelfAttention - 多头自注意力

```python
class BertSelfAttention(nn.Module):
    """
    多头自注意力机制
    """
    
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查 hidden_size 必须是 num_attention_heads 的倍数
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        
        # 配置参数
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Q/K/V 投影矩阵
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 位置编码类型 (相对位置编码等)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        
        # 缩放因子 (1/sqrt(d_k))
        self.scale = self.attention_head_size ** -0.5
    
    def transpose_for_scores(self, x: torch.Tensor) -&gt; torch.Tensor:
        """
        将输入 tensor 转换为多头注意力格式
        
        输入形状: (batch_size, seq_length, hidden_size)
        输出形状: (batch_size, num_heads, seq_length, head_size)
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -&gt; Tuple[torch.Tensor, Optional[Tuple[torch.FloatTensor]]]:
        """
        多头自注意力前向传播
        
        核心计算：
        Q = query(hidden_states)
        K = key(hidden_states)
        V = value(hidden_states)
        
        attention_scores = Q @ K.T / sqrt(d_k)
        attention_scores += attention_mask  # mask padding
        attention_probs = softmax(attention_scores)
        attention_probs = dropout(attention_probs)
        output = attention_probs @ V
        """
        # 处理交叉注意力 (如果是 decoder)
        if encoder_hidden_states is not None:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Q 投影
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        # 处理 KV 缓存 (用于自回归生成)
        if past_key_value is not None:
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        
        past_key_value = (key_layer, value_layer)
        
        # 计算注意力分数: (batch, heads, seq_length, seq_length)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores * self.scale
        
        # 应用注意力 mask (padding mask)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # 计算注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        # 应用 dropout
        attention_probs = self.dropout(attention_probs)
        
        # 应用 head mask (可选)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        # 计算输出
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # 调整形状回来
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = outputs + (past_key_value,)
        return outputs
```

### 3.3 BertSelfOutput - 注意力输出层

```python
class BertSelfOutput(nn.Module):
    """
    注意力层输出，包含线性投影、残差连接和 LayerNorm
    """
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        input_tensor: torch.Tensor
    ) -&gt; torch.Tensor:
        """
        前向传播：
        output = Dense(hidden_states)
        output = Dropout(output)
        output = LayerNorm(output + input_tensor)  # 残差连接
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
```

### 3.4 BertAttention - 注意力模块组合

```python
class BertAttention(nn.Module):
    """
    完整的注意力模块，包含 SelfAttention 和 SelfOutput
    """
    
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -&gt; Tuple[torch.Tensor, Optional[Tuple[torch.FloatTensor]]]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs
```

### 3.5 BertIntermediate - FFN 第一层

```python
class BertIntermediate(nn.Module):
    """
    Feed-Forward Network 的第一层
    Linear(hidden_size → intermediate_size) + GELU
    """
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]  # GELU
    
    def forward(self, hidden_states: torch.Tensor) -&gt; torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
```

### 3.6 BertOutput - FFN 第二层

```python
class BertOutput(nn.Module):
    """
    Feed-Forward Network 的第二层 + 残差连接
    Linear(intermediate_size → hidden_size) + Dropout + LayerNorm
    """
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        input_tensor: torch.Tensor
    ) -&gt; torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
```

### 3.7 BertLayer - 完整的编码器层

```python
class BertLayer(nn.Module):
    """
    完整的 Transformer Encoder 层
    包含：Attention → Residual/LayerNorm → FFN → Residual/LayerNorm
    """
    
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        
        # 交叉注意力 (如果是 decoder)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            self.crossattention = BertAttention(config)
        
        # FFN
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -&gt; Tuple[torch.Tensor, Optional[Tuple[torch.FloatTensor]]]:
        # 自注意力
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        
        # 交叉注意力 (如果是 decoder)
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]
        
        # FFN (支持分块处理)
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        
        outputs = (layer_output,) + outputs
        return outputs
    
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
```

### 3.8 BertEncoder - 编码器层堆叠

```python
class BertEncoder(nn.Module):
    """
    Transformer 编码器，堆叠 N 层 BertLayer
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -&gt; Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        """
        前向传播 through 所有编码器层
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.config.add_cross_attention) else None
        next_decoder_cache = () if use_cache else None
        
        # 遍历每一层
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            # 梯度检查点 (节省显存)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
```

### 3.9 BertPooler - <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> Token 池化

```python
class BertPooler(nn.Module):
    """
    提取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token 输出并进行池化
    用于序列分类任务
    """
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states: torch.Tensor) -&gt; torch.Tensor:
        """
        取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token (index 0) 的输出，然后线性投影 + Tanh
        """
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
```

## 4. 完整模型实现

### 4.1 BertModel - 基础 BERT 模型

```python
class BertModel(BertPreTrainedModel):
    """
    BERT 基础模型，输出隐藏状态和可选的池化输出
    """
    
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        
        # 核心组件
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        
        # 池化层 (可选)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        
        # 初始化权重
        self.gradient_checkpointing = False
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    
    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -&gt; Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
        """
        BERT 前向传播
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
        
        # 输入检查
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        # 构造 attention mask
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        
        # 构造 token_type_ids
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        
        # 扩展 attention mask 为 [batch_size, 1, 1, seq_length]
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape
        )
        
        # 处理 encoder attention mask
        encoder_extended_attention_mask = None
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        
        # 准备 head mask
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        # 嵌入层
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        
        # 编码器
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = encoder_outputs[0]
        
        # 池化
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
```

### 4.2 BertForSequenceClassification - 序列分类

```python
class BertForSequenceClassification(BertPreTrainedModel):
    """
    BERT 用于序列分类任务（如情感分析）
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        # BERT 基础模型
        self.bert = BertModel(config)
        
        # 分类头
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # 初始化权重
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -&gt; Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # BERT 前向传播
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token 输出
        pooled_output = outputs[1]
        
        # 分类
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # 计算 loss
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels &gt; 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```

## 5. 权重初始化

```python
class BertPreTrainedModel(PreTrainedModel):
    """
    BERT 模型基类，处理权重初始化
    """
    
    config_class = BertConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _supports_param_buffer_assignment = False
    
    def _init_weights(self, module):
        """
        初始化权重
        - Linear: 正态分布 (mean=0, std=initializer_range)
        - LayerNorm: weight=1, bias=0
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
```

## 6. 使用示例

### 6.1 基础使用

```python
from transformers import BertTokenizer, BertModel

# 加载 tokenizer 和 model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 编码文本
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

# 前向传播
outputs = model(**inputs)

# 获取输出
last_hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
pooled_output = outputs.pooler_output            # (batch, hidden_size)
```

### 6.2 序列分类

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

# 推理
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)
```

### 6.3 特征提取（获取各层隐藏状态）

```python
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# outputs.hidden_states 是 tuple，包含各层隐藏状态
# hidden_states[0] = embedding 输出
# hidden_states[1..12] = 12 层 encoder 输出
all_hidden_states = outputs.hidden_states
```

## 7. 关键技术要点

### 7.1 残差连接与 LayerNorm 位置

Transformer 采用 Pre-LN 结构：
```
x → LayerNorm → Attention → Dropout → +x → LayerNorm → FFN → Dropout → +x
```

### 7.2 位置编码

BERT 使用可学习的位置编码，不是正弦余弦位置编码。

### 7.3 Attention Mask

Padding mask: `[1, 1, 1, 0, 0]` (1 表示有效 token，0 表示 padding)
转换为 logit 空间: `[0, 0, 0, -10000, -10000]` (加到 attention scores)

### 7.4 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> Token 的作用

<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token 是用于聚合整个序列信息的特殊 token，用于分类任务。

## 代码参考

- [modeling_bert.py](../src/transformers/models/bert/modeling_bert.py)
- [configuration_bert.py](../src/transformers/models/bert/configuration_bert.py)
- [modeling_outputs.py](../src/transformers/modeling_outputs.py)

