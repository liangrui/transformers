
# Transformers 文本生成系统分析

## 概述图

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              文本生成系统架构                                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  GenerationConfig - 生成配置                                                                 │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ • max_new_tokens: 最大新生成 token 数                                                  │ │ │
│  │  │ • do_sample: 是否采样 (vs 贪婪解码)                                                    │ │ │
│  │  │ • temperature: 温度参数 (采样随机性)                                                   │ │ │
│  │  │ • top_k: Top-K 采样                                                                   │ │ │
│  │  │ • top_p: Nucleus (Top-P) 采样                                                         │ │ │
│  │  │ • repetition_penalty: 重复惩罚                                                         │ │ │
│  │  │ • num_beams: Beam Search 束数                                                         │ │ │
│  │  │ • early_stopping: Beam Search 提前停止                                                 │ │ │
│  │  │ • use_cache: 使用 KV 缓存 (加速解码)                                                   │ │ │
│  │  │ • eos_token_id: 结束 token ID                                                         │ │ │
│  │  │ • pad_token_id: Padding token ID                                                      │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────┬────────────────────────────────────────────────────┘ │
│                                           ↓                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  GenerationMixin - 生成 Mixin 基类                                                           │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ • generate(): 主入口方法                                                               │ │ │
│  │  │ • _validate_model_class(): 验证模型类型                                               │ │ │
│  │  │ • _prepare_special_tokens(): 准备特殊 token (eos, pad)                                │ │ │
│  │  │ • _get_logits_warper(): 获取 logits 处理器 (temperature, top_k, top_p)                │ │ │
│  │  │ • _get_logits_processor(): 获取 logits 处理器 (repetition penalty, etc)             │ │ │
│  │  │ • _sample(): 采样解码                                                                 │ │ │
│  │  │ • _greedy_search(): 贪婪解码                                                          │ │ │
│  │  │ • _beam_search(): Beam Search 解码                                                    │ │ │
│  │  │ • _beam_sample(): Beam Sample 解码                                                    │ │ │
│  │  │ • _group_beam_search(): Group Beam Search                                             │ │ │
│  │  │ • _contrastive_search(): 对比搜索                                                     │ │ │
│  │  │ • _assisted_decoding(): 辅助解码 (投机解码)                                           │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────┬────────────────────────────────────────────────────┘ │
│                                           ↓                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  LogitsProcessorList - Logits 处理流水线                                                    │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ ┌──────────────────────────────────┐  ┌──────────────────────────────────┐            │ │ │
│  │  │ │ TemperatureLogitsWarper          │  │ TopKLogitsWarper                 │            │ │ │
│  │  │ │ - 按 temperature 缩放 logits     │  │ - 只保留 top_k 个 logits          │            │ │ │
│  │  │ │ - 增大/减小随机性                │  │ - 其他设为 -inf                    │            │ │ │
│  │  │ └──────────────────────────────────┘  └──────────────────────────────────┘            │ │ │
│  │  │ ┌──────────────────────────────────┐  ┌──────────────────────────────────┐            │ │ │
│  │  │ │ TopPLogitsWarper                 │  │ RepetitionPenaltyLogitsProcessor │            │ │ │
│  │  │ │ - Nucleus 采样                   │  │ - 惩罚已出现 token                │            │ │ │
│  │  │ │ - 累计概率达到 p 的最小集合       │  │ - 抑制重复生成                     │            │ │ │
│  │  │ └──────────────────────────────────┘  └──────────────────────────────────┘            │ │ │
│  │  │ ┌──────────────────────────────────┐  ┌──────────────────────────────────┐            │ │ │
│  │  │ │ TypicalLogitsWarper              │  │ NoRepeatNGramLogitsProcessor     │            │ │ │
│  │  │ │ - 典型采样 (Typical)              │  │ - 禁止重复 n-gram                 │            │ │ │
│  │  │ │ - 局部典型性                     │  │ - 避免循环                         │            │ │ │
│  │  │ └──────────────────────────────────┘  └──────────────────────────────────┘            │ │ │
│  │  │ ┌──────────────────────────────────┐  ┌──────────────────────────────────┐            │ │ │
│  │  │ │ EpsilonLogitsWarper              │  │ EtaLogitsWarper                   │            │ │ │
│  │  │ │ - Epsilon 采样                   │  │ - Eta 采样                         │            │ │ │
│  │  │ └──────────────────────────────────┘  └──────────────────────────────────┘            │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────┬────────────────────────────────────────────────────┘ │
│                                           ↓                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  StoppingCriteria - 停止条件                                                                 │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ • MaxLengthCriteria: 达到最大长度停止                                                  │ │ │
│  │  │ • MaxNewTokensCriteria: 达到最大新 token 数停止                                       │ │ │
│  │  │ • MaxTimeCriteria: 达到最大时间停止                                                    │ │ │
│  │  │ • StoppingCriteriaList: 组合多个条件                                                   │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────┬────────────────────────────────────────────────────┘ │
│                                           ↓                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  Cache - KV 缓存系统                                                                         │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ • Cache 基类                                                                         │ │ │
│  │  │ • DynamicCache: 动态缓存 (默认)                                                       │ │ │
│  │  │ • SinkCache: 滑动窗口缓存 (保留最近 token)                                            │ │ │
│  │  │ • StaticCache: 静态缓存 (编译优化)                                                    │ │ │
│  │  │ • OffloadedCache: Offload 到磁盘                                                      │ │ │
│  │  │ • EncoderDecoderCache: 编码器-解码器缓存                                              │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  解码策略 (Decoding Strategies)                                                             │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ 1. Greedy Search: 每步选概率最大的 token                                               │ │ │
│  │  │ 2. Sampling: 从概率分布中采样 (带 temperature, top_k, top_p)                          │ │ │
│  │  │ 3. Beam Search: 保留 top-k 个候选序列 (Beams)                                        │ │ │
│  │  │ 4. Beam Sample: Beam Search + Sampling                                                │ │ │
│  │  │ 5. Group Beam Search: 分组 Beam Search (提高多样性)                                  │ │ │
│  │  │ 6. Contrastive Search: 对比搜索 (提高一致性和多样性)                                  │ │ │
│  │  │ 7. Assisted Decoding: 投机解码 (小模型草稿，大模型验证)                               │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 1. 文本生成系统概述

Transformers 的文本生成系统是一个高度可配置的框架，支持多种解码策略和优化技术。核心是 `GenerationMixin` 类，它为模型提供了 `generate()` 方法。

### 1.1 核心设计理念

1. **统一接口**：所有生成模型共享相同的 `generate()` 接口
2. **策略可插拔**：多种解码策略可以灵活选择
3. **丰富的配置**：30+ 个配置参数控制生成行为
4. **性能优化**：KV 缓存、编译优化、投机解码等
5. **类型安全**：`GenerationConfig` 提供强类型配置

### 1.2 支持的模型类型

- **自回归模型 (Causal LM)**: GPT2, LLaMA, Mistral, Qwen, etc.
- **序列到序列模型 (Seq2Seq)**: T5, BART, MarianMT, etc.
- **编码器-解码器模型**: 机器翻译、摘要等任务

## 2. GenerationConfig - 生成配置详解

**位置**: [generation/configuration_utils.py](../src/transformers/generation/configuration_utils.py)

### 2.1 GenerationConfig 类定义

```python
@dataclass
class GenerationConfig(PushToHubMixin):
    """
    生成配置类，控制所有生成行为
    
    参数分为几个类别：
    - 控制长度
    - 控制生成策略
    - 控制随机性
    - 控制输出
    - 其他优化
    """
    
    # ==================== 长度控制 ====================
    max_length: Optional[int] = field(default=20, metadata={"help": "最大长度 (包括输入)"})
    max_new_tokens: Optional[int] = field(default=None, metadata={"help": "最大新生成 token 数 (推荐)"})
    min_length: Optional[int] = field(default=0, metadata={"help": "最小长度"})
    min_new_tokens: Optional[int] = field(default=None, metadata={"help": "最小新生成 token 数"})
    early_stopping: Optional[Union[bool, str]] = field(default=False, metadata={"help": "Beam Search 提前停止"})
    max_time: Optional[float] = field(default=None, metadata={"help": "最大生成时间 (秒)"})
    
    # ==================== 生成策略 ====================
    do_sample: Optional[bool] = field(default=False, metadata={"help": "是否采样 (否则贪婪)"})
    num_beams: Optional[int] = field(default=1, metadata={"help": "Beam Search 束数"})
    num_beam_groups: Optional[int] = field(default=1, metadata={"help": "Group Beam Search 组数"})
    penalty_alpha: Optional[float] = field(default=None, metadata={"help": "对比搜索参数"})
    use_cache: Optional[bool] = field(default=True, metadata={"help": "使用 KV 缓存"})
    
    # ==================== 随机性控制 (采样用) ====================
    temperature: Optional[float] = field(default=1.0, metadata={"help": "温度参数"})
    top_k: Optional[int] = field(default=50, metadata={"help": "Top-K 采样"})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top-P (Nucleus) 采样"})
    typical_p: Optional[float] = field(default=1.0, metadata={"help": "典型采样"})
    epsilon_cutoff: Optional[float] = field(default=0.0, metadata={"help": "Epsilon 采样"})
    eta_cutoff: Optional[float] = field(default=0.0, metadata={"help": "Eta 采样"})
    
    # ==================== 重复控制 ====================
    repetition_penalty: Optional[float] = field(default=1.0, metadata={"help": "重复惩罚"})
    encoder_repetition_penalty: Optional[float] = field(default=1.0, metadata={"help": "编码器重复惩罚"})
    no_repeat_ngram_size: Optional[int] = field(default=0, metadata={"help": "禁止重复 n-gram"})
    bad_words_ids: Optional[List[List[int]]] = field(default=None, metadata={"help": "禁止的 token 序列"})
    force_words_ids: Optional[Union[List[List[int]], List[List[List[int]]]]] = field(default=None, metadata={"help": "强制出现的 token"})
    
    # ==================== 多样化输出 ====================
    num_return_sequences: Optional[int] = field(default=1, metadata={"help": "返回序列数"})
    diversity_penalty: Optional[float] = field(default=0.0, metadata={"help": "多样性惩罚 (Group Beam Search)"})
    
    # ==================== 特殊 token ====================
    pad_token_id: Optional[int] = field(default=None, metadata={"help": "Padding token ID"})
    bos_token_id: Optional[int] = field(default=None, metadata={"help": "Begin-of-sequence token ID"})
    eos_token_id: Optional[Union[int, List[int]]] = field(default=None, metadata={"help": "End-of-sequence token ID"})
    
    # ==================== 输出控制 ====================
    output_attentions: Optional[bool] = field(default=False, metadata={"help": "返回 attention weights"})
    output_hidden_states: Optional[bool] = field(default=False, metadata={"help": "返回 hidden states"})
    output_scores: Optional[bool] = field(default=False, metadata={"help": "返回 logits scores"})
    return_dict_in_generate: Optional[bool] = field(default=False, metadata={"help": "返回字典格式"})
    
    # ==================== 序列到序列特定 ====================
    encoder_no_repeat_ngram_size: Optional[int] = field(default=0, metadata={"help": "编码器端禁止 n-gram"})
    decoder_start_token_id: Optional[int] = field(default=None, metadata={"help": "解码器起始 token"})
    
    # ==================== 辅助解码 ====================
    assistant_model: Optional["PreTrainedModel"] = field(default=None, metadata={"help": "辅助模型"})
    
    # ==================== 其他 ====================
    renormalize_logits: Optional[bool] = field(default=False, metadata={"help": "重归一化 logits"})
    remove_invalid_values: Optional[bool] = field(default=False, metadata={"help": "移除无效值 (nan/inf)"})
    exponential_decay_length_penalty: Optional[Tuple[int, float]] = field(default=None, metadata={"help": "长度衰减惩罚"})
    suppress_tokens: Optional[List[int]] = field(default=None, metadata={"help": "抑制的 token"})
    begin_suppress_tokens: Optional[List[int]] = field(default=None, metadata={"help": "开始时抑制的 token"})
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        从预训练模型加载配置
        """
        # 省略代码...
    
    def save_pretrained(self, save_directory, **kwargs):
        """
        保存配置到文件
        """
        # 省略代码...
    
    def update(self, **kwargs):
        """
        更新配置
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def validate(self, is_encoder_decoder: bool = False):
        """
        验证配置是否合法
        """
        # 省略代码...
```

### 2.2 使用示例

```python
from transformers import GenerationConfig

# 创建配置
generation_config = GenerationConfig(
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
    eos_token_id=2,
    pad_token_id=0,
)

# 从模型加载
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
generation_config = GenerationConfig.from_model_config(model.config)

# 保存和加载
generation_config.save_pretrained("./my_config")
loaded_config = GenerationConfig.from_pretrained("./my_config")
```

## 3. GenerationMixin - 生成基类详解

**位置**: [generation/utils.py](../src/transformers/generation/utils.py)

### 3.1 GenerationMixin 类定义

```python
class GenerationMixin:
    """
    生成功能 Mixin，为模型提供 generate() 方法
    
    核心是 generate() 方法，支持多种解码策略
    """
    
    @torch.no_grad()
    @add_start_docstrings(
        GENERATE_ARGS_DOCSTRING,
        GENERATE_RETURN_DOCSTRING,
    )
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ):
        """
        生成主入口方法
        
        工作流程:
        1. 准备配置和参数
        2. 准备输入和特殊 token
        3. 准备 logits 处理器和停止条件
        4. 根据配置选择解码策略
        5. 执行解码
        6. 返回结果
        
        参数:
            inputs: 输入 tokens (input_ids)
            generation_config: 生成配置
            logits_processor: 自定义 logits 处理器
            stopping_criteria: 自定义停止条件
            streamer: 流式输出
            **kwargs: 其他参数 (覆盖 generation_config)
        """
        # 1. 处理配置
        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        
        # 2. 验证模型类
        self._validate_model_class(generation_config, synced_gpus, **model_kwargs)
        
        # 3. 准备特殊 tokens
        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, **model_kwargs
        )
        
        # 4. 准备 logits 处理器和停止条件
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids.shape[-1],
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )
        
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
        )
        
        # 5. 准备 logits warper (采样相关)
        logits_warper = self._get_logits_warper(generation_config)
        
        # 6. 准备 KV 缓存
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        
        # 7. 根据配置选择解码策略
        if generation_config.num_beams == 1:
            if generation_config.do_sample:
                # 采样解码
                result = self._sample(
                    input_ids,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                    logits_warper=logits_warper,
                    **model_kwargs,
                )
            else:
                # 贪婪解码
                result = self._greedy_search(
                    input_ids,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                    **model_kwargs,
                )
        else:
            if generation_config.num_beam_groups &gt; 1:
                # Group Beam Search
                result = self._group_beam_search(
                    input_ids,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                    **model_kwargs,
                )
            elif generation_config.do_sample:
                # Beam Sample
                result = self._beam_sample(
                    input_ids,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                    **model_kwargs,
                )
            else:
                # Beam Search
                result = self._beam_search(
                    input_ids,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                    **model_kwargs,
                )
        
        # 8. 后处理返回结果
        if generation_config.return_dict_in_generate:
            return result
        else:
            return result.sequences if hasattr(result, "sequences") else result
```

### 3.2 贪婪解码 (_greedy_search)

```python
def _greedy_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    **model_kwargs,
):
    """
    贪婪解码：每步选择概率最大的 token
    
    特点:
    - 确定性
    - 速度快
    - 可能陷入局部最优
    """
    
    # 初始化
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    
    # 准备 KV 缓存
    past_key_values = model_kwargs.get("past_key_values")
    use_cache = model_kwargs.get("use_cache", True)
    
    # 解码循环
    while True:
        # 前向传播
        model_inputs = self.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            **model_kwargs,
        )
        outputs = self(
            **model_inputs,
            return_dict=True,
        )
        
        # 获取下一个 token 的 logits
        next_token_logits = outputs.logits[:, -1, :]
        
        # 处理 logits
        next_token_logits = logits_processor(input_ids, next_token_logits)
        
        # 贪婪选择：argmax
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        
        # 追加到输入
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        
        # 更新 KV 缓存
        if use_cache:
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
            )
        
        # 检查停止条件
        if stopping_criteria(input_ids, None):
            break
    
    return GenerateOutput(sequences=input_ids)
```

### 3.3 采样解码 (_sample)

```python
def _sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    **model_kwargs,
):
    """
    采样解码：从概率分布中采样
    
    特点:
    - 随机性
    - 多样性高
    - 可能不连贯
    """
    
    # 初始化
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    
    # 解码循环
    while True:
        # 前向传播
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = self(**model_inputs, return_dict=True)
        
        # 获取 logits
        next_token_logits = outputs.logits[:, -1, :]
        
        # 处理 logits
        next_token_logits = logits_processor(input_ids, next_token_logits)
        
        # 应用 warper (temperature, top_k, top_p)
        next_token_scores = logits_warper(input_ids, next_token_logits)
        
        # 采样
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        # 追加
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        
        # 更新缓存
        model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
        
        # 检查停止
        if stopping_criteria(input_ids, None):
            break
    
    return GenerateOutput(sequences=input_ids)
```

### 3.4 Beam Search 解码 (_beam_search)

```python
def _beam_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    **model_kwargs,
):
    """
    Beam Search 解码：同时保留 top-k 个候选序列 (beams)
    
    特点:
    - 比贪婪解码更好
    - 搜索空间更大
    - 计算成本更高
    """
    
    # 初始化 beams
    batch_size = input_ids.shape[0]
    num_beams = model_kwargs.get("num_beams", 1)
    
    # 扩展输入到 num_beams 份
    input_ids = input_ids.repeat(num_beams, 1)
    
    beam_scores = torch.zeros(batch_size, num_beams, device=input_ids.device)
    beam_scores[:, 1:] = -1e9  # 初始化，只有第一个 beam 有效
    beam_scores = beam_scores.view(-1)  # flatten
    
    # 解码循环
    while True:
        # 前向传播
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = self(**model_inputs, return_dict=True)
        
        # 获取 logits
        next_token_logits = outputs.logits[:, -1, :]
        
        # 处理 logits
        next_token_logits = logits_processor(input_ids, next_token_logits)
        
        # 计算得分
        next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)
        
        # 累加 beam 得分
        next_token_scores = next_token_scores + beam_scores[:, None]
        
        # 选择 top 2*num_beams 个候选
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
        next_token_scores, next_tokens = torch.topk(
            next_token_scores,
            2 * num_beams,
            dim=1,
            largest=True,
            sorted=True,
        )
        
        # 计算 beam 索引和 token 索引
        next_indices = next_tokens // vocab_size
        next_tokens = next_tokens % vocab_size
        
        # 更新 beams
        beam_scores = next_token_scores
        beam_indices = next_indices
        
        # 更新输入
        input_ids = torch.cat([
            input_ids.view(batch_size, num_beams, -1)[range(batch_size), beam_indices],
            next_tokens[:, :, None],
        ], dim=-1)
        input_ids = input_ids.view(batch_size * num_beams, -1)
        
        # 检查停止
        if stopping_criteria(input_ids, None):
            break
    
    # 返回最优 beam
    final_beam_scores = beam_scores.view(batch_size, num_beams)
    best_beam_indices = final_beam_scores.argmax(dim=1)
    final_sequences = input_ids.view(batch_size, num_beams, -1)
    best_sequences = final_sequences[range(batch_size), best_beam_indices]
    
    return GenerateOutput(sequences=best_sequences)
```

### 3.5 对比搜索 (_contrastive_search)

```python
def _contrastive_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    **model_kwargs,
):
    """
    对比搜索：平衡模型置信度和 token 多样性
    
    核心思想:
    - 选 token 时同时考虑:
        1. 模型对该 token 的置信度 (越大越好)
        2. 该 token 与之前生成 token 的相似度 (越小越好)
    
    参数:
        penalty_alpha: 控制两个目标的平衡
    """
    
    # 省略复杂实现...
    # 关键步骤:
    # 1. 计算 top-k 候选
    # 2. 计算每个候选与历史的相似度
    # 3. 结合置信度和相似度得分
    # 4. 选择最优 token
    
    return GenerateOutput(sequences=input_ids)
```

## 4. LogitsProcessor - Logits 处理器系统

**位置**: [generation/logits_process.py](../src/transformers/generation/logits_process.py)

### 4.1 LogitsProcessor 基类

```python
class LogitsProcessor:
    """
    Logits 处理器基类
    
    在每步生成时处理 logits，修改概率分布
    """
    
    @abstractmethod
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """
        处理 logits
        
        参数:
            input_ids: 当前已生成的 token 序列
            scores: 当前步的 logits
        
        返回:
            处理后的 logits
        """
        raise NotImplementedError
```

### 4.2 常用 LogitsProcessor 实现

#### 4.2.1 TemperatureLogitsWarper

```python
class TemperatureLogitsWarper(LogitsWarper):
    """
    温度缩放 logits
    
    原理:
        logits = logits / temperature
    
    效果:
        - temperature &lt; 1: 更确定性
        - temperature &gt; 1: 更随机
    """
    
    def __init__(self, temperature: float):
        temperature = float(temperature)
        if not 0.0 &lt; temperature:
            raise ValueError(f"temperature has to be &gt; 0, but is {temperature}")
        self.temperature = temperature
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        scores = scores / self.temperature
        return scores
```

#### 4.2.2 TopKLogitsWarper

```python
class TopKLogitsWarper(LogitsWarper):
    """
    Top-K 采样：只保留概率最大的 k 个 token
    
    原理:
        - 找出 top-k logits
        - 其他设为 -inf
    """
    
    def __init__(self, top_k: int, filter_value: float = -float("inf"), min_tokens_to_keep: int = 1):
        top_k = int(top_k)
        if not (top_k &gt;= 0 or top_k == -1):
            raise ValueError(f"`top_k` has to be &gt;= 0, but is {top_k}")
        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.size(-1))
        
        # 找出不在 top-k 的索引
        indices_to_remove = scores &lt; torch.topk(scores, top_k)[0][..., -1, None]
        
        # 设为 -inf
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores
```

#### 4.2.3 TopPLogitsWarper

```python
class TopPLogitsWarper(LogitsWarper):
    """
    Top-P (Nucleus) 采样：保留累计概率达到 p 的最小 token 集合
    
    特点:
        - 自适应 k
        - 概率低时 k 小
        - 概率均匀时 k 大
    """
    
    def __init__(self, top_p: float, filter_value: float = -float("inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if not 0.0 &lt; top_p &lt;= 1.0:
            raise ValueError(f"`top_p` has to be between 0 and 1, but is {top_p}")
        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # 排序
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = torch.cumsum(nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 移除超过 top_p 的 token
        sorted_indices_to_remove = cumulative_probs &gt; self.top_p
        
        # 至少保留 min_tokens_to_keep
        if self.min_tokens_to_keep &gt; 1:
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
        
        # 保持一个 token (第一个)
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # 原始顺序中移除
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores
```

#### 4.2.4 RepetitionPenaltyLogitsProcessor

```python
class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """
    重复惩罚：抑制已出现的 token
    
    原理:
        如果 token 已出现:
            score[token] = score[token] / penalty
    """
    
    def __init__(self, penalty: float):
        if not 0.0 &lt;= penalty:
            raise ValueError(f"penalty has to be &gt;= 0, but is {penalty}")
        self.penalty = penalty
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score &lt; 0, score * self.penalty, score / self.penalty)
        scores.scatter_(1, input_ids, score)
        return scores
```

#### 4.2.5 NoRepeatNGramLogitsProcessor

```python
class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """
    禁止重复 n-gram：防止出现重复短语
    
    原理:
        - 记录所有已生成的 n-gram
        - 对于当前状态，检查哪些 token 会导致重复 n-gram
        - 将这些 token 的 score 设为 -inf
    """
    
    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size &lt;= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        
        # 找出禁止的 token
        banned_tokens = self._get_banned_ngram_tokens(input_ids, num_batch_hypotheses, cur_len)
        
        # 设为 -inf
        for batch_idx in range(num_batch_hypotheses):
            scores[batch_idx, banned_tokens[batch_idx]] = -float("inf")
        
        return scores
    
    def _get_banned_ngram_tokens(self, prev_input_ids, num_hypos, cur_len):
        """
        找出导致重复 n-gram 的 token
        """
        banned_tokens = [[] for _ in range(num_hypos)]
        if cur_len + 1 &lt; self.ngram_size:
            return banned_tokens
        
        # 检查所有 n-gram
        for hypo_idx in range(num_hypos):
            # 生成 ngram (size n-1)
            ngram_idx = tuple(
                prev_input_ids[hypo_idx][cur_len - self.ngram_size + 1 : cur_len].tolist()
            )
            
            # 找出以这个 ngram 开头的所有 token
            for start_idx in range(cur_len - self.ngram_size + 1):
                prev_ngram = tuple(
                    prev_input_ids[hypo_idx][start_idx : start_idx + self.ngram_size - 1].tolist()
                )
                if prev_ngram == ngram_idx:
                    banned_tokens[hypo_idx].append(
                        prev_input_ids[hypo_idx][start_idx + self.ngram_size - 1].item()
                    )
        
        return banned_tokens
```

### 4.3 LogitsProcessorList

```python
class LogitsProcessorList(list):
    """
    LogitsProcessor 列表，按顺序应用所有处理器
    """
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for processor in self:
            scores = processor(input_ids, scores)
        return scores
```

## 5. StoppingCriteria - 停止条件系统

**位置**: [generation/stopping_criteria.py](../src/transformers/generation/stopping_criteria.py)

### 5.1 停止条件基类

```python
class StoppingCriteria:
    """
    停止条件基类
    """
    
    @abstractmethod
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """
        检查是否应该停止生成
        
        返回:
            True 表示停止
        """
        raise NotImplementedError
```

### 5.2 常用停止条件

```python
class MaxLengthCriteria(StoppingCriteria):
    """
    达到最大长度停止
    """
    
    def __init__(self, max_length: int):
        self.max_length = max_length
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        return input_ids.shape[-1] &gt;= self.max_length


class MaxNewTokensCriteria(StoppingCriteria):
    """
    达到最大新生成 token 数停止
    """
    
    def __init__(self, start_length: int, max_new_tokens: int):
        self.start_length = start_length
        self.max_new_tokens = max_new_tokens
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        return input_ids.shape[-1] &gt;= self.start_length + self.max_new_tokens


class MaxTimeCriteria(StoppingCriteria):
    """
    达到最大时间停止
    """
    
    def __init__(self, max_time: float):
        self.max_time = max_time
        self.start_time = time.time()
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        return (time.time() - self.start_time) &gt;= self.max_time


class StoppingCriteriaList(list):
    """
    停止条件列表，只要有一个满足就停止
    """
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        return any(criteria(input_ids, scores) for criteria in self)
```

## 6. Cache - KV 缓存系统

**位置**: [cache_utils.py](../src/transformers/cache_utils.py)

### 6.1 Cache 基类

```python
class Cache(ABC):
    """
    KV 缓存基类
    
    存储注意力的 K 和 V，避免重复计算
    """
    
    @abstractmethod
    def __getitem__(self, layer_idx: int):
        """
        获取某一层的缓存 (key, value)
        """
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self):
        """
        层数
        """
        raise NotImplementedError
```

### 6.2 DynamicCache - 动态缓存

```python
class DynamicCache(Cache):
    """
    动态缓存：每步追加新 token 的 KV
    
    最常用的缓存方式
    """
    
    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def __getitem__(self, layer_idx: int):
        if layer_idx &lt; len(self.key_cache):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise IndexError(f"Layer index {layer_idx} out of range")
    
    def __len__(self):
        return len(self.key_cache)
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        更新缓存
        """
        if layer_idx == 0:
            # 记录形状
            pass
        
        if layer_idx &lt; len(self.key_cache):
            # 追加
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        else:
            # 新层
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def get_seq_length(self, layer_idx: int = 0):
        if len(self.key_cache) &lt;= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]
```

### 6.3 SinkCache - 滑动窗口缓存

```python
class SinkCache(DynamicCache):
    """
    滑动窗口缓存：保留前面几个 token (Sink) + 最近 token
    
    处理超长序列，限制内存使用
    """
    
    def __init__(self, window_length: int, num_sink_tokens: int = 4):
        super().__init__()
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        更新缓存，超过窗口长度时移除中间的 token
        """
        if layer_idx &lt; len(self.key_cache):
            # 追加
            keys = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            values = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            
            # 检查长度
            if keys.shape[-2] &gt; self.window_length:
                # 保留 sink tokens + 最近的
                self.key_cache[layer_idx] = torch.cat(
                    [keys[..., : self.num_sink_tokens, :], keys[..., -self.window_length + self.num_sink_tokens :, :]],
                    dim=-2,
                )
                self.value_cache[layer_idx] = torch.cat(
                    [values[..., : self.num_sink_tokens, :], values[..., -self.window_length + self.num_sink_tokens :, :]],
                    dim=-2,
                )
            else:
                self.key_cache[layer_idx] = keys
                self.value_cache[layer_idx] = values
        else:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
```

## 7. Streamer - 流式输出

**位置**: [generation/streamers.py](../src/transformers/generation/streamers.py)

```python
class BaseStreamer:
    """
    流式输出基类
    
    用于实时显示生成的 token
    """
    
    def put(self, value):
        """
        接收新 token
        """
        raise NotImplementedError
    
    def end(self):
        """
        结束流
        """
        raise NotImplementedError


class TextStreamer(BaseStreamer):
    """
    文本流输出器
    
    实时打印生成的文本
    """
    
    def __init__(self, tokenizer: "PreTrainedTokenizerBase", skip_prompt: bool = False, **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs
        self.token_cache = []
        self.print_len = 0
    
    def put(self, value):
        """
        接收新 token 并打印
        """
        if len(value.shape) == 2:
            value = value[0]
        
        self.token_cache.extend(value.tolist())
        
        # 解码
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        
        if self.skip_prompt and len(self.token_cache) == 1:
            self.print_len = len(text)
            return
        
        # 打印新部分
        printable_text = text[self.print_len :]
        print(printable_text, end="", flush=True)
        self.print_len = len(text)
    
    def end(self):
        """
        结束，打印剩余部分
        """
        if len(self.token_cache) &gt; 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            printable_text = text[self.print_len :]
            print(printable_text, flush=True)
```

## 8. 使用示例

### 8.1 基础生成

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = tokenizer("In a world where AI can", return_tensors="pt")

# 贪婪解码
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 8.2 采样生成

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
)
```

### 8.3 Beam Search

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    num_beams=5,
    early_stopping=True,
    no_repeat_ngram_size=2,
    num_return_sequences=3,
)
```

### 8.4 使用 GenerationConfig

```python
from transformers import GenerationConfig

gen_config = GenerationConfig(
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.pad_token_id,
)

outputs = model.generate(**inputs, generation_config=gen_config)
```

### 8.5 流式输出

```python
from transformers import TextStreamer

streamer = TextStreamer(tokenizer, skip_prompt=True)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    streamer=streamer,
)
```

### 8.6 对比搜索

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    penalty_alpha=0.6,
    top_k=4,
)
```

### 8.7 自定义 LogitsProcessor

```python
from transformers import LogitsProcessor, LogitsProcessorList

class MyLogitsProcessor(LogitsProcessor):
    def __init__(self, banned_token_ids):
        self.banned_token_ids = banned_token_ids
    
    def __call__(self, input_ids, scores):
        scores[:, self.banned_token_ids] = -float("inf")
        return scores

logits_processor = LogitsProcessorList([
    MyLogitsProcessor(banned_token_ids=[100, 200]),
])

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    logits_processor=logits_processor,
)
```

### 8.8 自定义 StoppingCriteria

```python
from transformers import StoppingCriteria, StoppingCriteriaList

class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id
    
    def __call__(self, input_ids, scores):
        return (input_ids[0, -1] == self.stop_token_id).any()

stopping_criteria = StoppingCriteriaList([
    MyStoppingCriteria(stop_token_id=198),  # 换行
])

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    stopping_criteria=stopping_criteria,
)
```

## 9. 关键技术要点

### 9.1 KV 缓存的工作原理

```
第1步:
Input: [A, B, C]
Compute KV for A, B, C
Generate next token D

第2步:
Input: [A, B, C, D]
Reuse KV for A, B, C
Only compute KV for D
Generate next token E
```

### 9.2 解码策略选择指南

- **贪婪解码**: 速度最快，适合确定性任务
- **采样解码**: 多样性高，适合创意写作
- **Beam Search**: 质量较高，适合翻译、摘要
- **对比搜索**: 平衡质量和多样性，最新技术
- **投机解码**: 速度快，适合大模型

### 9.3 常用参数组合

```python
# 创意写作
creative_config = GenerationConfig(
    max_new_tokens=200,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1,
)

# 事实回答
fact_config = GenerationConfig(
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    top_k=10,
    repetition_penalty=1.0,
)

# 翻译 (Beam Search)
translation_config = GenerationConfig(
    max_new_tokens=200,
    do_sample=False,
    num_beams=5,
    early_stopping=True,
)
```

## 代码参考

- [generation/utils.py](../src/transformers/generation/utils.py)
- [generation/configuration_utils.py](../src/transformers/generation/configuration_utils.py)
- [generation/logits_process.py](../src/transformers/generation/logits_process.py)
- [generation/stopping_criteria.py](../src/transformers/generation/stopping_criteria.py)
- [cache_utils.py](../src/transformers/cache_utils.py)
- [generation/streamers.py](../src/transformers/generation/streamers.py)

