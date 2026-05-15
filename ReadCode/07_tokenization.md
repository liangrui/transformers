
# Transformers Tokenization 系统分析

## 概述图

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                            Tokenization 系统架构                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  基类层级 (Base Classes)                                                                     │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ PreTrainedTokenizerBase: 所有分词器基类                                                  │ │ │
│  │  │  • __call__(): 主入口                                                                 │ │ │
│  │  │  • encode(): 编码                                                                     │ │ │
│  │  │  • decode(): 解码                                                                     │ │ │
│  │  │  • tokenize(): 分词                                                                   │ │ │
│  │  │  • convert_tokens_to_ids(): Token → ID                                                │ │ │
│  │  │  • convert_ids_to_tokens(): ID → Token                                                │ │ │
│  │  │  • save_pretrained(): 保存                                                             │ │ │
│  │  │  • from_pretrained(): 加载                                                             │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ PreTrainedTokenizer: 慢速分词器基类 (Python 实现)                                      │ │ │
│  │  │ PreTrainedTokenizerFast: 快速分词器基类 (Rust 实现, 基于 tokenizers 库)               │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────┬────────────────────────────────────────────────────┘ │
│                                           ↓                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  核心算法 (Core Algorithms)                                                                 │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ 1. Byte-Pair Encoding (BPE): 最常用                                                    │ │ │
│  │  │  2. WordPiece: BERT 系列                                                              │ │ │
│  │  │  3. Unigram: T5 系列                                                                  │ │ │
│  │  │  4. SentencePiece: 基于子词                                                          │ │ │
│  │  │  5. Byte-Level BPE: GPT2 系列                                                         │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────┬────────────────────────────────────────────────────┘ │
│                                           ↓                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  预处理与后处理 (Pre/Post Processing)                                                       │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ • PreTokenizer: 预分词 (按空格、标点等分割)                                            │ │ │
│  │  │ • Normalizer: 归一化 (小写、NFC、去掉重音等)                                          │ │ │
│  │  │ • TokenizerModel: 分词模型 (BPE/WordPiece/etc)                                        │ │ │
│  │  │ • PostProcessor: 后处理 (添加特殊 token、截断等)                                      │ │ │
│  │  │ • Decoder: 解码 (合并子词)                                                           │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────┬────────────────────────────────────────────────────┘ │
│                                           ↓                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  特殊 Token (Special Tokens)                                                                │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ • bos_token: Beginning-of-sequence (&lt;s&gt;)                                              │ │ │
│  │  │ • eos_token: End-of-sequence (&lt;/s&gt;)                                                   │ │ │
│  │  │ • pad_token: Padding token (&lt;pad&gt;)                                                   │ │ │
│  │  │ • unk_token: Unknown token (&lt;unk&gt;)                                                    │ │ │
│  │  │ • sep_token: Separator token (&lt;sep&gt;)                                                 │ │ │
│  │  │ • cls_token: Classification token<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>&gt;                                                │ │ │
│  │  │ • mask_token: Mask token [MASK]                                                      │ │ │
│  │  │ • additional_special_tokens: 额外特殊 token                                          │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────┬────────────────────────────────────────────────────┘ │
│                                           ↓                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  批处理与填充 (Batch Processing &amp; Padding)                                                  │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ • PaddingStrategy: 填充策略 (longest, max_length, do_not_pad)                       │ │ │
│  │  │ • pad_to_multiple_of: 填充到倍数 (2, 8, 128 等)                                       │ │ │
│  │  │ • truncation: 截断策略 (only_first, only_second, longest_first)                      │ │ │
│  │  │ • return_overflowing_tokens: 返回溢出 token                                          │ │ │
│  │  │ • return_offsets_mapping: 返回字符偏移量 (用于问答)                                   │ │ │
│  │  │ • return_token_type_ids: 返回 token type ids (BERT)                                  │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  常用分词器实现 (Popular Tokenizers)                                                         │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ • BertTokenizer / BertTokenizerFast: WordPiece                                      │ │ │
│  │  │ • GPT2Tokenizer / GPT2TokenizerFast: Byte-Level BPE                                  │ │ │
│  │  │ • T5Tokenizer / T5TokenizerFast: Unigram                                             │ │ │
│  │  │ • LlamaTokenizer / LlamaTokenizerFast: SentencePiece                                 │ │ │
│  │  │ • CodeGenTokenizer: 代码专用                                                          │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 1. Tokenization 系统概述

Tokenization 是将文本转换为模型可处理的 token 序列的过程，是 NLP 流水线的第一步。

### 1.1 核心功能

- **分词 (Tokenization)**: 将文本切分为子词/词/字符
- **编码 (Encoding)**: 将 token 转换为 ID
- **解码 (Decoding)**: 将 ID 转换回文本
- **填充与截断 (Padding &amp; Truncation)**: 统一序列长度
- **特殊 Token 处理**: 添加 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>&gt;, &lt;sep&gt;, [MASK] 等

### 1.2 设计理念

1. **统一接口**: 所有分词器共享相同 API
2. **快慢双实现**: 提供 Python (慢速) 和 Rust (快速) 两种实现
3. **可扩展性**: 支持自定义分词器
4. **向后兼容**: 保持与旧版本的兼容性

## 2. 基类详解

### 2.1 PreTrainedTokenizerBase - 基础基类

**位置**: [tokenization_utils_base.py](../src/transformers/tokenization_utils_base.py)

```python
class PreTrainedTokenizerBase:
    """
    所有分词器的基类，定义通用接口
    """
    
    # 特殊 token 属性
    pad_token: Optional[str] = None
    eos_token: Optional[str] = None
    bos_token: Optional[str] = None
    unk_token: Optional[str] = None
    sep_token: Optional[str] = None
    cls_token: Optional[str] = None
    mask_token: Optional[str] = None
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    unk_token_id: Optional[int] = None
    sep_token_id: Optional[int] = None
    cls_token_id: Optional[int] = None
    mask_token_id: Optional[int] = None
    
    # 其他属性
    model_max_length: Optional[int] = None
    padding_side: str = "right"
    truncation_side: str = "right"
    
    def __call__(
        self,
        text: Union[str, List[str], List[List[str]]],
        text_pair: Optional[Union[str, List[str], List[List[str]]]] = None,
        text_target: Union[str, List[str], List[List[str]]] = None,
        text_pair_target: Optional[Union[str, List[str], List[List[str]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -&gt; BatchEncoding:
        """
        主调用方法，处理编码
        
        参数:
            text: 输入文本
            text_pair: 第二输入文本 (用于 BERT 句子对)
            padding: 填充策略
            truncation: 截断策略
            max_length: 最大长度
            return_tensors: 返回 tensor 格式 (pt, tf, np)
            return_token_type_ids: 是否返回 token_type_ids
            return_attention_mask: 是否返回 attention_mask
            return_offsets_mapping: 是否返回字符偏移量 (问答任务用)
        """
        # 省略实现...
        return self._call_one(text=text, ...)
    
    def _call_one(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        **kwargs,
    ) -&gt; BatchEncoding:
        """
        处理单个文本
        """
        # 省略实现...
    
    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -&gt; List[int]:
        """
        编码文本为 ID 列表
        """
        return self.encode_plus(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            return_tensors=return_tensors,
            return_token_type_ids=False,
            return_attention_mask=False,
            **kwargs,
        )["input_ids"]
    
    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -&gt; BatchEncoding:
        """
        编码文本，返回字典格式
        """
        # 省略实现...
    
    def decode(
        self,
        token_ids: Union[int, List[int], np.ndarray, "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs,
    ) -&gt; str:
        """
        将 ID 列表解码为文本
        """
        # 省略实现...
    
    def tokenize(
        self,
        text: str,
        pair: Optional[str] = None,
        add_special_tokens: bool = False,
        **kwargs,
    ) -&gt; List[str]:
        """
        分词，返回 token 列表
        """
        raise NotImplementedError
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -&gt; Union[int, List[int]]:
        """
        将 token 转换为 ID
        """
        # 省略实现...
    
    def convert_ids_to_tokens(
        self,
        ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
    ) -&gt; Union[str, List[str]]:
        """
        将 ID 转换为 token
        """
        # 省略实现...
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *init_inputs,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ) -&gt; "PreTrainedTokenizerBase":
        """
        从预训练加载分词器
        """
        # 省略实现...
    
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        保存分词器
        """
        # 省略实现...
    
    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -&gt; List[int]:
        """
        构建带特殊 token 的输入序列
        
        例如 BERT: [CLS] + token_ids_0 + [SEP] + token_ids_1 + [SEP]
        """
        raise NotImplementedError
    
    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -&gt; List[int]:
        """
        获取特殊 token 的 mask (1 表示是特殊 token)
        """
        raise NotImplementedError
    
    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -&gt; List[int]:
        """
        创建 token_type_ids (用于 BERT 区分两个句子)
        """
        raise NotImplementedError
```

### 2.2 PreTrainedTokenizer - 慢速分词器基类

**位置**: [tokenization_utils.py](../src/transformers/tokenization_utils.py)

```python
class PreTrainedTokenizer(PreTrainedTokenizerBase):
    """
    慢速分词器基类（纯 Python 实现）
    """
    
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab: Dict[str, int] = {}
        self.ids_to_tokens: Dict[int, str] = {}
    
    def tokenize(
        self,
        text: str,
        **kwargs,
    ) -&gt; List[str]:
        """
        分词方法，需要子类实现
        """
        raise NotImplementedError
    
    def _tokenize(self, text: str) -&gt; List[str]:
        """
        实际分词逻辑
        """
        raise NotImplementedError
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -&gt; Union[int, List[int]]:
        """
        Token → ID
        """
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.unk_token_id)
        else:
            return [self.vocab.get(token, self.unk_token_id) for token in tokens]
    
    def convert_ids_to_tokens(
        self,
        ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
    ) -&gt; Union[str, List[str]]:
        """
        ID → Token
        """
        if isinstance(ids, int):
            return self.ids_to_tokens.get(ids, self.unk_token)
        else:
            tokens = [self.ids_to_tokens.get(id, self.unk_token) for id in ids]
            if skip_special_tokens:
                tokens = [token for token in tokens if token not in self.all_special_tokens]
            return tokens
    
    def convert_tokens_to_string(self, tokens: List[str]) -&gt; str:
        """
        合并 token 为字符串
        """
        raise NotImplementedError
    
    def _convert_token_to_id(self, token: str) -&gt; int:
        """
        单个 token → ID
        """
        return self.vocab.get(token, self.unk_token_id)
    
    def _convert_id_to_token(self, index: int) -&gt; str:
        """
        单个 ID → token
        """
        return self.ids_to_tokens.get(index, self.unk_token)
```

### 2.3 PreTrainedTokenizerFast - 快速分词器基类

**位置**: [tokenization_utils_fast.py](../src/transformers/tokenization_utils_fast.py)

```python
class PreTrainedTokenizerFast(PreTrainedTokenizerBase):
    """
    快速分词器基类（基于 Hugging Face 的 tokenizers 库，Rust 实现）
    
    比慢速分词器快 10-100 倍！
    """
    
    def __init__(
        self,
        tokenizer_object: Optional["Tokenizer"] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._tokenizer = tokenizer_object
    
    @property
    def tokenizer(self) -&gt; "Tokenizer":
        """
        底层 tokenizers 库的 Tokenizer 对象
        """
        return self._tokenizer
    
    def tokenize(
        self,
        text: str,
        **kwargs,
    ) -&gt; List[str]:
        """
        分词（快速）
        """
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        return encoding.tokens
    
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput], List[PreTokenizedInput], List[Tuple[TextInput, TextInput]], List[Tuple[PreTokenizedInput, PreTokenizedInput]]
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -&gt; BatchEncoding:
        """
        批处理编码（快速）
        """
        # 调用底层 tokenizers 库
        encodings = self._tokenizer.encode_batch(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            is_pretokenized=is_split_into_words,
        )
        
        # 转换为 BatchEncoding
        return BatchEncoding(
            encodings,
            encoding=self._tokenizer,
            return_tensors=return_tensors,
        )
    
    def _convert_token_to_id(self, token: str) -&gt; int:
        """
        Token → ID
        """
        return self._tokenizer.token_to_id(token)
    
    def _convert_id_to_token(self, index: int) -&gt; str:
        """
        ID → Token
        """
        return self._tokenizer.id_to_token(index)
    
    @classmethod
    def _from_pretrained_fast(
        cls,
        resolved_vocab_files,
        **kwargs,
    ):
        """
        从预训练加载快速分词器
        """
        tokenizer = Tokenizer.from_file(resolved_vocab_files["tokenizer_file"])
        return cls(tokenizer, **kwargs)
```

## 3. BatchEncoding - 编码结果容器

**位置**: [tokenization_utils_base.py](../src/transformers/tokenization_utils_base.py)

```python
class BatchEncoding:
    """
    编码结果容器，同时支持字典和属性访问
    
    示例:
        encoding = tokenizer("Hello world")
        print(encoding["input_ids"])
        print(encoding.input_ids)
    """
    
    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        encoding: Optional["Encoding"] = None,
        tensor_type: Optional[Union[str, TensorType]] = None,
        prepend_batch_axis: bool = False,
        **kwargs,
    ):
        self.data = dict(data) if data is not None else {}
        self._encodings = encoding if encoding is not None else []
        self.tensor_type = tensor_type
        self._n_sequences = None
        self._batch_size = None
    
    def __getitem__(self, item: Union[int, str, slice]) -&gt; Any:
        """
        支持索引访问
        """
        if isinstance(item, str):
            return self.data[item]
        else:
            # 索引或切片
            return BatchEncoding(
                {k: v[item] for k, v in self.data.items()},
                tensor_type=self.tensor_type,
            )
    
    def __getattr__(self, item: str) -&gt; Any:
        """
        支持属性访问
        """
        if item in self.data:
            return self.data[item]
        raise AttributeError(f"'BatchEncoding' object has no attribute '{item}'")
    
    def __len__(self):
        """
        返回 batch 大小
        """
        return len(next(iter(self.data.values())))
    
    def to(self, device: Union[str, "torch.device"]) -&gt; "BatchEncoding":
        """
        将 tensor 移动到指定设备
        """
        return BatchEncoding(
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in self.data.items()
            },
            tensor_type=self.tensor_type,
        )
    
    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None) -&gt; "BatchEncoding":
        """
        转换为 tensor 格式
        """
        # 省略实现...
        return self
    
    def word_ids(self, batch_index: int = 0) -&gt; List[Optional[int]]:
        """
        获取 word id (用于问答任务，将 token 对齐到原词)
        """
        return self._encodings[batch_index].word_ids
    
    def token_to_word(self, batch_or_token_index: int, token_index: Optional[int] = None) -&gt; Optional[int]:
        """
        Token 索引 → word 索引
        """
        # 省略实现...
    
    def word_to_tokens(self, batch_or_word_index: int, word_index: Optional[int] = None) -&gt; Optional[TokenSpan]:
        """
        Word 索引 → token 索引范围
        """
        # 省略实现...
    
    def char_to_token(self, batch_or_char_index: int, char_index: Optional[int] = None) -&gt; Optional[int]:
        """
        字符索引 → token 索引
        """
        # 省略实现...
    
    def token_to_chars(self, batch_or_token_index: int, token_index: Optional[int] = None) -&gt; Optional[CharSpan]:
        """
        Token 索引 → 字符范围
        """
        # 省略实现...
```

## 4. 分词算法详解

### 4.1 Byte-Pair Encoding (BPE)

**核心思想**:
- 从字符开始
- 迭代合并最频繁出现的字符对
- 得到子词词汇表

**算法**:
```
1. 初始化：每个字符作为初始 token
2. 计算所有相邻 token 对的频率
3. 合并频率最高的 token 对
4. 重复 2-3 直到达到词汇表大小
```

**示例**:
```
初始: "low", "low", "low", "new", "new", "year", "year"
       → l o w </w>, l o w </w>, l o w </w>, n e w </w>, n e w </w>, y e a r </w>, y e a r </w>

合并 "l" + "o" → "lo"
合并 "lo" + "w" → "low"
合并 "y" + "e" → "ye"
...
```

### 4.2 WordPiece (BERT 系列)

**核心思想**:
- 类似 BPE，但使用语言模型 likelihood 而非频率选择合并
- 使用 "##" 前缀表示子词

**示例**:
```
"unhappiness" → "un", "##happ", "##iness"
```

### 4.3 Unigram (T5 系列)

**核心思想**:
- 从大词汇表开始
- 迭代删除对语言模型影响最小的 token
- 保留子词概率

**优点**:
- 可以返回多个分词候选（n-best）
- 适合机器翻译等任务

### 4.4 Byte-Level BPE (GPT2 系列)

**核心思想**:
- 使用字节作为基础单位（256 个 token）
- 避免 OOV（所有 Unicode 字符都可表示）
- 不需要单独的预处理

**示例**:
```
"你好" → 字节表示 → 分词
```

## 5. 预处理流水线

### 5.1 组件介绍

快速分词器的预处理包含几个组件：

```python
# tokenizers 库的架构
Tokenizer(
    normalizer=Normalizer(...),       # 归一化
    pre_tokenizer=PreTokenizer(...),   # 预分词
    model=Model(...),                  # 分词模型
    post_processor=PostProcessor(...), # 后处理
    decoder=Decoder(...),              # 解码器
)
```

### 5.2 Normalizer - 归一化

**常见操作**:
```python
# 小写转换
Lowercase()

# Unicode 归一化 (NFC/NFD/NFKC/NFKD)
NFD()
NFC()

# 去掉重音
StripAccents()

# 统一替换
Replace("  ", " ")
```

### 5.3 PreTokenizer - 预分词

**常见预分词器**:
```python
# 按空格分割
Whitespace()

# 按空白和标点分割
WhitespaceSplit()

# Byte-Level
ByteLevel()

# 按标点分割
Punctuation()

# 数字分割
Digits()
```

### 5.4 TokenizerModel - 分词模型

**支持的模型**:
```python
BPE(...)
WordPiece(...)
Unigram(...)
WordLevel(...)
```

### 5.5 PostProcessor - 后处理

**常见操作**:
```python
# 添加特殊 token
TemplateProcessing(
    single="<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>> $A </s>",
    pair="<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>> $A </s> $B </s>",
    special_tokens=[
        ("<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>>", cls_token_id),
        ("</s>", sep_token_id),
    ]
)

# RoBERTa 风格
RobertaProcessing(...)
```

### 5.6 Decoder - 解码器

**常见解码器**:
```python
# Byte-Level 解码
ByteLevel()

# WordPiece 解码 (去掉 ##)
WordPiece()

# Meta (组合多个解码器)
Metaspace()
```

## 6. 填充与截断详解

### 6.1 PaddingStrategy - 填充策略

```python
class PaddingStrategy(ExplicitEnum):
    DO_NOT_PAD = "do_not_pad"       # 不填充
    LONGEST = "longest"             # 填充到 batch 内最长序列
    MAX_LENGTH = "max_length"       # 填充到指定 max_length
```

### 6.2 TruncationStrategy - 截断策略

```python
class TruncationStrategy(ExplicitEnum):
    DO_NOT_TRUNCATE = "do_not_truncate"       # 不截断
    ONLY_FIRST = "only_first"                 # 只截断第一个序列
    ONLY_SECOND = "only_second"               # 只截断第二个序列
    LONGEST_FIRST = "longest_first"           # 优先截断较长的
```

### 6.3 填充实现

```python
def pad(
    self,
    encoded_inputs: Union[Dict[str, List[int]], List[Dict[str, List[int]]], BatchEncoding],
    padding: Union[bool, str, PaddingStrategy] = True,
    max_length: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
    return_attention_mask: Optional[bool] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    verbose: bool = True,
) -&gt; BatchEncoding:
    """
    填充函数
    
    工作流程:
    1. 计算需要填充到的长度
    2. 使用 pad_token 填充
    3. 生成 attention_mask
    4. 可选: 生成 special_tokens_mask
    """
    # 计算最大长度
    if padding == PaddingStrategy.LONGEST:
        max_length = max(len(seq) for seq in encoded_inputs["input_ids"])
    elif padding == PaddingStrategy.MAX_LENGTH:
        max_length = max_length
    
    # 填充到倍数
    if pad_to_multiple_of is not None:
        max_length = ((max_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    
    # 填充
    padded = {}
    for key, value in encoded_inputs.items():
        if key == "input_ids":
            padded[key] = [
                seq + [self.pad_token_id] * (max_length - len(seq))
                for seq in value
            ]
        elif key == "attention_mask":
            padded[key] = [
                [1] * len(seq) + [0] * (max_length - len(seq))
                for seq in value
            ]
        elif key == "token_type_ids":
            padded[key] = [
                seq + [0] * (max_length - len(seq))
                for seq in value
            ]
        else:
            padded[key] = value
    
    return BatchEncoding(padded)
```

## 7. 常用分词器实现

### 7.1 BertTokenizer / BertTokenizerFast

```python
class BertTokenizer(PreTrainedTokenizer):
    """
    BERT 分词器 (WordPiece)
    """
    
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>>",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        
        # 加载词汇表
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
        # 基本分词器
        self.basic_tokenizer = BasicTokenizer(
            do_lower_case=do_lower_case,
            never_split=never_split,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
        )
        
        # WordPiece 分词器
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab,
            unk_token=unk_token,
        )
    
    def _tokenize(self, text):
        """
        分词: 先基本分词，再 WordPiece
        """
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens
    
    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -&gt; List[int]:
        """
        构建输入:<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>> + seq1 + [SEP] + seq2 + [SEP]
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep
    
    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -&gt; List[int]:
        """
        创建 token_type_ids: 0 表示第一句，1 表示第二句
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
```

### 7.2 GPT2Tokenizer / GPT2TokenizerFast

```python
class GPT2Tokenizer(PreTrainedTokenizer):
    """
    GPT2 分词器 (Byte-Level BPE)
    """
    
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="&lt;|endoftext|&gt;",
        bos_token="&lt;|endoftext|&gt;",
        eos_token="&lt;|endoftext|&gt;",
        pad_token=None,
        add_prefix_space=False,
        **kwargs,
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )
        
        # 加载词汇表和合并规则
        self.bpe_ranks = bytes_to_unicode()
        with open(vocab_file, encoding="utf-8") as f:
            self.vocab = json.load(f)
        with open(merges_file, encoding="utf-8") as f:
            bpe_merges = f.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.add_prefix_space = add_prefix_space
    
    def _tokenize(self, text):
        """
        Byte-Level BPE 分词
        """
        # 预处理文本
        if self.add_prefix_space:
            text = " " + text
        
        # 转换为字节
        text = text.encode("utf-8")
        text = "".join([self.byte_encoder[b] for b in text])
        
        # BPE 分词
        tokens = self.bpe(text).split(" ")
        return tokens
    
    def bpe(self, token):
        """
        BPE 分词核心
        """
        if token in self.cache:
            return self.cache[token]
        
        word = tuple(token)
        pairs = get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            # 找出频率最高的 pair
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            
            # 合并
            first, second = bigram
            new_word = []
            i = 0
            while i &lt; len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
                
                if word[i] == first and i &lt; len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        
        word = " ".join(word)
        self.cache[token] = word
        return word
```

## 8. 使用示例

### 8.1 基础编码

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 编码
text = "Hello, my dog is cute!"
encoding = tokenizer(text)

print(encoding["input_ids"])
print(encoding["attention_mask"])
print(encoding["token_type_ids"])

# 解码
decoded = tokenizer.decode(encoding["input_ids"])
print(decoded)
```

### 8.2 句子对编码

```python
text1 = "Hello, how are you?"
text2 = "I'm fine, thank you."

encoding = tokenizer(text1, text2, return_tensors="pt")
print(encoding["input_ids"])
print(tokenizer.decode(encoding["input_ids"][0]))
```

### 8.3 批处理

```python
texts = [
    "Hello world",
    "How are you",
    "I love transformers",
]

encoding = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=10,
    return_tensors="pt",
)

print(encoding["input_ids"])
print(encoding["attention_mask"])
```

### 8.4 获取偏移量 (问答任务)

```python
encoding = tokenizer(
    "Hello world",
    return_offsets_mapping=True,
    return_special_tokens_mask=True,
)

print(encoding["offset_mapping"])  # 每个 token 对应的字符范围
```

### 8.5 Word ID 对齐

```python
encoding = tokenizer("Hello world", return_offsets_mapping=True)
print(encoding.word_ids())          # [None, 0, 0, 1, 1, None]
print(encoding.token_to_word(1))    # 0
print(encoding.word_to_tokens(0))   # (1, 2)
```

### 8.6 快速分词器 vs 慢速分词器

```python
# 快速分词器 (推荐)
tokenizer_fast = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

# 慢速分词器
tokenizer_slow = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)

# 速度对比：快速分词器快 10-100 倍！
```

### 8.7 添加新的特殊 token

```python
# 添加特殊 token
special_tokens_dict = {
    "additional_special_tokens": ["[URL]", "[EMAIL]"]
}
tokenizer.add_special_tokens(special_tokens_dict)

# 调整模型的 token embedding 大小
model.resize_token_embeddings(len(tokenizer))
```

### 8.8 保存与加载

```python
# 保存
tokenizer.save_pretrained("./my_tokenizer")

# 加载
tokenizer = AutoTokenizer.from_pretrained("./my_tokenizer")
```

## 9. 关键技术要点

### 9.1 特殊 Token 设计

| Token | Purpose | BERT | GPT2 | T5 |
|-------|---------|------|------|----|
| <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>> | Classification | ✓ | ✗ | ✗ |
| &lt;sep&gt; | Separator | ✓ | ✗ | ✗ |
| [MASK] | Mask | ✓ | ✗ | ✗ |
| &lt;s&gt; | Begin | ✗ | ✗ | ✓ |
| &lt;/s&gt; | End | ✗ | ✓ | ✓ |
| &lt;pad&gt; | Padding | ✓ | ✓ | ✓ |
| &lt;unk&gt; | Unknown | ✓ | ✓ | ✓ |

### 9.2 处理长文本策略

```python
# 策略 1: 截断
encoding = tokenizer(text, truncation=True, max_length=512)

# 策略 2: 滑动窗口 (overflowing tokens)
encoding = tokenizer(
    text,
    truncation=True,
    max_length=512,
    stride=128,
    return_overflowing_tokens=True,
)

# 策略 3: 使用支持长文本的分词器
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
```

### 9.3 多语言分词

```python
# XLM-RoBERTa (多语言)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# mT5
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
```

### 9.4 代码分词

```python
# CodeGen
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-mono")

# CodeLlama
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
```

## 代码参考

- [tokenization_utils_base.py](../src/transformers/tokenization_utils_base.py)
- [tokenization_utils.py](../src/transformers/tokenization_utils.py)
- [tokenization_utils_fast.py](../src/transformers/tokenization_utils_fast.py)
- [models/bert/tokenization_bert.py](../src/transformers/models/bert/tokenization_bert.py)
- [models/gpt2/tokenization_gpt2.py](../src/transformers/models/gpt2/tokenization_gpt2.py)

