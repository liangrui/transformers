
# Transformers 量化系统分析

## 概述图

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                            量化系统架构                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  量化类型 (Quantization Types)                                                               │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ • 8-bit 量化 (bitsandbytes, HQQ)                                                        │ │ │
│  │  │ • 4-bit 量化 (bitsandbytes, AWQ, GPTQ)                                                  │ │ │
│  │  │ • 2-bit 量化 (GPTQ, QuIP)                                                               │ │ │
│  │  │ • FP8 量化 (Hugging Face)                                                               │ │ │
│  │  │ • 动态量化 (PyTorch)                                                                   │ │ │
│  │  │ • 静态量化 (PyTorch)                                                                   │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────┬────────────────────────────────────────────────────┘ │
│                                           ↓                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  核心类 (Core Classes)                                                                       │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ HfQuantizer: 量化器基类                                                                 │ │ │
│  │  │  • adjust_max_memory: 调整内存限制                                                      │ │ │
│  │  │  • adjust_target_dtype: 调整目标类型                                                     │ │ │
│  │  │  • create_quantized_param: 创建量化参数                                                   │ │ │
│  │  │  • process_weights_after_loading: 加载后处理权重                                        │ │ │
│  │  │  • update_torch_dtype: 更新 torch dtype                                                 │ │ │
│  │  │  • postprocess_model: 后处理模型                                                         │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ BitsAndBytesConfig: bitsandbytes 配置                                                    │ │ │
│  │  │  • load_in_8bit: 8-bit 加载                                                             │ │ │
│  │  │  • load_in_4bit: 4-bit 加载                                                             │ │ │
│  │  │  • bnb_4bit_quant_type: 4-bit 类型 (nf4, fp4)                                           │ │ │
│  │  │  • bnb_4bit_compute_dtype: 计算 dtype                                                   │ │ │
│  │  │  • bnb_4bit_use_double_quant: 双重量化                                                  │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ GPTQConfig: GPTQ 配置                                                                   │ │ │
│  │  │  • bits: 量化位数                                                                      │ │ │
│  │  │  • dataset: 校准数据集                                                                 │ │ │
│  │  │  • group_size: 分组大小                                                                │ │ │
│  │  │  • desc_act: 是否使用激活降序                                                          │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ AwqConfig: AWQ 配置                                                                   │ │ │
│  │  │  • bits: 量化位数                                                                     │ │ │
│  │  │  • group_size: 分组大小                                                                │ │ │
│  │  │  • zero_point: 是否使用零点                                                           │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────┬────────────────────────────────────────────────────┘ │
│                                           ↓                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  量化方法详解 (Detailed Methods)                                                             │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ BitsAndBytes (8-bit, 4-bit)                                                             │ │ │
│  │  │  • LLM.int8(): 混合精度量化                                                              │ │ │
│  │  │  • QLoRA: 4-bit 量化 + LoRA                                                              │ │ │
│  │  │  • NF4 (Normalized Float 4): 优化 4-bit 格式                                              │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ GPTQ (Gradient-based Post-Training Quantization)                                       │ │ │
│  │  │  • 基于梯度的量化                                                                      │ │ │
│  │  │  • 最小化均方误差                                                                      │ │ │
│  │  │  • 支持组量化 (group quantization)                                                      │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ AWQ (Activation-aware Weight Quantization)                                             │ │ │
│  │  │  • 激活感知的权重量化                                                                  │ │ │
│  │  │  • 跳过重要权重不量化                                                                  │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ HQQ (Half-Quadratic Quantization)                                                      │ │ │
│  │  │  • 二次量化                                                                           │ │ │
│  │  │  • 不依赖校准数据                                                                      │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────┬────────────────────────────────────────────────────┘ │
│                                           ↓                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  集成与应用 (Integration &amp; Usage)                                                         │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ • from_pretrained() 集成: 直接加载量化模型                                              │ │ │
│  │  │ • Trainer 集成: 支持量化训练                                                            │ │ │
│  │  │ • Pipeline 集成: 支持量化推理                                                            │ │ │
│  │  │ • AutoGPTQ, AutoAWQ, AutoHQQ: 第三方库                                                 │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 1. 量化系统概述

量化是通过降低模型权重和激活的数值精度来减少内存使用和提高推理速度的技术。Transformers 库内置了多种量化方法的支持。

### 1.1 为什么量化？

| 量化类型 | 内存减少 | 速度提升 | 精度损失 |
|---------|---------|---------|---------|
| 8-bit   | 50%     | 2-4x    | 小      |
| 4-bit   | 75%     | 3-6x    | 小-中等 |
| 3-bit   | 81.25%  | 4-8x    | 中等    |
| 2-bit   | 87.5%   | 5-10x   | 大      |

### 1.2 量化类型

1. **Post-Training Quantization (PTQ)**: 训练后量化，不需要重新训练
2. **Quantization-Aware Training (QAT)**: 量化感知训练，需要微调
3. **Dynamic Quantization**: 动态量化，运行时量化
4. **Static Quantization**: 静态量化，需要校准数据
5. **Weight-Only Quantization**: 仅量化权重，激活不变

## 2. 核心类详解

### 2.1 HfQuantizer - 量化器基类

**位置**: [quantizers/base.py](../src/transformers/quantizers/base.py)

```python
class HfQuantizer:
    """
    所有量化器的基类
    
    提供通用的量化接口
    """
    
    def __init__(self, quantization_config: "QuantizationConfigMixin", **kwargs):
        self.quantization_config = quantization_config
        self.pre_quantized = False
        self.training = False
    
    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -&gt; Dict[str, Union[int, str]]:
        """
        调整 max_memory 配置
        
        量化需要额外的内存
        """
        return max_memory
    
    def adjust_target_dtype(self, target_dtype: "torch.dtype") -&gt; "torch.dtype":
        """
        调整目标 dtype
        """
        return target_dtype
    
    def update_torch_dtype(self, torch_dtype: "torch.dtype") -&gt; "torch.dtype":
        """
        更新 torch dtype
        """
        return torch_dtype
    
    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: List[str],
    ):
        """
        创建量化参数
        
        子类必须实现此方法
        """
        raise NotImplementedError
    
    def process_weights_after_loading(self, model: "PreTrainedModel") -&gt; "PreTrainedModel":
        """
        加载后处理权重
        """
        return model
    
    def postprocess_model(self, model: "PreTrainedModel", **kwargs) -&gt; "PreTrainedModel":
        """
        后处理模型
        """
        return model
    
    def validate_environment(self, *args, **kwargs):
        """
        验证环境是否支持量化
        """
        pass
    
    def check_quantized_param(self, param: "torch.Tensor", param_name: str) -&gt; bool:
        """
        检查参数是否被量化
        """
        raise NotImplementedError
```

### 2.2 BitsAndBytesConfig - bitsandbytes 配置

**位置**: [quantizers/bitsandbytes.py](../src/transformers/quantizers/bitsandbytes.py)

```python
@dataclass
class BitsAndBytesConfig(QuantizationConfigMixin):
    """
    bitsandbytes 量化配置
    
    支持 8-bit 和 4-bit 量化
    """
    
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "是否使用 8-bit 量化"},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "是否使用 4-bit 量化"},
    )
    
    # 4-bit 配置
    bnb_4bit_compute_dtype: Optional[Union[str, torch.dtype]] = field(
        default=None,
        metadata={"help": "4-bit 计算时使用的 dtype"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="fp4",
        metadata={"help": "4-bit 量化类型: fp4, nf4"},
    )
    bnb_4bit_use_double_quant: bool = field(
        default=False,
        metadata={"help": "是否使用双重量化"},
    )
    
    # 8-bit 配置
    llm_int8_threshold: float = field(
        default=6.0,
        metadata={"help": "LLM.int8() 阈值"},
    )
    llm_int8_skip_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "跳过量化的模块"},
    )
    llm_int8_enable_fp32_cpu_offload: bool = field(
        default=False,
        metadata={"help": "是否启用 FP32 CPU offload"},
    )
    
    def to_dict(self) -&gt; Dict[str, Any]:
        """
        转换为字典
        """
        output = asdict(self)
        
        # 转换 dtype
        if output["bnb_4bit_compute_dtype"] is not None:
            output["bnb_4bit_compute_dtype"] = str(output["bnb_4bit_compute_dtype"])
        
        return output
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -&gt; "BitsAndBytesConfig":
        """
        从字典创建
        """
        # 转换 dtype
        if config_dict.get("bnb_4bit_compute_dtype") is not None:
            config_dict["bnb_4bit_compute_dtype"] = getattr(
                torch, config_dict["bnb_4bit_compute_dtype"].replace("torch.", "")
            )
        
        return cls(**config_dict)
```

### 2.3 GPTQConfig - GPTQ 配置

**位置**: [quantizers/gptq.py](../src/transformers/quantizers/gptq.py)

```python
@dataclass
class GPTQConfig(QuantizationConfigMixin):
    """
    GPTQ 量化配置
    """
    
    bits: int = field(
        default=4,
        metadata={"help": "量化位数"},
    )
    dataset: Optional[Union[str, List[str]]] = field(
        default=None,
        metadata={"help": "校准数据集"},
    )
    tokenizer: Optional[str] = field(
        default=None,
        metadata={"help": "分词器"},
    )
    group_size: int = field(
        default=128,
        metadata={"help": "分组大小"},
    )
    damp_percent: float = field(
        default=0.1,
        metadata={"help": "扰动百分比"},
    )
    desc_act: bool = field(
        default=True,
        metadata={"help": "是否使用激活降序"},
    )
    sym: bool = field(
        default=True,
        metadata={"help": "是否使用对称量化"},
    )
    true_sequential: bool = field(
        default=True,
        metadata={"help": "是否使用 true sequential"},
    )
    use_cuda_fp16: bool = field(
        default=True,
        metadata={"help": "是否使用 CUDA FP16"},
    )
    model_seqlen: Optional[int] = field(
        default=None,
        metadata={"help": "模型序列长度"},
    )
    block_name_to_quantize: Optional[str] = field(
        default=None,
        metadata={"help": "要量化的块名称"},
    )
    module_name_preceding_first_block: Optional[List[str]] = field(
        default=None,
        metadata={"help": "第一个块前的模块名称"},
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "batch size"},
    )
    pad_token_id: Optional[int] = field(
        default=None,
        metadata={"help": "pad token id"},
    )
    use_exllama: bool = field(
        default=False,
        metadata={"help": "是否使用 ExLlama 内核"},
    )
    use_exllama_v2: bool = field(
        default=False,
        metadata={"help": "是否使用 ExLlama V2 内核"},
    )
```

### 2.4 AwqConfig - AWQ 配置

**位置**: [quantizers/awq.py](../src/transformers/quantizers/awq.py)

```python
@dataclass
class AwqConfig(QuantizationConfigMixin):
    """
    AWQ 量化配置
    """
    
    bits: int = field(
        default=4,
        metadata={"help": "量化位数"},
    )
    group_size: int = field(
        default=128,
        metadata={"help": "分组大小"},
    )
    zero_point: bool = field(
        default=True,
        metadata={"help": "是否使用零点"},
    )
    version: str = field(
        default="gemm",
        metadata={"help": "版本: gemm, gemv"},
    )
    backend: str = field(
        default="auto",
        metadata={"help": "后端: auto, exllama, exllama_v2"},
    )
    do_fuse: bool = field(
        default=False,
        metadata={"help": "是否融合"},
    )
    fuse_max_seq_len: Optional[int] = field(
        default=None,
        metadata={"help": "融合最大序列长度"},
    )
    modules_to_not_convert: Optional[List[str]] = field(
        default=None,
        metadata={"help": "不转换的模块"},
    )
```

### 2.5 HqqConfig - HQQ 配置

**位置**: [quantizers/hqq.py](../src/transformers/quantizers/hqq.py)

```python
@dataclass
class HqqConfig(QuantizationConfigMixin):
    """
    HQQ 量化配置
    """
    
    nbits: int = field(
        default=4,
        metadata={"help": "量化位数"},
    )
    group_size: int = field(
        default=64,
        metadata={"help": "分组大小"},
    )
    quant_zero: bool = field(
        default=True,
        metadata={"help": "是否量化零点"},
    )
    quant_scale: bool = field(
        default=True,
        metadata={"help": "是否量化缩放因子"},
    )
    offload_meta: bool = field(
        default=False,
        metadata={"help": "是否 offload 元数据到 CPU"},
    )
    view_as_float: bool = field(
        default=False,
        metadata={"help": "是否查看为 float"},
    )
    axis: Optional[int] = field(
        default=None,
        metadata={"help": "量化轴"},
    )
    dynamic_config: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "动态配置"},
    )
```

## 3. 量化方法详解

### 3.1 BitsAndBytes (8-bit, 4-bit)

**原理**: LLM.int8() 和 QLoRA

```python
# bitsandbytes 量化流程
def llm_int8_quantize(w):
    """
    LLM.int8() 量化算法
    
    关键思想:
    - 异常值 (outliers) 保持 FP16
    - 其余值量化为 INT8
    - 混合精度计算
    """
    # 1. 检测异常值 (阈值通常为 6)
    threshold = 6.0
    outlier_mask = abs(w) &gt; threshold
    
    # 2. 量化非异常值
    w_int8 = quantize_to_int8(w[~outlier_mask])
    
    # 3. 混合精度计算
    return HybridMatrix(w_fp16=w[outlier_mask], w_int8=w_int8)


# 4-bit 量化 (NF4)
def nf4_quantize(w):
    """
    NF4 (Normalized Float 4) 量化
    
    基于正态分布的优化 4-bit 格式
    """
    # 1. 归一化
    w_normalized = normalize(w)
    
    # 2. 量化到 NF4
    w_nf4 = quantize_to_nf4(w_normalized)
    
    # 3. 双重量化 (可选)
    w_double_quantized = double_quantize(w_nf4)
    
    return w_double_quantized
```

**使用示例**:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 配置 8-bit
bnb_8bit_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=["lm_head"],
)

# 配置 4-bit
bnb_4bit_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_4bit_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# 推理
inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**QLoRA**: 4-bit 量化 + LoRA 微调

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 包装模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# output: trainable params: 5,242,880 || all params: 7,247,602,688 || trainable%: 0.07233911207613068

# 训练
# 使用 Trainer 或普通训练循环
```

### 3.2 GPTQ (Gradient-based Post-Training Quantization)

**原理**: 基于梯度的量化

```python
# GPTQ 算法伪代码
def gptq_quantize(w, hessian, bits=4, group_size=128):
    """
    GPTQ 量化算法
    
    参数:
        w: 权重矩阵
        hessian: 海森矩阵 (Hessian)
        bits: 量化位数
        group_size: 分组大小
    """
    # 1. 分组量化
    groups = split_into_groups(w, group_size)
    
    quantized_groups = []
    for group in groups:
        # 2. 对每组进行量化
        # 使用牛顿法优化
        w_quantized = newton_raphson_quantization(
            group,
            hessian,
            bits=bits,
        )
        quantized_groups.append(w_quantized)
    
    # 3. 合并结果
    w_quantized = concatenate(quantized_groups)
    
    return w_quantized


def newton_raphson_quantization(w, hessian, bits):
    """
    牛顿法量化
    
    最小化均方误差
    """
    # 初始化
    w_int = round_to_int(w, bits)
    
    # 迭代优化
    for i in range(max_iter):
        # 计算梯度
        grad = compute_gradient(w, w_int, hessian)
        
        # 更新
        w_int = w_int - lr * grad
        
        # 投影到量化范围
        w_int = project_to_quant_range(w_int, bits)
    
    return w_int
```

**使用示例**:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# 在线量化 (需要校准数据)
gptq_config = GPTQConfig(
    bits=4,
    dataset="wikitext2",
    group_size=128,
    desc_act=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=gptq_config,
    device_map="auto",
)

# 或者直接加载预量化模型
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-v0.1-GPTQ",
    device_map="auto",
    use_exllama=True,
)

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-v0.1-GPTQ")

# 推理
inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 3.3 AWQ (Activation-aware Weight Quantization)

**原理**: 激活感知的权重量化

```python
# AWQ 算法伪代码
def awq_quantize(w, activations, bits=4, group_size=128):
    """
    AWQ 量化算法
    
    关键思想:
    - 分析激活的重要性
    - 跳过重要的权重不量化
    - 量化不重要的权重
    """
    # 1. 计算重要性得分
    importance = compute_importance(activations)
    
    # 2. 选择不量化的权重
    # (保留 1% 最重要的权重)
    keep_mask = importance &gt; percentile(importance, 99)
    
    # 3. 量化剩余权重
    w_quantized = w.clone()
    w_quantized[~keep_mask] = quantize(w_quantized[~keep_mask], bits)
    
    return w_quantized


def compute_importance(activations):
    """
    计算权重重要性
    
    基于激活的大小和频率
    """
    return mean(abs(activations))  # 简化版
```

**使用示例**:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig

# AWQ 配置
awq_config = AwqConfig(
    bits=4,
    group_size=128,
    version="gemm",
    backend="exllama_v2",
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-v0.1-AWQ",
    quantization_config=awq_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-v0.1-AWQ")

# 推理
inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 3.4 HQQ (Half-Quadratic Quantization)

**原理**: 半二次量化

```python
# HQQ 算法伪代码
def hqq_quantize(w, bits=4, group_size=64):
    """
    HQQ 量化算法
    
    关键思想:
    - 半二次优化
    - 不需要校准数据
    - 快速量化
    """
    # 1. 分组
    groups = split_into_groups(w, group_size)
    
    quantized_groups = []
    for group in groups:
        # 2. 对每组进行量化
        scale = compute_scale(group, bits)
        zero = compute_zero(group, bits)
        
        # 3. 量化
        w_quant = round((group / scale) + zero)
        
        # 4. 可选: 量化 scale 和 zero
        scale_quant = quantize_scale(scale) if quant_scale else scale
        zero_quant = quantize_zero(zero) if quant_zero else zero
        
        quantized_groups.append((w_quant, scale_quant, zero_quant))
    
    return quantized_groups
```

**使用示例**:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig

# HQQ 配置
hqq_config = HqqConfig(
    nbits=4,
    group_size=64,
    quant_zero=True,
    quant_scale=True,
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=hqq_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# 推理
inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 3.5 FP8 量化

**原理**: 使用 FP8 格式

```python
# FP8 配置
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    fp8=True,
    device_map="auto",
)
```

## 4. 量化方法对比

| 方法 | 位数 | 需要校准 | 需要训练 | 速度 | 质量 | 内存 |
|------|------|---------|---------|------|------|------|
| BitsAndBytes (8-bit) | 8 | 否 | 否 | 快 | 高 | 50% |
| BitsAndBytes (4-bit) | 4 | 否 | 否 | 中 | 高 | 75% |
| GPTQ | 2-4 | 是 | 否 | 快 | 高 | 75-87.5% |
| AWQ | 4 | 是 | 否 | 快 | 高 | 75% |
| HQQ | 2-8 | 否 | 否 | 快 | 中 | 75-87.5% |
| QLoRA | 4-8 | 否 | 是 | 快 | 高 | 75-90% |

## 5. 集成与应用

### 5.1 from_pretrained() 集成

```python
from transformers import AutoModelForCausalLM

# 直接加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-v0.1-GPTQ",
    device_map="auto",
)

# 或者加载未量化模型时量化
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=gptq_config,
    device_map="auto",
)
```

### 5.2 Pipeline 集成

```python
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="TheBloke/Mistral-7B-v0.1-GPTQ",
    device_map="auto",
)

output = generator("Hello world", max_new_tokens=50)
print(output[0]["generated_text"])
```

### 5.3 Trainer 集成

```python
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig

# 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 训练
training_args = TrainingArguments(
    output_dir="./quantized_model",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
)
trainer.train()
```

## 6. 使用指南

### 6.1 选择量化方法

**选择指南**:
- **最快**: BitsAndBytes 4-bit (NF4)
- **最高质量**: GPTQ
- **平衡**: AWQ
- **无需校准**: BitsAndBytes, HQQ
- **需要微调**: QLoRA

### 6.2 最佳实践

```python
# 推荐配置: 4-bit NF4 + 双重量化 + BF16 计算
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 设备映射
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto",
)
```

### 6.3 内存使用估算

```python
# 模型内存 (GB) 估算
def estimate_memory(params, bits, dtype):
    """
    估算内存使用
    
    参数:
        params: 参数数量
        bits: 量化位数
        dtype: 激活 dtype
    """
    # 权重内存
    weight_memory = (params * bits) / (8 * 1024**3)  # GB
    
    # 激活内存 (估算)
    # 取决于 batch size 和序列长度
    seq_len = 2048
    batch_size = 1
    hidden_size = 4096
    activation_memory = (batch_size * seq_len * hidden_size * 2) / (1024**3)  # FP16
    
    return weight_memory + activation_memory

# 7B 模型
print(estimate_memory(7e9, bits=4, dtype="bf16"))  # ~3.5 GB (权重) + ~0.016 GB (激活) = ~3.5 GB
```

## 7. 常见问题

### 7.1 量化后质量下降

**解决方案**:
- 使用更高位数 (8-bit vs 4-bit)
- 增加 group_size (128 vs 32)
- 使用校准数据集 (GPTQ, AWQ)
- 使用 QLoRA 微调

### 7.2 推理速度慢

**解决方案**:
- 使用优化的内核 (ExLlama, ExLlamaV2)
- 使用 CUDA FP16
- 增加 batch size
- 使用 Flash Attention

### 7.3 不支持的模型

**解决方案**:
- 检查是否有预量化版本
- 使用通用的量化方法 (BitsAndBytes, HQQ)
- 等待官方支持

## 代码参考

- [quantizers/base.py](../src/transformers/quantizers/base.py)
- [quantizers/bitsandbytes.py](../src/transformers/quantizers/bitsandbytes.py)
- [quantizers/gptq.py](../src/transformers/quantizers/gptq.py)
- [quantizers/awq.py](../src/transformers/quantizers/awq.py)
- [quantizers/hqq.py](../src/transformers/quantizers/hqq.py)

