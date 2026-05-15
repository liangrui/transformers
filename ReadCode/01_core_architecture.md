# Hugging Face Transformers 完整核心架构分析

## 概述架构图

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                             Hugging Face Transformers 完整架构图                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│  ┌──────────────────────────────────┐ ┌────────────────────────────────────────────────┐  │
│  │         入口层 Entry Layer              │ │          高级 API 层                                │  │
│  │  ┌──────────────────────────┐ │ │  ┌──────────┐ ┌──────────┐ ┌──────────┐        │  │
│  │  │ __init__.py (LazyModule) │ │ │  │Pipeline│ │ Trainer │ │TrainerCLI│        │  │
│  │  └──────────────────────────┘ │ │  └────┬─────┘ └────┬─────┘ └──────────┘        │  │
│  └──────────────────────────────────┘ │         │            │                         │  │
│                              │         │            │                         │  │
│  ┌───────────────────────────┴─────────┴────────────┴───────────────────┐  │
│  │                     Auto 工厂类系统 (Auto System)                        │  │
│  │  ┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐ ┌─────┐ │  │
│  │  │ AutoConfig│ │  AutoTokenizer    │ │  AutoModel    │ │...  │ │  │
│  │  └──────────┬───────┴───────────┴─────────┘ │  │
│  └─────────────┬───────────────────────────────────────────────────┘  │
│                │                                                         │
│  ┌─────────────┴───────────────────────────────────────────────────┐  │
│  │                   核心基类层 (Core Base Classes)                │  │
│  │  ┌──────────────────────────────────────────────────────────┐  │
│  │  │  ┌──────────────────┐  ┌──────────────────┐            │  │
│  │  │  │PreTrainedConfig  │  │PreTrainedModel   │            │  │
│  │  │  │(配置基类)       │  │(模型基类)        │            │  │
│  │  │  └──────────────────┘  └────────┬─────────┘            │  │
│  │  │  ┌───────────────────────────┴──────────────┐  │  │
│  │  │  │   生成能力 (GenerationMixin)                 │  │  │
│  │  │  ┌──────────────────┐  ┌──────────────────┐  │  │
│  │  │  │PreTrainedTokenizer │  │   Cache 系统        │  │  │
│  │  │  └──────────────────┘  └──────────────────┘  │  │
│  │  └──────────────────────────────────────────────────────────┘  │
│                │                                                         │
│  ┌─────────────┴───────────────────────────────────────────────────┐  │
│  │                    模型实现层 (Model Implementation)                  │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────┐  │  │
│  │  │   BERT     │ │   GPT-2    │ │   LLaMA    │ │ ...  │  │  │
│  │  │   ViT      │ │   Whisper  │ │   T5       │ │140+  │  │  │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └─────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                   工具与集成层 (Utils & Integrations)              │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │  │
│  │  │Quantiz-│ │FlashAttn │ │Accelerate│ │DeepSpeed│         │  │
│  │  │ation   │ │  量化    │ │  分布式  │ │  Zero  │         │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘         │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │  │
│  │  │  PEFT   │ │  Safet- │ │  Hub    │ │  Optimizers        │  │
│  │  │  (LoRA) │ │  ensors │ │         │ │                 │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                   外部依赖层 (External Dependencies)             │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │  │
│  │  │   PyTorch  │ │  TensorFlow  │ │     JAX      │             │  │
│  │  └──────────────┘ └──────────────┘ └──────────────┘             │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 入口层：_LazyModule 懒加载机制

### 1.1 核心技术：_LazyModule 类

**位置**: [src/transformers/utils/import_utils.py#L2094-L2300](../src/transformers/utils/import_utils.py#L2094-L2300)

`_LazyModule` 是 Transformers 库的核心技术，它使库启动快速，只在需要时才真正导入实际模块。

```python
class _LazyModule(ModuleType):
    """
    模块类，表面暴露所有对象，但只在对象被请求时才执行实际导入。
    
    设计理念 inspired by optuna.integration._IntegrationModule
    """
    
    def __init__(
        self,
        name: str,
        module_file: str,
        import_structure: IMPORT_STRUCTURE_T,
        module_spec: importlib.machinery.ModuleSpec | None = None,
        extra_objects: dict[str, object] | None = None,
        explicit_import_shortcut: dict[str, list[str]] | None = None,
    ):
        super().__init__(name)
        self._object_missing_backend = {}
        self._explicit_import_shortcut = explicit_import_shortcut if explicit_import_shortcut else {}
        
        # 处理多后端依赖
        if any(isinstance(key, frozenset) for key in import_structure):
            self._modules = set()
            self._class_to_module = {}
            self.__all__ = []
            ...
            
            # 后端检查逻辑
            for backends, module in import_structure.items():
                missing_backends = []
                for backend in backends:
                    if backend in BACKENDS_MAPPING:
                        callable, _ = BACKENDS_MAPPING[backend]
                    else:
                        ...
                    try:
                        if not callable():
                            missing_backends.append(backend)
                    except (ModuleNotFoundError, RuntimeError):
                        missing_backends.append(backend)
                
                # 构建模块和类映射
                self._modules = self._modules.union(module_keys)
                for key, values in module.items():
                    if missing_backends:
                        self._object_missing_backend[key] = missing_backends
                    for value in values:
                        self._class_to_module[value] = key
                        if missing_backends:
                            self._object_missing_backend[value] = missing_backends
        
        # 设置基本属性
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    def __getattr__(self, name: str) -&gt; Any:
        """
        当访问模块属性时触发的钩子函数
        
        核心流程：
        1. 检查是否在额外对象中
        2. 检查是否缺失后端
        3. 从 _class_to_module 查找模块
        4. 动态导入模块并获取对象
        5. 缓存结果
        """
        if name in self._objects:
            return self._objects[name]
        if name in self._object_missing_backend:
            # 处理缺失后端的情况
            ...
        elif name in self._class_to_module:
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
            setattr(self, name, value)  # 缓存
            return value
        ...
```

### 1.2 __init__.py 入口实现

**位置**: [src/transformers/__init__.py#L62-L801](../src/transformers/__init__.py#L62-L801)

```python
# 1. 定义导入结构
_import_structure = {
    "configuration_utils": ["PreTrainedConfig", "PretrainedConfig"],
    "data": [...],
    "pipelines": [...],
    ...
}

# 2. 处理可选依赖（PyTorch, Vision, Tokenizers 等）
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_pt_objects
    _import_structure["utils.dummy_pt_objects"] = [...]
else:
    _import_structure["modeling_utils"] = ["PreTrainedModel", ...]
    ...

# 3. 定义类型提示
if TYPE_CHECKING:
    from .configuration_utils import PreTrainedConfig
    from .modeling_utils import PreTrainedModel
    ...

# 4. 运行时：替换 sys.modules[__name__] 为 _LazyModule
else:
    # 从 models 目录动态收集导入结构
    import_structure = define_import_structure(
        Path(__file__).parent / "models", 
        prefix="models"
    )
    import_structure[frozenset({})].update(_import_structure)
    
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
```

### 1.3 核心工作原理

```
用户导入 transformers
    ↓
立即返回 _LazyModule 代理对象
    ↓
用户访问 transformers.BertModel
    ↓
触发 _LazyModule.__getattr__("BertModel")
    ↓
查找 _class_to_module["BertModel"] → "models.bert.modeling_bert"
    ↓
动态导入 transformers.models.bert.modeling_bert
    ↓
获取 BertModel 类并缓存到模块
    ↓
返回给用户
```

---

## 2. Auto 工厂类系统

### 2.1 Auto 体系架构

**位置**: [src/transformers/models/auto/](../src/transformers/models/auto/)

Auto 类系统通过工厂模式实现，由以下组件构成：

```
AutoConfig           AutoTokenizer       AutoModel        AutoImageProcessor
    │                 │               │                   │
    └─────────────────┴───────────────┴───────────────────┘
                      │
              统一工厂模式
                      │
            ┌─────────┴─────────┐
            │                   │
    模型类型映射表      动态类查找
```

### 2.2 AutoConfig 实现

**位置**: [src/transformers/models/auto/configuration_auto.py](../src/transformers/models/auto/configuration_auto.py)

```python
class AutoConfig:
    """
    从预训练配置的工厂类，自动找到正确的配置类
    """
    _LazyConfigMapping = ...  # 延迟加载的配置映射表
    
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path, 
        **kwargs
    ) -&gt; PreTrainedConfig:
        """
        从预训练模型名称或路径加载配置
        
        流程：
        1. 下载 config.json
        2. 从 model_type 确定配置类
        3. 加载配置
        """
        ...
```

### 2.3 AutoModel 实现

**位置**: [src/transformers/models/auto/modeling_auto.py](../src/transformers/models/auto/modeling_auto.py)

```python
class AutoModel:
    """
    通用模型工厂类，自动找到正确的模型类
    """
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        **kwargs,
    ):
        """
        从预训练加载模型
        
        核心流程：
        1. 加载配置
        2. 根据 model_type 找对应模型类
        3. 调用该模型类的 from_pretrained 方法
        """
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs["_from_auto"] = True
        
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        
        if type(config) in MODEL_MAPPING:
            return MODEL_MAPPING[type(config)].from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )
        
        # 尝试其他查找逻辑...
```

### 2.4 Auto 工厂模式的优势

1. **统一接口**：所有模型使用相同的方式加载
2. **延迟加载**：只有使用时才导入对应模块
3. **扩展性**：新增模型只需注册到映射表
4. **类型安全**：类型检查时正常导入

---

## 3. 核心基类层

### 3.1 PreTrainedConfig：配置基类

**位置**: [src/transformers/configuration_utils.py#L123-L800](../src/transformers/configuration_utils.py#L123-L800)

```python
@dataclass_transform(kw_only_default=True)
@strict(accept_kwargs=True)
@dataclass(repr=False)
class PreTrainedConfig(PushToHubMixin, RotaryEmbeddingConfigMixin):
    """
    所有配置类的基类，使用 @dataclass 装饰器实现
    
    继承关系：
    - PushToHubMixin：支持推送到 Hugging Face Hub
    - RotaryEmbeddingConfigMixin：旋转嵌入配置支持
    """
    
    # 类属性（子类重写
    model_type: ClassVar[str]  # 模型类型标识
    has_no_defaults_at_init: ClassVar[bool] = False
    keys_to_ignore_at_inference: ClassVar[list[str]] = []
    attribute_map: ClassVar[dict[str, str]] = {}
    
    # 公共属性
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ) -&gt; "PreTrainedConfig":
        """
        从预训练配置加载
        
        核心流程：
        1. 下载/缓存 config.json
        2. 解析 JSON
        3. 反序列化为配置对象
        """
        ...
    
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        保存配置到文件
        """
        ...
    
    def to_dict(self) -&gt; Dict[str, Any]:
        """
        序列化为字典
        """
        ...
    
    def to_json_string(self, use_diff: bool = True) -&gt; str:
        """
        序列化为 JSON 字符串
        """
        ...
```

### 3.2 PreTrainedModel：模型基类

**位置**: [src/transformers/modeling_utils.py#L900-L2500](../src/transformers/modeling_utils.py#L900-L2500)

```python
class PreTrainedModel(nn.Module, PushToHubMixin, PeftAdapterMixin, GenerationMixin):
    """
    所有 PyTorch 模型的基类
    
    继承关系：
    - nn.Module：PyTorch 模块基类
    - PushToHubMixin：推送到 Hub
    - PeftAdapterMixin：PEFT 适配器支持
    - GenerationMixin：生成能力
    """
    
    # 核心属性
    config_class: PreTrainedConfig
    base_model_prefix: str
    _tied_weights_keys: List[str] = []
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        cache_dir=None,
        ignore_mismatched_sizes=False,
        state_dict=None,
        low_cpu_mem_usage=False,
        device_map=None,
        max_memory=None,
        offload_folder=None,
        offload_state_dict=False,
        load_in_4bit=False,
        load_in_8bit=False,
        quantization_config=None,
        **kwargs,
    ):
        """
        核心模型加载方法
        
        这是整个 Transformers 最复杂的方法之一，包含以下步骤：
        
        1. 配置加载与验证
        2. 量化配置检查
        3. 设备映射处理
        4. 模型实例化
        5. 权重加载
        6. 权重转换
        7. 最终设备移动
        8. 后处理
        
        支持 100+ 参数选项！
        """
        # 步骤 1：配置加载
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                **kwargs,
            )
        
        # 步骤 2：处理量化
        hf_quantizer = None
        if quantization_config is not None:
            hf_quantizer = get_hf_quantizer(quantization_config)
        
        # 步骤 3：设备映射
        if device_map is None:
            if low_cpu_mem_usage:
                device_map = "auto"
        
        # 步骤 4：模型实例化
        model = cls(config, *model_args, **model_kwargs)
        
        # 步骤 5：权重加载
        if state_dict is None:
            state_dict = load_state_dict(
                pretrained_model_name_or_path,
                ...
            )
        
        # 步骤 6：加载到模型
        model.load_state_dict(state_dict, ...)
        
        return model
    
    def save_pretrained(
        self,
        save_directory,
        is_main_process=True,
        state_dict=None,
        save_function=torch.save,
        **kwargs,
    ):
        """
        保存模型权重和配置
        """
        ...
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        前向传播（子类必须实现
        """
        ...
```

### 3.3 GenerationMixin：生成能力混入

**位置**: [src/transformers/generation/utils.py](../src/transformers/generation/utils.py)

```python
class GenerationMixin:
    """
    提供文本生成能力的混入类
    
    核心方法：
    - generate()：主生成方法
    - greedy_search()：贪心搜索
    - sample()：采样
    - beam_search()：光束搜索
    - contrastive_search()：对比搜索
    """
    
    def generate(
        self,
        inputs=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        streamer=None,
        **kwargs,
    ):
        """
        主生成方法
        
        支持多种生成策略：
        1. 贪心解码
        2. 采样解码
        3. 光束搜索
        4. 对比搜索
        5. 等等...
        """
        ...
```

---

## 4. 模型加载核心机制

### 4.1 核心模型加载：core_model_loading.py

**位置**: [src/transformers/core_model_loading.py](../src/transformers/core_model_loading.py)

这是处理权重转换、重命名等底层模块。

```python
class ConversionOps(ABC):
    """
    权重转换操作基类
    """
    
    @abstractmethod
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        """
        执行转换
        """
        ...

class WeightRenaming(ConversionOps):
    """
    权重重命名操作
    """
    ...

class WeightConverter:
    """
    权重转换器
    """
    ...

def convert_and_load_state_dict_in_model(
    model,
    state_dict,
    conversion_mapping=None,
    **kwargs,
):
    """
    转换并加载权重到模型
    """
    ...
```

### 4.2 模型加载完整流程

```
用户调用 AutoModel.from_pretrained()
    ↓
加载配置 (AutoConfig.from_pretrained)
    ↓
确定模型类 (MODEL_MAPPING)
    ↓
调用该模型类的 from_pretrained()
    ↓
初始化模型实例
    ↓
权重加载与转换
    ↓
    ├─ 下载/缓存检查
    ├─ 下载文件
    ├─ 读取权重文件
    ├─ 转换权重名称
    ├─ 调整权重形状
    └─ 加载到模型
    ↓
应用量化
    ↓
设备映射与分发
    ↓
后处理
    ↓
返回模型
```

---

## 5. Pipeline 系统

### 5.1 Pipeline 基类

**位置**: [src/transformers/pipelines/base.py](../src/transformers/pipelines/base.py)

```python
class Pipeline(ABC, PushToHubMixin):
    """
    Pipeline 基类
    
    高级推理 API
    """
    
    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        feature_extractor: Optional["PreTrainedFeatureExtractor"] = None,
        image_processor: Optional["BaseImageProcessor"] = None,
        ...
    ):
        ...
    
    def __call__(self, inputs, **kwargs):
        """
        Pipeline 调用
        """
        return self.run_single(inputs, **kwargs)
    
    def run_single(self, inputs, **kwargs):
        """
        单个输入运行
        """
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)
        
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self.forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs
    
    @abstractmethod
    def preprocess(self, inputs, **kwargs):
        """
        预处理
        """
        ...
    
    @abstractmethod
    def _forward(self, model_inputs, **kwargs):
        """
        前向
        """
        ...
    
    @abstractmethod
    def postprocess(self, model_outputs, **kwargs):
        """
        后处理
        """
        ...
```

### 5.2 Pipeline 流程

```
用户输入
    ↓
预处理 (preprocess)
    ├─ 文本 tokenization
    ├─ 图像处理
    ├─ 特征提取
    └─ 等等...
    ↓
前向传播 (_forward)
    └─ 调用模型
    ↓
后处理 (postprocess)
    ├─ 概率计算
    ├─ 标签映射
    └─ 等等...
    ↓
输出结果
```

---

## 6. 集成与工具系统

### 6.1 可选后端检查

**位置**: [src/transformers/utils/import_utils.py](../src/transformers/utils/import_utils.py)

```python
BACKENDS_MAPPING = {
    "torch": (is_torch_available, "torch"),
    "tensorflow": (is_tf_available, "tensorflow"),
    "jax": (is_flax_available, "flax"),
    "tokenizers": (is_tokenizers_available, "tokenizers"),
    "vision": (is_vision_available, "Pillow"),
    ...
}

def is_torch_available() -&gt; bool:
    """
    检查 PyTorch 是否可用，带版本检查
    """
    ...

def requires_backends(obj, backends):
    """
    装饰器/函数，要求特定后端可用
    """
    ...
```

### 6.2 量化集成

**位置**: [src/transformers/quantizers/](../src/transformers/quantizers/)

```python
class HfQuantizer(ABC):
    """
    量化器基类
    """
    
    @abstractmethod
    def validate_environment(self, *args, **kwargs):
        ...
    
    @abstractmethod
    def update_torch_dtype(self, torch_dtype):
        ...
    
    @abstractmethod
    def adapt_target_modules_for_save(self, model):
        ...
    
    @abstractmethod
    def adjust_target_modules(self, model):
        ...
```

### 6.3 分布式训练集成

- **Accelerate**: 简化分布式训练
- **DeepSpeed**: ZeRO 优化
- **FSDP**: 全分片数据并行
- **Tensor Parallel**: 张量并行

---

## 7. 完整包结构

```
transformers/
├── __init__.py                    # 入口，LazyModule
├── configuration_utils.py        # PreTrainedConfig
├── modeling_utils.py           # PreTrainedModel
├── tokenization_utils_base.py  # Tokenizer 基类
├── core_model_loading.py       # 权重加载
├── cache_utils.py             # KV Cache
│
├── generation/               # 生成相关
│   ├── utils.py            # GenerationMixin
│   ├── configuration_utils.py
│   └── ...
│
├── models/                 # 所有模型实现
│   ├── auto/             # Auto 类
│   │   ├── configuration_auto.py
│   │   ├── modeling_auto.py
│   │   ├── tokenization_auto.py
│   │   └── ...
│   ├── bert/
│   │   ├── configuration_bert.py
│   │   ├── modeling_bert.py
│   │   ├── tokenization_bert.py
│   │   └── ...
│   ├── gpt2/
│   ├── llama/
│   ├── t5/
│   └── ... 140+ 模型
│
├── pipelines/             # Pipeline 系统
│   ├── base.py
│   ├── text_generation.py
│   ├── text_classification.py
│   └── ...
│
├── utils/                 # 工具函数
│   ├── import_utils.py  # _LazyModule
│   ├── logging.py
│   ├── hub.py
│   └── ...
│
├── integrations/         # 第三方集成
│   ├── accelerate.py
│   ├── deepspeed.py
│   ├── peft.py
│   └── ...
│
├── quantizers/           # 量化系统
│   ├── base.py
│   ├── quantizer_bnb_4bit.py
│   ├── quantizer_gptq.py
│   └── ...
│
├── trainer.py             # Trainer API
├── training_args.py      # 训练参数
└── ...
```

---

## 8. 核心设计模式

### 8.1 核心架构设计理念

1. **配置-模型-分词器分离
   - PreTrainedConfig 独立于权重
   - 配置可以单独加载/保存
   - 支持任意组合

2. **工厂模式**
   - Auto 类统一接口
   - 延迟加载模块

3. **混入类设计
   - GenerationMixin
   - PeftAdapterMixin
   - PushToHubMixin

4. **可扩展性设计
   - 新模型遵循规范
   - 动态加载自定义代码
   - 插件式集成

### 8.2 依赖管理

- 可选依赖
- 条件导入
- 运行时检查

### 8.3 性能优化

- 延迟加载
- 内存映射加载
- 量化
- 设备映射
- Flash Attention
- Torch Compile

---

## 9. 核心文件索引

| 组件 | 文件 | 功能 |
|------|------|------|
| 入口 | [__init__.py](../src/transformers/__init__.py) | LazyModule 懒加载 |
| 配置基类 | [configuration_utils.py](../src/transformers/configuration_utils.py) | PreTrainedConfig |
| 模型基类 | [modeling_utils.py](../src/transformers/modeling_utils.py) | PreTrainedModel |
| 分词器基类 | [tokenization_utils_base.py](../src/transformers/tokenization_utils_base.py) | PreTrainedTokenizerBase |
| 核心加载 | [core_model_loading.py](../src/transformers/core_model_loading.py) | 权重加载转换 |
| 生成 | [generation/utils.py](../src/transformers/generation/utils.py) | GenerationMixin |
| Pipeline | [pipelines/base.py](../src/transformers/pipelines/base.py) | Pipeline 基类 |
| 训练 | [trainer.py](../src/transformers/trainer.py) | Trainer API |
| 工具 | [utils/import_utils.py](../src/transformers/utils/import_utils.py) | 导入工具 |
| 量化 | [quantizers/base.py](../src/transformers/quantizers/base.py) | 量化器 |
