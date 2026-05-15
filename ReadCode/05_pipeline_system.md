
# Transformers Pipeline 系统分析

## 概述图

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  Pipeline 系统架构                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │  User Interface (用户接口)                                                                  │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ pipeline() - 工厂函数，自动创建合适的 Pipeline                                      │  │  │
│  │  │ - 任务类型检测 ("text-classification", "question-answering", etc.)                 │  │  │
│  │  │ - 模型/分词器自动加载                                                                 │  │  │
│  │  │ - 设备自动配置 (CPU/GPU/MPS)                                                         │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────┬─────────────────────────────────────────────────┘  │
│                                            ↓                                                    │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │  Pipeline Base Class (基类)                                                                │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ Pipeline.__init__()                                                                   │  │  │
│  │  │ - 初始化模型、分词器、tokenizer、feature_extractor                                      │  │  │
│  │  │ - 设备映射                                                                             │  │  │
│  │  │ - Pipeline.__call__() - 主入口                                                         │  │  │
│  │  │   ├── preprocess()     - 数据预处理                                                   │  │  │
│  │  │   ├── _forward()       - 模型前向传播                                                  │  │  │
│  │  │   ├── postprocess()    - 结果后处理                                                   │  │  │
│  │  │   └── get_inference_context() - 推理上下文 (no_grad, autocast)                       │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────┬─────────────────────────────────────────────────┘  │
│                                            ↓                                                    │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │  Specific Pipeline Classes (具体 Pipeline)                                                │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ ┌──────────────────────────────────┐  ┌──────────────────────────────────┐         │  │  │
│  │  │ │ TextClassificationPipeline       │  │ QuestionAnsweringPipeline        │         │  │  │
│  │  │ │ - 情感分析                       │  │ - 抽取式问答                      │         │  │  │
│  │  │ │ - 文本分类                       │  │ - squad 格式                     │         │  │  │
│  │  │ └──────────────────────────────────┘  └──────────────────────────────────┘         │  │  │
│  │  │ ┌──────────────────────────────────┐  ┌──────────────────────────────────┐         │  │  │
│  │  │ │ TokenClassificationPipeline      │  │ TextGenerationPipeline           │         │  │  │
│  │  │ │ - NER (命名实体识别)              │  │ - 文本生成 (自回归)              │         │  │  │
│  │  │ │ - POS (词性标注)                  │  │ - 聊天机器人                      │         │  │  │
│  │  │ └──────────────────────────────────┘  └──────────────────────────────────┘         │  │  │
│  │  │ ┌──────────────────────────────────┐  ┌──────────────────────────────────┐         │  │  │
│  │  │ │ FillMaskPipeline                 │  │ SummarizationPipeline             │         │  │  │
│  │  │ │ - [MASK] 填充                    │  │ - 文本摘要                        │         │  │  │
│  │  │ │ - BERT MLM 任务                   │  │                                   │         │  │  │
│  │  │ └──────────────────────────────────┘  └──────────────────────────────────┘         │  │  │
│  │  │ ┌──────────────────────────────────┐  ┌──────────────────────────────────┐         │  │  │
│  │  │ │ TranslationPipeline              │  │ ZeroShotClassificationPipeline    │         │  │  │
│  │  │ │ - 机器翻译                       │  │ - 零样本分类                      │         │  │  │
│  │  │ │ - En→Fr, En→De, etc.             │  │ - 无需训练数据                    │         │  │  │
│  │  │ └──────────────────────────────────┘  └──────────────────────────────────┘         │  │  │
│  │  │ ┌──────────────────────────────────┐  ┌──────────────────────────────────┐         │  │  │
│  │  │ │ ConversationalPipeline           │  │ FeatureExtractionPipeline        │         │  │  │
│  │  │ │ - 多轮对话                       │  │ - 特征提取                        │         │  │  │
│  │  │ │ - Chat history 管理              │  │ - 向量检索                        │         │  │  │
│  │  │ └──────────────────────────────────┘  └──────────────────────────────────┘         │  │  │
│  │  │ ┌──────────────────────────────────┐  ┌──────────────────────────────────┐         │  │  │
│  │  │ │ ImageClassificationPipeline      │  │ ObjectDetectionPipeline          │         │  │  │
│  │  │ │ - 图像分类                       │  │ - 目标检测                        │         │  │  │
│  │  │ │ - ViT, ResNet 等                 │  │ - DETR, YOLO 等                   │         │  │  │
│  │  │ └──────────────────────────────────┘  └──────────────────────────────────┘         │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────┬─────────────────────────────────────────────────┘  │
│                                            ↓                                                    │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │  Chunking &amp; Batching (分块与批处理)                                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ - 处理长文本 (超过 max_seq_length)                                                    │  │  │
│  │  │ - 滑动窗口 (stride) 支持                                                              │  │  │
│  │  │ - 批处理 (batch_size) 优化                                                            │  │  │
│  │  │ - Pipeline.__call__ 中的 _sanitize_parameters()                                      │  │  │
│  │  │ - _get_iterator() - 数据迭代器                                                        │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │  Utilities (工具函数)                                                                      │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ - infer_framework_load_model() - 自动检测框架 (PyTorch/TensorFlow)                  │  │  │
│  │  │ - check_model_type() - 检查模型类型兼容性                                             │  │  │
│  │  │ - get_model_class() - 获取模型类                                                      │  │  │
│  │  │ - get_task() - 从模型名推断任务类型                                                   │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 1. Pipeline 系统概述

Pipeline 是 Transformers 库提供的高级 API，它封装了从模型加载、预处理、推理到后处理的完整流程。使得用户可以用几行代码完成常见的 NLP 任务，而无需关心底层细节。

### 1.1 核心设计理念

1. **简单易用**：一行代码完成任务
2. **自动选择**：自动检测任务类型和加载适当模型
3. **多框架支持**：同时支持 PyTorch 和 TensorFlow
4. **批量处理**：支持高效的批量推理
5. **多模态**：支持文本、图像、音频等

### 1.2 支持的任务类型

| 任务名 | 描述 | 模型示例 |
|--------|------|----------|
| `text-classification` | 文本分类/情感分析 | BERT, RoBERTa |
| `token-classification` | Token 分类/NER | BERT, RoBERTa |
| `question-answering` | 抽取式问答 | BERT, DistilBERT |
| `fill-mask` | Mask 填充 | BERT, RoBERTa |
| `summarization` | 文本摘要 | T5, BART, Pegasus |
| `translation` | 机器翻译 | T5, MarianMT |
| `text-generation` | 文本生成 | GPT2, LLaMA, Mistral |
| `conversational` | 对话/聊天机器人 | GPT2, LLaMA |
| `zero-shot-classification` | 零样本分类 | BART, MNLI |
| `feature-extraction` | 特征提取 | 任意模型 |

## 2. Pipeline 基类详解

### 2.1 Pipeline 类定义

**位置**: [pipelines/base.py](../src/transformers/pipelines/base.py)

```python
class Pipeline:
    """
    所有 Pipeline 的基类
    
    Pipeline 工作流程：
    Inputs → Preprocess → Model Forward → Postprocess → Outputs
    """
    
    default_input_names = None
    _has_default_input_handler = True
    # 用于标记是否可以使用自动转换
    _auto_model_class = None
    
    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        feature_extractor: Optional["FeatureExtractionMixin"] = None,
        image_processor: Optional["BaseImageProcessor"] = None,
        processor: Optional["ProcessorMixin"] = None,
        modelcard: Optional["ModelCard"] = None,
        framework: Optional[str] = None,
        task: str = "",
        args_parser: Optional["ArgumentHandler"] = None,
        device: Union[int, str, "torch.device"] = None,
        torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
        binary_output: bool = False,
        **kwargs,
    ):
        """
        Pipeline 初始化
        
        参数:
            model: 模型对象
            tokenizer: 分词器 (文本任务)
            feature_extractor: 特征提取器 (音频/图像)
            image_processor: 图像处理
            processor: 统一处理器
            framework: "pt" 或 "tf"
            device: 设备
            torch_dtype: 数据类型
        """
        self.model = model
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.image_processor = image_processor
        self.processor = processor
        self.modelcard = modelcard
        self.framework = framework
        self.task = task
        
        # 设备处理
        if device is None:
            device = -1
        self.device = get_device(device)
        
        # 模型移动到设备
        if self.framework == "pt":
            self.model.to(self.device)
            if torch_dtype is not None:
                self.model.to(dtype=torch_dtype)
        elif self.framework == "tf":
            pass  # TensorFlow 不需要移动
        
        # 更新 model_kwargs
        self.call_count = 0
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = kwargs.pop("num_workers", None)
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(
            **kwargs
        )
        self.binary_output = binary_output
    
    def __call__(
        self,
        inputs: Union[
            str,
            List[str],
            Dict[str, Any],
            List[Dict[str, Any]],
            List[List[str]],
            List[List[Dict[str, Any]]],
        ],
        *args,
        num_workers: Optional[int] = None,
        **kwargs,
    ):
        """
        Pipeline 主调用接口
        
        这是用户直接调用的方法，处理输入并返回结果
        
        工作流程:
        1. 预处理参数
        2. 创建数据集迭代器
        3. 遍历并推理
        4. 返回结果
        """
        # 处理单个 vs 多个输入
        is_single_input = False
        if not isinstance(inputs, list) or (len(inputs) &gt; 0 and not isinstance(inputs[0], list)):
            if self.default_input_names is None or len(self.default_input_names) != 1:
                is_single_input = True
            inputs = [inputs]
        
        # 参数处理
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(
            **kwargs
        )
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}
        
        # 模型评估模式
        if self.framework == "pt":
            self.model.eval()
        
        # 获取数据迭代器
        dataset = self._get_iterator(
            inputs,
            num_workers if num_workers is not None else self._num_workers,
            preprocess_params,
        )
        
        # 推理并收集结果
        final_outputs = []
        for model_outputs, _ in dataset:
            outputs = self.postprocess(model_outputs, **postprocess_params)
            final_outputs.extend(outputs)
        
        # 处理单个输入
        if is_single_input:
            return final_outputs[0]
        
        return final_outputs
    
    def _sanitize_parameters(self, **pipeline_parameters):
        """
        处理参数，将其分为 preprocess, forward, postprocess 三个阶段
        
        需要子类重写以支持具体参数
        """
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {}
        return preprocess_params, forward_params, postprocess_params
    
    def _get_iterator(
        self,
        dataset,
        num_workers: int,
        preprocess_params: Dict[str, Any],
    ):
        """
        获取数据迭代器，处理批处理
        
        关键功能:
        - 批处理 (batch)
        - 多进程预加载 (num_workers)
        - 设备间数据传输
        """
        # 创建 dataset
        if num_workers is None:
            num_workers = self._num_workers
        
        # PipelineDataset 包装输入数据
        dataloader = PipelineDataset(
            dataset,
            self,
            preprocess_params,
        )
        
        # 循环处理
        for model_inputs in dataloader:
            # Forward pass
            model_outputs = self.forward(model_inputs, **forward_params)
            yield model_outputs, model_inputs
    
    def preprocess(self, inputs, **kwargs):
        """
        数据预处理
        
        需要子类重写
        """
        raise NotImplementedError("preprocess not implemented")
    
    def _forward(self, model_inputs, **kwargs):
        """
        模型前向传播
        
        需要子类重写
        """
        raise NotImplementedError("_forward not implemented")
    
    def forward(self, model_inputs, **kwargs):
        """
        Forward pass (内部调用 _forward)
        """
        return self._forward(model_inputs, **kwargs)
    
    def postprocess(self, model_outputs, **kwargs):
        """
        结果后处理
        
        需要子类重写
        """
        raise NotImplementedError("postprocess not implemented")
    
    def get_inference_context(self):
        """
        获取推理上下文管理器 (torch.no_grad, autocast)
        
        提升推理性能
        """
        if self.framework == "pt":
            return torch.no_grad()
        else:
            import tensorflow as tf
            return tf.device("/cpu:0") if self.device.type == "cpu" else nullcontext()
```

### 2.2 ChunkPipeline - 支持分块处理

对于长文本处理，`ChunkPipeline` 提供了分块和滑动窗口功能：

```python
class ChunkPipeline(Pipeline):
    """
    支持分块处理的 Pipeline 基类
    
    用于处理超过模型最大长度的输入
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_model_type(dict(MODEL_FOR_QUESTION_ANSWERING_MAPPING.items()))
    
    def _get_chunk_size(
        self, stride: int, max_seq_len: Optional[int], input_ids: List[int]
    ):
        """
        计算分块大小
        """
        if max_seq_len is None:
            max_seq_len = self.tokenizer.model_max_length
        return min(max_seq_len, len(input_ids)) - stride
    
    def _split_inputs(
        self,
        input_ids: List[int],
        token_type_ids: List[int],
        attention_mask: List[int],
        max_seq_len: Optional[int],
        stride: int,
    ):
        """
        将长输入切分为多个 chunks
        
        使用滑动窗口策略，避免边界信息丢失
        """
        # 省略部分代码...
        return chunks
    
    def preprocess(
        self,
        inputs,
        max_seq_len: Optional[int] = None,
        stride: Optional[int] = None,
    ):
        """
        预处理，支持分块
        """
        # 省略代码...
    
    def postprocess(
        self,
        model_outputs,
        answer_start_score: float = 0.0,
        answer_end_score: float = 0.0,
        top_k: int = 1,
        max_answer_len: int = 20,
        align_to_words: bool = True,
    ):
        """
        后处理，合并多个 chunks 的结果
        """
        # 省略代码...
```

## 3. pipeline() 工厂函数详解

**位置**: [pipelines/__init__.py](../src/transformers/pipelines/__init__.py)

### 3.1 pipeline() 函数定义

```python
def pipeline(
    task: str,
    model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None,
    config: Optional[Union[str, "PretrainedConfig"]] = None,
    tokenizer: Optional[Union[str, "PreTrainedTokenizerBase"]] = None,
    feature_extractor: Optional[Union[str, "FeatureExtractionMixin"]] = None,
    image_processor: Optional[Union[str, "BaseImageProcessor"]] = None,
    processor: Optional[Union[str, "ProcessorMixin"]] = None,
    framework: Optional[str] = None,
    revision: Optional[str] = None,
    use_fast: bool = True,
    use_auth_token: Optional[Union[str, bool]] = None,
    device: Optional[Union[int, str, "torch.device"]] = None,
    device_map: Optional[Union[str, Dict[str, Union[int, str, "torch.device"]]]] = None,
    torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
    trust_remote_code: Optional[bool] = None,
    model_kwargs: Dict[str, Any] = None,
    pipeline_class: Optional[Type["Pipeline"]] = None,
    **kwargs,
) -&gt; Pipeline:
    """
    创建 Pipeline 的工厂函数
    
    参数:
        task: 任务类型 ("text-classification", 等)
        model: 模型名或模型对象
        config: 配置
        tokenizer: 分词器
        framework: "pt" 或 "tf"
        device: 设备
        device_map: 设备映射 (用于大模型)
        torch_dtype: 数据类型
    
    返回:
        初始化好的 Pipeline 对象
    """
    # 1. 推断 task (如果没有明确提供)
    if task is None:
        task = get_task(model)
    
    # 2. 检查 task 是否支持
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unknown task {task}, available tasks are {SUPPORTED_TASKS}")
    
    # 3. 获取任务配置
    targeted_task = SUPPORTED_TASKS[task]
    pipeline_class, model_class = targeted_task["impl"], targeted_task["model"]
    
    # 4. 推断框架
    framework = infer_framework_load_model(
        task,
        model,
        framework,
        revision,
        use_auth_token,
        trust_remote_code,
        model_kwargs,
    )
    
    # 5. 加载 tokenizer/feature_extractor/processor
    tokenizer = load_tokenizer(
        tokenizer, model, revision, use_auth_token, trust_remote_code
    )
    feature_extractor = load_feature_extractor(
        feature_extractor, model, revision, use_auth_token, trust_remote_code
    )
    image_processor = load_image_processor(
        image_processor, model, revision, use_auth_token, trust_remote_code
    )
    processor = load_processor(
        processor, model, revision, use_auth_token, trust_remote_code
    )
    
    # 6. 加载模型
    model = load_model(
        model_class,
        model,
        config,
        framework,
        revision,
        use_auth_token,
        trust_remote_code,
        model_kwargs,
    )
    
    # 7. 处理 device_map
    if device_map is not None:
        if framework == "pt":
            kwargs["device_map"] = device_map
            model = model.to(device_map)
    
    # 8. 创建 Pipeline
    pipeline = pipeline_class(
        model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        image_processor=image_processor,
        processor=processor,
        framework=framework,
        task=task,
        **kwargs,
    )
    
    return pipeline
```

### 3.2 支持的任务映射

```python
# 任务类型到 Pipeline 类和 Model 类的映射
SUPPORTED_TASKS = {
    "text-classification": {
        "impl": TextClassificationPipeline,
        "tf": (TFAutoModelForSequenceClassification,),
        "pt": (AutoModelForSequenceClassification,),
        "default": {
            "model": {
                "pt": "distilbert-base-uncased-finetuned-sst-2-english",
                "tf": "distilbert-base-uncased-finetuned-sst-2-english",
            },
        },
        "type": "text",
    },
    "token-classification": {
        "impl": TokenClassificationPipeline,
        "tf": (TFAutoModelForTokenClassification,),
        "pt": (AutoModelForTokenClassification,),
        "default": {
            "model": {
                "pt": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "tf": "dbmdz/bert-large-cased-finetuned-conll03-english",
            },
        },
        "type": "text",
    },
    "question-answering": {
        "impl": QuestionAnsweringPipeline,
        "tf": (TFAutoModelForQuestionAnswering,),
        "pt": (AutoModelForQuestionAnswering,),
        "default": {
            "model": {
                "pt": "distilbert-base-cased-distilled-squad",
                "tf": "distilbert-base-cased-distilled-squad",
            },
        },
        "type": "text",
    },
    "fill-mask": {
        "impl": FillMaskPipeline,
        "tf": (TFAutoModelForMaskedLM,),
        "pt": (AutoModelForMaskedLM,),
        "default": {
            "model": {
                "pt": "distilroberta-base",
                "tf": "distilroberta-base",
            },
        },
        "type": "text",
    },
    "text-generation": {
        "impl": TextGenerationPipeline,
        "tf": (TFAutoModelForCausalLM,),
        "pt": (AutoModelForCausalLM,),
        "default": {
            "model": {
                "pt": "gpt2",
                "tf": "gpt2",
            },
        },
        "type": "text",
    },
    # ... 更多任务
}
```

## 4. 具体 Pipeline 实现示例

### 4.1 TextClassificationPipeline - 文本分类

**位置**: [pipelines/text_classification.py](../src/transformers/pipelines/text_classification.py)

```python
class TextClassificationPipeline(Pipeline):
    """
    文本分类 Pipeline
    
    用于情感分析、主题分类等任务
    """
    
    def _sanitize_parameters(
        self,
        return_all_scores: Optional[bool] = None,
        function_to_apply: Optional[str] = None,
        **kwargs,
    ):
        """
        预处理参数
        """
        preprocess_params = {}
        
        postprocess_params = {}
        if return_all_scores is not None:
            postprocess_params["return_all_scores"] = return_all_scores
        if function_to_apply is not None:
            postprocess_params["function_to_apply"] = function_to_apply
        
        return preprocess_params, {}, postprocess_params
    
    def preprocess(self, inputs, **tokenizer_kwargs) -&gt; Dict[str, Any]:
        """
        预处理：分词
        """
        return_tensors = self.framework
        return self.tokenizer(inputs, return_tensors=return_tensors, **tokenizer_kwargs)
    
    def _forward(self, model_inputs):
        """
        前向传播
        """
        return self.model(**model_inputs)
    
    def postprocess(
        self,
        model_outputs,
        function_to_apply: Optional[str] = None,
        return_all_scores: Optional[bool] = None,
    ):
        """
        后处理：softmax 或 sigmoid，排序
        
        返回:
            [
                {"label": "POSITIVE", "score": 0.9998},
                {"label": "NEGATIVE", "score": 0.0002},
            ]
        """
        # 获取 logits
        logits = model_outputs.logits
        
        # 确定激活函数
        if function_to_apply is None:
            if self.model.config.problem_type == "multi_label_classification":
                function_to_apply = "sigmoid"
            elif self.model.config.problem_type == "single_label_classification":
                function_to_apply = "softmax"
            else:
                # 自动推断
                if self.model.config.num_labels == 1:
                    function_to_apply = "sigmoid"
                else:
                    function_to_apply = "softmax"
        
        # 应用激活函数
        if function_to_apply == "sigmoid":
            scores = torch.sigmoid(logits)
        elif function_to_apply == "softmax":
            scores = torch.softmax(logits, dim=-1)
        else:
            scores = logits
        
        # 转换为 list
        scores = scores[0].tolist()
        
        # 构造输出
        dict_scores = [
            {"label": self.model.config.id2label[i], "score": score}
            for i, score in enumerate(scores)
        ]
        
        # 排序
        dict_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # 是否返回所有分数
        if return_all_scores:
            return dict_scores
        else:
            return dict_scores[0]
```

### 4.2 QuestionAnsweringPipeline - 问答

**位置**: [pipelines/question_answering.py](../src/transformers/pipelines/question_answering.py)

```python
class QuestionAnsweringPipeline(ChunkPipeline):
    """
    抽取式问答 Pipeline
    
    给定 context 和 question，从 context 中抽取答案
    """
    
    def _sanitize_parameters(
        self,
        topk: Optional[int] = None,
        top_k: Optional[int] = None,
        doc_stride: Optional[int] = None,
        max_question_len: Optional[int] = None,
        max_answer_len: Optional[int] = None,
        handle_impossible_answer: Optional[bool] = None,
        align_to_words: Optional[bool] = None,
        **kwargs,
    ):
        preprocess_params = {}
        if doc_stride is not None:
            preprocess_params["stride"] = doc_stride
        if max_question_len is not None:
            preprocess_params["max_question_len"] = max_question_len
        
        postprocess_params = {}
        if topk is not None:
            warnings.warn(
                "The `topk` parameter is deprecated, use `top_k` instead.",
                FutureWarning,
            )
            postprocess_params["top_k"] = topk
        if top_k is not None:
            postprocess_params["top_k"] = top_k
        if max_answer_len is not None:
            postprocess_params["max_answer_len"] = max_answer_len
        if handle_impossible_answer is not None:
            postprocess_params["handle_impossible_answer"] = handle_impossible_answer
        if align_to_words is not None:
            postprocess_params["align_to_words"] = align_to_words
        
        return preprocess_params, {}, postprocess_params
    
    def preprocess(
        self,
        inputs,
        stride: int = None,
        max_question_len: int = None,
    ):
        """
        预处理问题和上下文
        
        支持长上下文分块
        """
        # 解析输入
        if isinstance(inputs, dict):
            question = inputs.get("question", "")
            context = inputs.get("context", "")
        else:
            question, context = inputs
        
        # 分词
        tokenized_examples = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.tokenizer.model_max_length,
            stride=stride or 128,
            return_overflow_to_sample_mapping=True,
            return_offsets_mapping=True,
            return_tensors=self.framework,
        )
        
        # 省略部分代码...
        
        return model_inputs
    
    def _forward(self, model_inputs):
        """
        前向传播，得到 start_logits 和 end_logits
        """
        return self.model(
            **model_inputs,
        )
    
    def postprocess(
        self,
        model_outputs,
        top_k: Optional[int] = 1,
        max_answer_len: Optional[int] = 15,
        handle_impossible_answer: Optional[bool] = False,
        align_to_words: Optional[bool] = True,
    ):
        """
        后处理，从 start/end logits 中提取答案
        
        返回:
            {
                "score": 0.998,
                "start": 10,
                "end": 20,
                "answer": "answer text"
            }
        """
        start_logits = model_outputs.start_logits[0].numpy()
        end_logits = model_outputs.end_logits[0].numpy()
        
        # 找到最佳 start/end 位置
        # 省略复杂的计算代码...
        
        return [
            {
                "score": score,
                "start": start,
                "end": end,
                "answer": answer,
            }
        ]
```

### 4.3 TextGenerationPipeline - 文本生成

**位置**: [pipelines/text_generation.py](../src/transformers/pipelines/text_generation.py)

```python
class TextGenerationPipeline(Pipeline):
    """
    文本生成 Pipeline
    
    用于自回归文本生成
    """
    
    def _sanitize_parameters(
        self,
        return_full_text: Optional[bool] = None,
        clean_up_tokenization_spaces: Optional[bool] = None,
        prefix: Optional[str] = None,
        handle_long_generation: Optional[str] = None,
        **generate_kwargs,
    ):
        preprocess_params = {}
        if prefix is not None:
            preprocess_params["prefix"] = prefix
        if handle_long_generation is not None:
            preprocess_params["handle_long_generation"] = handle_long_generation
        
        postprocess_params = {}
        if return_full_text is not None:
            postprocess_params["return_full_text"] = return_full_text
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces
        
        forward_params = generate_kwargs
        
        return preprocess_params, forward_params, postprocess_params
    
    def preprocess(self, prompt_text, prefix="", handle_long_generation=None, **generate_kwargs):
        """
        预处理：编码提示词
        """
        tokenizer_kwargs = {}
        if self.tokenizer.pad_token_id is None:
            tokenizer_kwargs["padding"] = False
        else:
            tokenizer_kwargs["padding"] = True
        
        inputs = self.tokenizer(
            prefix + prompt_text,
            return_tensors=self.framework,
            **tokenizer_kwargs,
        )
        
        inputs["prompt_text"] = prompt_text
        return inputs
    
    def _forward(self, model_inputs, **generate_kwargs):
        """
        前向传播：调用 model.generate()
        """
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        
        # 调用 generate
        generated_sequence = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        
        return {
            "generated_sequence": generated_sequence,
            "prompt_text": model_inputs["prompt_text"],
        }
    
    def postprocess(
        self,
        model_outputs,
        return_full_text=True,
        clean_up_tokenization_spaces=True,
    ):
        """
        后处理：解码生成的 token
        """
        generated_sequence = model_outputs["generated_sequence"][0]
        prompt_text = model_outputs["prompt_text"]
        
        # 解码
        generated_sequence = generated_sequence.numpy().tolist()
        text = self.tokenizer.decode(
            generated_sequence,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        
        # 是否返回完整文本
        if return_full_text:
            return [{"generated_text": text}]
        else:
            return [{"generated_text": text[len(prompt_text):]}]
```

## 5. PipelineDataset - 数据处理

**位置**: [pipelines/base.py](../src/transformers/pipelines/base.py)

```python
class PipelineDataset:
    """
    Pipeline 的数据集包装类
    
    处理批处理、并行预处理等
    """
    
    def __init__(
        self,
        dataset,
        pipeline,
        preprocess_params,
    ):
        self.dataset = dataset
        self.pipeline = pipeline
        self.preprocess_params = preprocess_params
    
    def __getitem__(self, i):
        """
        获取单个元素并预处理
        """
        item = self.dataset[i]
        return self.pipeline.preprocess(item, **self.preprocess_params)
    
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        """
        迭代器
        """
        for i in range(len(self)):
            yield self[i]
```

## 6. 使用示例

### 6.1 基础用法

```python
from transformers import pipeline

# 创建情感分析 pipeline
classifier = pipeline("text-classification")

# 使用
result = classifier("I love this movie!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

### 6.2 指定模型

```python
# 使用指定模型
classifier = pipeline(
    "text-classification",
    model="roberta-large-mnli",
    tokenizer="roberta-large-mnli",
    device="cuda:0"  # 使用 GPU
)
```

### 6.3 批量处理

```python
# 批量输入
results = classifier([
    "I love this movie!",
    "I hate this movie!",
    "This is okay."
])

print(results)
# [
#   {'label': 'POSITIVE', 'score': 0.9998},
#   {'label': 'NEGATIVE', 'score': 0.9997},
#   {'label': 'NEUTRAL', 'score': 0.95}
# ]
```

### 6.4 问答示例

```python
qa_pipeline = pipeline("question-answering")

result = qa_pipeline(
    question="What is the capital of France?",
    context="Paris is the capital and most populous city of France."
)

print(result)
# {
#   'score': 0.999,
#   'start': 0,
#   'end': 5,
#   'answer': 'Paris'
# }
```

### 6.5 文本生成

```python
generator = pipeline("text-generation", model="gpt2")

result = generator(
    "In a world where AI can",
    max_length=50,
    num_return_sequences=3,
    temperature=0.7
)

for seq in result:
    print(seq["generated_text"])
```

## 7. 高级用法

### 7.1 零样本分类

```python
zero_shot = pipeline("zero-shot-classification")

result = zero_shot(
    "This is a course about Python programming.",
    candidate_labels=["education", "politics", "sports", "technology"]
)

print(result)
# {
#   'sequence': 'This is a course about Python programming.',
#   'labels': ['technology', 'education', 'politics', 'sports'],
#   'scores': [0.98, 0.01, 0.005, 0.005]
# }
```

### 7.2 使用 device_map 加载大模型

```python
generator = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-v0.1",
    device_map="auto",  # 自动分配到多个 GPU
    torch_dtype=torch.float16
)
```

### 7.3 自定义 Pipeline

```python
from transformers import Pipeline

class MyCustomPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}
    
    def preprocess(self, inputs):
        return self.tokenizer(inputs, return_tensors="pt")
    
    def _forward(self, model_inputs):
        return self.model(**model_inputs)
    
    def postprocess(self, model_outputs):
        return {"logits": model_outputs.logits}

# 使用
pipeline = MyCustomPipeline(
    model="bert-base-uncased",
    tokenizer="bert-base-uncased"
)
```

## 8. 关键技术要点

### 8.1 框架自动检测

`infer_framework_load_model()` 函数会自动检测可用的框架：
- 优先使用用户指定的 framework
- 否则检查 PyTorch 是否可用
- 再检查 TensorFlow 是否可用

### 8.2 长文本处理

- 使用 `ChunkPipeline` 和滑动窗口
- `stride` 参数控制窗口重叠
- 后处理时合并多个 chunks 的结果

### 8.3 设备管理

- 自动检测可用设备 (CUDA, MPS, CPU)
- `device_map` 支持多 GPU 分配
- 支持 CPU offload 和磁盘 offload

### 8.4 批量优化

- 使用 `batch_size` 参数控制批大小
- 多进程预处理 (`num_workers`)
- 高效的数据加载

## 代码参考

- [pipelines/base.py](../src/transformers/pipelines/base.py)
- [pipelines/__init__.py](../src/transformers/pipelines/__init__.py)
- [pipelines/text_classification.py](../src/transformers/pipelines/text_classification.py)
- [pipelines/question_answering.py](../src/transformers/pipelines/question_answering.py)
- [pipelines/text_generation.py](../src/transformers/pipelines/text_generation.py)

