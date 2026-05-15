
# Transformers Trainer API 分析

## 概述图

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             Trainer API 架构                                                      │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  核心类 (Core Classes)                                                                       │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ Trainer: 主训练循环                                                                     │ │ │
│  │  │  • train(): 训练                                                                       │ │ │
│  │  │  • evaluate(): 评估                                                                   │ │ │
│  │  │  • predict(): 预测                                                                   │ │ │
│  │  │  • save_model(): 保存                                                                 │ │ │
│  │  │  • push_to_hub(): 推送到 Hub                                                         │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ TrainingArguments: 训练参数配置                                                        │ │ │
│  │  │  • output_dir: 输出目录                                                               │ │ │
│  │  │  • num_train_epochs: 训练轮数                                                          │ │ │
│  │  │  • per_device_train_batch_size: batch size                                           │ │ │
│  │  │  • learning_rate: 学习率                                                              │ │ │
│  │  │  • logging_steps: 日志步数                                                            │ │ │
│  │  │  • save_steps: 保存步数                                                               │ │ │
│  │  │  • fp16/bf16: 混合精度训练                                                            │ │ │
│  │  │  • gradient_accumulation_steps: 梯度累积                                               │ │ │
│  │  │  • deepspeed: DeepSpeed 配置                                                         │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────┬────────────────────────────────────────────────────┘ │
│                                           ↓                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  回调系统 (Callback System)                                                                 │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ TrainerCallback: 回调基类                                                              │ │ │
│  │  │  ├─ PrinterCallback: 打印回调                                                         │ │ │
│  │  │  ├─ ProgressCallback: 进度条回调                                                       │ │ │
│  │  │  ├─ EarlyStoppingCallback: 早停回调                                                     │ │ │
│  │  │  ├─ TensorBoardCallback: TensorBoard 日志                                              │ │ │
│  │  │  ├─ WandbCallback: Weights &amp; Biases                                                    │ │ │
│  │  │  ├─ MLflowCallback: MLflow                                                             │ │ │
│  │  │  └─ CustomCallback: 自定义回调                                                         │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────┬────────────────────────────────────────────────────┘ │
│                                           ↓                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  数据处理 (Data Handling)                                                                   │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ DataCollator: 数据整理器                                                               │ │ │
│  │  │  • DataCollatorForLanguageModeling: 语言模型                                           │ │ │
│  │  │  • DataCollatorForTokenClassification: Token 分类                                     │ │ │
│  │  │  • DataCollatorForSeq2Seq: Seq2Seq                                                    │ │ │
│  │  │  • DataCollatorForWholeWordMask: 整词 Mask                                           │ │ │
│  │  │  • DataCollatorWithPadding: 通用填充                                                   │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────┬────────────────────────────────────────────────────┘ │
│                                           ↓                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  评估与指标 (Evaluation &amp; Metrics)                                                           │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ load_dataset: 加载数据集                                                               │ │ │
│  │  │ load_metric: 加载指标                                                                  │ │ │
│  │  │  • accuracy: 准确率                                                                    │ │ │
│  │  │  • f1: F1 分数                                                                         │ │ │
│  │  │  • bleu: BLEU                                                                          │ │ │
│  │  │  • rouge: ROUGE                                                                        │ │ │
│  │  │  • perpleity: 困惑度                                                                  │ │ │
│  │  │  • custom: 自定义指标                                                                 │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────┬────────────────────────────────────────────────────┘ │
│                                           ↓                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  分布式训练 (Distributed Training)                                                          │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ • PyTorch Distributed (torch.distributed)                                              │ │ │
│  │  │ • DeepSpeed: ZeRO, CPU-offload                                                         │ │ │
│  │  │ • FSDP: Fully Sharded Data Parallel                                                    │ │ │
│  │  │ • Accelerate: 底层库                                                                   │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────┬────────────────────────────────────────────────────┘ │
│                                           ↓                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  优化与调度 (Optimizers &amp; Schedulers)                                                         │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ • AdamW: Adam 优化器 (默认)                                                            │ │ │
│  │  │ • Adafactor: T5 使用                                                                  │ │ │
│  │  │ • SGD: 随机梯度下降                                                                   │ │ │
│  │  │ • 自定义: 自定义优化器                                                                 │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ • LinearSchedule: 线性衰减                                                            │ │ │
│  │  │ • CosineSchedule: 余弦退火                                                            │ │ │
│  │  │ • CosineWithRestarts: 余弦退火重启                                                    │ │ │
│  │  │ • ConstantSchedule: 常数                                                              │ │ │
│  │  │ • 自定义: 自定义调度器                                                                 │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 1. Trainer API 概述

Trainer API 是 Transformers 库提供的完整训练框架，封装了训练循环、评估、预测等功能，支持分布式训练、混合精度、梯度累积等高级特性。

### 1.1 核心特性

- **简单易用**: 几行代码完成训练
- **分布式训练**: 支持多 GPU、TPU、Deepspeed、FSDP
- **混合精度**: FP16、BF16 支持
- **灵活**: 支持自定义训练循环、损失函数
- **可扩展**: 回调系统支持自定义逻辑
- **集成**: 支持 TensorBoard、WandB、MLflow 等

## 2. 核心类详解

### 2.1 TrainingArguments - 训练参数配置

**位置**: [training_args.py](../src/transformers/training_args.py)

```python
@dataclass
class TrainingArguments:
    """
    训练参数配置类
    
    包含 100+ 参数！
    """
    
    # ========== 输出配置 ==========
    output_dir: str = field(
        metadata={"help": "输出目录，用于保存模型、检查点"}
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": "覆盖输出目录"}
    )
    
    # ========== 训练配置 ==========
    do_train: bool = field(default=False, metadata={"help": "是否训练"})
    do_eval: bool = field(default=None, metadata={"help": "是否评估"})
    do_predict: bool = field(default=False, metadata={"help": "是否预测"})
    
    # ========== 训练循环 ==========
    num_train_epochs: float = field(default=3.0, metadata={"help": "训练轮数"})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "每个设备的 batch size"})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "评估 batch size"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "梯度累积步数"})
    
    # ========== 优化器 ==========
    learning_rate: float = field(default=5e-5, metadata={"help": "学习率"})
    weight_decay: float = field(default=0.0, metadata={"help": "权重衰减"})
    adam_beta1: float = field(default=0.9, metadata={"help": "Adam beta1"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Adam beta2"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Adam epsilon"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "梯度裁剪范数"})
    
    # ========== 学习率调度 ==========
    lr_scheduler_type: SchedulerType = field(
        default="linear",
        metadata={"help": "学习率调度类型: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup"}
    )
    warmup_ratio: float = field(default=0.0, metadata={"help": "warmup 比例"})
    warmup_steps: int = field(default=0, metadata={"help": "warmup 步数"})
    
    # ========== 日志与保存 ==========
    logging_dir: Optional[str] = field(default=None, metadata={"help": "日志目录"})
    logging_strategy: IntervalStrategy = field(default="steps", metadata={"help": "日志策略: steps, epoch, no"})
    logging_steps: int = field(default=500, metadata={"help": "每 N 步记录日志"})
    save_strategy: IntervalStrategy = field(default="steps", metadata={"help": "保存策略"})
    save_steps: int = field(default=500, metadata={"help": "每 N 步保存"})
    save_total_limit: Optional[int] = field(default=None, metadata={"help": "最多保留的检查点数量"})
    
    # ========== 评估 ==========
    evaluation_strategy: IntervalStrategy = field(default="no", metadata={"help": "评估策略"})
    eval_steps: Optional[int] = field(default=None, metadata={"help": "每 N 步评估"})
    eval_accumulation_steps: Optional[int] = field(default=None, metadata={"help": "评估时的梯度累积"})
    metric_for_best_model: Optional[str] = field(default=None, metadata={"help": "选择最佳模型的指标"})
    greater_is_better: Optional[bool] = field(default=None, metadata={"help": "指标是否越大越好"})
    
    # ========== 混合精度 ==========
    fp16: bool = field(default=False, metadata={"help": "FP16 混合精度训练"})
    fp16_opt_level: str = field(default="O1", metadata={"help": "Apex 优化级别: O0, O1, O2, O3"})
    bf16: bool = field(default=False, metadata={"help": "BF16 混合精度训练"})
    half_precision_backend: str = field(default="auto", metadata={"help": "混合精度后端: auto, cpu_amp, cuda_amp, apex"})
    
    # ========== 分布式训练 ==========
    local_rank: int = field(default=-1, metadata={"help": "分布式本地 rank"})
    xpu_backend: Optional[str] = field(default=None, metadata={"help": "XPU 后端"})
    tpu_num_cores: Optional[int] = field(default=None, metadata={"help": "TPU 核心数"})
    tpu_metrics_debug: bool = field(default=False, metadata={"help": "TPU 调试"})
    dataloader_pin_memory: bool = field(default=True, metadata={"help": "DataLoader pin_memory"})
    dataloader_num_workers: int = field(default=0, metadata={"help": "DataLoader worker 数"})
    
    # ========== DeepSpeed ==========
    deepspeed: Optional[Union[str, os.PathLike]] = field(
        default=None,
        metadata={"help": "DeepSpeed 配置文件路径"}
    )
    
    # ========== FSDP ==========
    fsdp: bool = field(default=False, metadata={"help": "使用 FSDP"})
    fsdp_min_num_params: int = field(default=0, metadata={"help": "FSDP 最小参数数"})
    fsdp_config: Optional[str] = field(default=None, metadata={"help": "FSDP 配置文件"})
    
    # ========== 其他 ==========
    seed: int = field(default=42, metadata={"help": "随机种子"})
    data_seed: Optional[int] = field(default=None, metadata={"help": "数据种子"})
    remove_unused_columns: bool = field(default=True, metadata={"help": "移除模型不需要的列"})
    label_names: Optional[List[str]] = field(default=None, metadata={"help": "标签列名"})
    include_inputs_for_metrics: bool = field(default=False, metadata={"help": "指标计算时包含输入"})
    
    def __post_init__(self):
        """
        后处理，设置默认值
        """
        # 省略...
    
    def get_process_log_level(self) -> int:
        """
        获取日志级别
        """
        return logging.WARNING
    
    def get_warmup_steps(self, num_training_steps: int) -> int:
        """
        获取 warmup 步数
        """
        if self.warmup_steps &gt; 0:
            return self.warmup_steps
        elif self.warmup_ratio &gt; 0:
            return int(self.warmup_ratio * num_training_steps)
        else:
            return 0
    
    def to_dict(self) -&gt; Dict[str, Any]:
        """
        转换为字典
        """
        return asdict(self)
    
    def to_json_string(self) -&gt; str:
        """
        转换为 JSON
        """
        return json.dumps(self.to_dict(), indent=2)
    
    def save_pretrained(self, output_dir: str):
        """
        保存配置
        """
        with open(os.path.join(output_dir, "training_args.json"), "w") as f:
            f.write(self.to_json_string())
```

### 2.2 Trainer - 主训练类

**位置**: [trainer.py](../src/transformers/trainer.py)

```python
class Trainer:
    """
    Trainer 类，实现完整的训练循环
    
    核心方法: train(), evaluate(), predict()
    """
    
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Trainer 初始化
        
        参数:
            model: 模型
            args: 训练参数
            data_collator: 数据整理器
            train_dataset: 训练集
            eval_dataset: 评估集
            tokenizer: 分词器 (用于保存)
            model_init: 模型初始化函数 (超参数搜索用)
            compute_metrics: 指标计算函数
            callbacks: 回调列表
            optimizers: (优化器, 调度器)
            preprocess_logits_for_metrics: logits 预处理函数
        """
        self.args = args
        self.model = model
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.model_init = model_init
        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        
        # 回调系统
        self.callback_handler = DefaultFlowCallbackHandler()
        if callbacks is not None:
            for callback in callbacks:
                self.add_callback(callback)
        
        # 优化器和调度器
        self.optimizer, self.lr_scheduler = optimizers
        
        # 训练状态
        self.state = TrainerState()
        self.control = TrainerControl()
        
        # 设备
        self.device = args.device
        
        # 初始化混合精度
        if args.fp16 or args.bf16:
            self.use_amp = True
            self.scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
        else:
            self.use_amp = False
            self.scaler = None
        
        # DeepSpeed
        self.deepspeed = None
        if args.deepspeed:
            self.deepspeed = True
            # 初始化 DeepSpeed...
        
        # FSDP
        self.fsdp = None
        if args.fsdp:
            self.fsdp = True
            # 初始化 FSDP...
    
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Optional["optuna.Trial"] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ) -&gt; TrainOutput:
        """
        训练主循环
        
        完整的训练流程:
        
        1. 数据加载与准备
        2. 初始化模型、优化器、调度器
        3. 训练循环:
           - 前向传播
           - 损失计算
           - 后向传播
           - 梯度累积
           - 优化器更新
           - 评估、保存、日志
        4. 返回结果
        """
        # ========== 1. 训练前准备 ==========
        self.state = TrainerState()
        self.callback_handler.on_train_begin(self.args, self.state, self.control)
        
        # 加载检查点
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
        
        # ========== 2. 数据加载 ==========
        train_dataloader = self.get_train_dataloader()
        
        # 计算训练步数
        len_dataloader = len(train_dataloader)
        num_update_steps_per_epoch = len_dataloader // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
        num_train_epochs = math.ceil(self.args.num_train_epochs)
        
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        
        # ========== 3. 初始化优化器和调度器 ==========
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        
        # ========== 4. 训练循环 ==========
        tr_loss = torch.tensor(0.0).to(self.args.device)
        self.model.zero_grad()
        
        for epoch in range(num_train_epochs):
            # 训练一个 epoch
            self.callback_handler.on_epoch_begin(self.args, self.state, self.control)
            
            for step, inputs in enumerate(train_dataloader):
                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / len_dataloader
                
                # 训练一步
                tr_loss_step = self.training_step(model, inputs)
                tr_loss += tr_loss_step
                
                # 梯度累积
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or step == len_dataloader - 1:
                    # 梯度裁剪
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm &gt; 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    
                    # 优化器更新
                    self.optimizer.step()
                    
                    # 调度器更新
                    self.lr_scheduler.step()
                    
                    # 清零梯度
                    model.zero_grad()
                
                # 评估
                if self.control.should_evaluate:
                    self.evaluate()
                    self.control.should_evaluate = False
                
                # 保存
                if self.control.should_save:
                    self.save_checkpoint()
                    self.control.should_save = False
                
                # 日志
                if self.control.should_log:
                    logs = {}
                    logs["loss"] = tr_loss.item()
                    logs["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                    self.callback_handler.on_log(self.args, self.state, self.control, logs)
                    self.control.should_log = False
                
                # 早停
                if self.control.should_training_stop:
                    break
                
                self.state.global_step += 1
            
            self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            
            if self.control.should_training_stop:
                break
        
        # ========== 5. 训练后 ==========
        self.callback_handler.on_train_end(self.args, self.state, self.control)
        
        return TrainOutput(
            global_step=self.state.global_step,
            training_loss=tr_loss.item(),
            metrics={},
        )
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -&gt; torch.Tensor:
        """
        执行一个训练步
        
        包含:
        - 前向传播
        - 损失计算
        - 后向传播
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu &gt; 1:
            loss = loss.mean()
        
        if self.args.gradient_accumulation_steps &gt; 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        loss.backward()
        
        return loss.detach()
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算损失
        
        默认调用模型的 forward 方法并返回 loss
        """
        if not hasattr(self, 'label_smoother'):
            outputs = model(**inputs)
        else:
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = self.label_smoother(outputs, labels)
            outputs.loss = loss
        
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -&gt; Dict[str, float]:
        """
        评估模型
        
        返回指标字典
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        for step, inputs in enumerate(eval_dataloader):
            inputs = self._prepare_inputs(inputs)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                labels = inputs.get("labels")
                
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                
                all_preds.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # 合并预测
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        
        # 计算指标
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=labels))
        else:
            metrics = {}
        
        # 添加前缀
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
        
        self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        
        return metrics
    
    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
    ) -&gt; PredictionOutput:
        """
        预测
        """
        test_dataloader = self.get_test_dataloader(test_dataset)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        for step, inputs in enumerate(test_dataloader):
            inputs = self._prepare_inputs(inputs)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                labels = inputs.get("labels")
                
                all_preds.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy() if labels is not None else None)
        
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels) if all_labels[0] is not None else None
        
        metrics = {}
        if labels is not None and self.compute_metrics is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=labels))
        
        return PredictionOutput(
            predictions=preds,
            label_ids=labels,
            metrics=metrics,
        )
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        保存模型
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
    
    def save_checkpoint(self):
        """
        保存检查点
        """
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
        self.save_model(checkpoint_dir)
        
        # 保存训练状态
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
        
        # 删除旧检查点
        if self.args.save_total_limit is not None:
            checkpoints = list_sorted_checkpoints(self.args.output_dir)
            if len(checkpoints) &gt; self.args.save_total_limit:
                for checkpoint in checkpoints[:-self.args.save_total_limit]:
                    shutil.rmtree(checkpoint)
    
    def create_optimizer(self):
        """
        创建优化器 (默认 AdamW)
        """
        if self.optimizer is None:
            decay_parameters = [p for n, p in self.model.named_parameters() if "bias" not in n and "LayerNorm" not in n]
            no_decay_parameters = [p for n, p in self.model.named_parameters() if "bias" in n or "LayerNorm" in n]
            
            optimizer_grouped_parameters = [
                {
                    "params": decay_parameters,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": no_decay_parameters,
                    "weight_decay": 0.0,
                },
            ]
            
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        
        return self.optimizer
    
    def create_scheduler(self, num_training_steps: int):
        """
        创建学习率调度器 (默认线性衰减)
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        return self.lr_scheduler
    
    def add_callback(self, callback: TrainerCallback):
        """
        添加回调
        """
        self.callback_handler.add_callback(callback)
    
    def remove_callback(self, callback: TrainerCallback):
        """
        移除回调
        """
        self.callback_handler.remove_callback(callback)
```

## 3. 回调系统

### 3.1 TrainerCallback - 回调基类

**位置**: [trainer_callback.py](../src/transformers/trainer_callback.py)

```python
class TrainerCallback:
    """
    回调基类
    
    可以自定义回调来实现各种功能
    """
    
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        初始化结束
        """
        pass
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        训练开始
        """
        pass
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        训练结束
        """
        pass
    
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Epoch 开始
        """
        pass
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Epoch 结束
        """
        pass
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Step 开始
        """
        pass
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Step 结束
        """
        pass
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float], **kwargs):
        """
        日志记录
        """
        pass
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Dict[str, float], **kwargs):
        """
        评估结束
        """
        pass
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        保存检查点
        """
        pass
    
    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        预测步
        """
        pass
```

### 3.2 常用回调

```python
class EarlyStoppingCallback(TrainerCallback):
    """
    早停回调
    
    如果验证集指标没有提升，提前停止训练
    """
    
    def __init__(
        self,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.0,
    ):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.best_metric = None
        self.patience_counter = 0
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """
        评估后检查是否需要早停
        """
        current_metric = metrics.get(args.metric_for_best_model)
        
        if self.best_metric is None:
            self.best_metric = current_metric
        else:
            if args.greater_is_better:
                improved = current_metric &gt; self.best_metric + self.early_stopping_threshold
            else:
                improved = current_metric &lt; self.best_metric - self.early_stopping_threshold
            
            if improved:
                self.best_metric = current_metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter &gt;= self.early_stopping_patience:
                    control.should_training_stop = True


class TensorBoardCallback(TrainerCallback):
    """
    TensorBoard 日志回调
    """
    
    def __init__(self, tb_writer=None):
        self.tb_writer = tb_writer
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        for key, value in logs.items():
            self.tb_writer.add_scalar(key, value, state.global_step)


class ProgressCallback(TrainerCallback):
    """
    进度条回调
    """
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.training_bar = tqdm(total=state.max_steps)
    
    def on_step_end(self, args, state, control, **kwargs):
        self.training_bar.update(1)
    
    def on_train_end(self, args, state, control, **kwargs):
        self.training_bar.close()
```

## 4. DataCollator - 数据整理器

**位置**: [data/data_collator.py](../src/transformers/data/data_collator.py)

```python
class DataCollatorForLanguageModeling:
    """
    语言模型数据整理器
    
    实现 MLM (Masked Language Modeling) 或 CLM (Causal Language Modeling)
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: Optional[int] = None,
        tf_experimental_compatibility: bool = False,
    ):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, examples: List[Dict[str, Any]]) -&gt; Dict[str, torch.Tensor]:
        """
        整理数据
        
        对于 MLM: 随机 mask token
        """
        # 转换为 tensor
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        
        # 如果是 CLM，labels = input_ids
        if not self.mlm:
            batch["labels"] = batch["input_ids"].clone()
            return batch
        
        # MLM: 随机 mask
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        
        return batch
    
    def mask_tokens(
        self,
        inputs: torch.Tensor,
        special_tokens_mask: Optional[torch.BoolTensor] = None,
    ) -&gt; Tuple[torch.Tensor, torch.Tensor]:
        """
        随机 mask token
        
        策略:
        - 15% 的 token 被选中
        - 80% 替换为 [MASK]
        - 10% 替换为随机 token
        - 10% 保持不变
        """
        labels = inputs.clone()
        
        # 概率矩阵
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # 不 mask 特殊 token
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # 采样 mask 位置
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 只有 mask 的位置计算 loss
        labels[~masked_indices] = -100
        
        # 80% 替换为 [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10% 替换为随机 token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        # 10% 保持不变
        return inputs, labels


class DataCollatorForTokenClassification:
    """
    Token 分类数据整理器
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
    
    def __call__(self, examples: List[Dict[str, Any]]) -&gt; Dict[str, torch.Tensor]:
        label_name = "label" if "label" in examples[0] else "labels"
        labels = [example[label_name] for example in examples]
        batch = self.tokenizer.pad(
            examples,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # 填充 labels
        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = torch.tensor(
                [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            )
        else:
            batch[label_name] = torch.tensor(
                [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            )
        
        return batch


class DataCollatorForSeq2Seq:
    """
    Seq2Seq 数据整理器 (翻译、摘要等)
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: Optional[PreTrainedModel] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
    
    def __call__(self, examples: List[Dict[str, Any]]) -&gt; Dict[str, torch.Tensor]:
        # 类似 DataCollatorForTokenClassification
        # 处理 decoder_input_ids
        pass
```

## 5. 分布式训练

### 5.1 PyTorch Distributed

```python
# 使用 accelerate 启动分布式训练
accelerate launch train.py

# 在训练脚本中
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
```

### 5.2 DeepSpeed

```python
# DeepSpeed 配置文件 (ds_config.json)
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "fp16": {
        "enabled": true
    }
}

# 训练脚本
training_args = TrainingArguments(
    deepspeed="ds_config.json",
)

trainer = Trainer(
    args=training_args,
)
trainer.train()

# 启动
deepspeed --num_gpus=4 train.py
```

### 5.3 FSDP

```python
training_args = TrainingArguments(
    fsdp=True,
    fsdp_min_num_params=1e8,
)

trainer = Trainer(
    args=training_args,
)
trainer.train()
```

## 6. 使用示例

### 6.1 简单的训练脚本

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import numpy as np
import evaluate

# 1. 加载数据
dataset = load_dataset("glue", "mrpc")
metric = evaluate.load("glue", "mrpc")

# 2. 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 3. 预处理数据
def preprocess_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4. 定义指标计算
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 5. 配置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
)

# 6. 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 7. 训练
trainer.train()

# 8. 评估
trainer.evaluate()

# 9. 预测
predictions = trainer.predict(tokenized_dataset["test"])
```

### 6.2 语言模型训练

```python
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./mlm_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
)

trainer.train()
```

### 6.3 使用回调

```python
from transformers import EarlyStoppingCallback, TrainerCallback

# 早停
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.0,
)

# 自定义回调
class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(f"Step {state.global_step}, Loss: {logs['loss']}")
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print(f"Evaluation: {metrics}")

trainer = Trainer(
    ...,
    callbacks=[early_stopping_callback, CustomCallback()],
)
```

### 6.4 混合精度训练

```python
# FP16
training_args = TrainingArguments(
    fp16=True,
    fp16_opt_level="O1",
)

# BF16 (Ampere+ GPU)
training_args = TrainingArguments(
    bf16=True,
)
```

### 6.5 梯度累积

```python
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    # 有效 batch size = 4 * 8 = 32
)
```

## 7. 关键技术要点

### 7.1 损失计算

Trainer 默认使用模型返回的 loss，模型的 forward 方法应该返回 (loss, ...) 或包含 loss 的字典。

### 7.2 梯度裁剪

```python
training_args = TrainingArguments(
    max_grad_norm=1.0,  # 梯度裁剪范数
)
```

### 7.3 加载最佳模型

```python
training_args = TrainingArguments(
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
)
```

### 7.4 断点续训

```python
# 从检查点恢复训练
trainer.train(resume_from_checkpoint="./results/checkpoint-1000")
```

### 7.5 推送到 Hub

```python
from huggingface_hub import notebook_login

notebook_login()

trainer.push_to_hub("my-user/my-model")
```

## 代码参考

- [trainer.py](../src/transformers/trainer.py)
- [training_args.py](../src/transformers/training_args.py)
- [trainer_callback.py](../src/transformers/trainer_callback.py)
- [data/data_collator.py](../src/transformers/data/data_collator.py)

