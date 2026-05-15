# Hub 交互

本文件详细解析 Transformers 库与 Hugging Face Hub 的交互机制。

---

## 1. Hugging Face Hub 概述

Hugging Face Hub 是模型、数据集和演示应用的平台，Transformers 库与之深度集成。

---

## 2. 加载模型和分词器

### 2.1 from_pretrained 方法

```python
from transformers import AutoModel, AutoTokenizer

# 从 Hub 加载
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

### 2.2 相关参数

```python
AutoModel.from_pretrained(
    "bert-base-uncased",
    cache_dir="./cache",           # 缓存目录
    force_download=False,         # 强制重新下载
    resume_download=False,        # 断点续传
    proxies=None,                 # 代理
    token=None                    # Hub token（私有模型）
)
```

---

## 3. Push to Hub

### 3.1 保存到 Hub

```python
# 保存模型
model.push_to_hub("my-username/my-model")

# 保存分词器
tokenizer.push_to_hub("my-username/my-model")
```

### 3.2 使用 Trainer 推送

```python
training_args = TrainingArguments(
    output_dir="./results",
    push_to_hub=True,
    hub_model_id="my-username/my-model"
)

trainer = Trainer(..., args=training_args)
trainer.train()
trainer.push_to_hub()
```

---

## 4. 私有模型

使用 token 访问私有模型：

```python
# 方法 1：使用 token 参数
model = AutoModel.from_pretrained(
    "my-username/my-private-model",
    token="hf_..."
)

# 方法 2：使用 login 函数
from huggingface_hub import login
login(token="hf_...")
model = AutoModel.from_pretrained("my-username/my-private-model")
```

---

## 5. 版本控制

```python
# 加载特定版本
model = AutoModel.from_pretrained(
    "bert-base-uncased",
    revision="main"  # 分支、标签或 commit hash
)

# 推送时创建新标签
model.push_to_hub(
    "my-username/my-model",
    commit_message="v1.0 release"
)
```

---

## 6. 缓存机制

```python
import os
from transformers import TRANSFORMERS_CACHE

# 查看缓存目录
print(TRANSFORMERS_CACHE)

# 自定义缓存位置
os.environ["TRANSFORMERS_CACHE"] = "/path/to/cache"
```

---

## 7. 离线模式

```python
import os

# 设置离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 此时会只使用本地缓存的模型
model = AutoModel.from_pretrained("bert-base-uncased")
```

---

## 8. 常用功能

### 8.1 列出模型文件

```python
from huggingface_hub import list_repo_files

files = list_repo_files("bert-base-uncased")
print(files)
```

### 8.2 下载单个文件

```python
from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    "bert-base-uncased",
    "config.json"
)
```
