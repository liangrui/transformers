
# 分词
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

###
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---


# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`


# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
-
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
-
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sent
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")


# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

#
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）

# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = token
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids,
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]

# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
-
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` |
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`:
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token 映射

### 词汇表大小

通过 `tokenizer.vocab_size` 获取。


# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token 映射

### 词汇表大小

通过 `tokenizer.vocab_size` 获取。

### 特殊 Token 管理

```python
# 添加新的特殊 token
tokenizer.add
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token 映射

### 词汇表大小

通过 `tokenizer.vocab_size` 获取。

### 特殊 Token 管理

```python
# 添加新的特殊 token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 调整模型嵌入层

# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token 映射

### 词汇表大小

通过 `tokenizer.vocab_size` 获取。

### 特殊 Token 管理

```python
# 添加新的特殊 token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 调整模型嵌入层
model.resize_token_embeddings(len(tokenizer))
```

---

## 分词算法


# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token 映射

### 词汇表大小

通过 `tokenizer.vocab_size` 获取。

### 特殊 Token 管理

```python
# 添加新的特殊 token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 调整模型嵌入层
model.resize_token_embeddings(len(tokenizer))
```

---

## 分词算法

### 1. WordPiece

- 用于 BERT、RoBERTa 等

# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token 映射

### 词汇表大小

通过 `tokenizer.vocab_size` 获取。

### 特殊 Token 管理

```python
# 添加新的特殊 token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 调整模型嵌入层
model.resize_token_embeddings(len(tokenizer))
```

---

## 分词算法

### 1. WordPiece

- 用于 BERT、RoBERTa 等
- 基于子词的分词方法
- 平衡了词汇表大小和词表覆盖
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token 映射

### 词汇表大小

通过 `tokenizer.vocab_size` 获取。

### 特殊 Token 管理

```python
# 添加新的特殊 token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 调整模型嵌入层
model.resize_token_embeddings(len(tokenizer))
```

---

## 分词算法

### 1. WordPiece

- 用于 BERT、RoBERTa 等
- 基于子词的分词方法
- 平衡了词汇表大小和词表覆盖度

### 2. BPE (Byte-Pair Encoding)

- 用于 G
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token 映射

### 词汇表大小

通过 `tokenizer.vocab_size` 获取。

### 特殊 Token 管理

```python
# 添加新的特殊 token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 调整模型嵌入层
model.resize_token_embeddings(len(tokenizer))
```

---

## 分词算法

### 1. WordPiece

- 用于 BERT、RoBERTa 等
- 基于子词的分词方法
- 平衡了词汇表大小和词表覆盖度

### 2. BPE (Byte-Pair Encoding)

- 用于 GPT、GPT-2 等
- 迭代合并最常见的字符对
-
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token 映射

### 词汇表大小

通过 `tokenizer.vocab_size` 获取。

### 特殊 Token 管理

```python
# 添加新的特殊 token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 调整模型嵌入层
model.resize_token_embeddings(len(tokenizer))
```

---

## 分词算法

### 1. WordPiece

- 用于 BERT、RoBERTa 等
- 基于子词的分词方法
- 平衡了词汇表大小和词表覆盖度

### 2. BPE (Byte-Pair Encoding)

- 用于 GPT、GPT-2 等
- 迭代合并最常见的字符对
- 可预测的分词

### 3. Unigram

- 用于 T5 等

# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token 映射

### 词汇表大小

通过 `tokenizer.vocab_size` 获取。

### 特殊 Token 管理

```python
# 添加新的特殊 token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 调整模型嵌入层
model.resize_token_embeddings(len(tokenizer))
```

---

## 分词算法

### 1. WordPiece

- 用于 BERT、RoBERTa 等
- 基于子词的分词方法
- 平衡了词汇表大小和词表覆盖度

### 2. BPE (Byte-Pair Encoding)

- 用于 GPT、GPT-2 等
- 迭代合并最常见的字符对
- 可预测的分词

### 3. Unigram

- 用于 T5 等
- 基于概率的子词分词
- 支持多种分词方式

### 4.
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token 映射

### 词汇表大小

通过 `tokenizer.vocab_size` 获取。

### 特殊 Token 管理

```python
# 添加新的特殊 token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 调整模型嵌入层
model.resize_token_embeddings(len(tokenizer))
```

---

## 分词算法

### 1. WordPiece

- 用于 BERT、RoBERTa 等
- 基于子词的分词方法
- 平衡了词汇表大小和词表覆盖度

### 2. BPE (Byte-Pair Encoding)

- 用于 GPT、GPT-2 等
- 迭代合并最常见的字符对
- 可预测的分词

### 3. Unigram

- 用于 T5 等
- 基于概率的子词分词
- 支持多种分词方式

### 4. SentencePiece

- 无监督分词
- 将空格视为普通字符
- 用于
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token 映射

### 词汇表大小

通过 `tokenizer.vocab_size` 获取。

### 特殊 Token 管理

```python
# 添加新的特殊 token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 调整模型嵌入层
model.resize_token_embeddings(len(tokenizer))
```

---

## 分词算法

### 1. WordPiece

- 用于 BERT、RoBERTa 等
- 基于子词的分词方法
- 平衡了词汇表大小和词表覆盖度

### 2. BPE (Byte-Pair Encoding)

- 用于 GPT、GPT-2 等
- 迭代合并最常见的字符对
- 可预测的分词

### 3. Unigram

- 用于 T5 等
- 基于概率的子词分词
- 支持多种分词方式

### 4. SentencePiece

- 无监督分词
- 将空格视为普通字符
- 用于多语言模型

---

## 优化与特性

### 1. 快速分词器
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token 映射

### 词汇表大小

通过 `tokenizer.vocab_size` 获取。

### 特殊 Token 管理

```python
# 添加新的特殊 token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 调整模型嵌入层
model.resize_token_embeddings(len(tokenizer))
```

---

## 分词算法

### 1. WordPiece

- 用于 BERT、RoBERTa 等
- 基于子词的分词方法
- 平衡了词汇表大小和词表覆盖度

### 2. BPE (Byte-Pair Encoding)

- 用于 GPT、GPT-2 等
- 迭代合并最常见的字符对
- 可预测的分词

### 3. Unigram

- 用于 T5 等
- 基于概率的子词分词
- 支持多种分词方式

### 4. SentencePiece

- 无监督分词
- 将空格视为普通字符
- 用于多语言模型

---

## 优化与特性

### 1. 快速分词器 (Fast Tokenizers)

优势：
- 性能更高（10-100
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token 映射

### 词汇表大小

通过 `tokenizer.vocab_size` 获取。

### 特殊 Token 管理

```python
# 添加新的特殊 token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 调整模型嵌入层
model.resize_token_embeddings(len(tokenizer))
```

---

## 分词算法

### 1. WordPiece

- 用于 BERT、RoBERTa 等
- 基于子词的分词方法
- 平衡了词汇表大小和词表覆盖度

### 2. BPE (Byte-Pair Encoding)

- 用于 GPT、GPT-2 等
- 迭代合并最常见的字符对
- 可预测的分词

### 3. Unigram

- 用于 T5 等
- 基于概率的子词分词
- 支持多种分词方式

### 4. SentencePiece

- 无监督分词
- 将空格视为普通字符
- 用于多语言模型

---

## 优化与特性

### 1. 快速分词器 (Fast Tokenizers)

优势：
- 性能更高（10-100x）
- 支持偏移量映射（offset mapping）
- 更好的批处理
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token 映射

### 词汇表大小

通过 `tokenizer.vocab_size` 获取。

### 特殊 Token 管理

```python
# 添加新的特殊 token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 调整模型嵌入层
model.resize_token_embeddings(len(tokenizer))
```

---

## 分词算法

### 1. WordPiece

- 用于 BERT、RoBERTa 等
- 基于子词的分词方法
- 平衡了词汇表大小和词表覆盖度

### 2. BPE (Byte-Pair Encoding)

- 用于 GPT、GPT-2 等
- 迭代合并最常见的字符对
- 可预测的分词

### 3. Unigram

- 用于 T5 等
- 基于概率的子词分词
- 支持多种分词方式

### 4. SentencePiece

- 无监督分词
- 将空格视为普通字符
- 用于多语言模型

---

## 优化与特性

### 1. 快速分词器 (Fast Tokenizers)

优势：
- 性能更高（10-100x）
- 支持偏移量映射（offset mapping）
- 更好的批处理
- 支持 Rust 后端

### 2. 偏移量映射

将 token
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token 映射

### 词汇表大小

通过 `tokenizer.vocab_size` 获取。

### 特殊 Token 管理

```python
# 添加新的特殊 token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 调整模型嵌入层
model.resize_token_embeddings(len(tokenizer))
```

---

## 分词算法

### 1. WordPiece

- 用于 BERT、RoBERTa 等
- 基于子词的分词方法
- 平衡了词汇表大小和词表覆盖度

### 2. BPE (Byte-Pair Encoding)

- 用于 GPT、GPT-2 等
- 迭代合并最常见的字符对
- 可预测的分词

### 3. Unigram

- 用于 T5 等
- 基于概率的子词分词
- 支持多种分词方式

### 4. SentencePiece

- 无监督分词
- 将空格视为普通字符
- 用于多语言模型

---

## 优化与特性

### 1. 快速分词器 (Fast Tokenizers)

优势：
- 性能更高（10-100x）
- 支持偏移量映射（offset mapping）
- 更好的批处理
- 支持 Rust 后端

### 2. 偏移量映射

将 token 位置映射回原始文本位置：

```python
output = tokenizer("Hello, world!",
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token 映射

### 词汇表大小

通过 `tokenizer.vocab_size` 获取。

### 特殊 Token 管理

```python
# 添加新的特殊 token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 调整模型嵌入层
model.resize_token_embeddings(len(tokenizer))
```

---

## 分词算法

### 1. WordPiece

- 用于 BERT、RoBERTa 等
- 基于子词的分词方法
- 平衡了词汇表大小和词表覆盖度

### 2. BPE (Byte-Pair Encoding)

- 用于 GPT、GPT-2 等
- 迭代合并最常见的字符对
- 可预测的分词

### 3. Unigram

- 用于 T5 等
- 基于概率的子词分词
- 支持多种分词方式

### 4. SentencePiece

- 无监督分词
- 将空格视为普通字符
- 用于多语言模型

---

## 优化与特性

### 1. 快速分词器 (Fast Tokenizers)

优势：
- 性能更高（10-100x）
- 支持偏移量映射（offset mapping）
- 更好的批处理
- 支持 Rust 后端

### 2. 偏移量映射

将 token 位置映射回原始文本位置：

```python
output = tokenizer("Hello, world!", return_offsets_mapping=True)
offsets = output["offset_mapping"]
#
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token 映射

### 词汇表大小

通过 `tokenizer.vocab_size` 获取。

### 特殊 Token 管理

```python
# 添加新的特殊 token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 调整模型嵌入层
model.resize_token_embeddings(len(tokenizer))
```

---

## 分词算法

### 1. WordPiece

- 用于 BERT、RoBERTa 等
- 基于子词的分词方法
- 平衡了词汇表大小和词表覆盖度

### 2. BPE (Byte-Pair Encoding)

- 用于 GPT、GPT-2 等
- 迭代合并最常见的字符对
- 可预测的分词

### 3. Unigram

- 用于 T5 等
- 基于概率的子词分词
- 支持多种分词方式

### 4. SentencePiece

- 无监督分词
- 将空格视为普通字符
- 用于多语言模型

---

## 优化与特性

### 1. 快速分词器 (Fast Tokenizers)

优势：
- 性能更高（10-100x）
- 支持偏移量映射（offset mapping）
- 更好的批处理
- 支持 Rust 后端

### 2. 偏移量映射

将 token 位置映射回原始文本位置：

```python
output = tokenizer("Hello, world!", return_offsets_mapping=True)
offsets = output["offset_mapping"]
# [(0, 5), (5, 6), (7, 12), (
# 分词器系统 (Tokenizer System) 分析

## 概述

分词器系统负责将文本转换为模型可以理解的 token 序列，以及反向转换。

Transformers 库支持多种分词器后端，以适应不同的需求。

---

## 分层架构

### 1. 基类层

#### `PreTrainedTokenizerBase`

这是所有分词器的抽象基类，定义了通用接口。

**位置**：`tokenization_utils_base.py`

**核心功能**：
- `__call__()`: 主要编码方法
- `encode()`: 编码文本
- `decode()`: 解码 token ids
- `tokenize()`: 分词
- `convert_tokens_to_ids()`: token 转 id
- `convert_ids_to_tokens()`: id 转 token
- `save_pretrained()`: 保存
- `from_pretrained()`: 加载

---

### 2. 后端实现层

库支持多种分词器后端：

#### Python 后端 (`PreTrainedTokenizer`)

**位置**：`tokenization_python.py`

- 纯 Python 实现
- 兼容性最好
- 适合研究和调试
- 性能一般

#### Tokenizers 后端 (`PreTrainedTokenizerFast`)

**位置**：`tokenization_utils_tokenizers.py`

- 基于 HuggingFace Tokenizers 库（Rust 实现）
- 高性能
- 支持更高级功能
- 推荐用于生产

#### SentencePiece 后端

**位置**：`tokenization_utils_sentencepiece.py`

- 基于 Google SentencePiece
- 用于特定模型（如 T5、ALBERT）

#### Mistral Common 后端

**位置**：`tokenization_mistral_common.py`

- 用于 Mistral 模型
- 特殊的分词器实现

---

## 核心功能

### 1. 编码 (Encoding)

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 调用方式 1
inputs = tokenizer("Hello, world!")

# 调用方式 2
inputs = tokenizer.encode("Hello, world!", return_tensors="pt")

# 成对输入
inputs = tokenizer("Hello", "world", padding=True)
```

**输出**：
- `input_ids`: token ids
- `attention_mask`: 注意力掩码
- `token_type_ids`: 分段 id（如 BERT）
- ...

### 2. 解码 (Decoding)

```python
text = tokenizer.decode(input_ids)

# 跳过特殊 token
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

### 3. 批处理 (Batch Processing)

```python
texts = ["Hello", "World", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**参数**：
- `padding`: 填充到相同长度
- `truncation`: 截断到最大长度
- `max_length`: 最大长度
- `return_tensors`: 返回张量格式

---

## 特殊 Token

每个分词器都有自己的特殊 token 集合：

| Token | 用途 |
|------|------|
| `[CLS]` | 分类标记（BERT 等） |
| `[SEP]` | 分隔标记 |
| `[PAD]` | 填充标记 |
| `[UNK]` | 未知标记 |
| `[MASK]` | 掩码标记 |
| `&lt;s&gt;` | 开始标记（GPT 等） |
| `&lt;/s&gt;` | 结束标记 |

---

## 词汇表管理

### 词汇表文件

分词器通常包含以下文件：
- `vocab.txt` / `vocab.json`: 词汇表
- `merges.txt`: BPE 合并规则
- `tokenizer.json`: 完整配置（Fast 分词器）
- `special_tokens_map.json`: 特殊 token 映射

### 词汇表大小

通过 `tokenizer.vocab_size` 获取。

### 特殊 Token 管理

```python
# 添加新的特殊 token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 调整模型嵌入层
model.resize_token_embeddings(len(tokenizer))
```

---

## 分词算法

### 1. WordPiece

- 用于 BERT、RoBERTa 等
- 基于子词的分词方法
- 平衡了词汇表大小和词表覆盖度

### 2. BPE (Byte-Pair Encoding)

- 用于 GPT、GPT-2 等
- 迭代合并最常见的字符对
- 可预测的分词

### 3. Unigram

- 用于 T5 等
- 基于概率的子词分词
- 支持多种分词方式

### 4. SentencePiece

- 无监督分词
- 将空格视为普通字符
- 用于多语言模型

---

## 优化与特性

### 1. 快速分词器 (Fast Tokenizers)

优势：
- 性能更高（10-100x）
- 支持偏移量映射（offset mapping）
- 更好的批处理
- 支持 Rust 后端

### 2. 偏移量映射

将 token 位置映射回原始文本位置：

```python
output = tokenizer("Hello, world!", return_offsets_mapping=True)
offsets = output["offset_mapping"]
# [(0, 5), (5, 6), (7, 12), (12, 13)]
```

### 3. 溢出标记处理

当