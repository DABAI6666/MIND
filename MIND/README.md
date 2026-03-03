# MIND 🧠: 面向精神科问诊的统一问询-诊断强化学习框架

**MIND: Unified Inquiry–Diagnosis RL with Criteria-Grounded Clinical Supports for Psychiatric Consultation**

## 目录

- [介绍](#介绍)
- [主要特性](#主要特性)
- [方法论](#方法论)
- [实验](#实验)
- [设置](#设置)
- [实验脚本](#实验脚本)
- [引用](#引用)

## 介绍

**MIND** 是一个面向精神科问诊的**证据支撑、过程监督**统一问询-诊断强化学习框架。精神科问诊面临特有挑战：主观模糊性、共病复杂性，以及从不完整、不一致的患者陈述中持续提取精神病理线索的需求。现有方法存在两大根本缺陷：（1）缺乏标准支撑时易产生无据可查的临床断言；（2）多轮交互中难以抑制**问询漂移**（偏题或低效问询）。

为解决上述问题，MIND 提出三项核心机制：

1. **标准支撑的精神科推理库（PRB）**：将多轮对话上下文提炼为临床检索状态，检索语义相近的参考问诊案例，蒸馏出可复用的标准对齐临床支撑，引导问询与差异诊断。
2. **显式临床推理与 Rubric 过程监督**：强制生成结构化推理链（症状分析→鉴别诊断→决策逻辑），并以 LLM 评判器提供逐轮细粒度的过程奖励。
3. **价值感知轨迹纠正**：检测低效轮次，自适应触发自我重试或 PRB 引导回退，抑制问询漂移，提升多轮信息获取效率。

## 主要特性

- 🏦 **标准支撑的精神科推理库（PRB）**：以结构化临床检索状态实现可靠检索，提供标准对齐的问询提示，避免直接从隐喻性精神叙述中检索的不可靠性。
- 🔍 **两阶段检索-推理转格式**：每轮先生成 `<rag_query>` 检索 PRB，再基于检索支撑生成 `<think>…</think><answer>…</answer>` 的显式推理与回复。
- 📊 **Rubric 过程奖励**：LLM 评判器从症状覆盖（$S^{\text{sym}}$）、鉴别诊断（$S^{\text{diff}}$）、决策逻辑（$S^{\text{dec}}$）三维度评分，提供逐轮密集监督信号。
- 🔄 **价值感知轨迹纠正**：基于规则检测重复、格式异常、预算违规等低效轮次，触发自我重试或 PRB 引导回退（SCID-5 风格参考问询）。
- 🎯 **混合奖励优化**：综合过程奖励、检索塑形奖励、信息增益奖励、操作惩罚与终态诊断准确性奖励，联合优化问询策略与诊断决策。
- ⚡ **SFT→GRPO 分阶段训练**：Kimi-K2 蒸馏的 SFT 冷启动（DeepSeek-R1 风格）建立稳定的两阶段轮格式，GRPO 进一步优化多轮决策。

## 方法论



MIND 框架由三个核心模块构成：

### 模块一：标准支撑的精神科推理库（PRB）

PRB 是一个将问诊状态与标准对齐问询支撑配对的知识库。每条历史案例被提炼为**临床检索状态** $q_i$（关键词密集、仅含事实、未明确字段标注为"未提及/不清楚"），并经由 Kimi-K2 综合教材与临床指南，蒸馏出知识支撑 $r_i$（涵盖已知事实、缺失检查项、下一步问询依据）。PRB 条目经 LLM 评判器进行质量评估（1–5 分可靠性打分）以确保临床严谨性。

### 模块二：显式临床推理与过程监督

每轮执行两阶段格式：

**阶段 I** — 检索查询生成：
$$y_t^{(1)} = \texttt{<rag\_query>} q_t \texttt{</rag\_query>}$$

**阶段 II** — 推理-回复生成（以检索支撑为条件）：
$$y_t^{(2)} = \texttt{<think>} z_t \texttt{</think>} \texttt{<answer>} a_t \texttt{</answer>}$$

推理链 $z_t$ 需显式覆盖：（1）症状分析（已确认/排除发现及最关键缺失信息）；（2）鉴别考量（竞争解释与排除线索）；（3）决策逻辑（说明为何该问题是当前最具信息量的步骤）。

**Rubric 过程奖励**：LLM 评判器对推理链进行三维评分：
$$\mathbf{S}_t = (S_t^{\text{sym}}, S_t^{\text{diff}}, S_t^{\text{dec}}), \quad r_t^{\text{proc}} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \frac{S_t^c}{S_{\max}}$$

### 模块三：价值感知轨迹纠正

当动作表现出低效（重复、格式错误、预算违规）时，系统触发**自我重试**（更严格约束下的反思重生成）；持续失败则调用 **PRB 引导回退**，检索最近 PRB 条目的参考问询：
$$i^* = \arg\max_i \cos(q_t, q_i), \quad a_t^{\text{ref}} = \text{RefInq}(i^*)$$

检索注入受可靠性门控：仅当检索匹配强度超过预设阈值时才注入外部提示。

### 奖励聚合

| 奖励组件 | 计算方式 | 权重 |
|----------|----------|------|
| 诊断准确性（终态） | $\mathbb{1}[d_p = d^*]$ | 5.0 |
| 信息增益 | $\lambda_{\text{gain}} \cdot \Delta_t$（新揭示临床线索数） | 0.005 |
| 临床推理（过程） | Rubric 三维归一化均值 | 0.01 |
| 格式合规（惩罚） | 语法错误、重复、预算违规 | 0.1 |

总奖励：$r_t = r_t^{\text{proc}} + r_t^{\text{retr}} + r_t^{\text{gain}} + r_t^{\text{pen}}$，$R = \sum_{t=1}^T \alpha r_t + \beta r^{\text{term}}$

## 实验

### 患者智能体仿真评估

患者智能体从信息控制（IC）、响应完整性（RC）、事实冲突率（FC）、拟人度（HL）四个维度评估，经 LLM 评判器与领域专家双重验证。

### 医生智能体诊断性能



MIND 在两种患者模拟器（PsySim-Std 和 PsySim-Adapt）下均取得最优表现：

| 方法 | PsySim-Std Acc | PsySim-Std F1 | PsySim-Adapt Acc | PsySim-Adapt F1 |
|------|---------------|---------------|-----------------|-----------------|
| GPT-4o | 49.5 | — | 40.5 | — |
| DDO | 53.0 | — | 45.7 | — |
| DoctorAgent-RL | 56.5 | — | 47.8 | — |
| **MIND-8B（Ours）** | **71.5** | **72.5** | **62.5** | **63.1** |

### 支撑忠实度评估

MIND 在事实一致性（FC）、支撑接地性（SG）、患者忠实度（PF）三维度均取得最优均分 **8.6/10**，优于 DDO（8.1）和 DoctorAgent-RL（8.0）。

### 性能对比


### 消融研究

关键发现：去除 thinking 监督导致最大衰减（F1 下降 ~12–14%）；去除 PRB 导致 F1 下降 ~5–6%；去除回退机制导致 F1 下降 ~3%。

### 动态轮次预算分析


---

## 设置

### 1. 环境配置

```bash
# 创建 conda 环境
conda create -n mind python=3.10
conda activate mind

# 安装 PyTorch（CUDA 11.8）
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install -r verl/requirements.txt
pip install vllm==0.6.4.post1
pip install ray[default]==2.10.0
pip install hydra-core omegaconf
pip install transformers accelerate peft
pip install pandas pyarrow rouge-score
pip install wandb
```

### 2. 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | 4× A100 (40GB) | 8× A100 (80GB) |
| 内存 | 128GB | 256GB |
| 存储 | 50GB | 200GB |

### 3. 下载模型

- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)（医生智能体基座 MIND-8B / 患者智能体固定权重）
- [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)（医生智能体基座 MIND-4B）

## 实验脚本

### 1. 数据预处理

数据集基于 1,000 条脱敏电子病历（EMR），分为四类（抑郁 335 / 焦虑 200 / 混合 265 / 其他 200），训练数据放置于 `data/` 目录下：

| 划分 | 焦虑 | 抑郁 | 混合 | 其他 | 合计 |
|------|------|------|------|------|------|
| SFT  | 2,410 | 4,120 | 3,640 | 3,720 | **13,890** |
| RL   | 884 | 1,929 | 1,141 | 1,485 | **5,399** |
| Test | 115 | 182 | 140 | 165 | **602** |

```bash
# 从 EMR 重建患者档案并生成对话
python scripts/data_process/extract_medical_data.py
python scripts/data_process/convert_dialog_format.py

# 生成训练/验证划分
python scripts/data_process/split_train_val.py
```

### 2. 构建精神科推理库（PRB）

```bash
# 生成临床检索状态并蒸馏支撑注释
python scripts/prb/build_clinical_states.py
python scripts/prb/distill_supports.py

# 质量评估与过滤（LLM 评判器打分 1–5）
python scripts/prb/quality_filter.py
```

### 3. 训练医生智能体

**阶段一：SFT 冷启动（Kimi-K2 蒸馏，DeepSeek-R1 风格）**

| 超参数 | 值 |
|--------|-----|
| 训练方法 | LoRA |
| LoRA rank | 64 |
| LoRA alpha | 32 |
| 学习率 | 1e-4 |
| Warmup ratio | 0.03 |

```bash
# SFT 冷启动（建立两阶段轮格式与临床行为）
bash scripts/training/train_sft_warmstart.sh
```

**阶段二：GRPO 强化学习**

| 超参数 | 值 |
|--------|-----|
| 算法 | GRPO |
| 计算资源 | 8× NVIDIA A100 |
| Actor 学习率 | 5e-6 |
| KL 系数 | 0.02 |
| Clip ratio | 0.10–0.18 |
| 诊断准确性权重 | 5.0 |
| 信息增益权重 | 0.005 |
| 临床推理权重 | 0.01 |
| 格式合规权重 | 0.1 |

```bash
# GRPO 强化学习训练
bash scripts/training/train_grpo.sh

# 从基线开始训练（无 SFT 冷启动）
bash scripts/training/train_grpo_baseline.sh
```

训练过程可通过 WandB 监控，最大对话轮次 $L=10$。

### 4. 运行评估

```bash
# 基于患者 LLM 的端到端推理评估（两种模拟器）
python ragen/env/med_dialogue/evaluation/inference_fast_for_patientllm_zh_1018_3_best.py \
    --model_path /path/to/checkpoint \
    --data_path data/test.parquet \
    --output_dir results/ \
    --patient_sim psysim_std   # 或 psysim_adapt

# 支撑忠实度评估（DeepSeek-V3 评判器）
python ragen/env/med_dialogue/evaluation/evaluation_support_faithfulness.py \
    --result_path results/inference_output.json

# 语义精确评分（LLM 评估器）
python ragen/env/med_dialogue/evaluation/evaluation_for_patientllm_category_zh_optimized_best.py \
    --result_path results/inference_output.json
```

### 5. 检查点转换

```bash
# 将分布式训练检查点转换为 HuggingFace 格式
bash scripts/convert_best_checkpoint.sh
```

## 引用

如果 MIND 对您的研究有所帮助，请引用我们的工作：

```bibtex
@inproceedings{mind2025,
  title={MIND: Unified Inquiry--Diagnosis RL with Criteria-Grounded Clinical Supports for Psychiatric Consultation},
  author={},
  booktitle={Proceedings of the ACM Conference},
  year={2025}
}
```

## 致谢

- [veRL](https://github.com/volcengine/verl) — GRPO 训练基础设施
- [vLLM](https://github.com/vllm-project/vllm) — 高效 LLM 推理引擎
- [Ray](https://github.com/ray-project/ray) — 分布式计算框架
- [Qwen3](https://huggingface.co/Qwen) — 基座语言模型（Qwen3-4B/8B 用于医生智能体，Qwen3-8B 用于患者智能体）
- [Kimi-K2](https://arxiv.org/abs/2507.20534) — PRB 构建与 SFT 蒸馏教师模型
