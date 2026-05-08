# qwen_rl

基于 `Qwen2.5-VL-3B-Instruct`，在 Thyme-SFT 转换得到的单图多选 VQA 数据上实现：

- SFT warm start
- PPO 训练
- 后续可扩展到 GRPO

当前仓库默认使用 `LoRA`，不做 full fine-tuning。

## 1. 项目现状

当前已完成：

- 数据转换脚本：`scripts/data_process/convert_thyme_sft_to_qwen_vl_rl.py`
- PPO 数据：`data/wo_thinking_thyme_single_round-00000-of-00146.qwen_vl_ppo.jsonl`
- GRPO 数据：`data/wo_thinking_thyme_single_round-00000-of-00146.qwen_vl_grpo.jsonl`
- SFT 训练脚本：`scripts/train/train_sft_qwen_vl_lora.py`
- PPO 训练脚本：`scripts/train/train_ppo_qwen_vl_lora.py`

基础模型目录：

`../Qwen/Qwen2.5-VL-3B-Instruct`

## 2. 数据说明

原始数据为 Thyme-SFT 的 wo_thinking_thyme_single_round-00000-of-00146.parquet

当前数据的关键事实：

- 样本数：`1242`
- 任务类型：单图、多选视觉问答
- `choice_letter` 覆盖率：`100%`
- 训练目标可以直接做 `A/B/C/D` exact match

两份 RL 数据的区别：

- PPO 文件使用 `messages`
- GRPO 文件使用 `prompt`

## 3. 推荐环境

```bash
conda create -n qwen_rl python=3.11 -y
conda activate qwen_rl

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install \
  transformers==4.56.2 \
  accelerate==1.13.0 \
  trl==1.3.0 \
  peft==0.19.1 \
  datasets==4.8.5 \
  bitsandbytes==0.49.2 \
  sentencepiece \
  matplotlib \
  plotly
```

## 4. 文件管理规范

训练不会修改原始 `Qwen2.5-VL-3B-Instruct` 权重。

原始模型目录只作为输入：

- tokenizer
- processor
- base model weights

训练输出统一写到：

```text
data/
outputs/
  sft/
  ppo/
  grpo/
```

推荐规则：

- SFT 输出到 `outputs/sft/<run_name>/`
- PPO 输出到 `outputs/ppo/<run_name>/`
- GRPO 输出到 `outputs/grpo/<run_name>/`
- 不要覆盖原始 Qwen 目录
- 项目名已经是 `qwen_rl`，不要再套一层同名目录

## 5. 最佳实践训练路线

推荐路线：

1. 先做 `SFT warm start`
2. 再做 `PPO`

原因很简单：

- 你的 reward 是 `exact match`，非常稀疏
- 如果直接从 base instruct 做 PPO，前期更容易只学到“格式修正”而不是“答对题”
- 先用 SFT 让模型学会稳定输出 `A/B/C/D`，再做 PPO 会更稳

## 6. 训练前检查

先确认 GPU 可见：

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## 7. 执行命令

### 7.0 准备
进入项目目录：
```bash
git clone Qwen-RFT
cd Qwen-RFT
```
数据集下载：
```bash
modelscope download --dataset Kwai-Keye/Thyme-SFT data/wo_thinking_thyme_single_round-00000-of-00146.parquet --local_dir ../Thyme-SFT
```
基础模型下载：
```bash
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct --local_dir ../Qwen/Qwen2.5-VL-3B-Instruct
```

默认配置使用的模型路径就是：`../Qwen/Qwen2.5-VL-3B-Instruct`

### 7.1 一开始的数据生成

先把 Thyme-SFT 的 parquet 转成 PPO / GRPO 数据：

```bash
python scripts/data_process/convert_thyme_sft_to_qwen_vl_rl.py \
  --input ../Thyme-SFT/data/wo_thinking_thyme_single_round-00000-of-00146.parquet \
  --output-dir data \
  --output-prefix wo_thinking_thyme_single_round-00000-of-00146 \
  --export-targets both \
  --image-mode first \
  --image-format data_uri
```

生成结果会放在 `data/` 下：

- `data/wo_thinking_thyme_single_round-00000-of-00146.qwen_vl_ppo.jsonl`
- `data/wo_thinking_thyme_single_round-00000-of-00146.qwen_vl_grpo.jsonl`
- `data/wo_thinking_thyme_single_round-00000-of-00146.qwen_vl_rl_manifest.json`

### 7.2 先做 SFT warm start

最小 smoke test：

```bash
python scripts/train/train_sft_qwen_vl_lora.py \
  --config configs/sft_qwen_vl_lora.yaml \
  --max-steps 2
```

正式训练：

```bash
python scripts/train/train_sft_qwen_vl_lora.py \
  --config configs/sft_qwen_vl_lora.yaml
```

### 7.3 再做 PPO

先确认 `configs/ppo_qwen_vl_lora.yaml` 中的：

- `model.sft_adapter_path`

已经指向你要接续的 SFT checkpoint，例如：

```yaml
sft_adapter_path: outputs/sft/default/checkpoint-2/adapter
```

PPO smoke test：

```bash
python scripts/train/train_ppo_qwen_vl_lora.py \
  --config configs/ppo_qwen_vl_lora.yaml \
  --max-steps 1
```

PPO 正式训练：

```bash
python scripts/train/train_ppo_qwen_vl_lora.py \
  --config configs/ppo_qwen_vl_lora.yaml
```

## 8. 当前默认配置

### SFT

配置文件：`configs/sft_qwen_vl_lora.yaml`

默认关键设置：

- 输出目录：`outputs/sft/default/`
- `train_size=1000`
- `eval_size=121`
- `test_size=121`
- `load_in_4bit=true`
- `gradient_checkpointing=true`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=4`

### PPO

配置文件：`configs/ppo_qwen_vl_lora.yaml`

默认关键设置：

- 输出目录：`outputs/ppo/default/`
- 从 SFT adapter 接续训练
- `load_in_4bit=true`
- `gradient_checkpointing=true`
- `per_device_prompt_batch_size=1`
- `per_device_minibatch_size=1`

## 9. 如何查看训练效果

### SFT

SFT 训练结束后，重点看：

- `outputs/sft/default/metrics.jsonl`
- `outputs/sft/default/training_curve.png`
- `outputs/sft/default/train_summary.json`

含义：

- `metrics.jsonl`：逐步记录 `train loss` 和 `eval metrics`
- `training_curve.png`：训练曲线图
- `train_summary.json`：本次训练汇总

说明：

- `checkpoint-XX/` 主要用于恢复训练或接 PPO
- `checkpoint-XX/` 本身不直观展示训练效果

### PPO

PPO 训练结束后，重点看：

- `reward_mean`
- `accuracy`
- `valid_option_rate`
- `response_length_mean`
- `kl_mean`
- `value_loss`

这些会直接打印在训练日志中。

## 10. 做完 SFT 之后具体做什么

做完 SFT 后，下一步就是：

1. 选择一个 SFT checkpoint
2. 把这个 checkpoint 的 `adapter/` 路径填到 `configs/ppo_qwen_vl_lora.yaml` 的 `model.sft_adapter_path`
3. 先跑 PPO smoke test
4. smoke test 正常后，再跑正式 PPO

简化版流程：

```bash
export PROJECT_DIR=$(pwd)
export THYME_PARQUET=../Thyme-SFT/data/wo_thinking_thyme_single_round-00000-of-00146.parquet
export QWEN_MODEL_DIR=../Qwen/Qwen2.5-VL-3B-Instruct

python scripts/data_process/convert_thyme_sft_to_qwen_vl_rl.py \
  --input "$THYME_PARQUET" \
  --output-dir data \
  --output-prefix wo_thinking_thyme_single_round-00000-of-00146 \
  --export-targets both \
  --image-mode first \
  --image-format data_uri

python scripts/train/train_sft_qwen_vl_lora.py \
  --config configs/sft_qwen_vl_lora.yaml

python scripts/train/train_ppo_qwen_vl_lora.py \
  --config configs/ppo_qwen_vl_lora.yaml \
  --max-steps 1

python scripts/train/train_ppo_qwen_vl_lora.py \
  --config configs/ppo_qwen_vl_lora.yaml
```

## 11. 目录结构

```text
configs/
  sft_qwen_vl_lora.yaml
  ppo_qwen_vl_lora.yaml
data/
  wo_thinking_thyme_single_round-00000-of-00146.qwen_vl_ppo.jsonl
  wo_thinking_thyme_single_round-00000-of-00146.qwen_vl_grpo.jsonl
  wo_thinking_thyme_single_round-00000-of-00146.qwen_vl_rl_manifest.json
scripts/
  data_process/
    convert_thyme_sft_to_qwen_vl_rl.py
  train/
    train_sft_qwen_vl_lora.py
    train_ppo_qwen_vl_lora.py
src/
  qwen_vl_rl/
outputs/
  sft/
  ppo/
  grpo/
```
