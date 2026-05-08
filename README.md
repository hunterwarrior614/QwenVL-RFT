# qwen_rl

基于 `Qwen2.5-VL-3B-Instruct`，在 Thyme-SFT 转换得到的单图多选 VQA 数据上实现：

- SFT warm start
- PPO 训练
- GRPO 训练

当前仓库默认使用 `LoRA`，不做 full fine-tuning。

## 1. 项目现状

当前已完成：

- 数据转换脚本：`scripts/data_process/convert_thyme_sft_to_qwen_vl_rl.py`
- PPO 数据：`data/wo_thinking_thyme_single_round-00000-of-00146.qwen_vl_ppo.jsonl`
- GRPO 数据：`data/wo_thinking_thyme_single_round-00000-of-00146.qwen_vl_grpo.jsonl`
- SFT 训练脚本：`scripts/train/train_sft_qwen_vl_lora.py`
- PPO 训练脚本：`scripts/train/train_ppo_qwen_vl_lora.py`
- GRPO 训练脚本：`scripts/train/train_grpo_qwen_vl_lora.py`

基础模型目录：

`../Qwen/Qwen2.5-VL-3B-Instruct`

## 2. 数据说明

原始数据为 Thyme-SFT 的 wo_thinking_thyme_single_round-00000-of-00146.parquet

当前数据的关键事实：

- 样本数：`1242`
- 任务类型：单图、多选视觉问答
- `choice_letter` 覆盖率：`100%`
- 训练目标可以直接做 `<answer>A</answer>` 格式下的 `A/B/C/D` exact match

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
2. 再做 `PPO` 或 `GRPO`

原因很简单：

- 你的 reward 是 `exact match`，非常稀疏
- 如果直接从 base instruct 做 PPO，前期更容易只学到“格式修正”而不是“答对题”
- 先用 SFT 让模型学会稳定输出 `<answer>A</answer>`，再做 PPO / GRPO 会更稳

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

如果服务器上有多张卡，`python scripts/train/...py` 仍然通常只会使用 `cuda:0`。原因是当前训练脚本虽然使用了 `Accelerator()`，但只有在通过 `accelerate launch` 启动为多进程时，才会真正启用多卡分布式训练。

## 7. 训练

### 7.1 准备

先下载数据和基础模型，然后把 Thyme-SFT parquet 转成 RL 数据：

```bash
export THYME_FILE=data/wo_thinking_thyme_single_round-00000-of-00146.parquet
export THYME_PARQUET=../Thyme-SFT/$THYME_FILE
export QWEN_MODEL_DIR=../Qwen/Qwen2.5-VL-3B-Instruct

# 下载 SFT 数据集
modelscope download --dataset Kwai-Keye/Thyme-SFT "$THYME_FILE" --local_dir ../Thyme-SFT
# 下载模型
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct --local_dir "$QWEN_MODEL_DIR"

# 将 SFT 轨迹格式数据转换为 RL 问答格式数据
python scripts/data_process/convert_thyme_sft_to_qwen_vl_rl.py \
  --input "$THYME_PARQUET" \
  --output-dir data \
  --output-prefix wo_thinking_thyme_single_round-00000-of-00146 \
  --export-targets both \
  --image-mode first \
  --image-format data_uri
```

### 7.2 训练命令

`--max-steps` 用来做 smoke test，不加就是正式训练。

| 阶段 | 单卡 | 4 卡 |
|---|---|---|
| SFT | `python scripts/train/train_sft_qwen_vl_lora.py --config configs/sft_qwen_vl_lora.yaml [--max-steps 2]` | `accelerate launch --num_processes 4 scripts/train/train_sft_qwen_vl_lora.py --config configs/sft_qwen_vl_lora.yaml [--max-steps 2]` |
| PPO | `python scripts/train/train_ppo_qwen_vl_lora.py --config configs/ppo_qwen_vl_lora.yaml [--max-steps 1]` | `accelerate launch --num_processes 4 scripts/train/train_ppo_qwen_vl_lora.py --config configs/ppo_qwen_vl_lora.yaml [--max-steps 1]` |
| GRPO | `python scripts/train/train_grpo_qwen_vl_lora.py --config configs/grpo_qwen_vl_lora.yaml [--max-steps 1]` | `accelerate launch --num_processes 4 scripts/train/train_grpo_qwen_vl_lora.py --config configs/grpo_qwen_vl_lora.yaml [--max-steps 1]` |

PPO / GRPO 的 `model.sft_adapter_path` 要先指向你要接续的 SFT checkpoint。

## 8. 推理与结果查看

三种训练都会在 `final_eval` 阶段自动跑测试集，并生成逐样本报告：

- `outputs/sft/<run_name>/final_test_predictions.html`
- `outputs/sft/<run_name>/final_test_predictions.jsonl`
- `outputs/ppo/<run_name>/final_test_predictions.html`
- `outputs/ppo/<run_name>/final_test_predictions.jsonl`
- `outputs/grpo/<run_name>/final_test_predictions.html`
- `outputs/grpo/<run_name>/final_test_predictions.jsonl`

PPO / GRPO 还支持只跑评估：

```bash
python scripts/train/train_ppo_qwen_vl_lora.py --config configs/ppo_qwen_vl_lora.yaml --eval-only
python scripts/train/train_grpo_qwen_vl_lora.py --config configs/grpo_qwen_vl_lora.yaml --eval-only
```

训练效果主要看：

- SFT：`outputs/sft/<run_name>/metrics.jsonl`、`training_curve.png`、`train_summary.json`
- PPO / GRPO：日志中的 `reward_mean`、`accuracy`、`valid_option_rate`、`response_length_mean`、`kl_mean`
- 测试集直观效果：相应目录下的 `final_test_predictions.html`

## 9. 当前默认配置

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

### GRPO

配置文件：`configs/grpo_qwen_vl_lora.yaml`

默认关键设置：

- 输出目录：`outputs/grpo/default/`
- 从 SFT adapter 接续训练
- `load_in_4bit=true`
- `gradient_checkpointing=true`
- `per_device_prompt_batch_size=1`
- `num_generations=4`
- `per_device_minibatch_size=1`

## 10. 目录结构

```text
configs/
  sft_qwen_vl_lora.yaml
  ppo_qwen_vl_lora.yaml
  grpo_qwen_vl_lora.yaml
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
    train_grpo_qwen_vl_lora.py
src/
  qwen_vl_rl/
outputs/
  sft/
  ppo/
  grpo/
```
