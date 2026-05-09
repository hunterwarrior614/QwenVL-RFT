# Refactor Plan

本文件用于记录当前仓库的分阶段重构计划，目标是：

1. 提升可读性
2. 提升模块复用率
3. 在不破坏现有功能正确性的前提下，逐步收敛重复实现

本次重构遵循以下原则：

- 不删除你现有的中文注释。已有注释默认保留，仅在必要时补充少量辅助注释。
- 每一步重构都先保证正确性，再考虑结构美化。
- 每一步重构都应尽量小而可验证，避免一次性大改。
- 每完成一个阶段，都应运行对应测试或最小验证命令。
- 如果发现两个实现“看起来相同，但语义已经漂移”，优先统一行为，再统一结构。

## 当前识别出的高重复区域

### 1. 答案/选项字母解析

当前仓库中存在多处“从模型输出或标注文本中提取选项字母”的实现，且行为不完全一致：

- `scripts/data_process/convert_thyme_sft_to_qwen_vl_rl.py`
- `src/qwen_vl_rl/reward.py`
- `scripts/train/train_sft_qwen_vl_lora.py`
- `src/qwen_vl_rl/reports.py`

风险：

- 数据转换、奖励计算、训练评估、测试报告对同一条输出的理解可能不一致。

### 2. 图像提取与消息遍历

当前仓库中存在多处“从 messages 中提取第一张图像”的实现：

- `src/qwen_vl_rl/data.py`
- `src/qwen_vl_rl/sft.py`
- `src/qwen_vl_rl/reports.py`

风险：

- 修改消息格式后，多个实现需要同步更新。

### 3. Tensor 搬运到 device

当前仓库中存在多处“把 batch 中 tensor 搬到设备上”的实现：

- `src/qwen_vl_rl/ppo.py`
- `src/qwen_vl_rl/reports.py`
- `scripts/train/train_sft_qwen_vl_lora.py`

风险：

- 小差异会导致行为难以统一维护。

### 4. Collator 共享逻辑

当前仓库中的 `QwenVLPPOCollator` 与 `QwenVLSFTCollator` 有较多重复逻辑：

- tokenizer padding 设置
- prompt 构造
- 图像解码
- `processor(...)` 调用
- 批次元数据整理

风险：

- 相同输入约定在 SFT 和 PPO/GRPO 路线下逐渐分叉。

### 5. 训练脚本外壳

当前训练脚本存在较大面积的平行实现：

- `scripts/train/train_sft_qwen_vl_lora.py`
- `scripts/train/train_ppo_qwen_vl_lora.py`
- `scripts/train/train_grpo_qwen_vl_lora.py`

重复点包括：

- metric 写入
- 曲线绘制
- checkpoint 保存
- evaluate 流程
- 日志打印

风险：

- 某条训练路径修复后，其他路径仍保留旧行为。

### 6. 配置与模型初始化

当前 SFT 与 PPO/GRPO 使用两套不同配置加载方式，量化配置和 LoRA target 解析也有重复实现。

风险：

- 参数字段演进时容易遗漏一侧。

## 分阶段执行计划

### Phase 0: 建立护栏

目标：

- 在重构前用测试锁住关键行为

任务：

1. 补充或加强以下测试：
   - 裸字母输出与 `<answer>...</answer>` 输出的解析行为
   - reward 解析行为
   - report 中 `pred_letter` 提取行为
   - SFT eval 中的解析行为
2. 如果测试无法覆盖完整链路，则补充最小集成测试

验收标准：

- 新增测试全部通过
- 能明确说明“当前行为是什么”

### Phase 1: 统一答案解析

目标：

- 全仓库只保留一套核心答案解析逻辑

建议新增模块：

- `src/qwen_vl_rl/answering.py`

建议抽出的公共函数：

- `extract_answer_tag_content(text: str) -> str | None`
- `extract_choice_letter(text: str, require_answer_tag: bool = False) -> str | None`
- `format_choice_answer(letter: str, with_answer_tag: bool = True) -> str`

改动策略：

1. 先新增共享函数
2. 让旧调用点逐步改为调用共享函数
3. 不立刻删除旧包装函数，必要时先做兼容层

验收标准：

- 数据转换、reward、SFT eval、report 在同一输入上的解析行为可预测且一致

### Phase 2: 统一底层通用 helper

目标：

- 合并最基础、最机械的重复逻辑

建议收敛内容：

- `move_tensors_to_device`
- `extract_first_image_uri`
- `decode_first_image_from_messages`
- `build_prompt_text_from_messages`

建议位置：

- `src/qwen_vl_rl/utils.py`
- 或新建 `src/qwen_vl_rl/message_utils.py`

验收标准：

- 至少减少 2 到 3 处重复实现
- 不改变原有训练/测试行为

### Phase 3: 收敛 Collator 共享逻辑

目标：

- 降低 SFT/PPO/GRPO 预处理逻辑分叉

建议做法：

1. 先抽纯函数，而不是直接强行做类继承
2. 抽出：
   - tokenizer padding 初始化 helper
   - prompt/image 收集 helper
   - processor 输入构建 helper
3. 在行为稳定后，再判断是否需要基类

验收标准：

- `QwenVLSFTCollator` 和 `QwenVLPPOCollator` 只保留各自真正不同的部分

### Phase 4: 收敛训练脚本公共外壳

目标：

- 合并训练脚本中高重复、低算法相关性的代码

建议优先抽离：

- `append_metric`
- `render_training_curve`
- `log_metrics`
- checkpoint metadata 写入
- 通用 generation eval helper

不建议在这一阶段做的事：

- 不要急于把 SFT/PPO/GRPO 训练主循环强行合成一个大 Trainer

验收标准：

- 训练脚本行数显著下降
- 各脚本职责更集中

### Phase 5: 统一配置与模型初始化

目标：

- 收敛配置体系和模型初始化路径

建议方向：

1. SFT 配置逐步迁移到 dataclass 风格
2. 量化配置统一从共享 helper 构造
3. LoRA target module 解析统一
4. adapter/checkpoint 路径解析统一

验收标准：

- 新增字段时，只需要维护一套主入口

## 推荐执行顺序

推荐严格按以下顺序推进：

1. Phase 0: 建立护栏
2. Phase 1: 统一答案解析
3. Phase 2: 统一底层 helper
4. Phase 3: 收敛 Collator
5. Phase 4: 收敛训练脚本外壳
6. Phase 5: 统一配置与模型初始化

## 每一步的操作要求

每个阶段开始前：

- 明确本阶段不修改的范围
- 先阅读相关测试

每个阶段改动后：

- 运行最小必要测试
- 若影响训练/推理链路，补一次最小人工验证

如果某一步发现行为不一致：

- 先停在“行为对齐”层
- 不要继续做结构合并

## 当前建议的下一步

建议从 Phase 0 和 Phase 1 开始：

1. 先补测试，固定当前答案解析行为
2. 再抽 `answering.py`
3. 然后让 `reward.py`、`reports.py`、SFT eval、数据转换脚本逐步改用共享实现

## 当前进度快照

以下内容记录截至当前轮次已经完成的重构收敛点，便于后续继续推进时快速对齐上下文。

### 已完成：Phase 0

已经补充并固定了以下测试护栏：

- `tests/test_answering.py`
- `tests/test_reward.py`
- `tests/test_reports.py`
- `tests/test_convert_thyme_sft_to_qwen_vl_rl.py`
- `tests/test_collator_utils.py`
- `tests/test_training_io.py`

当前基线测试状态：

- `22 passed`

### 已完成：Phase 1 第一轮

已经新增共享答案解析模块：

- `src/qwen_vl_rl/answering.py`

当前已接入共享答案解析逻辑的调用点包括：

- `src/qwen_vl_rl/reward.py`
- `src/qwen_vl_rl/reports.py`
- `scripts/train/train_sft_qwen_vl_lora.py`
- `scripts/data_process/convert_thyme_sft_to_qwen_vl_rl.py`

当前策略说明：

- 共享模块负责核心解析能力
- 旧入口函数在必要处仍然保留，作为兼容层
- 严格模式与非严格模式通过参数显式表达，不再依赖隐式模块语义

### 已完成：Phase 2

已经收敛到底层通用 helper 的共享模块：

- `src/qwen_vl_rl/utils.py`

当前已共享的能力包括：

- `move_tensors_to_device`
- `extract_first_image_uri`
- `decode_data_uri_image`
- `resize_image_longest_edge`
- `decode_first_image_from_messages`

当前已接入这些 helper 的调用点包括：

- `src/qwen_vl_rl/reports.py`
- `src/qwen_vl_rl/ppo.py`
- `src/qwen_vl_rl/data.py`
- `src/qwen_vl_rl/sft.py`
- `scripts/train/train_sft_qwen_vl_lora.py`

### 已完成：Phase 3 前两小段

已经新增 collator 共享 helper 模块：

- `src/qwen_vl_rl/collator_utils.py`

当前已共享的能力包括：

- `prepare_tokenizer_for_padding`
- `build_generation_prompt_texts`
- `decode_prompt_images`
- `build_processor_inputs`
- `collect_prompt_metadata`

当前已接入这些 helper 的调用点包括：

- `src/qwen_vl_rl/data.py` 中的 `QwenVLPPOCollator`
- `src/qwen_vl_rl/sft.py` 中的 `QwenVLSFTCollator`

当前状态说明：

- 已经把公共骨架抽出
- 各自真正不同的逻辑仍保留在原 collator 中
- 暂未强行引入复杂继承层次

### 已完成：Phase 4 前三小段

已经新增训练外壳共享模块：

- `src/qwen_vl_rl/training_io.py`

当前已共享的能力包括：

- `append_metric`
- `log_metrics`
- `prepare_checkpoint_dir`
- `save_optimizer_and_training_state`

当前已接入这些 helper 的调用点包括：

- `scripts/train/train_ppo_qwen_vl_lora.py`
- `scripts/train/train_grpo_qwen_vl_lora.py`
- `scripts/train/train_sft_qwen_vl_lora.py`

当前状态说明：

- PPO/GRPO 的训练外壳收敛程度最高
- SFT 已经共享 checkpoint 外壳，但图表与其余流程仍保留自身实现
- `render_training_curve(...)` 暂未收敛，原因是三条训练路径展示指标不同，过早抽象收益不一定高

### 已完成：Phase 5 前三小段

已经新增模型初始化共享模块：

- `src/qwen_vl_rl/modeling_common.py`

当前已共享的能力包括：

- `get_torch_dtype`
- `build_quantization_config_from_fields`
- `match_module_names`
- `resolve_lora_target_modules`

当前已接入这些 helper 的调用点包括：

- `src/qwen_vl_rl/modeling_ppo.py`
- `scripts/train/train_sft_qwen_vl_lora.py`

另外，配置路径归一化已经收敛到：

- `src/qwen_vl_rl/utils.py`

当前已共享的配置路径 helper 包括：

- `resolve_config_paths_in_dict`
- `resolve_object_paths`

当前已接入这些 helper 的调用点包括：

- `scripts/train/train_sft_qwen_vl_lora.py`
- `scripts/train/train_ppo_qwen_vl_lora.py`
- `scripts/train/train_grpo_qwen_vl_lora.py`

当前状态说明：

- 量化配置构造不再有双实现
- LoRA target module 解析不再有双实现
- 训练入口的路径归一化不再各自手写
- SFT 仍未迁移到 dataclass 配置体系，但维护成本已经下降

### 文件数量控制说明

当前重构过程中新增的核心共享文件为：

- `src/qwen_vl_rl/answering.py`
- `src/qwen_vl_rl/collator_utils.py`
- `src/qwen_vl_rl/training_io.py`
- `src/qwen_vl_rl/modeling_common.py`

这些文件都已经承载了跨模块共享职责，不是临时堆叠出来的过渡文件。

测试文件方面，后续遵循以下策略：

- 优先向已有测试文件追加测试
- 仅在主题明显独立时新增测试文件
- 如果某些测试只服务于一次性迁移、且共享模块已被更稳定测试覆盖，再考虑删除

## 当前剩余的主要重复源

### 1. 配置与模型初始化双体系

这是当前最值得优先处理的高重复区：

- `scripts/train/train_sft_qwen_vl_lora.py`
- `src/qwen_vl_rl/config.py`
- `src/qwen_vl_rl/modeling_ppo.py`

残留问题包括：

- SFT 仍使用 dict/YAML loader
- PPO/GRPO 使用 dataclass config
- adapter 路径处理仍有部分分叉

说明：

- 量化配置构造重复已经收敛
- LoRA target module 解析重复已经收敛
- 路径归一化重复已经收敛
- 剩下更核心的是配置体系本身是否要统一

### 2. 训练曲线绘制

`render_training_curve(...)` 仍分别存在于：

- `scripts/train/train_sft_qwen_vl_lora.py`
- `scripts/train/train_ppo_qwen_vl_lora.py`
- `scripts/train/train_grpo_qwen_vl_lora.py`

说明：

- 这部分确实重复
- 但三条路径图表指标并不完全相同
- 建议放在配置/模型初始化之后，除非后续对图表输出有统一需求

### 3. 训练主循环外壳

PPO/GRPO/SFT 的主循环结构仍有相似部分，但目前不建议继续强行合并。

原因：

- 算法相关逻辑已经开始明显分叉
- 若过早抽象，容易损失可读性
- 当前项目更需要的是“明确边界”，而不是“一个超大共享 Trainer”

## 当前推荐的下一步

推荐先进行一次“低风险收尾 + 是否继续统一配置体系”的判断：

1. 继续清理已经失去调用价值的兼容包装
2. 保持测试全绿
3. 再决定是否让 SFT 迁移到 dataclass 配置体系

如果继续深入 Phase 5，仍建议保持当前风格：

- 先抽共享 helper
- 再做薄迁移
- 最后才考虑删除兼容层
