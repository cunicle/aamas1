# Silent Dissent：多智能体辩论中的沉默异议

研究问题：智能体在多数压力下改口时，内部是真的被说服了，还是嘴上从众、内部仍然偏向原答案？

做法：选择题的答案是单个字母 token，我们在强制前缀 `The answer is` 之后的位置，用 lens 逐层读出 A/B/C/D 的 logit，并与嘴上说的答案对比。

## 安装

```bash
pip install -e ".[dev]"
pytest -q                      # 14 个测试，不需要下载任何东西
```

## 流程

以 `configs/qwen7b.yaml` 为例。换模型时复制一份，改 `model.name`、`out_dir`、`prereg` 三项。

| 步骤 | 命令 | 产出（`results/<run>/`） |
|---|---|---|
| 1. 独立作答（第 0 轮） | `python scripts/run_baseline.py --config C` | `baseline_all.jsonl`；按初始对/错分层后的 `baseline.jsonl` |
| 2. **预注册读出层** | `python scripts/select_layer.py --config C`，然后 **commit `prereg/*.json`** | `prereg/qwen7b.json` |
| 3. 施压实验 | `python scripts/run_pressure.py --config C [--grid main]` | `pressure_main.jsonl`、`pressure_controls.jsonl` |
| 4. 因果注入 | `python scripts/run_intervention.py --config C` | `intervention.jsonl`、`intervention_baseline.jsonl` |
| 5. 真实辩论 / 集体决策 | `python scripts/run_debate.py --config C` | `debate.jsonl` |
| 6. 分析 | `python scripts/analyze.py --config C` | `tables/*.csv`、`figures/original_top1_by_layer.png` |

第 2 步只使用第 0 轮数据。读出层、阈值和"沉默异议"的定义都写进 prereg 文件，**必须在看施压数据之前 commit**。脚本拒绝覆盖已有的 prereg 文件。

`run_pressure.py` 和 `run_intervention.py` 支持 `--limit N`，辩论的题量由 `debate.n_items` 控制。建议先用小样本跑通，再上全量。

### 离线调试（不需要 GPU，也不需要下载）

```bash
python -m silent_dissent.debug results/debug_model
for s in run_baseline select_layer run_pressure run_intervention run_debate analyze; do
  python scripts/$s.py --config configs/debug.yaml; done
```

调试模型是随机初始化的，得到的数字没有意义，只用来检查流程是否能跑通。

## 实验条件（`silent_dissent/prompts.py`）

| 条件 | 同伴怎么说 | 用途 |
|---|---|---|
| `pressure` | 全员一致给出 target | 主实验 |
| `agree` | 全员给出智能体的原答案 | 无压力对照 |
| `split` | 一半给 target，一半给原答案 | 压力强度 |
| `mention_control` | 同 pressure，同时让对照字母 M 在同伴发言中被提及的次数与原答案在上下文中出现的次数相等 | 排除"只是上下文里出现过" |
| `remove_original` | 同 pressure，但智能体自己之前的回答从上下文中删除 | 排除从上下文复制原答案 |

`target_mode`：`wrong` 表示推一个既不是原答案也不是正确答案的选项；`correct` 表示推正确答案（只对初始答错的题有效，对应有益更新）。同一道题的 target 和对照字母在所有条件下保持一致，由 `(seed, item_id)` 决定，方便做配对比较。

`mention_control` 做比较时，应该看 `original_correct == False` 这一层：此时原答案和对照字母都是错误选项，只有"是否是自己说过的答案"这一个差别。

## 指标（`silent_dissent/metrics.py`）

- **沉默异议**（在预注册层 L 上判断）：
  - strict：智能体改口了，并且原答案在各字母中仍排第 1。
  - lenient：原答案排前 2，并且 `logit(嘴上答案) − logit(原答案) < δ`。
- **决策翻转层**：从哪一层开始，嘴上答案的 logit 一直高于原答案，直到最后一层。
- **注入恢复率**：在层 L 的答案位置加上 `alpha·‖h‖·d`，d 是对比方向（该字母方向减去其余字母方向的均值）。对照方向包括多数答案方向、其他字母方向、随机方向。`intervention_baseline` 在无压力的题目上做同样的注入，用来检查注入是否只是让输出偏向某个字母。
- **集体决策**：真实辩论中比较三种投票规则的群体准确率：嘴上答案投票、层 L 的 lens top-1 投票、第 0 轮独立答案投票。

## Lens（`silent_dissent/lenses.py`）

- `logit`：`unembed(final_norm(h))`，已实现，作为基线。测试保证它在最后一层与模型输出一致。
- `affine`：每层一个线性映射 `A_l h + b_l`，再接 logit lens。tuned lens 这一类方法都可以导出成这个格式直接使用。
- `jlens`：**TODO**。实现 `logits(h, layer)` 和 `direction(token_id, layer)` 两个方法即可，其余代码不需要改。如果 J-lens 本身就是逐层的线性映射，直接导出成 affine 格式最快。

## 代码结构

```
silent_dissent/
  data.py         MMLU / ARC / CSQA → 统一的 MCQItem（字母标签）
  prompts.py      DebateState：多轮对话构造，5 种条件
  model.py        模型加载、逐层残差捕获、字母读出
  lenses.py       Lens 接口
  intervene.py    注入 hook 和方向构造
  experiments.py  baseline / pressure / intervention 主循环，按长度分批
  debate.py       真实 N 智能体辩论
  metrics.py      所有指标（只读 JSONL，不需要模型）
  debug.py        离线调试用的小模型
scripts/          每个步骤一个命令行脚本
configs/          qwen7b.yaml（主实验）、debug.yaml
prereg/           预注册文件（需要 commit）
tests/            单元测试和端到端冒烟测试
```

## 分工建议

- **A（实验管线）**：`prompts.py`、`experiments.py`、`debate.py`，负责跑实验。
- **B（lens / 干预）**：`lenses.py` 中的 JLens、`intervene.py`、`select_layer.py`。
- **C（数据 / 分析）**：`data.py`、`metrics.py`、`analyze.py`、画图。

`metrics.py` 只依赖 JSONL 文件，所以 C 可以先用调试模型的输出开发分析代码，不需要等 GPU 实验跑完。

## 已知注意事项

- 默认只保留 4 个选项的题，保证各条件的随机水平相同。CSQA 有 5 个选项，而且 test 集没有标签，使用时要设置 `split: validation` 和 `n_choices: 5`。
- `require_letter_format: true` 会丢掉第 0 轮首选 token 不是字母的题。
- 用 logit lens 时，τ=0.9 选出的层通常比较靠后，这是一个偏严格的检验。如果还想报告中间层，请把第二个层号也写进 prereg，不要事后挑层。
- 带理由的同伴（`with_reason`）使用同一个模型生成一句论证，缓存在 `reasons.json`。
- 在 `intervention_baseline` 中，`majority` 方向与 `original` 方向相同，因为无压力时没有多数答案。
