# A09 本地检索可行性实验：复现与证据

日期：2026-11-05｜状态：本地离线实验已执行；XIPU AI 平台未执行。作者：组员 A 的 AI 辅助实验，待本人和团队复核。结果、样本和代码不属于生产知识库。

## 实验对象

比较本地实现的 BM25、中文语义向量、两者的 RRF 融合。BM25 使用正 IDF 公式（`k1=1.5, b=0.75`）、jieba 及型号完整词项；向量使用 [Xenova 的 BGE-small-zh-v1.5 量化 ONNX](https://huggingface.co/Xenova/bge-small-zh-v1.5/blob/main/onnx/model_quantized.onnx)，CPU 推理，CLS pooling、L2 归一化；按 [BAAI 模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5)给短查询加中文检索前缀。RRF `c=60`，每路最多取前 20 个候选。三种方法在本地实现，不代表 XIPU AI 的内部实现。

1. **虚构语料**：8 份人工编写的资料、18 个问题（16 个有证据题、2 个无答案题），增加 0、40、200 个程序生成的虚构干扰单元；随机种子 11、19、29，另比较开启与关闭本地项目字段筛选，共 18 个条件。每个条件运行三种检索器，共 972 条原始记录。`observed_results.json` 与 `observed_repeat.json` 是同一固定实验的两次实际运行。
2. **团队文件语料**：来自仓库的 B01—B05、A03、A06 共 7 份文件，按 Markdown 二级标题切成 41 个本地段落；12 个问题，3 种方法，36 条记录，重复运行一次。另做每块前加来源编号的事后对照。检索 B 的原文是为了查证其表述，不表示认可其中可能有问题的建议。
3. **单题排查 R10**：对“比较 B03 与 A06”一题，在单次查询取前 6 个片段与拆成两个查询各取前 3 个片段的条件下对照。拆问是看到失败后、参考目标段落手动设计的，包含已知答案线索；它不是盲测或独立留出集，也没有实现自动拆问。

来源、原文依据和问题标注见 `cases.json`、`real_source_cases.json`；生成噪声的函数和随机种子见 `run_benchmark.py`，文件/模型/数据哈希在结果中。人工标注由实验设计者完成，尚未经过独立审阅。虚构电压及器件编号不得用于真实设备。

## 主要结果

在**开启本地项目字段筛选**、200 个虚构干扰单元时，16 道有证据题各在 3 个不同种子下运行一次，所以每种方法统计 48 条题目运行记录。8 份基础文件加 200 个干扰单元为 208 个本地索引单元，**不是 208 个已上传到 XIPU AI 的文件**。

| 方法 | 首个结果相关 Hit@1 | Top 3 覆盖全部必要证据 | 3 道多证据题的 Top 3 全覆盖 |
|---|---:|---:|---:|
| BM25 | 45/48 | 48/48 | 9/9 |
| 中文向量 | 32/48 | 33/48 | 6/9 |
| RRF 混合 | 39/48 | 39/48 | 6/9 |

本地筛选是数据字段条件，不是身份授权或安全隔离；BM25 的文档频率和平均长度在完整测试语料上计算，然后按项目筛选排序结果，这不等同于为每个项目单独建索引。0 个干扰时三个种子的语料相同，不构成三个独立样本。无答案题不计入以上分母；这些测试没有运行语言模型，不能验证拒答或最终答案。

在 7 份现有团队文件的 41 个本地段落上，12 个问题的 Top 3 全部证据覆盖均为 **11/12**：BM25 的 Hit@1 为 11/12，中文向量为 10/12，RRF 混合为 11/12。三种方法都漏掉比较 B03 与 A06 的 R10 所需的一个来源；给每块加文件编号后，该题仍未通过 Top 3 检查，整体全证据覆盖仍为 11/12。

对 R10 进行事后诊断时，单个原问题取 Top 6 时三种方法都未覆盖两份证据；将问题手动拆为两个有针对性的问题后，不带源文件筛选时每个子问题各取 Top 3，三种方法在合并结果中均覆盖了两份证据。按文件编号分别筛选时也覆盖了两份。两次调用的成本与单次不同，且这是围绕已知失败题的单题分析，不能证明自动拆问或全局效果。

样本有限、人工合成干扰较规则、标注者与测试设计者相同、BGE 是一个特定的中文模型、模型及切块方式与平台可能不同；这些数字不能推断真实用户准确率、统计显著性、生产延迟或 XIPU AI 性能。短列表的 Hit@1 只说明首个相关段落，不证明多证据齐全。真实文档和虚构文档的内容可能被索引截断；本实验会在输入达到 512 token 时停止，而不是悄悄截断。

本实验的 BM25 为便于检查而逐个扫描文档并重新统计词频，向量侧也使用全量相似度计算；两者没有实现生产级倒排索引或近似向量索引。因此实验仅验证本地评分、融合和小语料排序机制，不能据此证明处理大规模知识库时的速度、成本或可扩展性；也不应把实验脚本直接作为生产检索服务。

## 文件

- `core.py`、`test_core.py`：BM25、RRF 和多证据指标。BM25 与课件例题及 30 组随机输入下的 `rank-bm25` 库结果进行了对照（统一 IDF 定义后）。
- `run_benchmark.py`、`cases.json`：虚构语料实验代码与人工标注。
- `observed_results.json`、`observed_repeat.json`：两次虚构实验的原始结果和语料清单；`audit_repeat.py`、`repeat_audit.json`：重复排序、筛选及指标复核。
- `real_source_benchmark.py`、`real_source_cases.json`：团队文档实验；`real_source_results.json`、`real_source_repeat.json`、`real_source_with_id.json` 与 `real_source_audit.json`：原始数据、重复结果、文件编号对照及核查。
- `r10_diagnostic.py` 与两个结果文件：R10 事后拆问诊断及复测。
- [A09 项目方案](../../../01_official_sources/retrieval_design/A09_BM25混合检索_可行性方案.md)：结论、决策与平台验证步骤。

## 在 Windows 上复现

需要 Python 3.12、CPU、网络下载约 24 MB 的公开模型以及这些 Python 依赖。创建独立环境后，从本目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force model | Out-Null
Invoke-WebRequest 'https://huggingface.co/Xenova/bge-small-zh-v1.5/resolve/75c43b0/onnx/model_quantized.onnx' -OutFile model/model_quantized.onnx
Invoke-WebRequest 'https://huggingface.co/Xenova/bge-small-zh-v1.5/resolve/75c43b0/tokenizer.json' -OutFile model/tokenizer.json
.\.venv\Scripts\python.exe -m unittest -v test_core
.\.venv\Scripts\python.exe run_benchmark.py --model-dir model --output observed_local.json
.\.venv\Scripts\python.exe real_source_benchmark.py --model-dir model --output real_source_local.json
.\.venv\Scripts\python.exe r10_diagnostic.py --model-dir model --output r10_local.json
.\.venv\Scripts\python.exe audit_repeat.py
.\.venv\Scripts\python.exe audit_real.py
```

脚本核对模型权重 SHA-256 `15b717c382bcb518ba457b93ea6850ede7f4f1cd8937454aa06972366cd19bcc` 及 tokenizer 哈希。模型和依赖不包含在 Git 仓库；如链接或版本变化导致哈希不符，应先核对来源，不要跳过校验。以上复核脚本检查提交的原始记录；不同平台、库版本或 CPU 可能产生细小数值差异，须实际比较结果。原始文件不含真实学生私人数据。
