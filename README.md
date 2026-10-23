# LLM-Password

通过 API 研究大语言模型生成弱口令的倾向，比较提示策略、输出强度与模型指纹，并为长期观测提供采集基础。

当前具备批量采集与离线分析代码；**尚未实现完整的长期定时监测服务**。详细缺口见 [项目检查](docs/project-audit.md)。

## 安装和采集

Python 3.9+；以下命令在项目根目录执行：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cp env.example .env
```

Windows 复制配置使用 `Copy-Item env.example .env`。在本地 `.env` 填写 OpenRouter 或 SiliconFlow API key。模型 ID 由你的供应商账户可用模型决定。

```bash
python -m src.cli.generate --help
python -m src.cli.generate --provider openrouter --model YOUR_PROVIDER_MODEL_ID --prompt-type few-shot --examples 'password123;;qwerty123;;welcome123' --count 20 --output data/llm_passwords.csv
```

提示类型：`benign`（正常要求）、`few-shot`（示例诱导）、`jailbreak`（弱模式提示）。供应商仅支持 `openrouter` 和 `siliconflow`；配置模板中的其他供应商尚无客户端实现。

输出 CSV 包含口令、模型、提示词、时间戳和评分相关字段。仅用于研究生成的实验口令，不应将其用于真实账户。API 调用会消耗供应商额度。

## 目录

- `src/clients/`：API 调用、重试与文本提取。
- `src/prompts/`：提示构建。
- `src/utils/`：CSV 与评分工具。
- `src/cli/generate.py`：批量采集入口。
- `scripts/`：数据处理、分析、特征和分类训练脚本，见 [脚本说明](scripts/README.md)。
- `docs/`：实验设计与项目完整性检查。

离线分析依赖另装：`python -m pip install -r requirements-analysis.txt`。DNN 脚本还需要适合目标平台的 TensorFlow。分析脚本依赖本地采集数据，分类预测需要先训练模型；数据、权重与历史图表不随代码包发布。

现有代码应从根目录以模块运行；pyproject.toml 的安装布局尚未校正，不建议以 `pip install .` 作为本项目部署方式。

## 代码发布范围

仓库只包含源代码、依赖清单、无密钥配置模板及必要运行文档。不包含 `.env`、实验口令数据、训练权重、项目书、压缩存档、LaTeX 编译文件和本机登录凭证。

目前已完成 Python 源文件语法检查；研究 API 和历史分析流程尚未通过完整端到端测试。
