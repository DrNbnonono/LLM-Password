# Scripts Directory - 脚本说明文档

本目录包含了项目中所有的数据处理、分析、模型训练和评估脚本。所有脚本均按照阶段和任务编号命名，便于追溯和复现。

---

## 📋 目录结构

```
scripts/
├── 数据准备阶段
│   ├── batch_convert_txt_to_csv.py          # TXT转CSV批处理
│   ├── check_missing_fewshot.py             # 检查缺失的少样本数据
│   ├── collect_fewshot_for_model.py         # 收集少样本提示数据
│   ├── split_api_csv_auto.py                # 拆分API收集的CSV
│   ├── cleanup_redundant_files.py           # 清理冗余文件
│   └── check_collection_progress.py         # 检查数据收集进度
│
├── 阶段一：口令强度评估
│   ├── add_entropy_to_csv.py                # 添加熵值计算
│   ├── 1-1-entropy_analysis.py              # 1.1 口令熵分析与可视化
│   ├── add_zxcvbn_to_csv.py                 # 添加zxcvbn评估
│   ├── 1-2-zxcvbn_analysis.py               # 1.2 zxcvbn分析与可视化
│   └── 1-3-bias_analysis.py                 # 1.3 倾向性指标分析与可视化
│
├── 阶段二：LLM指纹识别
│   ├── 2-1-prepare_classification_dataset.py  # 2.1 数据集构建
│   ├── 2-2-feature_engineering.py             # 2.2 特征工程（39维特征）
│   ├── 2-3-train_fingerprint_models.py        # 2.3 模型训练（主版本）
│   ├── 2-3-train_fingerprint_models_v2.py     # 2.3 模型训练（实验版本）
│   ├── 2-3-train_models.py                    # 2.3 模型训练（旧版本，已弃用）
│   ├── 2-4-analyze_and_visualize_results.py   # 2.4 结果分析与可视化
│   ├── 2-5-classify_passwords.py              # 2.5 批量口令分类（旧版本）
│   ├── 2-5-classify_passwords_v2.py           # 2.5 批量口令分类（多分类）
│   ├── 2-6-demo_fingerprint_identification.py # 2.6 完整演示程序
│   └── 2-6-predict_single_password.py         # 2.6 单口令预测工具★
│
└── 辅助工具
    ├── check_data_stats.py                  # 数据统计检查
    └── (其他工具脚本)
```

---

## 🚀 快速开始

### 环境要求

```bash
# Python 3.10+
# 依赖包（见 requirements.txt）
pandas
numpy
scikit-learn
tensorflow (for GPU acceleration)
joblib
matplotlib
seaborn
zxcvbn
```

### 基本用法

#### 1. 单口令预测（推荐）

```bash
# 预测单个口令的来源模型
python scripts/2-6-predict_single_password.py <password>

# 示例
python scripts/2-6-predict_single_password.py mnbvcxz
```

**输出**：
- 控制台显示：Top-3预测结果和置信度（带可视化进度条）
- 文件保存：`data/classification/predictions/prediction_YYYYMMDD_HHMMSS.txt`

**输出示例**：
```
【RandomForest Model】
  Rank 1: Doubao        ██████████████████████████████████ 66.24%
  Rank 2: gemma_2_9b_it ████████░░░░░░░░░░░░░░░░░░░░░░░░░ 16.09%
  Rank 3: Hunyuan_MT_7B ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  7.54%
```

#### 2. 批量口令预测

```bash
# 从文件读取多个口令
python scripts/2-5-classify_passwords_v2.py --txt_file passwords.txt --model_type RandomForest
```

#### 3. 完整演示

```bash
# 运行15个测试案例的完整演示
python scripts/2-6-demo_fingerprint_identification.py
```

---

## 📚 详细脚本说明

### 阶段一：口令强度评估（3.1.3节）

#### 1-1-entropy_analysis.py
**功能**：计算并分析口令的信息熵
- **输入**：`data/*/llm_passwords_*.csv`
- **输出**：
  - `latex/figures/1-1-entropy_distribution.png` - 熵分布图
  - `latex/figures/1-1-entropy_vs_length.png` - 熵与长度关系图
- **核心算法**：Shannon熵 $H = L \times \log_2(R)$
- **运行时间**：~2分钟

#### 1-2-zxcvbn_analysis.py
**功能**：使用zxcvbn库评估口令强度
- **输入**：带有`Entropy`列的CSV文件
- **输出**：
  - `latex/figures/1-2-score_distribution.png` - 评分分布
  - `latex/figures/1-2-theoretical_vs_effective_entropy.png` - 理论熵vs等效熵
  - `latex/figures/1-2-crack_time_distribution.png` - 破解时间分布
- **评估维度**：
  - zxcvbn评分（0-4）
  - 估计破解时间
  - 等效熵（log2(guesses)）
- **运行时间**：~5分钟

#### 1-3-bias_analysis.py
**功能**：分析LLM生成口令的倾向性指标
- **输入**：带有`zxcvbn_*`列的CSV文件
- **输出**：
  - `latex/figures/1-3-character_distribution.png` - 字符类型分布
  - `latex/figures/1-3-length_distribution.png` - 长度分布
  - `latex/figures/1-3-positional_bias.png` - 位置偏好
  - 以及其他7个分析图表
- **分析维度**：
  - 字符类型分布
  - 长度分布
  - 位置偏好
  - 归一化熵
  - Jensen-Shannon散度
- **运行时间**：~3分钟

---

### 阶段二：LLM指纹识别（3.2节）

#### 2-1-prepare_classification_dataset.py
**功能**：构建16分类任务的数据集
- **输入**：`data/良性提示/` 和 `data/少样本提示/` 下的所有CSV
- **输出**：
  - `data/classification/train_passwords.csv` (102,596条)
  - `data/classification/val_passwords.csv` (21,985条)
  - `data/classification/test_passwords.csv` (21,984条)
  - `data/classification/dataset_statistics.txt`
- **关键决策**：
  - 保留所有重复口令（用于多模型指纹识别）
  - 分层采样（70%/15%/15%）
  - 16个LLM模型的多分类标注
- **运行时间**：~30秒

#### 2-2-feature_engineering.py
**功能**：提取39维口令特征
- **输入**：`data/classification/*_passwords.csv`
- **输出**：
  - `data/classification/train_features.csv` (102,596×40)
  - `data/classification/val_features.csv` (21,985×40)
  - `data/classification/test_features.csv` (21,984×40)
  - `data/classification/feature_description.txt`
- **特征类别**：
  1. **统计特征**（11个）：长度、字符类型计数、字符类型比例、字符多样性
  2. **序列模式特征**（12个）：连续字符、键盘模式、位置偏好、数字集中度
  3. **N-gram特征**（8个）：2-gram和3-gram的数量、唯一性、多样性
  4. **模式识别特征**（8个）：单词-数字模式、Leetspeak、年份/日期模式
- **运行时间**：~2分钟

#### 2-3-train_fingerprint_models.py ★
**功能**：训练16分类指纹识别模型（主版本）
- **输入**：`data/classification/*_features.csv`
- **输出**：
  - `data/classification/fingerprint_models/RandomForest_Fingerprint.pkl`
  - `data/classification/fingerprint_models/GradientBoosting_Fingerprint.pkl`
  - `data/classification/fingerprint_models/DNN_Fingerprint.keras`
  - `data/classification/fingerprint_models/scaler.pkl`
  - `data/classification/fingerprint_models/label_encoder.pkl`
  - `data/classification/fingerprint_models/label_mapping.pkl`
  - `data/classification/fingerprint_evaluation_results.csv`
  - `data/classification/fingerprint_evaluation_detailed.json`
  - `data/classification/fingerprint_comparison_report.txt`
- **训练模型**：
  1. **Random Forest**：200棵树，最大深度30，类别权重平衡
  2. **Gradient Boosting**：150轮迭代，学习率0.1，类别权重平衡
  3. **DNN**：4层全连接（256-128-64-16），BatchNorm + Dropout + EarlyStopping
- **性能**：
  - Random Forest: Top-1准确率 **93.95%**, Top-3准确率 **99.52%**
  - Gradient Boosting: Top-1准确率 **93.88%**, Top-3准确率 **99.59%**
  - DNN: Top-1准确率 **92.09%**, Top-3准确率 **99.55%**
- **运行时间**：~20分钟（含GPU加速）
- **环境要求**：`conda activate tensorflow`

#### 2-4-analyze_and_visualize_results.py
**功能**：分析训练结果并生成可视化图表
- **输入**：
  - `data/classification/fingerprint_evaluation_results.csv`
  - `data/classification/fingerprint_evaluation_detailed.json`
- **输出**：
  - `latex/figures/2-4-model_performance_comparison.png` (4子图布局)
  - `latex/figures/2-4-per_class_performance.png` (2子图布局)
  - `latex/figures/2-4-model_performance_summary.csv`
  - `latex/figures/2-4-per_class_performance_detailed.csv`
- **可视化内容**：
  - Top-K准确率对比
  - F1-Score对比
  - 各LLM模型识别性能
  - 样本规模vs召回率散点图
- **运行时间**：~30秒

#### 2-5-classify_passwords_v2.py
**功能**：批量口令分类工具
- **输入**：`.txt`文件（每行一个口令）
- **输出**：CSV格式的预测结果
- **支持模型**：RandomForest, GradientBoosting, DNN
- **用法**：
  ```bash
  python scripts/2-5-classify_passwords_v2.py \
      --txt_file passwords.txt \
      --model_type RandomForest
  ```

#### 2-6-demo_fingerprint_identification.py
**功能**：完整演示程序（15个测试案例）
- **输入**：无（内置15个测试口令）
- **输出**：
  - `data/classification/demo_fingerprint_results.csv`
  - `data/classification/demo_fingerprint_report.txt`
- **测试案例**：
  - 少样本提示风格（8个）：`p@ssw0rd123`, `admin12345`, `Welcome2024!`等
  - 良性提示风格（4个）：`Kx7#mQ2$nL9&pR4`, `G8!vZ#4mT@9qY$2`等
  - 混合风格（3个）：`MyP@ssw0rd2024!`, `Secure#Pass123`等
- **运行时间**：~1分钟

#### 2-6-predict_single_password.py ★★★（推荐使用）
**功能**：单口令预测工具（命令行接口）
- **输入**：命令行参数（单个口令）
- **输出**：
  - 控制台：实时显示Top-3预测结果（带可视化进度条）
  - 文件：`data/classification/predictions/prediction_YYYYMMDD_HHMMSS.txt`
- **特点**：
  - ✓ 简单易用（一行命令）
  - ✓ 实时反馈（带进度条）
  - ✓ 自动保存（带时间戳）
  - ✓ 智能解释（置信度分析）
- **用法**：
  ```bash
  python scripts/2-6-predict_single_password.py <password>
  ```
- **示例**：
  ```bash
  python scripts/2-6-predict_single_password.py mnbvcxz
  ```
- **运行时间**：<1秒

---

## 📊 数据流图

```
数据采集
    ↓
TXT文件 → batch_convert_txt_to_csv.py → CSV文件
    ↓
阶段一分析
    ├→ add_entropy_to_csv.py → 1-1-entropy_analysis.py → 熵分析图表
    ├→ add_zxcvbn_to_csv.py → 1-2-zxcvbn_analysis.py → zxcvbn分析图表
    └→ 1-3-bias_analysis.py → 倾向性分析图表
    ↓
阶段二分析
    ├→ 2-1-prepare_classification_dataset.py → 数据集（train/val/test）
    ├→ 2-2-feature_engineering.py → 特征数据集（39维）
    ├→ 2-3-train_fingerprint_models.py → 训练好的模型
    ├→ 2-4-analyze_and_visualize_results.py → 性能分析图表
    ├→ 2-6-demo_fingerprint_identification.py → 演示报告
    └→ 2-6-predict_single_password.py → 单口令预测★
```

---

## 🔬 关键技术说明

### 特征工程（2-2）
- **时间复杂度**：O(L) per password, L为口令长度
- **空间复杂度**：O(N×39), N为口令数量
- **优化技术**：向量化计算、并行处理

### 模型训练（2-3）
- **类别权重**：$w_i = \frac{N_{total}}{N_{classes} \times N_i}$
- **早停策略**：监控验证集损失，patience=10
- **GPU加速**：TensorFlow + CUDA 12.0
- **训练稳定性**：BatchNormalization + Dropout(0.2-0.3)

### 模型评估（2-4）
- **Top-K准确率**：K=1/3/5
- **F1-Score**：宏平均和加权平均
- **召回率**：各LLM模型的识别准确率（核心指标）

---

## 🛠️ 故障排除

### 常见问题

**Q1: `ModuleNotFoundError: No module named 'xxx'`**
```bash
# 解决方案：安装缺失的依赖
pip install -r requirements.txt
```

**Q2: `FileNotFoundError: models not found`**
```bash
# 解决方案：先训练模型
conda activate tensorflow
python scripts/2-3-train_fingerprint_models.py
```

**Q3: `UnicodeDecodeError` on Windows**
```python
# 解决方案：确保文件使用UTF-8编码
# 在脚本中添加：encoding='utf-8'
```

**Q4: GPU内存不足**
```bash
# 解决方案：减小batch size或使用CPU
# 修改2-3脚本中的batch_size参数
```

---

## 📄 输出文件说明

### 模型文件（data/classification/fingerprint_models/）
- `RandomForest_Fingerprint.pkl` - 随机森林模型（~150MB）
- `GradientBoosting_Fingerprint.pkl` - 梯度提升模型（~50MB）
- `DNN_Fingerprint.keras` - 深度神经网络模型（~5MB）
- `scaler.pkl` - 标准化器
- `label_encoder.pkl` - 标签编码器
- `label_mapping.pkl` - 标签映射字典
- `feature_columns.pkl` - 特征列名列表

### 预测结果（data/classification/predictions/）
- `prediction_YYYYMMDD_HHMMSS.txt` - 单次预测结果
- 格式：
  ```
  Password: xxx
  RandomForest: Top-1预测 (置信度)
  GradientBoosting: Top-1预测 (置信度)
  Interpretation: 解释说明
  ```

### 可视化图表（latex/figures/）
- 阶段一：7个PNG图表（熵、zxcvbn、倾向性）
- 阶段二：2个PNG图表（模型性能、各LLM识别性能）
- 所有图表：300 DPI，英文标签，适合论文插图

---

## 📖 参考文献

本项目实现参考了以下文献：
1. **Password Guessing Using Large Language Models** (2025)
2. **zxcvbn: Low-Budget Password Strength Estimation** (USENIX Security 2016)
3. **Random Forest** - Breiman, L. (2001)
4. **Gradient Boosting** - Friedman, J. H. (2001)

---

## 👨‍💻 开发者信息

**项目名称**：大语言模型生成口令的安全性评估  
**开发时间**：2025年10月  
**Python版本**：3.10+  
**主要依赖**：TensorFlow 2.15, scikit-learn 1.3, pandas, numpy

---

## 📝 版本历史

### v1.0.0 (2025-10-20)
- ✅ 完成阶段一：口令强度评估（熵、zxcvbn、倾向性）
- ✅ 完成阶段二：LLM指纹识别（16分类，93.95%准确率）
- ✅ 添加单口令预测工具（2-6-predict_single_password.py）
- ✅ 生成完整文档和README

---

## ⭐ 推荐使用

对于日常使用，推荐以下脚本：

1. **🥇 单口令预测**：`2-6-predict_single_password.py` ← 最常用
2. **🥈 批量预测**：`2-5-classify_passwords_v2.py`
3. **🥉 完整演示**：`2-6-demo_fingerprint_identification.py`

---

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 项目目录：`E:\Competition\密码学技术竞赛\Source\`
- 文档位置：`scripts/README.md`

---

**最后更新时间**：2025-10-20  
**文档版本**：v1.0.0

