# 数据集规范草案

## 建议核心字段

- `Password`
- `Model_ID`
- `Prompt_Type`
- `Timestamp`
- `Length`
- `Character_Diversity`
- `Entropy`
- `Pattern_Score`
- `Crack_Time`

## 建议补充字段

- `Provider`
- `Source`
- `Collection_Batch`
- `Is_Deduplicated`
- `zxcvbn_score`
- `zxcvbn_guesses`
- `effective_entropy_bits`

## 清洗建议

- 去除空值
- 去除明显非口令文本
- 标准化引号、空格和换行
- 统一模型名与提示类型
- 保留是否去重的记录
