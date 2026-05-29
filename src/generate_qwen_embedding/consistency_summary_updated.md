# 8B和8B_Ablation系列模型一致性分析报告（更新版）

## 概述

本报告分析了8B和8B_Ablation系列模型在ver_synchronized系列数据集上的一致性情况，统计了以下几种情况的占比：
1. 8B对了Ablation没对
2. Ablation对了8B没对
3. 两者都对
4. 两者都错

新增了对ver_8B_Ablation_onlyts系列数据集的分析。

## 主要发现

### 1. Trendstrength数据集（趋势+强度）

在test数据集中：
- Step级别：
  - 都正确: 1.11%
  - 仅8B正确: 7.07%
  - 仅Ablation正确: 9.19%
  - 都错误: 82.63%
- 强度级别：
  - 都正确: 3.71%
  - 仅8B正确: 12.14%
  - 仅Ablation正确: 13.89%
  - 都错误: 70.26%

**分析**：Ablation模型在趋势和强度预测上都略优于8B模型，但整体准确率仍然很低。

### 2. Trendonly数据集（仅趋势）

在test数据集中：
- Step级别：
  - 都正确: 20.38%
  - 仅8B正确: 19.77%
  - 仅Ablation正确: 30.68%
  - 都错误: 29.16%

**分析**：Ablation模型在趋势预测上明显优于8B模型，两者都对的比例也较高。

### 3. Globalonly数据集（仅全局趋势）

在test数据集中：
- Record级别：
  - 都正确: 39.89%
  - 仅8B正确: 28.53%
  - 仅Ablation正确: 18.44%
  - 都错误: 13.15%

**分析**：在全局趋势预测上，8B模型表现优于Ablation模型，两者都对的比例最高。

### 4. Trendstrength_onlyts数据集（基于onlyts的trendstrength）

在test数据集中：
- Step级别：
  - 都正确: 3.05%
  - 仅8B正确: 5.13%
  - 仅Ablation正确: 6.24%
  - 都错误: 85.58%
- 强度级别：
  - 都正确: 7.22%
  - 仅8B正确: 8.63%
  - 仅Ablation正确: 11.03%
  - 都错误: 73.12%

**分析**：基于onlyts的trendstrength模型在趋势和强度预测上都优于原始8B模型，但略逊于Ablation模型。

### 5. Trendonly_onlyts数据集（基于onlyts的trendonly）

在test数据集中：
- Step级别：
  - 都正确: 34.32%
  - 仅8B正确: 5.83%
  - 仅Ablation正确: 13.68%
  - 都错误: 46.16%

**分析**：基于onlyts的trendonly模型在趋势预测上表现最好，显著优于其他模型组合。

### 6. Globalonly_onlyts数据集（基于onlyts的globalonly）

在test数据集中：
- Record级别：
  - 都正确: 58.01%
  - 仅8B正确: 10.40%
  - 仅Ablation正确: 12.57%
  - 都错误: 19.02%

**分析**：基于onlyts的globalonly模型在全局趋势预测上表现最好，两者都对的比例最高。

## 结论

1. **Trendonly_onlyts表现最佳**：在所有数据集中，基于onlyts的trendonly模型表现最好，准确率显著高于其他模型。

2. **Trendstrength任务最具挑战性**：包含趋势和强度的联合预测任务准确率最低，即使两个模型都难以准确预测。

3. **基于onlyts的模型整体表现更好**：在大多数任务中，基于onlyts的模型变体表现优于原始8B和Ablation模型。

4. **Ablation模型在细粒度预测上表现更好**：在trendonly和trendstrength任务中，Ablation模型的表现优于8B模型。

5. **8B模型在全局预测上表现更好**：在仅预测全局趋势的任务中，8B模型的表现优于Ablation模型，但不如基于onlyts的模型。

这些结果表明，不同的模型变体在不同类型的预测任务上具有不同的优势。基于onlyts的模型在大多数任务上表现最好，这可能是因为它只使用时间序列信息而不依赖文本信息，使得模型更加专注于时间序列模式的学习。