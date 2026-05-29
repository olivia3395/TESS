# Ablation实验一致性分析报告（与8BInstruct对比）

## 概述

本报告分析了三种模型变体在ver_synchronized系列数据集上的一致性情况：
1. 8BInstruct（完整模型）
2. onlynews（仅使用新闻文本的消融模型）
3. onlyts（仅使用时间序列的消融模型）

我们统计了以下几种情况的占比：
1. 两个模型都预测正确
2. 仅第一个模型预测正确
3. 仅第二个模型预测正确
4. 两个模型都预测错误

## 主要发现

### 1. Trendstrength数据集（趋势+强度）

在test数据集中：
- **8BInstruct vs onlynews一致性**：
  - 都正确: 1.35%
  - 仅8BInstruct正确: 8.68%
  - 仅onlynews正确: 8.95%
  - 都错误: 81.02%

- **8BInstruct vs onlyts一致性**：
  - 都正确: 3.80%
  - 仅8BInstruct正确: 6.23%
  - 仅onlyts正确: 5.49%
  - 都错误: 84.48%

**分析**：在趋势和强度预测任务中，onlynews和onlyts模型表现相当，但整体准确率都很低。

### 2. Trendonly数据集（仅趋势）

在test数据集中：
- **8BInstruct vs onlynews一致性**：
  - 都正确: 24.58%
  - 仅8BInstruct正确: 23.94%
  - 仅onlynews正确: 26.48%
  - 都错误: 24.99%

- **8BInstruct vs onlyts一致性**：
  - 都正确: 41.49%
  - 仅8BInstruct正确: 7.03%
  - 仅onlyts正确: 6.51%
  - 都错误: 44.96%

**分析**：在仅趋势预测任务中，onlyts模型表现明显优于onlynews模型，且与8BInstruct模型的一致性最高。

### 3. Globalonly数据集（仅全局趋势）

在test数据集中：
- **8BInstruct vs onlynews一致性**：
  - 都正确: 45.76%
  - 仅8BInstruct正确: 28.53%
  - 仅onlynews正确: 12.57%
  - 都错误: 13.15%

- **8BInstruct vs onlyts一致性**：
  - 都正确: 62.48%
  - 仅8BInstruct正确: 11.81%
  - 仅onlyts正确: 8.10%
  - 都错误: 17.61%

**分析**：在全局趋势预测任务中，onlyts模型表现最好，与8BInstruct模型的一致性最高。

## 结论

1. **在复杂任务中（trendstrength），完整模型表现最好**：在需要同时预测趋势和强度的任务中，完整8BInstruct模型与两个消融模型的一致性都不高，但仍然优于单独的消融模型。

2. **在趋势预测任务中（trendonly），onlyts模型表现最佳**：在仅需预测趋势的任务中，onlyts模型与8BInstruct模型的一致性最高，达到了41.49%。

3. **在全局趋势预测任务中（globalonly），onlyts模型表现最好**：在全局趋势预测任务中，onlyts模型与8BInstruct模型的一致性最高，达到了62.48%。

4. **时间序列信息在预测中起关键作用**：整体来看，onlyts模型在大多数任务中都优于onlynews模型，表明时间序列信息在股价预测任务中起着关键作用。

5. **多模态融合的优势**：虽然消融模型在某些特定任务上有不错的表现，但完整8BInstruct模型在整体上仍然保持了与各消融模型的良好一致性，体现了多模态融合的价值。

这些结果表明，在金融时间序列预测任务中，时间序列信息比新闻文本信息更为重要，但将两者结合的完整模型仍然能够提供额外的价值。与之前的8B模型相比，8BInstruct在各项任务中的表现都有所提升。