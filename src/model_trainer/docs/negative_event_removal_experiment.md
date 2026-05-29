# Negative Event Removal Experiment Design

## 1. Experiment Overview

This experiment aims to quantify the impact of removing negative influence events on time series forecasting performance.

### Key Questions:
1. How much do negative events degrade prediction performance?
2. Can removing negative events from training improve model performance?
3. What is the optimal strategy for handling negative events?

## 2. Experimental Setup

### 2.1 Data Preparation
- **Holdout Set**: 2,052 samples with 9,896 events
- **Negative Events**: Events with contribution_score < -0.01
- **Positive Events**: Events with contribution_score > 0.01  
- **Neutral Events**: Events with |contribution_score| ≤ 0.01

### 2.2 Experiment Phases

#### Phase 1: Direct Impact on Holdout Set
1. Evaluate base model on complete holdout set (baseline)
2. Evaluate base model on holdout set without negative events
3. Compare MSE, MAE, RMSE metrics

#### Phase 2: Dynamic Model Update
1. Fine-tune base model using holdout set without negative events
2. Evaluate updated model on test set
3. Compare with baseline model on test set

#### Phase 3: Ablation Studies
1. Remove only high-impact negative events (score < -1.0)
2. Remove negative events by percentiles (top 10%, 25%, 50%)
3. Test different thresholds for negative event definition

## 3. Implementation Strategy

### 3.1 Data Filtering
```python
def filter_negative_events(data, contribution_scores, threshold=-0.01):
    """Remove events with contribution scores below threshold"""
    filtered_data = []
    for sample in data:
        events = sample['event']
        filtered_events = {}
        for event_key, event_text in events.items():
            score = get_contribution_score(event_key, event_text)
            if score >= threshold:
                filtered_events[event_key] = event_text
        sample_filtered = sample.copy()
        sample_filtered['event'] = filtered_events
        filtered_data.append(sample_filtered)
    return filtered_data
```

### 3.2 Model Evaluation
```python
def evaluate_with_events(model, data_loader, use_events=True):
    """Evaluate model with or without events"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_events, batch_y in data_loader:
            if use_events:
                outputs = model(batch_x, events=batch_events)
            else:
                outputs = model(batch_x, events=None)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    return total_loss / len(data_loader)
```

### 3.3 Dynamic Model Update
```python
def update_model_with_filtered_data(model, filtered_data, epochs=5):
    """Fine-tune model on filtered data"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(epochs):
        for batch in filtered_data:
            optimizer.zero_grad()
            loss = model.calculate_loss(batch)
            loss.backward()
            optimizer.step()
    return model
```

## 4. Metrics

### Primary Metrics:
- **MSE Reduction**: (MSE_baseline - MSE_filtered) / MSE_baseline * 100
- **MAE Reduction**: (MAE_baseline - MAE_filtered) / MAE_baseline * 100
- **RMSE Reduction**: (RMSE_baseline - RMSE_filtered) / RMSE_baseline * 100

### Secondary Metrics:
- Number of events removed
- Distribution of removed events
- Impact on different prediction horizons

## 5. Expected Outcomes

### Hypotheses:
1. **H1**: Removing negative events will improve prediction accuracy by 5-15%
2. **H2**: Dynamic model update will show greater improvement than simple filtering
3. **H3**: There exists an optimal threshold for negative event removal

### Risk Factors:
- Over-filtering may remove important contextual information
- Model may become over-optimistic without negative signals
- Performance gains may not generalize to new data

## 6. Implementation Timeline

1. **Step 1**: Load contribution scores and identify negative events (5 min)
2. **Step 2**: Create filtered datasets (10 min)
3. **Step 3**: Evaluate on holdout set (15 min)
4. **Step 4**: Fine-tune model (30 min)
5. **Step 5**: Evaluate on test set (15 min)
6. **Step 6**: Generate report (10 min)

Total estimated time: ~85 minutes