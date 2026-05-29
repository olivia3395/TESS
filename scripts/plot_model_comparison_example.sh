#!/bin/bash
# Example usage of plot_model_comparison.py

python scripts/plot_model_comparison.py \
    --files \
        saved/MultiModal_Baseline/Electricity/ver_camf/best/test_samples.jsonl \
        saved/MultiModal_Baseline/Electricity/ver_global_temporal_shape_volatility_natural/best/test_samples.jsonl \
        saved/UniModal_Baseline/Electricity/ver_camf/best/test_samples.jsonl \
    --labels \
        "MultiModal_Baseline (ver_camf)" \
        "MultiModal_Baseline (ver_global_temporal_shape_volatility_natural)" \
        "UniModal_Baseline (ver_camf)" \
    --output model_comparison_electricity.png \
    --samples 10 \
    --show
