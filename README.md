# kpt2-multisignal-intelligence
Multi-signal Kitchen Prep Time (KPT) prediction system with bias correction, kitchen load estimation, composite readiness scoring, and shadow monitoring for large-scale food delivery platforms.
# KPT 2.0 – Multi-Signal Kitchen Intelligence System

## Problem
Kitchen Prep Time (KPT) prediction at Zomato relies heavily on merchant-marked Food Order Ready (FOR) signals.
These signals are noisy due to:
- Rider-influenced marking
- Human bias
- No visibility into non-Zomato kitchen load
- Rush-hour distortion

This introduces systematic label noise and ETA volatility.

## Solution Overview

We propose a Multi-Signal Readiness Inference System that enhances KPT prediction by:

1. Merchant Bias Index (MBI)
2. Kitchen Load Score (KLS)
3. Rider Wait Cluster Detection
4. Adaptive Composite Readiness Score (CRS)
5. Shadow KPT Model (FOR-independent)

This system improves signal quality without requiring major model redesign.

## Key Features

- Realistic timestamp simulation
- Behavioral bias correction
- Rush detection
- Congestion-adaptive weighting
- Drift monitoring
- A/B evaluation framework
- Visualization of error distributions
