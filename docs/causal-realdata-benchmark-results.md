# E5 Causal Benchmark with Real HuggingFace Data

**Generated:** 2026-01-21T22:15:48.419153532+00:00

## Configuration

| Parameter | Value |
|-----------|-------|
| Total Chunks | 500 |
| Documents | 2437 |
| Topics | 100 |
| E5 Coverage | 100.0% |
| Asymmetric E5 | true |

## Direction Detection

| Metric | Value |
|--------|-------|
| Total Samples | 100 |
| Cause Detected | 22 |
| Effect Detected | 19 |
| Detection Rate | 41.0% |

## Asymmetric Retrieval

| Metric | Value | Target |
|--------|-------|--------|
| MRR Cause→Effect | 1.0000 | - |
| MRR Effect→Cause | 1.0000 | - |
| MRR Symmetric (E1) | 1.0000 | - |
| **Asymmetry Ratio** | **1.00** | ~1.5 |
| Improvement over E1 | 0.0% | >0% |

## COPA-Style Reasoning

| Metric | Value | Target |
|--------|-------|--------|
| **E5 Asymmetric Accuracy** | **74.0%** | >70% |
| E1 Symmetric Accuracy | 84.0% | - |
| Random Baseline | 50.0% | - |
| Improvement over E1 | -11.9% | >0% |

## E5 Contribution Analysis

| Metric | Value | Target |
|--------|-------|--------|
| MRR with E5 | 1.0000 | - |
| MRR without E5 | 1.0000 | - |
| **E5 Contribution** | **0.0%** | >5% |

## Recommendations

- Asymmetry ratio below target (1.5). Consider tuning E5 direction modifiers.
- E5 contribution below 5%. Consider increasing E5 weight in fusion formula.

