# Replication of Nonlinear Classification by Choquet Integral with Signed Fuzzy Measure

This repository contains a replication of Figure 3(a)/(b) from the paper "Nonlinear Classification by Genetic Algorithm with Signed Fuzzy Measure" by Wang et al. (2007).

## Overview

This project implements a nonlinear classifier based on the Choquet integral with a signed fuzzy measure, as described in the referenced paper. The classifier uses a Choquet-hyperplane decision boundary to separate two classes in a 2D feature space.

## Files

- `replicate_choquet_classifier.py` - Main implementation script
- `replication_report.md` - Detailed technical report documenting the mathematical implementation
- `README.md` - This file

## Requirements

- Python 3.10+
- numpy
- matplotlib

## Installation

```bash
pip install numpy matplotlib
```

## Usage

1. Run the main script:
   ```bash
   python replicate_choquet_classifier.py
   ```

2. To replicate different figures, modify the `which` variable in the script:
   - Set `which = "3a"` to replicate Figure 3(a)
   - Set `which = "3b"` to replicate Figure 3(b)

## Output

The script generates:
- `fig3a_replication.png` - Replication of Figure 3(a) parameters
- `fig3b_replication.png` - Replication of Figure 3(b) parameters

Each figure shows:
- Scatter plot of the 200 generated data points
- Class A points (circles) and Class A' points (crosses)
- Nonlinear decision boundary (contour line at H(x) = 0)
- Class counts in the title

## Mathematical Background

The classifier implements a Choquet-hyperplane:
```
H(x) = (C)∫ (a + b ∘ f(x)) dμ − B
```

Where:
- `f(x) ∈ R²` are two attributes
- `a, b ∈ R²` are matching vectors
- `μ` is a signed fuzzy measure
- `B` is a scalar offset

Points are classified as Class A if H(x) > 0, otherwise as Class A'.

## Parameters

The implementation uses the exact parameters from the paper:

**Figure 3(a):**
- μ₁₂ = 0.1389, μ₁ = 0.1802, μ₂ = 0.5460
- B = 0.0917
- a = (0, 0), b = (1, 1)

**Figure 3(b):**
- μ₁₂ = 0.3830, μ₁ = 0.6683, μ₂ = 0.5713
- B = 0.2633
- a = (0.4420, 0.7021), b = (0.3614, -0.154)

## Author

Robert Ngabo Mugisha (PhD applicant)

## Reference

H. Wang et al., Nonlinear Classification by Genetic Algorithm with Signed Fuzzy Measure, IEEE 2007.
