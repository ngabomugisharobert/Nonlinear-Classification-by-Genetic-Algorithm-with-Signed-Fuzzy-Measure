
# Replication of Figure 3(a)/(b): Nonlinear Classification by Choquet Integral with a Signed Fuzzy Measure

**Author:** Robert (PhD applicant)  
**Goal:** Replicate Fig. 3(a) or Fig. 3(b) from *Nonlinear Classification by Genetic Algorithm with Signed Fuzzy Measure* by Wang et al. (2007).  
**Deliverables:** Replication code and this brief report documenting the mathematical implementation and steps.

## 1. Problem Setup

The paper defines a nonlinear classifier as a **Choquet-hyperplane**:
H(x) = (C)∫ (a + b ∘ f(x)) dμ − B,
where f(x)∈R² are two attributes, a,b∈R² are matching vectors (scaling/phase-matching), μ is a **signed fuzzy measure** on subsets of attributes, and B is a scalar offset. A point is labeled **Class A** if H(x)>0 and **Class A'** otherwise. Figures 3(a)/(b) in the paper list concrete parameter values (μ components, a,b, B).

## 2. Choquet Integral (n = 2) with a Signed Measure

For two attributes with transformed values g₁, g₂ and a signed measure with singleton masses μ({x₁})=μ₁, μ({x₂})=μ₂ and pair mass μ({x₁,x₂})=μ₁₂, the discrete Choquet integral used is:
(C)∫ (g₁,g₂) dμ = g_(1)·μ({i_(1), i_(2)}) + (g_(2)−g_(1))·μ({i_(2)}),
where g_(1)≤g_(2) are the sorted values and i_(2) indexes the larger component. This is equivalent to the standard Σ (g_(j)−g_(j−1)) μ(A_(j)) definition with A_(j) being the top-j index set. The implementation follows the formula described in the paper’s Section II.

**Matching transform:** g_i = a_i + b_i f_i. This is applied elementwise prior to the Choquet integral, per the generalized weighted Choquet integral described in the paper.

## 3. Data Generation and Labeling

Following the simulation section, a 2D synthetic dataset of size N=200 is generated via a RNG on [0,1]². Each point is labeled using the sign of H(x) with the exact parameters printed under each figure in the paper:
- Fig. 3(a): μ₁₂=0.1389, μ₁=0.1802, μ₂=0.5460, B=0.0917, a=(0,0), b=(1,1).
- Fig. 3(b): μ₁₂=0.3830, μ₁=0.6683, μ₂=0.5713, B=0.2633, a=(0.4420,0.7021), b=(0.3614,−0.154).

The class boundary is the level set H(x)=0. It is visualized by drawing a contour at level 0 on a dense grid over [0,1]².

## 4. Penalized Distance (for completeness)

The paper’s training objective maximizes a **penalized total signed Choquet distance** to address class imbalance. This replication focuses on reproducing the figure given known parameters; the code can be extended to implement the exact objective and a GA as in the paper.

## 5. Results

Running the provided script produces two figures:
- fig3a_replication.png — replication of Fig. 3(a) parameters.
- fig3b_replication.png — replication of Fig. 3(b) parameters.

Counts may differ slightly from Table I/II because the RNG sample differs, but the qualitative structure (two clearly separable classes and a smooth nonlinear boundary) is replicated.

## 6. How to Run

1. Python 3.10+ with matplotlib and numpy.
2. Execute replicate_choquet_classifier.py. Inside, set `which = "3a"` or `"3b"` to select the target figure.
3. The script saves `fig3a_replication.png` or `fig3b_replication.png` and prints the class counts.
