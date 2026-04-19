# Low-Rank Matrix Factorization: Optimization & Applications
## 15-Minute Presentation - Table of Contents

---

## I. INTRODUCTION & MOTIVATION (3 minutes)

### Slide 1: Title Slide
- **Title**: Topic 3: Low-Rank Matrix Factorization
- **Subtitle**: Optimization Algorithms for Recommender Systems & Data Analysis
- **Group**: Nicole Chen, Jiuyi (Joy) Cheng, Sirui (Michelle) Li

### Slide 2: Project Overview & Roadmap

#### **Project Goal**
Evaluate and compare three optimization algorithms for low-rank matrix factorization problem:
$$\min_{U, V} \frac{1}{2}\|UV^T - M\|_F^2 + \frac{\lambda}{2}(\|U\|_F^2 + \|V\|_F^2)$$

#### **Three Algorithms Compared**
| Algorithm | Type | Strength |
|-----------|------|----------|
| **Gradient Descent + Armijo** | 1st-order | Simple, stable, pedagogical |
| **L-BFGS** | Quasi-Newton | Fast convergence, few iterations |
| **Alternating Least Squares** | Block coordinate descent | Ideal for sparse data |

#### **Three Data Types Evaluated**
| Dataset | Type | Size | Sparsity | Purpose |
|---------|------|------|----------|---------|
| **Synthetic** | Dense | 500×200, rank-5 | 0% | Controlled experiments, ground truth |
| **MovieLens 100K** | Sparse | 943×1,682 | 93.7% | Real recommender system benchmark |
| **Superconductivity** | Dense | 21,263×81 | 0% | Real-world feature reduction |

#### **Five Core Experiments**
1. **Exp 1 - Method Comparison**: Which algorithm is fastest & most accurate?
2. **Exp 2 - Initialization Sensitivity**: Does SVD initialization protect from bad local minima?
3. **Exp 3 - Rank Selection**: How does underestimating/overestimating rank affect quality?
4. **Exp 4 - Regularization Impact**: How does λ affect recovery under noise?
5. **Exp 5 - Optimization Landscape**: How nonconvex is the problem in practice?

### Slide 3: Why Matrix Factorization?
- **Real-World Problem**: High-dimensional data with missing or noisy entries
  - User-item ratings (Netflix problem)
  - Image pixels with noise
  - Feature discovery from incomplete observations
- **Key Insight**: Many datasets have underlying low-rank structure
- **Visual**: Show a rank-1 approximation of an image

### Slide 4: Problem Statement
- **Optimization Problem**:
  $$\min_{U \in \mathbb{R}^{m \times r}, V \in \mathbb{R}^{n \times r}} \frac{1}{2}\|UV^T - M\|_F^2 + \frac{\lambda}{2}(\|U\|_F^2 + \|V\|_F^2)$$
- **Key Components**:
  - **Factorization**: Decompose matrix M into two factor matrices U and V
  - **Reconstruction Loss**: $\frac{1}{2}\|UV^T - M\|_F^2$ (squared Frobenius norm)
  - **Regularization**: L2 penalties $\frac{\lambda}{2}(\|U\|_F^2 + \|V\|_F^2)$ prevent overfitting
  - **Parameter λ**: Regularization strength (trade-off hyperparameter)
- **Problem Properties**:
  - **Non-convex** in $(U, V)$ jointly (but biconvex in each factor separately)
  - **L-smooth**: gradients Lipschitz continuous (enables line search convergence)
  - Existence of multiple stationary points (local minima)
- **Motivation for Low-Rank**: $r \ll \min(m,n)$ compresses representation while capturing signal
  *(See appendix for smoothness constants and convergence analysis)* 

---

## II. APPLICATIONS (1 minute)

### Slide 5: Key Applications
1. **Recommender Systems**
   - Predict missing user ratings
   - MovieLens dataset: 100K ratings from 1K users on 1.7K movies
   - Rank typically 5-50

2. **Image Compression & Denoising**
   - Reduce storage via low-rank approximation
   - Remove noise from images (compression cleans noise)

3. **Latent Feature Discovery**
   - Uncover hidden patterns in data
   - Example: movie genres from ratings

4. **Missing Data Completion**
   - Inpainting, matrix recovery
   - Netflix Prize: predict unrated movies

### Slide 6: Why Low-Rank Works
- **Empirical Observation**: Real data often has intrinsic low-rank structure
  - User preferences: ~5-10 latent factors
  - Images: compressible via wavelets (low-rank in transform domain)
- **Degrees of Freedom**: $mr + nr$ factors vs. $mn$ entries
  - For M = 1000×1700, r=20: compress by 10× vs. 2M entries
- **Generalization**: Captures signal, discards noise

---

## III. DATASETS (1 minute)

### Slide 7: Datasets Used in Study
1. **Synthetic Low-Rank Matrix** (Controlled Experiment)
   - Generation: $M = U_{true} V_{true}^T + \sigma \cdot \text{noise}$
   - Default: 500×200, rank=5, σ=0.1
   - **Advantage**: Know true solution → can measure recovery error
   - **Use Case**: Test sensitivity to initialization & rank

2. **MovieLens 100K** (Real Sparse Data)
   - ~100K ratings from 943 users on 1,682 movies
   - **Sparse**: ~6.3% of entries observed
   - Train/test split: evaluate RMSE on held-out ratings
   - **Challenge**: Only observe ~100K entries, predict rest

3. **Superconductivity Dataset** (Dense Real Data)
   - 21,263 compounds × 81 features
   - Features standardized (zero mean, unit variance)
   - **Dense**: All entries present
   - **Application**: Dimensionality reduction for materials science

### Slide 8: Dataset Characteristics Table
| Dataset | Type | Size | Sparsity | Rank |
|---------|------|------|----------|------|
| Synthetic | Dense | 500×200 | 0% | 5 |
| MovieLens | Sparse | 943×1,682 | 93.7% | 20-30 |
| Superconductivity | Dense | 21,263×81 | 0% | ~20 |

---

## IV. OPTIMIZATION METHODS (3-4 minutes)

### Slide 9: Method 1 - Gradient Descent with Armijo Backtracking
- **Update Rule**: $x_{k+1} = x_k - \alpha_k \nabla f(x_k)$
- **Armijo Line Search**:
  - Start with α = 1
  - If sufficient decrease not met, halve α
  - Continue until: $f(x_k - \alpha_k \nabla f(x_k)) \leq f(x_k) - c \alpha_k \|\nabla f(x_k)\|^2$
- **Gradient**:
  - $\nabla_U = (UV^T - M)V + \lambda U$
  - $\nabla_V = (UV^T - M)^T U + \lambda V$
- **Pros**: Simple, intuitive, first-order method
- **Cons**: Slow convergence, hyperparameter tuning (backtracking constant)

### Slide 10: Method 2 - L-BFGS (Limited-memory BFGS)
- **Quasi-Newton Method**: Approximate Hessian with limited history
  - Maintains m historical gradient/step pairs
  - Typically m=10 suffices
- **Key Idea**: Second-order information from first-order data
  - Better convergence than GD (~superlinear)
  - Lower memory than full BFGS
- **Implementation**: Wrapped via scipy's `minimize(method='L-BFGS-B')`
- **Factor Packing**: Flatten $U, V$ into vector $x = [U; V] \in \mathbb{R}^{mr+nr}$
- **Pros**: Fast convergence, robust, few hyperparameters
- **Cons**: More complex, requires gradient & Hessian-vector products

### Slide 11: Method 3 - Alternating Least Squares (ALS)
- **Block Coordinate Descent**: Optimize U and V alternately
  - **Fix V, optimize U**: Decouples into m independent ridge regressions
    $$\min_U \frac{1}{2}\|UV^T - M\|_F^2 + \frac{\lambda}{2}\|U\|_F^2$$
    Closed form: $U = MV(V^TV + \lambda I)^{-1}$
  - **Fix U, optimize V**: Symmetric
    $$V = M^T U(U^T U + \lambda I)^{-1}$$
- **Dense Implementation**: Direct Gram matrix solves
- **Sparse Implementation**: Row-by-row, only using observed entries
- **Pros**: 
  - Closed-form updates (no line search)
  - Natural handling of missing data
  - Fast per-iteration (especially sparse)
- **Cons**: May converge to different local minima than GD/L-BFGS

### Slide 12: Method Comparison Summary
| Method | Update | Convergence Class | Per-Iteration Cost | Sparse-Friendly |
|--------|--------|-------------------|-------------------|-----------------|
| GD | First-order line search | Linear (α-linear) | O(mn + r²) | Limited |
| L-BFGS | Quasi-Newton, memory=10 | Superlinear | O((m+n)r·history) | Possible |
| ALS | Block coordinate descent | Sublinear (provable) | O(nnz·r² + mr + nr) | **Yes** |

**Note**: Convergence rates are theoretical worst-case. See appendix for derivation of convergence classes and per-iteration complexity analysis.
---

## V. EXPERIMENTS & INVESTIGATIONS (5 minutes)

### Slide 13: Experiment 1 - Method Comparison
- **Question**: Which algorithm is fastest and most accurate?
- **Setup**: Run all three methods on synthetic, MovieLens, Superconductivity
- **Metrics**: 
  - Iterations to convergence
  - Wall-clock time
  - Final objective value
  - RMSE (sparse) / Reconstruction error (dense)
- **Key Results**:
  - ALS typically fastest per iteration but may need more iterations
  - L-BFGS convergence is superlinear (fewer iterations)
  - GD is slowest but stable

### Slide 14: Experiment 2 - Initialization Sensitivity
- **Question**: How sensitive is the problem to initialization?
- **Initializations Tested** (3 types):
  1. **Small random**: $U, V \sim \mathcal{N}(0, 0.01^2)$ × 10 seeds
  2. **Large random**: $U, V \sim \mathcal{N}(0, 1^2)$ × 10 seeds
  3. **SVD-based**: $U, V$ from truncated SVD of M (single run)
- **Experimental Design**: 
  - Total: 21 configurations (20 random + 1 SVD)
  - For each: run GD, L-BFGS, ALS to convergence
  - Track final objective, gradient norm, iterations, wall-clock time
- **Key Question**: Can poor initialization trap us in bad local minima? Does SVD protect from this?
- **Hypothesis**: SVD initialization should be more robust and reach better solutions

### Slide 15: Experiment 3 - Rank Selection
- **Question**: How does rank choice affect optimization and reconstruction quality?
- **Rank Grid**: [2, 3, 5, 8, 10, 20, 50] depending on dataset
- **Metrics**:
  - Training objective at convergence
  - Reconstruction error / RMSE
  - Singular value decay of true factors (synthetic)
- **Expected Behavior**:
  - Underfitting (low rank): high training error, poor reconstruction
  - Overfitting (high rank): may fit noise, especially with sparse data
  - Sweet spot: matches intrinsic rank
- **Diagnostic**: Plot error vs. rank (U-shaped curve expected)

### Slide 16: Experiment 4 - Regularization Sensitivity
- **Question**: How does λ affect optimization and generalization on noisy data?
- **Dataset**: **Noisy synthetic** (σ = 0.5, higher noise than standard σ = 0.1)
  - True rank-5 matrix with added Gaussian noise
  - All entries observed (dense)
- **Parameter Grid**: $\lambda \in [10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0]$ (5 values)
- **Metrics Tracked**:
  - Final training objective
  - **Recovery error**: $\|UV^T - M_{clean}\|_F / \|M_{clean}\|_F$
  - Relative reconstruction error on noisy matrix
- **Expected Findings**:
  - Small λ: fits noise, high recovery error to clean matrix
  - Large λ: underfitting, high training objective
  - Optimal λ recovers clean signal under noise

### Slide 17: Experiment 5 - Optimization Landscape
- **Question**: How do nonconvex effects appear in practice?
- **Setup** (using one method, e.g., GD):
  - Generate 8 solutions from different random seeds (scale=1.0)
  - Each runs to convergence on synthetic problem
  - Identify best & worst final objectives
  - Interpolate: evaluate objective along segment $f((1-t)U_{best} + tU_{worst}, (1-t)V_{best} + tV_{worst})$ for $t \in [0,1]$
- **Observable Nonconvex Behavior**:
  - Non-monotonic objective curve (not linear) = nonconvex
  - Saddle points or local minima along interpolation
  - Quality variance: $\Delta f \approx f(U_{worst}) - f(U_{best})$
  - Solution clustering: are runs trapped in different local minima?
- **Visualization**: 100-point line plot showing landscape shape

---

## VI. KEY FINDINGS & NONCONVEX EFFECTS (1.5 minutes)

### Slide 18: Empirical Observations
1. **Nonconvexity Doesn't Kill Optimization**
   - All three methods converge reliably (possibly to different local minima)
   - With good initialization, convergence is fast
   - ALS especially robust to local minima (block coordinate descent)

2. **Rank & Initialization Matter Most**
   - Underestimating rank: underfitting bias dominates
   - Overestimating rank: overfitting with sparse data
   - SVD initialization outperforms random init significantly

3. **Regularization Critical for Sparse Data**
   - MovieLens: λ ∈ [0.1, 1] works well
   - Overly large λ kills performance
   - Synthetic dense data: less sensitive to λ

4. **Method Trade-offs**:
   - **ALS**: Best for sparse data, fastest per iteration, good for real-time systems
   - **L-BFGS**: Fewest iterations, good for dense offline computation
   - **GD**: Most stable, easiest to debug, good pedagogical choice

### Slide 19: Practical Recommendations
- **For Sparse Data (e.g., Recommender Systems)**:
  - Use ALS with SVD initialization
  - Regularization λ = 0.1 to 1.0
  - Rank 20-50 for typical collaborative filtering

- **For Dense Data (e.g., Image Compression)**:
  - L-BFGS preferred (fewer iterations, fast wall-clock)
  - SVD initialization essential
  - Rank determined by required compression ratio

- **Nonconvex Challenges**:
  - Multiple local minima exist but quality varies slightly
  - Initialization more important than algorithm choice
  - Well-regularized problem → few local minima

---

## VII. CONCLUSIONS & FUTURE WORK (1 minute)

### Slide 20: Takeaways
1. **Low-Rank Factorization**: Powerful, practical tool across many domains
2. **Nonconvex ≠ Unsolvable**: With proper initialization & regularization, optimization is reliable
3. **Algorithm Choice**: Depends on data structure (dense vs. sparse) and application
4. **Open Questions**:
   - How to choose rank automatically? (cross-validation, information criteria)
   - Can we avoid local minima theoretically? (restrictive RIP conditions)
   - How to scale to billion-scale matrices? (stochastic, distributed ALS)

### Slide 21: References & Questions
- **Key Papers**:
  - Zhou et al. (2016): Alternating Least Squares review
  - Candès & Recht (2009): Matrix completion via convex optimization
  - Beck & Teboulle (2009): First-order methods convergence theory
- **Software**: TensorFlow/PyTorch, scikit-learn, scipy.optimize
- **Q&A**: Open discussion

---

## APPENDIX: Detailed Technical Notes & Proofs

### Smoothness & Convergence Theory

**Gradient Lipschitz Continuity (L-smooth)**:
- The objective $f(U,V) = \frac{1}{2}\|UV^T - M\|_F^2 + \frac{\lambda}{2}(\|U\|_F^2 + \|V\|_F^2)$ is L-smooth
- Smoothness constant: $L \leq \sigma_{max}(M)^2 + \lambda$ (approximately)
- Consequence: GD with step size $\alpha \leq 1/L$ converges
- Armijo line search in code ensures automatic step size adaptation

**Convergence Rate Derivations**:

1. **Gradient Descent**: α-linear convergence
   - For L-smooth, μ-strongly convex: $f(x_k) - f^* \leq (1 - \frac{\mu}{L})^k (f(x_0) - f^*)$
   - Our problem: convex in each factor but joint non-convexity complicates analysis
   - Empirically: linear convergence to stationary point (not guaranteed optimum)

2. **L-BFGS**: Superlinear convergence
   - Quasi-Newton maintains low-rank Hessian approximation
   - Convergence: $\|x_k - x^*\| = o(\|x_{k-1} - x^*\|)$ superlinearly
   - Practical: 2-3x fewer iterations than GD

3. **Alternating Least Squares**: Sublinear but guaranteed descent
   - Each block update: optimal solution for fixed other variable
   - Objective decreases every iteration (stationary point of block coordinate descent)
   - Convergence rate: typically sublinear $O(1/k)$ but can accelerate
   - Advantage: natural for sparse data (no dense matrix multiplication needed)

**Biconvexity**:
- $f(U, V)$ is jointly non-convex BUT
- For fixed $V$: minimizing over $U$ is convex (quadratic)
- For fixed $U$: minimizing over $V$ is convex (quadratic)
- This biconvexity guarantees ALS converges to stationary point, not escape to $\infty$

### Gradient Derivations

**Dense Case** (all entries observed):
$$\\nabla_U f = \\frac{\\partial}{\\partial U} \\left[ \\frac{1}{2}\\|UV^T - M\\|_F^2 + \\frac{\\lambda}{2}\\|U\\|_F^2 \\right]$$

Let $R = UV^T - M$ (residual). Then:
$$\\nabla_U f = RV + \\lambda U = (UV^T - M)V + \\lambda U$$

Similarly:
$$\\nabla_V f = R^T U + \\lambda V = (UV^T - M)^T U + \\lambda V$$

**Sparse Case** (only observed entries):
- For each observed triplet $(i,j,M_{ij})$: prediction $\\hat{m}_{ij} = u_i^T v_j = \\sum_r u_{ir} v_{jr}$
- Residual: $\\epsilon_{ij} = \\hat{m}_{ij} - M_{ij}$
- Loss on observed entries: $f = \\frac{1}{2}\\sum_{(i,j) \\in \\Omega} \\epsilon_{ij}^2 + \\frac{\\lambda}{2}(\\|U\\|_F^2 + \\|V\\|_F^2)$

Gradient (accumulated via `np.add.at` in code):
$$\\nabla_{u_i} f = \\sum_{j: (i,j) \\in \\Omega} \\epsilon_{ij} v_j + \\lambda u_i$$
$$\\nabla_{v_j} f = \\sum_{i: (i,j) \\in \\Omega} \\epsilon_{ij} u_i + \\lambda v_j$$

### Per-Iteration Computational Complexity

| Method | Dense ($m \\times n$, rank $r$) | Sparse ($nnz$ nonzeros) |\n|--------|--------|----------|\n| **GD** | Matrix products: $O(mn r)$ for $U \\times V$ and residual | $O(nnz \\cdot r)$ for gradients |\n| **L-BFGS** | Hessian-vector: $O(mnr)$; memory 10 vectors: $O(mr + nr)$ | Same per-iteration as GD |\n| **ALS Dense** | QR/Gram: $O(r^3 + mr^2 + nr^2)$ per block update | N/A |\n| **ALS Sparse** | Row-by-row solves: $O(nnz \\cdot r^2 + mr^2 + nr^2)$ | Optimal for sparse |\n\n**Key Insight**: ALS per-iteration is dominated by the number of nonzeros (sparse) or full matrix products (dense). GD is linear in matrix entries; L-BFGS adds memory overhead.\n\n### ALS Update Formulas\n\n**Fix $V$, optimize $U$** (decouples into $m$ independent ridge regressions):\n$$\\min_U \\frac{1}{2}\\|UV^T - M\\|_F^2 + \\frac{\\lambda}{2}\\|U\\|_F^2$$\n\nTaking derivative w.r.t. $U$ and setting to zero:\n$$0 = MV - UV^T V + \\lambda U = MV - U(V^TV + \\lambda I)$$\n$$U = MV(V^TV + \\lambda I)^{-1}$$\n\n**Fix $U$, optimize $V$** (symmetric):\n$$V = M^T U(U^T U + \\lambda I)^{-1}$$\n\n**Sparse Case**: Only use observed entries for each row/column, accumulating gradients to affected rows/columns only.\n\n### Data Structure Details
- **Dense Problem**: Store full $m \times n$ matrix M
- **Sparse Problem**: Store only observed $(i, j, M_{ij})$ triplets
  - Lookup tables: for each row i, list of column indices
  - Memory: O(nnz) instead of O(mn)

### Gradient Verification
- Numerical gradient: $\frac{\partial f}{\partial \theta} \approx \frac{f(\theta + \epsilon e_i) - f(\theta - \epsilon e_i)}{2\epsilon}$
- Analytic gradient: Compute directly from formula
- Check: $\|\text{numerical} - \text{analytic}\| < 10^{-5}$ (indicates correct implementation)

### Convergence Criteria (Actual Implementation)
- **GD**: 
  - Stop when gradient norm $\|\nabla f\| < 10^{-6}$ (tolerance: 1e-6)
  - Or max iterations (default 500)
- **L-BFGS**: 
  - scipy `minimize` stops when gradient norm tolerance met or max iterations
  - Default gtol = 1e-6
- **ALS (Dense & Sparse)**:
  - Stop when gradient norm $\|\nabla f\| < 10^{-6}$ OR
  - Objective relative change: $|f^{(k)} - f^{(k-1)}| / \max(1, |f^{(k-1)}|) < 10^{-8}$
  - Whichever comes first
  - Or max iterations (default 500)

### Sparse RMSE Computation
- Train/test RMSE: computed only on observed entries
  $$\text{RMSE} = \sqrt{\frac{1}{|\text{test}|} \sum_{(i,j) \in \text{test}} (M_{ij} - (UV^T)_{ij})^2}$$

---

## TIME ALLOCATION

| Section | Duration | Slides |
|---------|----------|--------|
| Intro & Motivation | 3 min | 1-4 |
| Applications & Datasets | 2 min | 5-8 |
| Methods | 3-4 min | 9-12 |
| Experiments | 5 min | 13-17 |
| Findings & Conclusions | 2.5 min | 18-21 |
| **Total** | **~15 min** | **21 slides** |

---

## VISUALIZATION SUGGESTIONS FOR SLIDES

### Slide 3: Image showing rank-1, rank-5, rank-20 approximation progression
### Slide 2: Project overview - summary table with algorithms, datasets, experiments
### Slide 8: Dataset characteristics with size/sparsity comparison chart
### Slide 13: Wall-clock time & iterations bar charts for three methods
### Slide 14: Box plots of final objective across 10 random seeds for 3 initializations
### Slide 15: Error vs. Rank curve (U-shaped) for each dataset
### Slide 16: Training objective & recovery error vs. λ (two curves)
### Slide 17: Line plots showing optimization landscape along interpolated segment
### Slide 20: Summary table with recommendations for different use cases

---

- **Key Takeaway**: "Nonconvex doesn't mean hopeless: good initialization + regularization makes the problem tractable"
- **Interactive Elements**: Show live demo of method convergence
- **Backup Slides**: Include extra detail on gradient derivations if questions arise
