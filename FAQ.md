# Frequently Asked Questions (FAQ)

## 1. What is Low-Rank Matrix Factorization?

**Q: What is Low-Rank Matrix Factorization?**

A: Low-rank matrix factorization is a technique that decomposes a large matrix $M \in \mathbb{R}^{m \times n}$ into a product of two smaller matrices:

$$M \approx U V^T$$

where $U \in \mathbb{R}^{m \times r}$ and $V \in \mathbb{R}^{n \times r}$, with $r \ll \min(m, n)$ being the **rank** of the factorization.

**Why is this useful?**
- **Dimensionality reduction**: Compresses large matrices while preserving important structure
- **Noise removal**: Can recover underlying low-rank structure from noisy observations
- **Prediction**: Used in recommendation systems (e.g., MovieLens) to predict missing values
- **Computational efficiency**: Requires only $r(m + n)$ parameters instead of $mn$ for the full matrix

---

## 2. Problem Formulation and Mathematical Framework

**Q: What is the exact optimization problem being solved?**

A: We solve the **regularized low-rank matrix factorization problem**:

$$\min_{U, V} f(U, V) = \frac{1}{2}\left\|UV^T - M\right\|_F^2 + \frac{\lambda}{2}\left(\|U\|_F^2 + \|V\|_F^2\right)$$

where:
- $U \in \mathbb{R}^{m \times r}$ and $V \in \mathbb{R}^{n \times r}$ are the factor matrices
- $M \in \mathbb{R}^{m \times n}$ is the observed matrix (dense or sparse)
- $\|\cdot\|_F$ denotes the **Frobenius norm**: $\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2}$
- $\lambda \geq 0$ is the **regularization parameter** (controls model complexity)

**Q: What about the sparse version?**

A: For sparse matrices with observed entries $\{(i, j, M_{ij}) : (i,j) \in \Omega\}$, we solve:

$$\min_{U, V} f(U, V) = \frac{1}{2}\sum_{(i,j) \in \Omega}(U_i V_j^T - M_{ij})^2 + \frac{\lambda}{2}\left(\|U\|_F^2 + \|V\|_F^2\right)$$

This optimizes only over observed ratings (e.g., in recommendation systems) rather than the entire matrix.

---

## 3. Key Mathematical Symbols and Notation

**Q: What are the standard mathematical notations used in this project?**

A: Here are the key symbols and their meanings:

| Symbol | Meaning | Type |
|--------|---------|------|
| $U$ | First factor matrix | $\mathbb{R}^{m \times r}$ |
| $V$ | Second factor matrix | $\mathbb{R}^{n \times r}$ |
| $M$ | Target/observed matrix | $\mathbb{R}^{m \times n}$ |
| $r$ | Rank (latent dimensionality) | scalar $\ll \min(m,n)$ |
| $m, n$ | Dimensions of $M$ | scalars |
| $\lambda$ | Regularization parameter | $\lambda > 0$ |
| $\nabla_U f$ | Gradient with respect to $U$ | $\mathbb{R}^{m \times r}$ |
| $\nabla_V f$ | Gradient with respect to $V$ | $\mathbb{R}^{n \times r}$ |
| $\alpha$ or $\eta$ | Step size/learning rate | scalar |
| $f(U, V)$ | Objective function value | $\mathbb{R}$ |
| $\Omega$ | Set of observed entries (sparse case) | index set |
| $\|\cdot\|_F$ | Frobenius norm | $\mathbb{R} \to \mathbb{R}_{\geq 0}$ |

---

## 4. Optimization Methods: Algorithms and Approaches

**Q: What three optimization methods does the project implement?**

A: This project compares three state-of-the-art optimization algorithms:

### 4.1 Gradient Descent with Armijo Backtracking (GD)

**Algorithm Overview:**
At each iteration $k$:
1. Compute gradient: $\mathbf{g}_k = \nabla f(U_k, V_k) = [\nabla_U f, \nabla_V f]^T$
2. Perform line search to find step size $\alpha_k$
3. Update: $\theta_{k+1} = \theta_k - \alpha_k \mathbf{g}_k$

**Gradient Formulation:**

For dense problems:
$$\nabla_U f = (UV^T - M)V + \lambda U$$
$$\nabla_V f = (UV^T - M)^T U + \lambda V$$

For sparse problems:
$$\nabla_U f = \sum_{(i,j) \in \Omega} (U_i V_j^T - M_{ij})V_j + \lambda U$$
$$\nabla_V f = \sum_{(i,j) \in \Omega} (U_i V_j^T - M_{ij})U_i + \lambda V$$

**Armijo Line Search Condition:**
We accept step size $\alpha_k$ if:
$$f(\theta_{k+1}) \leq f(\theta_k) - c \alpha_k \|\mathbf{g}_k\|^2$$

where $c = 10^{-4}$ (default) is the **Armijo constant**. This ensures sufficient decrease in the objective function.

**Backtracking Strategy:**
- Start with initial guess $\alpha = 1.0$
- If condition fails, shrink: $\alpha \leftarrow 0.5 \cdot \alpha$ (backtracking with $\rho = 0.5$)
- Continue until line search succeeds or $\alpha < 10^{-12}$ (failure)

**Convergence Criterion:**
- Stops when gradient norm $\|\mathbf{g}_k\|_2 \leq 10^{-6}$ or max iterations reached

### 4.2 L-BFGS (Limited-memory BFGS)

**Algorithm Overview:**
L-BFGS is a **quasi-Newton method** that approximates the second-order Hessian information without storing the full matrix.

**Key Idea:**
Instead of storing the full approximate Hessian $\mathbf{B}_k$ (requires $O(d^2)$ memory where $d = mr + nr$), L-BFGS stores only the last $m$ pairs of gradient differences and update vectors.

**Update Rule:**
$$\theta_{k+1} = \theta_k - \alpha_k \mathbf{B}_k^{-1} \nabla f(\theta_k)$$

where $\mathbf{B}_k^{-1}$ is implicitly reconstructed from stored history.

**Why Use L-BFGS Over GD?**
- **Faster convergence**: Superlinear convergence vs. linear for GD
- **Adaptive step sizes**: Incorporates second-order information
- **Memory efficient**: $O(mr + nr)$ memory instead of $O((mr + nr)^2)$

**Implementation:**
We use `scipy.optimize.minimize` with `method='L-BFGS-B'` for box-constrained optimization.

### 4.3 Alternating Least Squares (ALS)

**Algorithm Overview:**
ALS exploits the **bilinear structure** of the problem by optimizing $U$ and $V$ separately in alternating fashion.

**ALS Updates:**

**Fix $V$, optimize $U$:**
$$U^* = \arg\min_U \frac{1}{2}\|UV^T - M\|_F^2 + \frac{\lambda}{2}\|U\|_F^2$$

This is a **least-squares problem** with closed-form solution:
$$U_i^* = (V^TV + \lambda I)^{-1} V^T M_i^T$$

Or in matrix form:
$$U^* = M(V + \lambda I)^{-1}$$

**Fix $U$, optimize $V$:**
$$V^* = \arg\min_V \frac{1}{2}\|UV^T - M\|_F^2 + \frac{\lambda}{2}\|V\|_F^2$$

Solution:
$$V_j^* = (U^TU + \lambda I)^{-1} U^T M_j$$

Or in matrix form:
$$V^* = M^T(U + \lambda I)^{-1}$$

**Sparse Version:**
For sparse data, we update row-by-row and column-by-column using only observed entries:

For row $i$:
$$U_i^* = (V_{\text{obs}}^T V_{\text{obs}} + \lambda I)^{-1} V_{\text{obs}}^T M_{i,\text{obs}}$$

where $V_{\text{obs}}$ contains only columns corresponding to observed entries in row $i$.

**Advantages:**
- **Exact solutions**: Each subproblem is solved exactly (or via pseudo-inverse)
- **Stable convergence**: No step size tuning needed
- **Highly efficient for sparse data**: Complexity depends on sparsity pattern

---

## 5. Why These Optimization Techniques?

**Q: Why use gradient descent instead of directly solving the normal equations?**

A: While the normal equations $\mathbf{A}^T\mathbf{A}\mathbf{x} = \mathbf{A}^Tb$ have closed-form solutions, directly solving them is problematic:

1. **Non-convex problem**: The factorization problem is **non-convex** in $(U, V)$ jointly. There are no global optimality conditions from standard linear algebra.
2. **Numerical stability**: For ill-conditioned systems, solving normal equations amplifies numerical errors (condition number is squared).
3. **Sparse data handling**: Gradient-based methods naturally handle sparse, high-dimensional problems.
4. **Local convergence guarantees**: Gradient descent with backtracking ensures convergence to stationary points.

**Q: Why use Armijo line search in gradient descent?**

A: The Armijo condition provides:
- **Guaranteed descent**: Each step reduces the objective by at least $c\alpha_k \|\mathbf{g}_k\|^2$
- **Adaptive step sizes**: Large steps when far from optimum, small steps when close
- **Efficiency**: Avoids expensive exact line search while maintaining convergence
- **Robustness**: Works without problem-specific knowledge

**Q: Why use L-BFGS?**

A: L-BFGS combines advantages of both GD and Newton methods:
- **Superlinear convergence**: Faster than GD, approaching Newton's quadratic convergence near optimum
- **Memory efficient**: Practical for high-dimensional problems $(mr + nr)$ parameters
- **Automatic step sizing**: Uses approximate Hessian information
- **Black-box solver**: No algorithm tuning needed (works well out-of-the-box)

**Q: When should we use ALS instead of GD or L-BFGS?**

A: Use ALS when:
- **Bilinear structure is critical**: Problem must be solved repeatedly with changing data
- **Sparse matrices dominate**: Row/column-wise updates exploit sparsity patterns
- **Computational efficiency matters**: ALS avoids expensive gradient computations per iteration
- **Convergence behavior is stable**: No step size tuning; often faster initial convergence

---

## 6. Computational Complexity and Convergence Analysis

**Q: What is the computational complexity per iteration for each method?**

A: 

| Method | Dense Complexity | Sparse Complexity | Comments |
|--------|-----------------|-------------------|----------|
| **GD** | $O(mnr)$ | $O(\text{nnz} \cdot r)$ | Dominated by gradient computation $(UV^T - M)V$ |
| **L-BFGS** | $O(mnr)$ | $O(\text{nnz} \cdot r)$ | Similar to GD; Hessian approximation is $O(mr)$ storage |
| **ALS (dense)** | $O(r^3 + mnr)$ | $O(r^3 + \text{nnz} \cdot r)$ | Linear system solve: $O(r^3)$; matrix multiply: $O(mnr)$ |
| **ALS (sparse)** | — | $O(\text{nnz} \cdot r^2 / r)$ | Much cheaper for sparse; updates per row/column only |

where $\text{nnz}$ is the number of nonzero entries in sparse matrix.

**Q: What are the convergence rates?**

A:

**Gradient Descent:**
- **Local convergence rate**: Linear (geometric)
- **Rate constant**: Depends on Lipschitz constant $L$ of gradient
- **Asymptotic bound**: $\|\mathbf{g}_k\| = O(\rho^k)$ where $\rho < 1$

**L-BFGS:**
- **Superlinear convergence**: $\|\mathbf{g}_k\| = o(\rho^k)$ for any $\rho > 0$
- **Typically faster**: Quadratic-like near optimum
- **No exact rate**: Depends on problem conditioning

**ALS:**
- **Linear convergence**: Similar to GD but often with better constants
- **Problem-dependent**: Depends on spectrum of $(V^TV + \lambda I)^{-1}(V^TV)$ type matrices
- **Fast initial convergence**: Often beats GD for first 5-10 iterations

**Q: How do we measure convergence in practice?**

A: We monitor:

1. **Objective function**: $f(U_k, V_k)$ should monotonically decrease
2. **Gradient norm**: 
   $$\|\nabla f\|_F = \sqrt{\|\nabla_U f\|_F^2 + \|\nabla_V f\|_F^2}$$
   Converged when $\|\nabla f\|_F \leq 10^{-6}$

3. **Relative change**:
   $$\frac{|f(U_k, V_k) - f(U_{k-1}, V_{k-1})|}{|f(U_{k-1}, V_{k-1})|}$$
   Converged when this is $\leq 10^{-8}$

4. **Wall-clock time**: Total execution time in seconds

---

## 7. Regularization and Its Role

**Q: Why do we use $L_2$ regularization with parameter $\lambda$?**

A: Regularization serves multiple purposes:

1. **Prevents overfitting**: 
   - Without regularization, factors $U, V$ can grow arbitrarily large while maintaining small loss
   - Regularization term $\lambda(\|U\|_F^2 + \|V\|_F^2)$ keeps them bounded

2. **Improves numerical stability**:
   - Makes the problem better conditioned
   - Helps with sparse data where some rows/columns have few observations

3. **Implicit smoothing**:
   - Encourages simpler, smoother solutions
   - Relevant for noisy data

4. **Connection to Bayesian interpretation**:
   - Equivalent to $L_2$ prior on $U, V$: $P(U,V) \propto \exp(-\lambda(\|U\|^2 + \|V\|^2))$

**Q: How do we choose $\lambda$?**

A: Common strategies:

1. **Cross-validation**: Split data into train/validation, choose $\lambda$ minimizing validation error
2. **Grid search**: Try $\lambda \in \{10^{-3}, 10^{-2}, 10^{-1}, 1.0\}$ and compare
3. **Problem-specific**: 
   - MovieLens: typically $\lambda = 10^{-2}$ works well
   - Superconductivity: may need larger $\lambda$ due to noise
4. **Generalized cross-validation**: Automatic selection without explicit validation set

---

## 8. Initialization Strategies

**Q: How do we initialize $U_0$ and $V_0$?**

A: We provide two strategies:

### Random Initialization:
$$U_0 \sim \mathcal{N}(0, \sigma^2 I), \quad V_0 \sim \mathcal{N}(0, \sigma^2 I)$$

where $\sigma = 0.1$ (small scale to avoid large initial gradients).

**Pros:** Simple, requires no prior information
**Cons:** May converge to poor local minima

### SVD-based Initialization:
1. Compute truncated SVD: $M = U_{\text{SVD}}\Sigma_r V_{\text{SVD}}^T$ (rank-$r$ approximation)
2. Initialize:
   $$U_0 = U_{\text{SVD}} \sqrt{\Sigma_r}, \quad V_0 = V_{\text{SVD}} \sqrt{\Sigma_r}$$

where $\sqrt{\Sigma_r}$ is element-wise square root.

**Pros:** 
- Starts very close to optimal solution
- Better local optimum quality
- Faster convergence

**Cons:** 
- Expensive for large matrices ($O(mn \min(m,n))$)
- Requires dense matrix in memory
- May overfit for sparse data

---

## 9. Handling Dense vs. Sparse Data

**Q: How does the implementation handle sparse matrices?**

A: The project supports both **dense** and **sparse** representations:

### Dense Problems:
- Store full $m \times n$ matrix in memory
- Compute gradients: $\nabla_U f = (UV^T - M)V + \lambda U$
- All matrix operations use standard linear algebra

### Sparse Problems:
- Store only nonzero entries as coordinate format: $(i, j, M_{ij})$
- Build **lookup tables**:
  - `row_to_cols[i]`: columns with nonzero entries in row $i$
  - `row_to_vals[i]`: corresponding values
  - `col_to_rows[j]`: rows with nonzero entries in column $j$  
  - `col_to_vals[j]`: corresponding values

### Gradient Computation (Sparse):
$$\nabla_{U_i} f = \sum_{j \in \text{row\_to\_cols}[i]} (U_i V_j^T - M_{ij})V_j + \lambda U_i$$

Only computes contributions from observed entries.

### ALS for Sparse:
For each row $i$:
$$U_i = (V_{\text{obs}}^T V_{\text{obs}} + \lambda I)^{-1} V_{\text{obs}}^T M_{i,\text{obs}}$$

where $V_{\text{obs}} = V[\text{row\_to\_cols}[i]]$ contains only relevant columns.

**Benefits:**
- Memory: $O(\text{nnz})$ instead of $O(mn)$
- Time: $O(\text{nnz} \cdot r)$ instead of $O(mnr)$
- Critical for massive datasets (MovieLens: ~100K ratings, but $943 \times 1682 \approx 1.6M$ entries)

---

## 10. Datasets and Evaluation Metrics

**Q: What datasets are used, and how do we evaluate performance?**

A: The project tests on three dataset types:

### 1. Synthetic Low-Rank Matrices:
**Generation:**
- Create true factors $U_* \in \mathbb{R}^{m \times r}, V_* \in \mathbb{R}^{n \times r}$
- Generate data: $M = U_* V_*^T + \text{noise}$, where noise $\sim \mathcal{N}(0, \sigma^2)$
- Default: $m=500, n=200, r=5, \sigma=0.1$

**Evaluation:**
$$\text{Reconstruction Error} = \|U_{\text{recovered}}V_{\text{recovered}}^T - M_{\text{original}}\|_F$$

### 2. Superconductivity Dataset:
**Source:** UCI Machine Learning Repository (Dataset ID: 464)
**Properties:** 
- 21,263 samples × 81 features
- Regression problem (predicting superconducting temperature)
- Treated as dense matrix for factorization

**Evaluation:** 
- Test on held-out features/samples using reconstruction error

### 3. MovieLens 100K:
**Source:** Grouplens MovieLens dataset
**Properties:**
- 943 users × 1,682 movies
- ~100,000 ratings (5-star scale)
- Extremely sparse: $\approx 6.3\%$ density
- Multiple train/test splits (u1.base/u1.test, etc.)

**Evaluation Metric - RMSE on Test Set:**
$$\text{RMSE}_{\text{test}} = \sqrt{\frac{1}{|\Omega_{\text{test}}|}\sum_{(i,j) \in \Omega_{\text{test}}}(U_i V_j^T - M_{ij})^2}$$

where $\Omega_{\text{test}}$ are the held-out test ratings.

---

## 11. Experiments and Comparisons

**Q: What experiments does the project run?**

A:

### Experiment 1: Method Comparison
**Purpose:** Compare GD, L-BFGS, and ALS on standard setting
**Variables:** Fixed rank, fixed $\lambda$, vary dataset
**Metric:** Time to convergence, final objective value, solution quality

### Experiment 2: Initialization Sensitivity
**Purpose:** Assess random vs. SVD initialization impact
**Variables:** Initialization method, vary dataset
**Finding:** SVD init typically gives better solutions with faster convergence

### Experiment 3: Rank Sensitivity  
**Purpose:** Study effect of chosen rank $r$ vs. true rank $r_{\text{true}}$
**Variables:** Range of $r$ values, fixed $\lambda$
**Findings:**
- Best performance at $r = r_{\text{true}}$
- Underfitting when $r < r_{\text{true}}$ (bias)
- Overfitting when $r > r_{\text{true}}$ (variance)

### Experiment 4: Regularization Sensitivity
**Purpose:** Optimize regularization parameter $\lambda$
**Variables:** Grid of $\lambda$ values
**Findings:** U-shaped error curve; optimal $\lambda$ balances bias-variance

### Experiment 5: Loss Landscape
**Purpose:** Visualize optimization landscape
**Exploration:** 2D slice through factor space
**Findings:** Multiple local minima; algorithms converge to different solutions from different initializations

---

## 12. Stopping Criteria and Convergence Diagnostics

**Q: When does the algorithm stop iterating?**

A: Algorithms stop when ANY of these conditions is met:

1. **Gradient convergence**:
   $$\|\nabla f(U_k, V_k)\|_F \leq 10^{-6}$$

2. **Objective stalling**:
   $$\frac{|f_k - f_{k-1}|}{|f_{k-1}|} \leq 10^{-8}$$
   (Relative change too small to matter)

3. **Maximum iterations reached**: 
   Default: 500 iterations

4. **Failure modes**:
   - Line search fails (GD only): Cannot find acceptable step size
   - Singular linear system (ALS only): Pseudo-inverse fallback used

**Status Field:** The history dictionary tracks termination reason:
- `"converged"`: Gradient converged to tolerance
- `"objective_stalled"`: Function value stopped improving
- `"max_iter"`: Hit iteration limit
- `"line_search_failed"`: GD backtracking failed

---

## 13. Implementation Details and Numerical Considerations

**Q: What numerical precautions are taken?**

A:

1. **Pseudo-inverse fallback (ALS)**:
   - Try to solve $\mathbf{A}x = b$ via `np.linalg.solve()`
   - If matrix is singular, use `np.linalg.pinv()` (pseudo-inverse)
   - Ensures robust behavior even with ill-conditioned systems

2. **Gradient norm computation**:
   $$\|\nabla f\|_2 = \sqrt{\sum_{i,j}(\nabla_U f)_{ij}^2 + (\nabla_V f)_{ij}^2}$$
   Computed efficiently using summed squared elements

3. **Step size bounds**:
   - Minimum: $\alpha_{\min} = 10^{-12}$ (GD backtracking)
   - Initial guess: $\alpha_0 = 1.0$
   - Never exceeds: $\alpha_{\max} = 2 \times \alpha_{\text{previous}}$

4. **Packing/unpacking factors**:
   - For optimization algorithms requiring vector input (L-BFGS), concatenate:
   $$\mathbf{x} = [U_{\text{vec}}; V_{\text{vec}}]$$
   - Allows standard optimizers to work with matrix problems

---

## 14. Performance Benchmarking and Timing

**Q: How is computational time measured?**

A: Using `time.perf_counter()` for wall-clock time:

```python
start = time.perf_counter()
# ... run optimization ...
elapsed = time.perf_counter() - start
```

**Typical timings** (on moderate hardware, synthetic data $500 \times 200$, rank 5):

| Method | Iterations | Time | Time/Iteration |
|--------|-----------|------|-----------------|
| GD | 150-300 | 2-5s | ~15ms |
| L-BFGS | 40-80 | 1-3s | ~30ms |
| ALS | 20-50 | 0.5-2s | ~25ms |

**Q: What affects convergence speed?**

A:

1. **Initialization quality**
   - SVD init: 5-10× faster than random
   
2. **Rank selection**
   - Too large: Overfitting, slower convergence
   - Too small: Underfitting, may not converge to good solution

3. **Regularization strength $\lambda$**
   - Small $\lambda$: Longer to converge, higher risk of overfitting
   - Large $\lambda$: Faster, but solution less accurate

4. **Data properties**
   - Sparsity: Sparse data usually converges slower
   - Noise: Higher noise requires more iterations

5. **Problem conditioning**
   - Ill-conditioned: All methods slow down
   - Well-conditioned: Superlinear methods (L-BFGS) excel

---

## 15. Practical Tips and Troubleshooting

**Q: Which method should I use for my problem?**

A:

| Problem Type | Recommended Method | Reason |
|--------------|-------------------|--------|
| **Dense, small-medium** | L-BFGS | Best all-around performance |
| **Dense, very large** | GD | Memory efficient |
| **Sparse, large** | ALS | Exploits sparsity; row/column updates |
| **Quick prototyping** | GD | Simple, reliable |
| **Production system** | ALS or L-BFGS | Stable, predictable convergence |

**Q: How do I avoid poor local minima?**

A:

1. **Use SVD initialization** (if computationally feasible)
2. **Try multiple random seeds** and pick best result
3. **Increase rank slightly** (may help escape bad minima)
4. **Reduce regularization** (but watch for overfitting)
5. **Increase iterations** (sometimes helps find better solution)

**Q: My solution has high reconstruction error. What should I do?**

A:

**Check these issues:**

1. **Is rank sufficient?** Try $r + 1$ or $r + 2$
2. **Is $\lambda$ too large?** Reduces model capacity
3. **Is data noisy?** May need larger $\lambda$ or accept higher error
4. **Did algorithm converge?** Check gradient norm and objective value
5. **Is initialization poor?** Try SVD or multiple random seeds

**Q: Convergence is very slow. How can I speed it up?**

A:

1. **Switch to L-BFGS** (superlinear vs. linear GD)
2. **Use SVD initialization** (much better starting point)
3. **Reduce tolerance** (converge to $10^{-5}$ instead of $10^{-6}$)
4. **Optimize $\lambda$** (may be poorly chosen)
5. **Check data quality** (noisy data requires more iterations)
6. **For sparse data, use ALS** (best for MovieLens-type problems)

---

## 16. Mathematical Background References

**Q: Where can I learn more about the underlying mathematics?**

A:

**Key Concepts:**

| Topic | Relevance | Key Papers |
|-------|-----------|-----------|
| **Matrix factorization** | Core | Eckart & Young (1936) on SVD; Lee & Seung (2001) on NMF |
| **Optimization** | Methods | Boyd & Vandenberghe (2004) "Convex Optimization" (Ch. 1-3) |
| **Gradient descent** | GD method | Nesterov (2003) on accelerated methods |
| **Quasi-Newton methods** | L-BFGS | Dennis & More (1977); Nocedal & Wright (2006) |
| **Alternating optimization** | ALS | Csiszár & Tusnády (1984); Razaviyayn et al. (2013) |
| **Line search** | Armijo backtracking | Armijo (1966); Nocedal & Wright Ch. 3 |

---

## 17. Common Questions About Results and Outputs

**Q: What does the history dictionary contain?**

A:

```python
history = {
    "method": str,                  # "GD", "L-BFGS", or "ALS"
    "obj": List[float],             # Objective values over iterations
    "grad_norm": List[float],       # Gradient norms over iterations
    "time": List[float],            # Cumulative time in seconds
    "step_size": List[float],       # Step sizes (GD only)
    "U_final": ndarray,             # Final U matrix
    "V_final": ndarray,             # Final V matrix
    "status": str,                  # Convergence status
    "iterations": int,              # Total iterations taken
}
```

**Q: How do I extract and visualize results?**

A:

```python
# Convergence plot
import matplotlib.pyplot as plt
for method, hist in results.items():
    plt.semilogy(hist["iterations"], hist["grad_norm"], label=method)
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Gradient Norm")
plt.show()

# Reconstruction error
U_final = history["U_final"]
V_final = history["V_final"]
reconstruction = U_final @ V_final.T
error = np.linalg.norm(reconstruction - original_matrix)
```

---

## 18. Frequently Encountered Errors

**Q: I get "LinAlgError: Singular matrix" during ALS. What should I do?**

A: This happens when $V^T V + \lambda I$ or $U^T U + \lambda I$ is singular. 

**Solutions:**
1. **Increase $\lambda$**: Use larger regularization parameter (e.g., $10 \lambda$)
2. **Algorithm fallback**: Code automatically uses pseudo-inverse (ensure `allow_pinv=True`)
3. **Check rank**: May be too large for dataset size
4. **Data validation**: Ensure no all-zero rows/columns in data

**Q: Gradient check fails. What does this mean?**

A: Gradient check compares analytical gradients with numerical finite-difference approximation:
$$\text{relative\_error} = \frac{\|\nabla f_{\text{analytical}} - \nabla f_{\text{numerical}}\|}{\|\nabla f_{\text{numerical}}\|}$$

**If check fails** ($> 10^{-6}$ typically):
1. **Bug in gradient code**: Review implementation
2. **Numerical precision issue**: Use higher precision (float64 if using float32)
3. **Step size too large**: Reduce for numerical differentiation

---

## 19. Relationship Between Methods

**Q: How do GD, L-BFGS, and ALS relate to each other?**

A:

```
All three optimize: min_{U,V} 0.5||UV^T - M||_F^2 + λ(||U||_F^2 + ||V||_F^2)

    ↓

GD: Uses only first-order (gradient) information
   - Simple, stable
   - Slow convergence (linear)
   
L-BFGS: Uses approximate second-order (Hessian) information
   - More complex, but faster convergence (superlinear)
   - Memory efficient (limited-memory variant)
   
ALS: Exploits bilinear structure; alternates optimization
   - Solves subproblems exactly via least-squares
   - Natural for factorization; very efficient for sparse data
```

**Key Insight:** Different methods make different trade-offs between:
- **Convergence speed** (iterations to convergence)
- **Per-iteration cost** (time per iteration)
- **Implementation complexity** (code simplicity)
- **Hyperparameter tuning** (algorithm parameters to set)

---

## 20. FAQ Meta: How to Use This Document

**Q: How should I use this FAQ?**

A:

- **For understanding concepts**: Start with Sections 1-3 (Low-Rank, Problem, Notation)
- **For algorithm details**: Read Sections 4-6 (Methods, Why, Complexity)
- **For practical usage**: Jump to Sections 13-18 (Implementation, Troubleshooting)
- **For comparison**: Use tables in Sections 4-6 and 15
- **For mathematical rigor**: Sections 2, 4-5, and 16 have detailed formulas

---

## Quick Reference Card

| Question | Section |
|----------|---------|
| What is low-rank factorization? | #1 |
| What problem are we solving? | #2, #3 |
| Which algorithm is fastest? | #4, #6, #15 |
| How do I choose hyperparameters? | #7, #15 |
| Why is my solution bad? | #15 |
| What datasets are available? | #10 |
| How do I interpret results? | #17 |
| Something went wrong! | #18 |

