# Mathematical Background

## Oscillator Lie Superalgebras

### Definition

The oscillator Lie superalgebra B(0,n) = osp(1|2n) is a Lie superalgebra
realized via the oscillator (Heisenberg–Clifford) construction. It consists of:

- **Even part** (bosonic): Dimension n² + n, including
  - Cartan subalgebra: H₁, ..., Hₙ
  - Positive/negative even root vectors: E_{±2δₖ}, E_{δₖ±δₗ} (k < l)
- **Odd part** (fermionic): Dimension 2n
  - Positive/negative odd root vectors: E_{±δₖ} (k = 1, ..., n)
- **Total dimension**: n² + 3n = n(n+3)

### Oscillator Realization

The generators are realized as quadratic expressions in oscillator operators:
- **Fermion**: a₀ (supplementary fermion with a₀² = 1/2)
- **Bosons**: bₖ⁺, bₖ⁻ (k = 1, ..., n) with [bₖ⁻, bₗ⁺] = δₖₗ

The odd generators have the form E_{±δₖ} = a₀ bₖ± and the even generators
are bilinear in the bosonic operators.

### Frappat Basis Convention

We follow the basis convention of Frappat, Sciarrino, and Sorba
(arXiv:hep-th/9607161), adapted to the oscillator realization.

## First-Order Deformations

### Setup

A first-order (inhomogeneous) deformation of B(0,n) is parameterized by
a matrix B ∈ M_{1×2n}(F) (for B(0,n)) that modifies the commutation
relations:

```
[bⱼ, aᵢ] = -B[i][j] · κ
```

where κ is a nilpotent element (κ² = 0) extending the base field.

### 2-Cocycle

The deformation determines a 2-cocycle γ: g ⊗ g → g via the deformed
bracket structure. Explicitly, γ(a,b) captures the first-order correction
to the Lie superbracket.

### Triviality Condition

A 2-cocycle γ is **trivial** (a coboundary) if there exists an odd linear
map f: g → g such that (Bakalov–Sullivan, Eq. 4.5):

```
γ(a,b) = (-1)^{p(a)} [a, f(b)] - (-1)^{(p(a)+1)·p(b)} [b, f(a)] - f([a,b])
```

where p(a) denotes the parity of a.

A trivial cocycle means the deformation is equivalent to the undeformed
algebra via a change of basis (automorphism of the extended algebra).

## Computational Approach

### Algorithm

For given n and deformation parameter B:

1. **Compute γ**: Use the oscillator realization to compute all
   γ(eᵢ, eⱼ) for basis vectors eᵢ, eⱼ.

2. **Build adjoint matrices**: For each basis vector x, compute the
   matrix (ad_x)ᵢⱼ defined by [x, eⱼ] = Σᵢ (ad_x)ᵢⱼ eᵢ.

3. **Solve for f**: The triviality condition becomes a linear system
   in the components of f. Solve via least-squares (the system is
   typically overdetermined).

4. **Check residual**: If ‖γ - δf‖ < ε for numerical tolerance ε,
   the cocycle is trivial.

### Key Results Verified

- **Dimension formula**: dim C¹_μ = 6n − 2 for each gh-weight sector μ
- **Certificate verification**: Algebraic certificates c satisfying
  c^T A = 0 that prove triviality symbolically
- **Numerical triviality**: For B(0,n) with n = 1, 2, 3, all first-order
  deformations are verified to be trivial

### Certificate Families

Three families of algebraic certificates are identified:

- **Family I** (E⁺ type): Certificates from positive root contributions
- **Family II** (E⁻ type): Certificates from negative root contributions
- **Family III** (Mixed type): Certificates from mixed root interactions

Each certificate c satisfies c^T A = 0 where A is the constraint matrix,
providing a symbolic (exact) proof of triviality independent of the
specific deformation parameter.

## References

1. Bakalov, B. and Sullivan, M. (2017). "Inhomogeneous variants of
   the oscillator Lie superalgebra and their deformations."
   arXiv:1612.09400v2.

2. Frappat, L., Sciarrino, A., and Sorba, P. (1996).
   "Dictionary on Lie Superalgebras." arXiv:hep-th/9607161.

3. Scheunert, M. (1979). "The Theory of Lie Superalgebras."
   Lecture Notes in Mathematics, Vol. 716, Springer.