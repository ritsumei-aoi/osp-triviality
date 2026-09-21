import InhomogeneousDeformations.SourceIndependence

/-!
# I106 R3 — `L0FamilyIndependent`, proved

Proves the **frozen** `Source.L0FamilyIndependent n` from `SourceIndependence.lean`, and
discharges `Source.IndependenceStatement n` unconditionally via the **frozen**
`independence_of_L0_and_F0`.

**Reduction (Step 0).** `L0 n u v = (1/4) • (Bu0 u * Bu0 v + Bu0 v * Bu0 u)` and
`Bu0 n u = B n u ᵍ⊗ₜ 1`. Since `1 ≠ 0` in `C`, a functional dual to `1` on `C` (exactly
`F0_linearIndependent`'s own device) collapses a vanishing combination of `L0`'s to the same
combination of `(1/4)•(B u * B v + B v * B u)` in `Module.End ℚ (WPoly n)` — the "purely Weyl
factor" statement the packet's own scope predicts.

**Sector identity (Step 1).** Rather than four separate hand-derived sector formulas, a single
general identity suffices, directly from the already-proved `A1` (`B_comm`):
`B u * B v = B v * B u + JnQ n u v • 1`, so `B u * B v + B v * B u = 2 • (B v * B u) + JnQ n u v •
1`. `JnQ n u v` is already fully computed by parity in `SourceWeyl.lean`
(`JnQ_eq_zero_of_same_parity`, `JnQ_evenIdx_oddIdx`, `JnQ_oddIdx_evenIdx`), so the correction term
falls out automatically without separate case lemmas.
-/

namespace InhomogeneousDeformations
namespace Source

open scoped TensorProduct

theorem Bu0_mul (n : ℕ) (u v : Fin (2 * n)) :
    Bu0 n u * Bu0 n v = (B n u * B n v) ᵍ⊗ₜ[ℚ] (1 : C) := by
  unfold Bu0
  exact GradedTensorProduct.tmul_one_mul_coe_tmul (𝒜 := WGrading n) (ℬ := CGrading)
    (B n u) (⟨B n v, trivial⟩ : WGrading n 0) (1 : C)

/-- `4 • L0 n u v`'s Weyl-factor image reduces, via `A1`, to `2 • (B v * B u) + JnQ n u v • 1` --
a single formula valid for every `u v`, with no case split on parity needed for the algebraic
identity itself. -/
theorem B_mul_add_B_mul_swap (n : ℕ) (u v : Fin (2 * n)) :
    B n u * B n v + B n v * B n u
      = (2 : ℚ) • (B n v * B n u) + JnQ n u v • (1 : Module.End ℚ (WPoly n)) := by
  have h : B n u * B n v = B n v * B n u + JnQ n u v • (1 : Module.End ℚ (WPoly n)) := by
    have := B_comm n u v
    unfold opComm at this
    rw [sub_eq_iff_eq_add] at this
    rw [this]
    abel
  rw [h]
  module

/-- **Step 0**: `L0FamilyIndependent` reduces to independence of the symmetrized `B`-products in
`Module.End ℚ (WPoly n)`. -/
theorem L0FamilyIndependent_of_weyl (n : ℕ)
    (h : LinearIndependent ℚ
      (fun p : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2} =>
        B n p.1.1 * B n p.1.2 + B n p.1.2 * B n p.1.1)) :
    L0FamilyIndependent n := by
  rw [L0FamilyIndependent, Fintype.linearIndependent_iff]
  intro g hg p
  obtain ⟨φ, hφ⟩ := Module.Projective.exists_dual_eq_one ℚ (one_ne_zero (α := C))
  set Ψ : A0 n →ₗ[ℚ] Module.End ℚ (WPoly n) :=
    (TensorProduct.rid ℚ (Module.End ℚ (WPoly n))).toLinearMap ∘ₗ
      (TensorProduct.map (LinearMap.id : Module.End ℚ (WPoly n) →ₗ[ℚ] _) φ) ∘ₗ
      (GradedTensorProduct.of ℚ (WGrading n) CGrading).symm.toLinearMap with hΨdef
  have hΨw : ∀ w : Module.End ℚ (WPoly n), Ψ (w ᵍ⊗ₜ[ℚ] (1 : C)) = w := by
    intro w
    show Ψ (GradedTensorProduct.of ℚ (WGrading n) CGrading (w ⊗ₜ (1 : C))) = w
    rw [hΨdef]
    simp [TensorProduct.map_tmul, hφ]
  have hΨL0 : ∀ q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2},
      Ψ (L0 n q.1.1 q.1.2) = (1 / 4 : ℚ) • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) := by
    intro q
    unfold L0
    rw [map_smul]
    congr 1
    rw [map_add, Bu0_mul, Bu0_mul, hΨw, hΨw]
  have key : ∑ q, g q • Ψ (L0 n q.1.1 q.1.2) =
      ∑ q, g q • ((1 / 4 : ℚ) • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1)) :=
    Finset.sum_congr rfl (fun q _ => by rw [hΨL0])
  simp only [← map_smul, ← map_sum] at key
  rw [hg, map_zero] at key
  have key2 : ∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) = 0 := by
    have h1 : (0 : Module.End ℚ (WPoly n)) =
        (1 / 4 : ℚ) • ∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) := by
      rw [key, Finset.smul_sum]
      exact Finset.sum_congr rfl (fun q _ => smul_comm (g q) (1 / 4 : ℚ) _)
    rcases smul_eq_zero.mp h1.symm with h2 | h2
    · norm_num at h2
    · exact h2
  exact (Fintype.linearIndependent_iff.mp h) g key2 p

/-- membership proof: `oddIdx k ≤ oddIdx l` when `k ≤ l`. -/
theorem oddIdx_le_oddIdx (n : ℕ) (k l : Fin n) (h : k ≤ l) :
    Indexed.oddIdx n k ≤ Indexed.oddIdx n l := by
  simp only [Fin.le_def, Indexed.oddIdx] at h ⊢
  omega

theorem evenIdx_le_evenIdx (n : ℕ) (k l : Fin n) (h : k ≤ l) :
    Indexed.evenIdx n k ≤ Indexed.evenIdx n l := by
  simp only [Fin.le_def, Indexed.evenIdx] at h ⊢
  omega

theorem evenIdx_le_oddIdx (n : ℕ) (k l : Fin n) (h : k ≤ l) :
    Indexed.evenIdx n k ≤ Indexed.oddIdx n l := by
  simp only [Fin.le_def, Indexed.evenIdx, Indexed.oddIdx] at h ⊢
  omega

theorem oddIdx_le_evenIdx_of_lt (n : ℕ) (k l : Fin n) (h : k < l) :
    Indexed.oddIdx n k ≤ Indexed.evenIdx n l := by
  simp only [Fin.lt_def, Indexed.oddIdx, Indexed.evenIdx] at h
  simp only [Fin.le_def, Indexed.oddIdx, Indexed.evenIdx]
  omega

/-- A polynomial value that is a scalar multiple of `1` has coefficient `0` at any nonzero
exponent. -/
theorem coeff_smul_one_eq_zero (n : ℕ) (c : ℚ) {m : Fin n →₀ ℕ} (hm : m ≠ 0) :
    MvPolynomial.coeff m (c • (1 : WPoly n)) = 0 := by
  rw [MvPolynomial.coeff_smul, MvPolynomial.coeff_one, if_neg (Ne.symm hm), smul_eq_mul, mul_zero]

theorem single_add_single_ne_zero (n : ℕ) (k l : Fin n) :
    Finsupp.single k (1 : ℕ) + Finsupp.single l 1 ≠ 0 := by
  intro h
  have h2 : (Finsupp.single k (1:ℕ) + Finsupp.single l 1 : Fin n →₀ ℕ) k = (0 : Fin n →₀ ℕ) k := by
    rw [h]
  simp only [Finsupp.add_apply, Finsupp.single_eq_same, Finsupp.coe_zero, Pi.zero_apply] at h2
  by_cases hlk : l = k <;> simp [hlk, Finsupp.single_apply] at h2

/-- **Step 2, stage 1 (evaluate at `1`)**: the value, for arbitrary `q`, of the coefficient of
`X_k X_l` in `(B u * B v + B v * B u) (1)`. Only the both-odd sector (`u = oddIdx k`,
`v = oddIdx l`) contributes; every other sector's value is a scalar multiple of `1`, hence has
zero coefficient at this (nonzero) exponent. -/
theorem coeff_single_add_single_X_mul_X (n : ℕ) (k l i j : Fin n) (hij : i ≤ j) (hkl : k ≤ l)
    (hne : ¬ (i = k ∧ j = l)) :
    MvPolynomial.coeff (Finsupp.single k 1 + Finsupp.single l 1)
      (MvPolynomial.X i * MvPolynomial.X j : WPoly n) = 0 := by
  have hX : (MvPolynomial.X i * MvPolynomial.X j : WPoly n)
      = MvPolynomial.monomial (Finsupp.single i 1 + Finsupp.single j 1) (1 : ℚ) := by
    rw [MvPolynomial.X, MvPolynomial.X, MvPolynomial.monomial_mul, mul_one]
  rw [hX, MvPolynomial.coeff_monomial]
  rw [if_neg]
  intro heq
  rw [Finsupp.single_add_single_eq_single_add_single (by norm_num : (1:ℕ) ≠ 0)
    (by norm_num : (1:ℕ) ≠ 0)] at heq
  rcases heq with ⟨hik, hjl⟩ | ⟨-, hil, hjk⟩ | ⟨h, -, -⟩
  · exact hne ⟨hik, hjl⟩
  · apply hne
    have h1 : (i:ℕ) = l := by rw [hil]
    have h2 : (j:ℕ) = k := by rw [hjk]
    have h3 : (i:ℕ) ≤ j := Fin.le_def.mp hij
    have h4 : (k:ℕ) ≤ l := Fin.le_def.mp hkl
    have h5 : (i:ℕ) = j := by omega
    have h6 : (k:ℕ) = l := by omega
    exact ⟨Fin.ext (by omega), Fin.ext (by omega)⟩
  · norm_num at h

theorem hval1 (n : ℕ) (k l : Fin n) (hkl : k ≤ l)
    (q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2}) :
    MvPolynomial.coeff (Finsupp.single k 1 + Finsupp.single l 1)
      ((B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) (1 : WPoly n))
      = if q.1.1 = Indexed.oddIdx n k ∧ q.1.2 = Indexed.oddIdx n l then (2 : ℚ) else 0 := by
  have oddIdx_eq_iff : ∀ i i' : Fin n, Indexed.oddIdx n i = Indexed.oddIdx n i' ↔ i = i' := by
    intro i i'; simp only [Fin.ext_iff, Indexed.oddIdx]; omega
  obtain ⟨⟨u, v⟩, huv⟩ := q
  simp only
  simp only [B_mul_add_B_mul_swap, LinearMap.add_apply, LinearMap.smul_apply,
    Module.End.mul_apply]
  by_cases hoo : (u : ℕ) % 2 = 1 ∧ (v : ℕ) % 2 = 1
  · obtain ⟨hu, hv⟩ := hoo
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.oddIdx n i :=
      ⟨wIndex n u, u_eq_oddIdx_of_odd n u (by omega)⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.oddIdx n j :=
      ⟨wIndex n v, u_eq_oddIdx_of_odd n v (by omega)⟩
    simp only [B_oddIdx, LinearMap.mulLeft_apply, mul_one, Module.End.one_apply,
      JnQ_eq_zero_of_same_parity n (Indexed.oddIdx n i) (Indexed.oddIdx n j)
        (by simp only [Indexed.oddIdx]; omega),
      zero_smul, add_zero, oddIdx_eq_iff, mul_comm (MvPolynomial.X j) (MvPolynomial.X i)]
    have hij : i ≤ j := by
      simp only [Fin.le_def, Indexed.oddIdx] at huv; simp only [Fin.le_def]; omega
    by_cases hik : i = k <;> by_cases hjl : j = l
    · subst hik; subst hjl
      rw [if_pos (And.intro rfl rfl)]
      simp only [MvPolynomial.coeff_smul, MvPolynomial.coeff_X_mul, MvPolynomial.coeff_single_X]
      norm_num
    · rw [if_neg (fun h => hjl h.2), MvPolynomial.coeff_smul,
        coeff_single_add_single_X_mul_X n k l i j hij hkl (fun h => hjl h.2), smul_zero]
    · rw [if_neg (fun h => hik h.1), MvPolynomial.coeff_smul,
        coeff_single_add_single_X_mul_X n k l i j hij hkl (fun h => hik h.1), smul_zero]
    · rw [if_neg (fun h => hik h.1), MvPolynomial.coeff_smul,
        coeff_single_add_single_X_mul_X n k l i j hij hkl (fun h => hik h.1), smul_zero]
  · rw [if_neg (by
      rintro ⟨rfl, rfl⟩
      apply hoo
      constructor <;> (simp only [Indexed.oddIdx]; omega))]
    by_cases hu : (u : ℕ) % 2 = 0 <;> by_cases hv : (v : ℕ) % 2 = 0
    · obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.evenIdx n i := ⟨wIndex n u, u_eq_evenIdx_of_even n u hu⟩
      obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.evenIdx n j := ⟨wIndex n v, u_eq_evenIdx_of_even n v hv⟩
      simp only [B_evenIdx, Derivation.coeFn_coe, MvPolynomial.pderiv_one, map_zero,
        Module.End.one_apply,
        JnQ_eq_zero_of_same_parity n (Indexed.evenIdx n i) (Indexed.evenIdx n j)
          (by simp only [Indexed.evenIdx]; omega),
        zero_smul, add_zero, smul_zero]
      exact MvPolynomial.coeff_zero _
    · obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.evenIdx n i := ⟨wIndex n u, u_eq_evenIdx_of_even n u hu⟩
      obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.oddIdx n j := ⟨wIndex n v, u_eq_oddIdx_of_odd n v hv⟩
      simp only [B_evenIdx, B_oddIdx, LinearMap.mulLeft_apply, Derivation.coeFn_coe,
        MvPolynomial.pderiv_one, mul_zero, Module.End.one_apply, JnQ_evenIdx_oddIdx, smul_zero,
        zero_add]
      exact coeff_smul_one_eq_zero n _ (single_add_single_ne_zero n k l)
    · obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.oddIdx n i := ⟨wIndex n u, u_eq_oddIdx_of_odd n u hu⟩
      obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.evenIdx n j := ⟨wIndex n v, u_eq_evenIdx_of_even n v hv⟩
      -- `u ≤ v` forces `i < j` strictly here (oddIdx i ≤ evenIdx j ⟺ 2i+1 ≤ 2j ⟺ i < j), so
      -- the JnQ correction (which only fires at i = j) never appears in this sector.
      have hij : i ≠ j := by
        simp only [Fin.le_def, Indexed.oddIdx, Indexed.evenIdx] at huv
        omega
      simp only [B_oddIdx, B_evenIdx, LinearMap.mulLeft_apply, mul_one, Derivation.coeFn_coe,
        MvPolynomial.pderiv_X_of_ne hij, Module.End.one_apply,
        JnQ_oddIdx_evenIdx, if_neg hij, zero_smul, add_zero, smul_zero]
      exact MvPolynomial.coeff_zero _
    · exact absurd ⟨by omega, by omega⟩ hoo

/-- **Stage 1**: the both-odd sector's coefficients all vanish. -/
theorem hOO (n : ℕ) (g : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2} → ℚ)
    (hg : ∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) = 0)
    (k l : Fin n) (hkl : k ≤ l) :
    g ⟨(Indexed.oddIdx n k, Indexed.oddIdx n l), oddIdx_le_oddIdx n k l hkl⟩ = 0 := by
  set target : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2} :=
    ⟨(Indexed.oddIdx n k, Indexed.oddIdx n l), oddIdx_le_oddIdx n k l hkl⟩ with htarget
  have hiff : ∀ q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2},
      (q.1.1 = Indexed.oddIdx n k ∧ q.1.2 = Indexed.oddIdx n l) ↔ q = target := by
    intro q
    rw [htarget, Subtype.ext_iff, Prod.ext_iff]
  have heval : (∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1)) (1 : WPoly n) = 0 := by
    rw [hg]; simp
  have hcoeff0 : MvPolynomial.coeff (Finsupp.single k 1 + Finsupp.single l 1)
      ((∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1)) (1 : WPoly n)) = 0 := by
    rw [heval]; simp
  rw [LinearMap.sum_apply] at hcoeff0
  simp only [LinearMap.smul_apply] at hcoeff0
  rw [MvPolynomial.coeff_sum] at hcoeff0
  simp only [MvPolynomial.coeff_smul, smul_eq_mul] at hcoeff0
  have hterm : ∀ q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2},
      g q * MvPolynomial.coeff (Finsupp.single k 1 + Finsupp.single l 1)
        ((B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) (1 : WPoly n))
      = if q = target then g q * 2 else 0 := by
    intro q
    rw [hval1 n k l hkl q]
    by_cases h : q.1.1 = Indexed.oddIdx n k ∧ q.1.2 = Indexed.oddIdx n l
    · rw [if_pos h, if_pos ((hiff q).mp h)]
    · rw [if_neg h, if_neg (fun h' => h ((hiff q).mpr h')), mul_zero]
  rw [Finset.sum_congr rfl (fun q _ => hterm q)] at hcoeff0
  rw [Finset.sum_ite_eq' Finset.univ target (fun q => g q * 2), if_pos (Finset.mem_univ _)]
    at hcoeff0
  linarith

/-- **Step 2, stage 2 (evaluate at `X_m`)**: the value, for arbitrary `q`, of the coefficient of
`X_p` in `(B u * B v + B v * B u) (X_m)`. The EE and OO sectors vanish automatically (degree
mismatch: EE gives `0`, OO gives a homogeneous degree-`3` polynomial, whose degree-`1`
coefficient is `0`); the EO sector contributes `2` when `(u,v) = (evenIdx m, oddIdx p)` and an
extra `1` exactly at the diagonal `(evenIdx m, oddIdx m)` with `p = m`; the OE sector
contributes `2` when `(u,v) = (oddIdx p, evenIdx m)`. -/
theorem hval2 (n : ℕ) (m p : Fin n) (hmp : m ≠ p)
    (q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2}) :
    MvPolynomial.coeff (Finsupp.single p 1)
      ((B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) (MvPolynomial.X m : WPoly n))
      = (if q.1.1 = Indexed.evenIdx n m ∧ q.1.2 = Indexed.oddIdx n p then (2 : ℚ) else 0)
        + (if q.1.1 = Indexed.oddIdx n p ∧ q.1.2 = Indexed.evenIdx n m then 2 else 0) := by
  have evenIdx_eq_iff : ∀ i i' : Fin n, Indexed.evenIdx n i = Indexed.evenIdx n i' ↔ i = i' := by
    intro i i'; simp only [Fin.ext_iff, Indexed.evenIdx]; omega
  have oddIdx_eq_iff : ∀ i i' : Fin n, Indexed.oddIdx n i = Indexed.oddIdx n i' ↔ i = i' := by
    intro i i'; simp only [Fin.ext_iff, Indexed.oddIdx]; omega
  have even_ne_odd : ∀ i j : Fin n, Indexed.evenIdx n i ≠ Indexed.oddIdx n j := by
    intro i j h; simp only [Fin.ext_iff, Indexed.evenIdx, Indexed.oddIdx] at h; omega
  obtain ⟨⟨u, v⟩, huv⟩ := q
  simp only
  simp only [B_mul_add_B_mul_swap, LinearMap.add_apply, LinearMap.smul_apply,
    Module.End.mul_apply]
  by_cases hu : (u : ℕ) % 2 = 0 <;> by_cases hv : (v : ℕ) % 2 = 0
  · -- EE: vanishes (second derivative of a linear polynomial), RHS also vanishes (parity clash)
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.evenIdx n i := ⟨wIndex n u, u_eq_evenIdx_of_even n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.evenIdx n j := ⟨wIndex n v, u_eq_evenIdx_of_even n v hv⟩
    have hpim : (MvPolynomial.pderiv i (MvPolynomial.X m : WPoly n)) = 0
        ∨ (MvPolynomial.pderiv i (MvPolynomial.X m : WPoly n)) = 1 := by
      by_cases him : i = m
      · subst him; exact Or.inr (MvPolynomial.pderiv_X_self i)
      · exact Or.inl (MvPolynomial.pderiv_X_of_ne (Ne.symm him))
    rw [if_neg (fun h => even_ne_odd j p h.2), if_neg (fun h => even_ne_odd i p h.1), add_zero]
    simp only [B_evenIdx, Derivation.coeFn_coe,
      Module.End.one_apply,
      JnQ_eq_zero_of_same_parity n (Indexed.evenIdx n i) (Indexed.evenIdx n j)
        (by simp only [Indexed.evenIdx]; omega),
      zero_smul, add_zero, smul_zero]
    rcases hpim with h | h <;> rw [h] <;> simp [MvPolynomial.pderiv_one]
  · -- EO: u = evenIdx i, v = oddIdx j, i ≤ j.
    -- Bv*Bu (applied to X_m) = mulLeft(X_j)(pderiv_i(X_m)) = X_j * (if i = m then 1 else 0).
    -- With hmp : m ≠ p, the diagonal JnQ correction term never contributes to this coefficient.
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.evenIdx n i := ⟨wIndex n u, u_eq_evenIdx_of_even n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.oddIdx n j := ⟨wIndex n v, u_eq_oddIdx_of_odd n v hv⟩
    rw [if_neg (fun h : Indexed.evenIdx n i = Indexed.oddIdx n p ∧ _ => even_ne_odd i p h.1)]
    simp only [B_evenIdx, B_oddIdx, LinearMap.mulLeft_apply, Derivation.coeFn_coe,
      Module.End.one_apply, evenIdx_eq_iff, JnQ_evenIdx_oddIdx, zero_add, true_and]
    by_cases him : i = m
    · subst him
      simp only [MvPolynomial.pderiv_X_self, mul_one, MvPolynomial.coeff_add,
        MvPolynomial.coeff_smul, MvPolynomial.coeff_single_X, smul_eq_mul, mul_one, mul_ite,
        mul_zero, mul_one, eq_self_iff_true, true_and]
      by_cases hjp : j = p <;> by_cases hij : i = p <;> simp_all [oddIdx_eq_iff]
    · rw [if_neg (fun h : i = m ∧ _ => him h.1)]
      simp only [MvPolynomial.pderiv_X_of_ne (Ne.symm him), mul_zero, zero_smul, zero_add]
      by_cases hij : i = j <;> by_cases hip : i = p <;>
        simp_all [oddIdx_eq_iff, MvPolynomial.coeff_single_X]
  · -- OE: u = oddIdx i, v = evenIdx j, i < j strictly.
    -- Bv*Bu (applied to X_m) = pderiv_j (mulLeft(X_i)(X_m)) = pderiv_j(X_i*X_m)
    --   = pderiv_j(X_i)*X_m + X_i*pderiv_j(X_m) = 0*X_m + X_i*(if j=m then 1 else 0)
    -- (the first term vanishes since i < j strictly forces i ≠ j), and the JnQ correction is 0
    -- for the same reason. So the value is `if j = m then 2•X_i else 0`.
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.oddIdx n i := ⟨wIndex n u, u_eq_oddIdx_of_odd n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.evenIdx n j := ⟨wIndex n v, u_eq_evenIdx_of_even n v hv⟩
    have hij : i ≠ j := by
      simp only [Fin.le_def, Indexed.oddIdx, Indexed.evenIdx] at huv; omega
    rw [if_neg (fun h : Indexed.oddIdx n i = Indexed.evenIdx n m ∧ _ =>
      even_ne_odd m i h.1.symm)]
    simp only [B_oddIdx, B_evenIdx, LinearMap.mulLeft_apply, Derivation.coeFn_coe,
      MvPolynomial.pderiv_mul, MvPolynomial.pderiv_X_of_ne hij, zero_mul,
      Module.End.one_apply, JnQ_oddIdx_evenIdx, if_neg hij, zero_smul, add_zero, zero_add,
      oddIdx_eq_iff]
    by_cases hjm : j = m
    · subst hjm
      simp only [MvPolynomial.pderiv_X_self, one_mul, MvPolynomial.coeff_smul,
        MvPolynomial.coeff_single_X, smul_eq_mul]
      by_cases hip : i = p <;> simp_all
    · simp only [MvPolynomial.pderiv_X_of_ne (Ne.symm hjm), mul_zero, smul_zero,
        MvPolynomial.coeff_zero, evenIdx_eq_iff]
      rw [if_neg (fun h => hjm h.2)]
  · -- OO: u = oddIdx i, v = oddIdx j. Value is `2•(X_i*X_j*X_m)`, a homogeneous degree-3
    -- polynomial, so its coefficient at the degree-1 exponent `single p 1` is always `0`.
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.oddIdx n i := ⟨wIndex n u, u_eq_oddIdx_of_odd n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.oddIdx n j := ⟨wIndex n v, u_eq_oddIdx_of_odd n v hv⟩
    rw [if_neg (fun h : Indexed.oddIdx n i = Indexed.evenIdx n m ∧ _ => even_ne_odd m i h.1.symm),
      if_neg (fun h : Indexed.oddIdx n i = Indexed.oddIdx n p ∧ _ => absurd h.2 (by
        intro h2; exact even_ne_odd m j h2.symm))]
    simp only [B_oddIdx, LinearMap.mulLeft_apply,
      JnQ_eq_zero_of_same_parity n (Indexed.oddIdx n i) (Indexed.oddIdx n j)
        (by simp only [Indexed.oddIdx]; omega),
      zero_smul, add_zero]
    have hne : (Finsupp.single i 1 + Finsupp.single j 1 + Finsupp.single m 1 : Fin n →₀ ℕ)
        ≠ Finsupp.single p 1 := by
      intro h
      have h3 : (Finsupp.single i 1 + Finsupp.single j 1 + Finsupp.single m 1 : Fin n →₀ ℕ).sum
          (fun _ v => v) = 3 := by
        simp [Finsupp.sum_add_index', Finsupp.sum_single_index]
      have h1 : (Finsupp.single p 1 : Fin n →₀ ℕ).sum (fun _ v => v) = 1 := by
        simp [Finsupp.sum_single_index]
      rw [h] at h3; omega
    rw [show (MvPolynomial.X j * (MvPolynomial.X i * MvPolynomial.X m) : WPoly n)
        = MvPolynomial.monomial (Finsupp.single i 1 + Finsupp.single j 1 + Finsupp.single m 1)
          (1 : ℚ) from by
      rw [MvPolynomial.X, MvPolynomial.X, MvPolynomial.X, MvPolynomial.monomial_mul,
        MvPolynomial.monomial_mul, mul_one, mul_one]
      congr 1
      abel,
      MvPolynomial.coeff_smul, MvPolynomial.coeff_monomial, if_neg hne, smul_eq_mul, mul_zero]

/-- **Step 2, stage 1b (evaluate at `1`, constant term)**: the value, for arbitrary `q`, of the
constant coefficient of `(B u * B v + B v * B u) (1)`. Only the diagonal EO pairs
(`u = evenIdx i`, `v = oddIdx i`, same `i`) contribute. -/
theorem hval0 (n : ℕ) (q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2}) :
    MvPolynomial.coeff 0 ((B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) (1 : WPoly n))
      = if ∃ i : Fin n, q.1.1 = Indexed.evenIdx n i ∧ q.1.2 = Indexed.oddIdx n i then (1 : ℚ)
        else 0 := by
  have evenIdx_eq_iff : ∀ i i' : Fin n, Indexed.evenIdx n i = Indexed.evenIdx n i' ↔ i = i' := by
    intro i i'; simp only [Fin.ext_iff, Indexed.evenIdx]; omega
  have oddIdx_eq_iff : ∀ i i' : Fin n, Indexed.oddIdx n i = Indexed.oddIdx n i' ↔ i = i' := by
    intro i i'; simp only [Fin.ext_iff, Indexed.oddIdx]; omega
  have even_ne_odd : ∀ i j : Fin n, Indexed.evenIdx n i ≠ Indexed.oddIdx n j := by
    intro i j h; simp only [Fin.ext_iff, Indexed.evenIdx, Indexed.oddIdx] at h; omega
  obtain ⟨⟨u, v⟩, huv⟩ := q
  simp only
  simp only [B_mul_add_B_mul_swap, LinearMap.add_apply, LinearMap.smul_apply,
    Module.End.mul_apply]
  by_cases hu : (u : ℕ) % 2 = 0 <;> by_cases hv : (v : ℕ) % 2 = 0
  · -- EE: pderiv_j (pderiv_i (1)) = 0. RHS: no i works (evenIdx ≠ oddIdx).
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.evenIdx n i := ⟨wIndex n u, u_eq_evenIdx_of_even n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.evenIdx n j := ⟨wIndex n v, u_eq_evenIdx_of_even n v hv⟩
    simp only [B_evenIdx, Derivation.coeFn_coe, MvPolynomial.pderiv_one, map_zero,
      Module.End.one_apply,
      JnQ_eq_zero_of_same_parity n (Indexed.evenIdx n i) (Indexed.evenIdx n j)
        (by simp only [Indexed.evenIdx]; omega),
      zero_smul, add_zero, smul_zero, MvPolynomial.coeff_zero]
    rw [if_neg]
    rintro ⟨k, -, hk⟩
    exact even_ne_odd j k hk
  · -- EO: value is `δ_ij` (a constant). RHS: `∃ k, i=k ∧ j=k` iff `i = j`.
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.evenIdx n i := ⟨wIndex n u, u_eq_evenIdx_of_even n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.oddIdx n j := ⟨wIndex n v, u_eq_oddIdx_of_odd n v hv⟩
    simp only [B_evenIdx, B_oddIdx, LinearMap.mulLeft_apply, Derivation.coeFn_coe,
      MvPolynomial.pderiv_one, mul_zero, Module.End.one_apply, JnQ_evenIdx_oddIdx, smul_zero,
      zero_add]
    by_cases hij : i = j
    · subst hij
      have hex : ∃ k : Fin n, Indexed.evenIdx n i = Indexed.evenIdx n k ∧
          Indexed.oddIdx n i = Indexed.oddIdx n k := ⟨i, rfl, rfl⟩
      simp [if_pos hex]
    · rw [if_neg hij]
      simp only [zero_smul, MvPolynomial.coeff_zero]
      symm
      rw [if_neg]
      rintro ⟨k, hk1, hk2⟩
      exact hij ((evenIdx_eq_iff i k).mp hk1 |>.trans ((oddIdx_eq_iff j k).mp hk2).symm)
  · -- OE: value is `0` (mulLeft(X i) applied to `1`, then pderiv_j, but the JnQ term is `0`
    -- since `i ≠ j` in this sector). RHS: no `k` works (evenIdx ≠ oddIdx).
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.oddIdx n i := ⟨wIndex n u, u_eq_oddIdx_of_odd n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.evenIdx n j := ⟨wIndex n v, u_eq_evenIdx_of_even n v hv⟩
    have hij : i ≠ j := by
      simp only [Fin.le_def, Indexed.oddIdx, Indexed.evenIdx] at huv; omega
    rw [if_neg (by rintro ⟨k, hk, -⟩; exact even_ne_odd k i hk.symm)]
    simp only [B_oddIdx, B_evenIdx, LinearMap.mulLeft_apply, mul_one, Derivation.coeFn_coe,
      MvPolynomial.pderiv_X_of_ne hij, Module.End.one_apply, JnQ_oddIdx_evenIdx,
      if_neg hij, zero_smul, add_zero, smul_zero, MvPolynomial.coeff_zero]
  · -- OO: value is `2•(X_i*X_j)`, a positive-degree polynomial, constant coeff `0`.
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.oddIdx n i := ⟨wIndex n u, u_eq_oddIdx_of_odd n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.oddIdx n j := ⟨wIndex n v, u_eq_oddIdx_of_odd n v hv⟩
    rw [if_neg (by rintro ⟨k, hk, -⟩; exact even_ne_odd k i hk.symm)]
    simp only [B_oddIdx, LinearMap.mulLeft_apply, mul_one,
      JnQ_eq_zero_of_same_parity n (Indexed.oddIdx n i) (Indexed.oddIdx n j)
        (by simp only [Indexed.oddIdx]; omega),
      zero_smul, add_zero, mul_comm (MvPolynomial.X j) (MvPolynomial.X i)]
    rw [show (MvPolynomial.X i * MvPolynomial.X j : WPoly n)
        = MvPolynomial.monomial (Finsupp.single i 1 + Finsupp.single j 1) (1 : ℚ) from by
      rw [MvPolynomial.X, MvPolynomial.X, MvPolynomial.monomial_mul, mul_one],
      MvPolynomial.coeff_smul, MvPolynomial.coeff_monomial,
      if_neg (by
        intro h
        have h2 : (Finsupp.single i 1 + Finsupp.single j 1 : Fin n →₀ ℕ).sum (fun _ v => v) = 2 := by
          simp [Finsupp.sum_add_index', Finsupp.sum_single_index]
        rw [h] at h2; simp at h2),
      smul_eq_mul, mul_zero]

/-- **Stage 1b**: the diagonal EO coefficients sum to zero. -/
theorem hS (n : ℕ) (g : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2} → ℚ)
    (hg : ∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) = 0) :
    ∑ i : Fin n, g ⟨(Indexed.evenIdx n i, Indexed.oddIdx n i), evenIdx_le_oddIdx n i i le_rfl⟩
      = 0 := by
  have heval : (∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1)) (1 : WPoly n) = 0 := by
    rw [hg]; simp
  have hcoeff0 : MvPolynomial.coeff 0
      ((∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1)) (1 : WPoly n)) = 0 := by
    rw [heval]; simp
  rw [LinearMap.sum_apply] at hcoeff0
  simp only [LinearMap.smul_apply] at hcoeff0
  rw [MvPolynomial.coeff_sum] at hcoeff0
  simp only [MvPolynomial.coeff_smul, smul_eq_mul] at hcoeff0
  rw [← hcoeff0]
  rw [show (∑ q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2}, g q *
        MvPolynomial.coeff 0 ((B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) (1 : WPoly n)))
      = ∑ q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2},
        (if ∃ i : Fin n, q.1.1 = Indexed.evenIdx n i ∧ q.1.2 = Indexed.oddIdx n i then g q else 0)
      from Finset.sum_congr rfl (fun q _ => by
        rw [hval0 n q]; by_cases h : ∃ i : Fin n, q.1.1 = Indexed.evenIdx n i ∧ q.1.2 = Indexed.oddIdx n i
        · rw [if_pos h, if_pos h, mul_one]
        · rw [if_neg h, if_neg h, mul_zero])]
  set φ : Fin n → {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2} :=
    fun i => ⟨(Indexed.evenIdx n i, Indexed.oddIdx n i), evenIdx_le_oddIdx n i i le_rfl⟩ with hφ
  have hφ_inj : Function.Injective φ := by
    intro i i' h
    have h1 : (φ i).1.1 = (φ i').1.1 := by rw [h]
    simp only [hφ] at h1
    simp only [Fin.ext_iff, Indexed.evenIdx] at h1
    exact Fin.ext (by omega)
  have hφ_apply : ∀ i, (φ i).1 = (Indexed.evenIdx n i, Indexed.oddIdx n i) := fun i => by rw [hφ]
  have hset : (Finset.univ : Finset {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2}).filter
      (fun q => ∃ i : Fin n, q.1.1 = Indexed.evenIdx n i ∧ q.1.2 = Indexed.oddIdx n i)
      = Finset.univ.image φ := by
    ext q
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_image]
    constructor
    · rintro ⟨i, hi1, hi2⟩
      refine ⟨i, ?_⟩
      apply Subtype.ext
      rw [hφ_apply i, ← hi1, ← hi2]
    · rintro ⟨i, hi⟩
      have h1 : q.1 = (Indexed.evenIdx n i, Indexed.oddIdx n i) := by rw [← hi, hφ_apply i]
      refine ⟨i, ?_, ?_⟩ <;> rw [h1]
  rw [← Finset.sum_filter, hset, Finset.sum_image (fun i _ i' _ h => hφ_inj h)]

/-- **Stage 2a**: the off-diagonal `EO` sector's coefficients vanish. -/
theorem hEO_offdiag (n : ℕ) (g : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2} → ℚ)
    (hg : ∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) = 0)
    (m p : Fin n) (hmp : m ≠ p) (hle : Indexed.evenIdx n m ≤ Indexed.oddIdx n p) :
    g ⟨(Indexed.evenIdx n m, Indexed.oddIdx n p), hle⟩ = 0 := by
  set target : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2} :=
    ⟨(Indexed.evenIdx n m, Indexed.oddIdx n p), hle⟩ with htarget
  have heval : (∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1))
      (MvPolynomial.X m : WPoly n) = 0 := by rw [hg]; simp
  have hcoeff0 : MvPolynomial.coeff (Finsupp.single p 1)
      ((∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1)) (MvPolynomial.X m)) = 0 := by
    rw [heval]; simp
  rw [LinearMap.sum_apply] at hcoeff0
  simp only [LinearMap.smul_apply] at hcoeff0
  rw [MvPolynomial.coeff_sum] at hcoeff0
  simp only [MvPolynomial.coeff_smul, smul_eq_mul] at hcoeff0
  have hterm : ∀ q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2},
      g q * MvPolynomial.coeff (Finsupp.single p 1)
        ((B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) (MvPolynomial.X m : WPoly n))
      = if q = target then g q * 2 else 0 := by
    intro q
    rw [hval2 n m p hmp q]
    have hiff1 : (q.1.1 = Indexed.evenIdx n m ∧ q.1.2 = Indexed.oddIdx n p) ↔ q = target := by
      rw [htarget, Subtype.ext_iff, Prod.ext_iff]
    have hiff2 : ¬ (q.1.1 = Indexed.oddIdx n p ∧ q.1.2 = Indexed.evenIdx n m) := by
      rintro ⟨h1, h2⟩
      have hle' := q.2
      rw [h1, h2] at hle'
      simp only [Fin.le_def, Indexed.oddIdx, Indexed.evenIdx] at hle'
      simp only [Fin.le_def, Indexed.evenIdx, Indexed.oddIdx] at hle
      have hmp' : (m : ℕ) ≠ (p : ℕ) := fun h => hmp (Fin.ext h)
      omega
    by_cases h : q = target
    · rw [if_pos ((hiff1).mpr h), if_pos h, if_neg hiff2, add_zero]
    · rw [if_neg (fun h' => h (hiff1.mp h')), if_neg h, if_neg hiff2, zero_add, mul_zero]
  rw [Finset.sum_congr rfl (fun q _ => hterm q)] at hcoeff0
  rw [Finset.sum_ite_eq' Finset.univ target (fun q => g q * 2), if_pos (Finset.mem_univ _)]
    at hcoeff0
  linarith

/-- **Stage 2b**: the `OE` sector's coefficients vanish (`i < j` strictly). -/
theorem hOE (n : ℕ) (g : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2} → ℚ)
    (hg : ∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) = 0)
    (p m : Fin n) (hpm : p < m) :
    g ⟨(Indexed.oddIdx n p, Indexed.evenIdx n m), oddIdx_le_evenIdx_of_lt n p m hpm⟩ = 0 := by
  set target : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2} :=
    ⟨(Indexed.oddIdx n p, Indexed.evenIdx n m), oddIdx_le_evenIdx_of_lt n p m hpm⟩ with htarget
  have heval : (∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1))
      (MvPolynomial.X m : WPoly n) = 0 := by rw [hg]; simp
  have hcoeff0 : MvPolynomial.coeff (Finsupp.single p 1)
      ((∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1)) (MvPolynomial.X m)) = 0 := by
    rw [heval]; simp
  rw [LinearMap.sum_apply] at hcoeff0
  simp only [LinearMap.smul_apply] at hcoeff0
  rw [MvPolynomial.coeff_sum] at hcoeff0
  simp only [MvPolynomial.coeff_smul, smul_eq_mul] at hcoeff0
  have hterm : ∀ q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2},
      g q * MvPolynomial.coeff (Finsupp.single p 1)
        ((B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) (MvPolynomial.X m : WPoly n))
      = if q = target then g q * 2 else 0 := by
    intro q
    rw [hval2 n m p (Ne.symm (ne_of_lt hpm)) q]
    have hiff1 : (q.1.1 = Indexed.oddIdx n p ∧ q.1.2 = Indexed.evenIdx n m) ↔ q = target := by
      rw [htarget, Subtype.ext_iff, Prod.ext_iff]
    have hiff2 : ¬ (q.1.1 = Indexed.evenIdx n m ∧ q.1.2 = Indexed.oddIdx n p) := by
      rintro ⟨h1, h2⟩
      have hle' := q.2
      rw [h1, h2] at hle'
      simp only [Fin.le_def, Indexed.evenIdx, Indexed.oddIdx] at hle'
      simp only [Fin.lt_def, Indexed.oddIdx, Indexed.evenIdx] at hpm
      omega
    by_cases h : q = target
    · rw [if_neg hiff2, if_pos ((hiff1).mpr h), if_pos h, zero_add]
    · rw [if_neg hiff2, if_neg (fun h' => h (hiff1.mp h')), if_neg h, zero_add, mul_zero]
  rw [Finset.sum_congr rfl (fun q _ => hterm q)] at hcoeff0
  rw [Finset.sum_ite_eq' Finset.univ target (fun q => g q * 2), if_pos (Finset.mem_univ _)]
    at hcoeff0
  linarith

/-- **Step 2, stage 2 (diagonal case, evaluate at `X_k`, coeff of `X_k` itself)**: the value, for
arbitrary `q`, of the coefficient of `X_k` in `(B u * B v + B v * B u) (X_k)`. Only the EO
sector contributes: `3` at the diagonal `q = (evenIdx k, oddIdx k)`, `1` at any other diagonal
`(evenIdx i, oddIdx i)`, `0` elsewhere. -/
theorem hval3 (n : ℕ) (k : Fin n) (q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2}) :
    MvPolynomial.coeff (Finsupp.single k 1)
      ((B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) (MvPolynomial.X k : WPoly n))
      = if q.1.1 = Indexed.evenIdx n k ∧ q.1.2 = Indexed.oddIdx n k then (3 : ℚ)
        else if ∃ i : Fin n, q.1.1 = Indexed.evenIdx n i ∧ q.1.2 = Indexed.oddIdx n i then 1
        else 0 := by
  have evenIdx_eq_iff : ∀ i i' : Fin n, Indexed.evenIdx n i = Indexed.evenIdx n i' ↔ i = i' := by
    intro i i'; simp only [Fin.ext_iff, Indexed.evenIdx]; omega
  have oddIdx_eq_iff : ∀ i i' : Fin n, Indexed.oddIdx n i = Indexed.oddIdx n i' ↔ i = i' := by
    intro i i'; simp only [Fin.ext_iff, Indexed.oddIdx]; omega
  have even_ne_odd : ∀ i j : Fin n, Indexed.evenIdx n i ≠ Indexed.oddIdx n j := by
    intro i j h; simp only [Fin.ext_iff, Indexed.evenIdx, Indexed.oddIdx] at h; omega
  obtain ⟨⟨u, v⟩, huv⟩ := q
  simp only
  simp only [B_mul_add_B_mul_swap, LinearMap.add_apply, LinearMap.smul_apply,
    Module.End.mul_apply]
  by_cases hu : (u : ℕ) % 2 = 0 <;> by_cases hv : (v : ℕ) % 2 = 0
  · -- EE: second derivative of a linear polynomial is `0`.
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.evenIdx n i := ⟨wIndex n u, u_eq_evenIdx_of_even n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.evenIdx n j := ⟨wIndex n v, u_eq_evenIdx_of_even n v hv⟩
    rw [if_neg (fun h => even_ne_odd j k h.2), if_neg (by rintro ⟨l, -, hl⟩; exact even_ne_odd j l hl)]
    have hpik : (MvPolynomial.pderiv i (MvPolynomial.X k : WPoly n)) = 0
        ∨ (MvPolynomial.pderiv i (MvPolynomial.X k : WPoly n)) = 1 := by
      by_cases hik : i = k
      · subst hik; exact Or.inr (MvPolynomial.pderiv_X_self i)
      · exact Or.inl (MvPolynomial.pderiv_X_of_ne (Ne.symm hik))
    simp only [B_evenIdx, Derivation.coeFn_coe,
      Module.End.one_apply,
      JnQ_eq_zero_of_same_parity n (Indexed.evenIdx n i) (Indexed.evenIdx n j)
        (by simp only [Indexed.evenIdx]; omega),
      zero_smul, add_zero, smul_zero]
    rcases hpik with h | h <;> rw [h] <;> simp [MvPolynomial.pderiv_one]
  · -- EO: value is `2•X_j•(if i=k then1else0) + (if i=j then1else0)•X_k`.
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.evenIdx n i := ⟨wIndex n u, u_eq_evenIdx_of_even n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.oddIdx n j := ⟨wIndex n v, u_eq_oddIdx_of_odd n v hv⟩
    simp only [B_evenIdx, B_oddIdx, LinearMap.mulLeft_apply, Derivation.coeFn_coe,
      Module.End.one_apply, evenIdx_eq_iff, JnQ_evenIdx_oddIdx]
    obtain rfl | hik := eq_or_ne i k
    · simp only [MvPolynomial.pderiv_X_self, mul_one, MvPolynomial.coeff_add,
        MvPolynomial.coeff_smul, MvPolynomial.coeff_single_X, smul_eq_mul, mul_one]
      obtain rfl | hij := eq_or_ne i j
      · have hex : ∃ l : Fin n, Indexed.evenIdx n i = Indexed.evenIdx n l ∧
            Indexed.oddIdx n i = Indexed.oddIdx n l := ⟨i, rfl, rfl⟩
        simp only [oddIdx_eq_iff, hex, if_pos, and_self]
        norm_num
      · have hji : j ≠ i := Ne.symm hij
        simp only [oddIdx_eq_iff]
        simp [hij, hji]
    · rw [MvPolynomial.pderiv_X_of_ne (Ne.symm hik), mul_zero, smul_zero, zero_add]
      obtain rfl | hij := eq_or_ne i j
      · simp only [if_pos rfl, one_smul, MvPolynomial.coeff_single_X]
        have hex : ∃ l : Fin n, Indexed.evenIdx n i = Indexed.evenIdx n l ∧
            Indexed.oddIdx n i = Indexed.oddIdx n l := ⟨i, rfl, rfl⟩
        simp [hik, hex]
      · have hne2 : ¬ ∃ l : Fin n, i = l ∧ Indexed.oddIdx n j = Indexed.oddIdx n l := by
          rintro ⟨l, hl1, hl2⟩
          exact hij ((((oddIdx_eq_iff j l).mp hl2).trans hl1.symm).symm)
        rw [if_neg hij, zero_smul, MvPolynomial.coeff_zero, if_neg (fun h => hik h.1),
          if_neg hne2]
  · -- OE: `i < j` strictly, so `i = j = k` is impossible; value is `2•X_i•(if j=k then1else0)`.
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.oddIdx n i := ⟨wIndex n u, u_eq_oddIdx_of_odd n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.evenIdx n j := ⟨wIndex n v, u_eq_evenIdx_of_even n v hv⟩
    have hij : i ≠ j := by
      simp only [Fin.le_def, Indexed.oddIdx, Indexed.evenIdx] at huv; omega
    rw [if_neg (fun h => even_ne_odd k i h.1.symm)]
    simp only [B_oddIdx, B_evenIdx, LinearMap.mulLeft_apply, Derivation.coeFn_coe,
      MvPolynomial.pderiv_mul, MvPolynomial.pderiv_X_of_ne hij, zero_mul,
      Module.End.one_apply, JnQ_oddIdx_evenIdx, if_neg hij, zero_smul, add_zero, zero_add]
    by_cases hjk : j = k
    · subst hjk
      simp only [MvPolynomial.pderiv_X_self, mul_one, MvPolynomial.coeff_smul,
        MvPolynomial.coeff_single_X, smul_eq_mul, mul_one]
      rw [if_neg (fun h => hij h.2), mul_zero,
        if_neg (by rintro ⟨l, hl1, -⟩; exact even_ne_odd l i hl1.symm)]
    · rw [MvPolynomial.pderiv_X_of_ne (Ne.symm hjk), mul_zero, smul_zero, MvPolynomial.coeff_zero]
      rw [if_neg (by rintro ⟨l, hl1, -⟩; exact even_ne_odd l i hl1.symm)]
  · -- OO: degree-3 mismatch, coefficient `0`.
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.oddIdx n i := ⟨wIndex n u, u_eq_oddIdx_of_odd n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.oddIdx n j := ⟨wIndex n v, u_eq_oddIdx_of_odd n v hv⟩
    rw [if_neg (fun h => even_ne_odd k i h.1.symm),
      if_neg (by rintro ⟨l, hl1, -⟩; exact even_ne_odd l i hl1.symm)]
    simp only [B_oddIdx, LinearMap.mulLeft_apply,
      JnQ_eq_zero_of_same_parity n (Indexed.oddIdx n i) (Indexed.oddIdx n j)
        (by simp only [Indexed.oddIdx]; omega),
      zero_smul, add_zero]
    have hne : (Finsupp.single i 1 + Finsupp.single j 1 + Finsupp.single k 1 : Fin n →₀ ℕ)
        ≠ Finsupp.single k 1 := by
      intro h
      have h3 : (Finsupp.single i 1 + Finsupp.single j 1 + Finsupp.single k 1 : Fin n →₀ ℕ).sum
          (fun _ v => v) = 3 := by
        simp [Finsupp.sum_add_index', Finsupp.sum_single_index]
      have h1 : (Finsupp.single k 1 : Fin n →₀ ℕ).sum (fun _ v => v) = 1 := by
        simp [Finsupp.sum_single_index]
      rw [h] at h3; omega
    rw [show (MvPolynomial.X j * (MvPolynomial.X i * MvPolynomial.X k) : WPoly n)
        = MvPolynomial.monomial (Finsupp.single i 1 + Finsupp.single j 1 + Finsupp.single k 1)
          (1 : ℚ) from by
      rw [MvPolynomial.X, MvPolynomial.X, MvPolynomial.X, MvPolynomial.monomial_mul,
        MvPolynomial.monomial_mul, mul_one, mul_one]
      congr 1
      abel,
      MvPolynomial.coeff_smul, MvPolynomial.coeff_monomial, if_neg hne, smul_eq_mul, mul_zero]

/-- **Stage 2c (diagonal EO)**: combining `hval3` (evaluate at `X_k`) with `hS` (`S = 0`), the
diagonal EO sector's coefficients vanish. -/
theorem hEO_diag (n : ℕ) (g : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2} → ℚ)
    (hg : ∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) = 0) (k : Fin n) :
    g ⟨(Indexed.evenIdx n k, Indexed.oddIdx n k), evenIdx_le_oddIdx n k k le_rfl⟩ = 0 := by
  set target : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2} :=
    ⟨(Indexed.evenIdx n k, Indexed.oddIdx n k), evenIdx_le_oddIdx n k k le_rfl⟩ with htarget
  have heval : (∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1))
      (MvPolynomial.X k : WPoly n) = 0 := by rw [hg]; simp
  have hcoeff0 : MvPolynomial.coeff (Finsupp.single k 1)
      ((∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1)) (MvPolynomial.X k)) = 0 := by
    rw [heval]; simp
  rw [LinearMap.sum_apply] at hcoeff0
  simp only [LinearMap.smul_apply] at hcoeff0
  rw [MvPolynomial.coeff_sum] at hcoeff0
  simp only [MvPolynomial.coeff_smul, smul_eq_mul] at hcoeff0
  have hiff : ∀ q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2},
      (q.1.1 = Indexed.evenIdx n k ∧ q.1.2 = Indexed.oddIdx n k) ↔ q = target := by
    intro q; rw [htarget, Subtype.ext_iff, Prod.ext_iff]
  have hterm : ∀ q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2},
      g q * MvPolynomial.coeff (Finsupp.single k 1)
        ((B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) (MvPolynomial.X k : WPoly n))
      = (if q = target then g q * 2 else 0)
        + (if ∃ i : Fin n, q.1.1 = Indexed.evenIdx n i ∧ q.1.2 = Indexed.oddIdx n i
            then g q else 0) := by
    intro q
    rw [hval3 n k q]
    by_cases h : q.1.1 = Indexed.evenIdx n k ∧ q.1.2 = Indexed.oddIdx n k
    · have hex : ∃ i : Fin n, q.1.1 = Indexed.evenIdx n i ∧ q.1.2 = Indexed.oddIdx n i :=
        ⟨k, h.1, h.2⟩
      rw [if_pos h, if_pos ((hiff q).mp h), if_pos hex]; ring
    · rw [if_neg h, if_neg (fun h' => h ((hiff q).mpr h'))]
      by_cases hex : ∃ i : Fin n, q.1.1 = Indexed.evenIdx n i ∧ q.1.2 = Indexed.oddIdx n i
      · simp [hex]
      · simp [hex]
  rw [Finset.sum_congr rfl (fun q _ => hterm q), Finset.sum_add_distrib] at hcoeff0
  rw [Finset.sum_ite_eq' Finset.univ target (fun q => g q * 2), if_pos (Finset.mem_univ _)]
    at hcoeff0
  have hφ_inj : Function.Injective
      (fun i : Fin n => (⟨(Indexed.evenIdx n i, Indexed.oddIdx n i),
        evenIdx_le_oddIdx n i i le_rfl⟩ : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2})) := by
    intro i i' h
    have h1 : Indexed.evenIdx n i = Indexed.evenIdx n i' := congrArg (fun x => x.1.1) h
    simp only [Fin.ext_iff, Indexed.evenIdx] at h1
    exact Fin.ext (by omega)
  have hset : (Finset.univ : Finset {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2}).filter
      (fun q => ∃ i : Fin n, q.1.1 = Indexed.evenIdx n i ∧ q.1.2 = Indexed.oddIdx n i)
      = Finset.univ.image (fun i : Fin n => (⟨(Indexed.evenIdx n i, Indexed.oddIdx n i),
          evenIdx_le_oddIdx n i i le_rfl⟩ : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2})) := by
    ext q
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_image]
    constructor
    · rintro ⟨i, hi1, hi2⟩
      refine ⟨i, ?_⟩
      apply Subtype.ext
      show (Indexed.evenIdx n i, Indexed.oddIdx n i) = q.1
      rw [← hi1, ← hi2]
    · rintro ⟨i, hi⟩
      have h1 : q.1 = (Indexed.evenIdx n i, Indexed.oddIdx n i) := by rw [← hi]
      refine ⟨i, ?_, ?_⟩ <;> rw [h1]
  rw [← Finset.sum_filter, hset, Finset.sum_image (fun i _ i' _ h => hφ_inj h)] at hcoeff0
  rw [hS n g hg] at hcoeff0
  linarith

/-- **Step 3, stage 1 (evaluate at `X_k * X_l`, constant term)**: the value, for arbitrary `q`, of
the constant coefficient of `(B u * B v + B v * B u) (X_k * X_l)`. Only the both-even sector
contributes (`u = evenIdx i`, `v = evenIdx j`); every other sector's value there is a polynomial
of positive degree, hence has zero constant coefficient. -/
theorem hval4 (n : ℕ) (k l : Fin n) (hkl : k ≤ l)
    (q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2}) :
    MvPolynomial.coeff 0
      ((B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1)
        (MvPolynomial.X k * MvPolynomial.X l : WPoly n))
      = if q.1.1 = Indexed.evenIdx n k ∧ q.1.2 = Indexed.evenIdx n l then
          (if k = l then (4 : ℚ) else 2) else 0 := by
  have evenIdx_eq_iff : ∀ i i' : Fin n, Indexed.evenIdx n i = Indexed.evenIdx n i' ↔ i = i' := by
    intro i i'; simp only [Fin.ext_iff, Indexed.evenIdx]; omega
  have oddIdx_eq_iff : ∀ i i' : Fin n, Indexed.oddIdx n i = Indexed.oddIdx n i' ↔ i = i' := by
    intro i i'; simp only [Fin.ext_iff, Indexed.oddIdx]; omega
  have even_ne_odd : ∀ i j : Fin n, Indexed.evenIdx n i ≠ Indexed.oddIdx n j := by
    intro i j h; simp only [Fin.ext_iff, Indexed.evenIdx, Indexed.oddIdx] at h; omega
  have coeff0_X_mul : ∀ (i : Fin n) (p : WPoly n),
      MvPolynomial.coeff (0 : Fin n →₀ ℕ) (MvPolynomial.X i * p) = 0 := by
    intro i p
    rw [← MvPolynomial.constantCoeff_eq, map_mul, MvPolynomial.constantCoeff_X, zero_mul]
  have coeff0_XkXl : MvPolynomial.coeff (0 : Fin n →₀ ℕ)
      (MvPolynomial.X k * MvPolynomial.X l : WPoly n) = 0 := coeff0_X_mul k (MvPolynomial.X l)
  have coeff0_mul_of_right : ∀ (A B : WPoly n), MvPolynomial.coeff (0 : Fin n →₀ ℕ) B = 0 →
      MvPolynomial.coeff (0 : Fin n →₀ ℕ) (A * B) = 0 := by
    intro A B hB
    rw [← MvPolynomial.constantCoeff_eq] at hB ⊢
    rw [map_mul, hB, mul_zero]
  have coeff0_mul_XkXl : ∀ A : WPoly n,
      MvPolynomial.coeff (0 : Fin n →₀ ℕ) (A * (MvPolynomial.X k * MvPolynomial.X l)) = 0 :=
    fun A => coeff0_mul_of_right A _ coeff0_XkXl
  obtain ⟨⟨u, v⟩, huv⟩ := q
  simp only
  simp only [B_mul_add_B_mul_swap, LinearMap.add_apply, LinearMap.smul_apply,
    Module.End.mul_apply]
  by_cases hu : (u : ℕ) % 2 = 0 <;> by_cases hv : (v : ℕ) % 2 = 0
  · -- EE: the interesting sector.
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.evenIdx n i := ⟨wIndex n u, u_eq_evenIdx_of_even n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.evenIdx n j := ⟨wIndex n v, u_eq_evenIdx_of_even n v hv⟩
    have hij : i ≤ j := by
      simp only [Fin.le_def, Indexed.evenIdx] at huv; simp only [Fin.le_def]; omega
    have hzero1 : ∀ P : Prop, [Decidable P] →
        MvPolynomial.pderiv j (if P then (1 : WPoly n) else 0) = 0 := by
      intro P _; split_ifs <;> simp
    have hpi : MvPolynomial.pderiv i (MvPolynomial.X k * MvPolynomial.X l : WPoly n)
        = (if i = k then (1 : WPoly n) else 0) * MvPolynomial.X l
          + MvPolynomial.X k * (if i = l then (1 : WPoly n) else 0) := by
      rw [MvPolynomial.pderiv_mul]
      have h1 : MvPolynomial.pderiv i (MvPolynomial.X k : WPoly n) = if i = k then 1 else 0 := by
        by_cases h : i = k
        · subst h; rw [if_pos rfl]; exact MvPolynomial.pderiv_X_self i
        · rw [if_neg h]; exact MvPolynomial.pderiv_X_of_ne (Ne.symm h)
      have h2 : MvPolynomial.pderiv i (MvPolynomial.X l : WPoly n) = if i = l then 1 else 0 := by
        by_cases h : i = l
        · subst h; rw [if_pos rfl]; exact MvPolynomial.pderiv_X_self i
        · rw [if_neg h]; exact MvPolynomial.pderiv_X_of_ne (Ne.symm h)
      rw [h1, h2]
    have hpjk : MvPolynomial.pderiv j (MvPolynomial.X k : WPoly n) = if j = k then 1 else 0 := by
      by_cases h : j = k
      · rw [if_pos h]; subst h; exact MvPolynomial.pderiv_X_self j
      · rw [if_neg h]; exact MvPolynomial.pderiv_X_of_ne (Ne.symm h)
    have hpjl : MvPolynomial.pderiv j (MvPolynomial.X l : WPoly n) = if j = l then 1 else 0 := by
      by_cases h : j = l
      · rw [if_pos h]; subst h; exact MvPolynomial.pderiv_X_self j
      · rw [if_neg h]; exact MvPolynomial.pderiv_X_of_ne (Ne.symm h)
    simp only [B_evenIdx, Derivation.coeFn_coe,
      Module.End.one_apply,
      JnQ_eq_zero_of_same_parity n (Indexed.evenIdx n i) (Indexed.evenIdx n j)
        (by simp only [Indexed.evenIdx]; omega),
      zero_smul, add_zero, evenIdx_eq_iff]
    rw [hpi, map_add, MvPolynomial.pderiv_mul, MvPolynomial.pderiv_mul, hzero1, hzero1,
      zero_mul, mul_zero, zero_add, add_zero, hpjl, hpjk]
    have hcoeff : ∀ P Q : Prop, [Decidable P] → [Decidable Q] →
        MvPolynomial.coeff (0 : Fin n →₀ ℕ)
          ((if P then (1 : WPoly n) else 0) * (if Q then (1 : WPoly n) else 0))
        = if P ∧ Q then (1 : ℚ) else 0 := by
      intro P Q _ _
      by_cases hP : P <;> by_cases hQ : Q <;> simp [hP, hQ]
    rw [MvPolynomial.coeff_smul, smul_eq_mul, MvPolynomial.coeff_add, hcoeff, hcoeff]
    have hij' : (i : ℕ) ≤ (j : ℕ) := Fin.le_def.mp hij
    have hkl' : (k : ℕ) ≤ (l : ℕ) := Fin.le_def.mp hkl
    by_cases hP : i = k ∧ j = l
    · rw [if_pos hP, if_pos hP]
      have e1 : (i : ℕ) = k := congrArg Fin.val hP.1
      have e2 : (j : ℕ) = l := congrArg Fin.val hP.2
      by_cases hkl2 : k = l
      · have hQ : j = k ∧ i = l := ⟨Fin.ext (by omega), Fin.ext (by omega)⟩
        rw [if_pos hkl2, if_pos hQ]; norm_num
      · have hQ : ¬ (j = k ∧ i = l) := by
          rintro ⟨h1, h2⟩
          exact hkl2 (Fin.ext (by
            have e3 := congrArg Fin.val h1; have e4 := congrArg Fin.val h2; omega))
        rw [if_neg hkl2, if_neg hQ]; norm_num
    · rw [if_neg hP, if_neg hP]
      have hQ : ¬ (j = k ∧ i = l) := by
        rintro ⟨h1, h2⟩
        apply hP
        have e1 : (j : ℕ) = k := congrArg Fin.val h1
        have e2 : (i : ℕ) = l := congrArg Fin.val h2
        exact ⟨Fin.ext (by omega), Fin.ext (by omega)⟩
      rw [if_neg hQ]; norm_num
  · -- EO: `B(evenIdx i) * B(oddIdx j)`'s image has an `X_j *` factor: coeff `0` vanishes.
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.evenIdx n i := ⟨wIndex n u, u_eq_evenIdx_of_even n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.oddIdx n j := ⟨wIndex n v, u_eq_oddIdx_of_odd n v hv⟩
    rw [if_neg (fun h => even_ne_odd l j (h.2.symm))]
    simp only [B_evenIdx, B_oddIdx, LinearMap.mulLeft_apply, Derivation.coeFn_coe,
      Module.End.one_apply, JnQ_evenIdx_oddIdx]
    simp only [MvPolynomial.coeff_add, MvPolynomial.coeff_smul, smul_eq_mul, coeff0_X_mul,
      coeff0_XkXl, mul_zero, add_zero]
  · -- OE: `pderiv j (X_i * (X_k * X_l))` splits by Leibniz into two coeff-`0`-vanishing terms.
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.oddIdx n i := ⟨wIndex n u, u_eq_oddIdx_of_odd n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.evenIdx n j := ⟨wIndex n v, u_eq_evenIdx_of_even n v hv⟩
    have hij : i ≠ j := by
      simp only [Fin.le_def, Indexed.oddIdx, Indexed.evenIdx] at huv; omega
    rw [if_neg (fun h => even_ne_odd k i (h.1.symm))]
    simp only [B_oddIdx, B_evenIdx, LinearMap.mulLeft_apply, Derivation.coeFn_coe,
      MvPolynomial.pderiv_mul, Module.End.one_apply, JnQ_oddIdx_evenIdx, if_neg hij, zero_smul,
      add_zero]
    simp only [MvPolynomial.coeff_add, MvPolynomial.coeff_smul, smul_eq_mul, coeff0_mul_XkXl,
      coeff0_X_mul, mul_zero, add_zero]
  · -- OO: `B(oddIdx i) * B(oddIdx j)`'s image has an `X_j *` factor: coeff `0` vanishes.
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.oddIdx n i := ⟨wIndex n u, u_eq_oddIdx_of_odd n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.oddIdx n j := ⟨wIndex n v, u_eq_oddIdx_of_odd n v hv⟩
    rw [if_neg (fun h => even_ne_odd k i (h.1.symm))]
    simp only [B_oddIdx, LinearMap.mulLeft_apply, Module.End.one_apply]
    simp only [MvPolynomial.coeff_add, MvPolynomial.coeff_smul, smul_eq_mul, coeff0_X_mul,
      coeff0_XkXl, mul_zero, add_zero]

/-- **Stage 3 (EE)**: the both-even sector's coefficients all vanish. -/
theorem hEE (n : ℕ) (g : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2} → ℚ)
    (hg : ∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) = 0)
    (k l : Fin n) (hkl : k ≤ l) :
    g ⟨(Indexed.evenIdx n k, Indexed.evenIdx n l), evenIdx_le_evenIdx n k l hkl⟩ = 0 := by
  set target : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2} :=
    ⟨(Indexed.evenIdx n k, Indexed.evenIdx n l), evenIdx_le_evenIdx n k l hkl⟩ with htarget
  have heval : (∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1))
      (MvPolynomial.X k * MvPolynomial.X l : WPoly n) = 0 := by rw [hg]; simp
  have hcoeff0 : MvPolynomial.coeff 0
      ((∑ q, g q • (B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1))
        (MvPolynomial.X k * MvPolynomial.X l : WPoly n)) = 0 := by
    rw [heval]; simp
  rw [LinearMap.sum_apply] at hcoeff0
  simp only [LinearMap.smul_apply] at hcoeff0
  rw [MvPolynomial.coeff_sum] at hcoeff0
  simp only [MvPolynomial.coeff_smul, smul_eq_mul] at hcoeff0
  have hiff : ∀ q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2},
      (q.1.1 = Indexed.evenIdx n k ∧ q.1.2 = Indexed.evenIdx n l) ↔ q = target := by
    intro q; rw [htarget, Subtype.ext_iff, Prod.ext_iff]
  have hterm : ∀ q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2},
      g q * MvPolynomial.coeff 0
        ((B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1)
          (MvPolynomial.X k * MvPolynomial.X l : WPoly n))
      = if q = target then g q * (if k = l then 4 else 2) else 0 := by
    intro q
    rw [hval4 n k l hkl q]
    by_cases h : q.1.1 = Indexed.evenIdx n k ∧ q.1.2 = Indexed.evenIdx n l
    · rw [if_pos h, if_pos ((hiff q).mp h)]
    · rw [if_neg h, if_neg (fun h' => h ((hiff q).mpr h')), mul_zero]
  rw [Finset.sum_congr rfl (fun q _ => hterm q)] at hcoeff0
  rw [Finset.sum_ite_eq' Finset.univ target (fun q => g q * (if k = l then 4 else 2)),
    if_pos (Finset.mem_univ _)] at hcoeff0
  by_cases hkl2 : k = l
  · rw [if_pos hkl2] at hcoeff0; linarith
  · rw [if_neg hkl2] at hcoeff0; linarith

/-- **Step 1 (assembled)**: the symmetrized `B`-products are linearly independent, combining all
four sector results (`hOO`, `hEO_offdiag` + `hEO_diag`, `hOE`, `hEE`). -/
theorem B_sum_linearIndependent (n : ℕ) :
    LinearIndependent ℚ
      (fun p : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2} =>
        B n p.1.1 * B n p.1.2 + B n p.1.2 * B n p.1.1) := by
  rw [Fintype.linearIndependent_iff]
  intro g hg q0
  obtain ⟨⟨u, v⟩, huv⟩ := q0
  by_cases hu : (u : ℕ) % 2 = 0 <;> by_cases hv : (v : ℕ) % 2 = 0
  · -- EE
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.evenIdx n i := ⟨wIndex n u, u_eq_evenIdx_of_even n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.evenIdx n j := ⟨wIndex n v, u_eq_evenIdx_of_even n v hv⟩
    have hij : i ≤ j := by
      simp only [Fin.le_def, Indexed.evenIdx] at huv; simp only [Fin.le_def]; omega
    exact hEE n g hg i j hij
  · -- EO
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.evenIdx n i := ⟨wIndex n u, u_eq_evenIdx_of_even n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.oddIdx n j := ⟨wIndex n v, u_eq_oddIdx_of_odd n v hv⟩
    have hij : i ≤ j := by
      simp only [Fin.le_def, Indexed.evenIdx, Indexed.oddIdx] at huv
      simp only [Fin.le_def]; omega
    rcases hij.lt_or_eq with hlt | heq
    · exact hEO_offdiag n g hg i j (ne_of_lt hlt) huv
    · subst heq
      exact hEO_diag n g hg i
  · -- OE
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.oddIdx n i := ⟨wIndex n u, u_eq_oddIdx_of_odd n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.evenIdx n j := ⟨wIndex n v, u_eq_evenIdx_of_even n v hv⟩
    have hij : i < j := by
      simp only [Fin.le_def, Indexed.oddIdx, Indexed.evenIdx] at huv
      simp only [Fin.lt_def]; omega
    exact hOE n g hg i j hij
  · -- OO
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.oddIdx n i := ⟨wIndex n u, u_eq_oddIdx_of_odd n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.oddIdx n j := ⟨wIndex n v, u_eq_oddIdx_of_odd n v hv⟩
    have hij : i ≤ j := by
      simp only [Fin.le_def, Indexed.oddIdx] at huv; simp only [Fin.le_def]; omega
    exact hOO n g hg i j hij

/-- **`L0FamilyIndependent`, proved unconditionally.** -/
theorem L0FamilyIndependent_proved (n : ℕ) : L0FamilyIndependent n :=
  L0FamilyIndependent_of_weyl n (B_sum_linearIndependent n)

/-- **`IndependenceStatement`, discharged unconditionally** via the frozen
`independence_of_L0_and_F0` splitting lemma. -/
theorem IndependenceStatement_proved (n : ℕ) : IndependenceStatement n :=
  independence_of_L0_and_F0 n (L0FamilyIndependent_proved n)
