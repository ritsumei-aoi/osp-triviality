import InhomogeneousDeformations.Native

/-!
# Independent oracle (T1), degree/skew (T2), Jacobi (T3)

`oracleFull` is an independent literal transcription of the fifteen-row table
in I105_R1A_N1_ORACLE_AND_TESTS.md §1, reversed to all 25 ordered pairs by the
document's own stated skew rule `[Y,X]=-(-1)^{p(X)p(Y)}[X,Y]`, applied here by
hand (not derived from `Native.bracket`/`bracketBasis`). `bracket_eq_oracle`
(T1) is the independent comparison; native formulas are not defined by this
table.
-/

namespace InhomogeneousDeformations

open Basis5

@[simp] lemma crat_smul_e (q : ℚ) (b : Basis5) (k : Basis5) :
    (crat q • e b) k = if k = b then crat q else 0 := by
  simp [crat, e, smul_eq_mul]

/-- `crat` is (the restriction of) a ring homomorphism `ℚ →+* Coeff`: these let
`simp` collapse any combination of `crat`-constants and raw `Coeff` numerals
down to a single `crat (single rational)`, closeable by `norm_num`/injectivity. -/
@[simp] lemma crat_zero : crat (0 : ℚ) = 0 := by simp [crat]
@[simp] lemma crat_one : crat (1 : ℚ) = 1 := by simp [crat]
@[simp] lemma crat_add (a b : ℚ) : crat a + crat b = crat (a + b) := by
  simp [crat]
@[simp] lemma crat_mul (a b : ℚ) : crat a * crat b = crat (a * b) := by
  simp [crat]
@[simp] lemma crat_neg (a : ℚ) : -crat a = crat (-a) := by
  simp [crat]
@[simp] lemma crat_natCast (n : ℕ) : (n : Coeff) = crat (n : ℚ) := by
  simp [crat]
lemma crat_injective : Function.Injective crat := by
  intro a b hab
  have h0 : MvPolynomial.coeff (0 : Fin 2 →₀ ℕ) (crat a) = MvPolynomial.coeff 0 (crat b) :=
    congrArg (MvPolynomial.coeff 0) hab
  simpa [crat, MvPolynomial.coeff_C] using h0
@[simp] lemma crat_eq_iff (a b : ℚ) : crat a = crat b ↔ a = b := crat_injective.eq_iff
@[simp] lemma crat_neg_one : crat (-1 : ℚ) = -1 := by
  rw [show (-1 : ℚ) = -(1 : ℚ) from rfl, ← crat_neg, crat_one]
lemma crat_two : (2 : Coeff) = crat (2 : ℚ) := by
  rw [show (2 : ℚ) = 1 + 1 from by norm_num, ← crat_add, crat_one]
  norm_num
@[simp] lemma crat_two_mul (a : ℚ) : (2 : Coeff) * crat a = crat (2 * a) := by
  rw [crat_two, crat_mul]
@[simp] lemma crat_mul_two (a : ℚ) : crat a * (2 : Coeff) = crat (a * 2) := by
  rw [crat_two, crat_mul]

/-- Independent literal fifteen-row oracle, reversed to all 25 ordered pairs. -/
noncomputable def oracleFull : Basis5 → Basis5 → Mod
  | .L11, .L11 => crat 0 • e .L11
  | .L11, .L12 => crat 1 • e .L11
  | .L11, .L22 => crat 2 • e .L12
  | .L11, .F1  => crat 0 • e .F1
  | .L11, .F2  => crat 1 • e .F1
  | .L12, .L11 => crat (-1) • e .L11
  | .L12, .L12 => crat 0 • e .L12
  | .L12, .L22 => crat 1 • e .L22
  | .L12, .F1  => crat (-1 / 2) • e .F1
  | .L12, .F2  => crat (1 / 2) • e .F2
  | .L22, .L11 => crat (-2) • e .L12
  | .L22, .L12 => crat (-1) • e .L22
  | .L22, .L22 => crat 0 • e .L22
  | .L22, .F1  => crat (-1) • e .F2
  | .L22, .F2  => crat 0 • e .F2
  | .F1,  .L11 => crat 0 • e .F1
  | .F1,  .L12 => crat (1 / 2) • e .F1
  | .F1,  .L22 => crat 1 • e .F2
  | .F1,  .F1  => crat (1 / 2) • e .L11
  | .F1,  .F2  => crat (1 / 2) • e .L12
  | .F2,  .L11 => crat (-1) • e .F1
  | .F2,  .L12 => crat (-1 / 2) • e .F2
  | .F2,  .L22 => crat 0 • e .F2
  | .F2,  .F1  => crat (1 / 2) • e .L12
  | .F2,  .F2  => crat (1 / 2) • e .L22

/-- The double-sum bilinear extension collapses to the basis table on basis vectors. -/
lemma bracket_e_e (i j : Basis5) : bracket (e i) (e j) = bracketBasis i j := by
  unfold bracket
  rw [Finset.sum_eq_single i]
  · rw [Finset.sum_eq_single j]
    · simp
    · intro b _ hb; simp [e_apply_ne hb]
    · intro h; exact absurd (Finset.mem_univ j) h
  · intro b _ hb
    have : e i b = 0 := e_apply_ne hb
    simp [this]
  · intro h; exact absurd (Finset.mem_univ i) h

/-- `T1`: the native bracket, evaluated on basis vectors, equals the
independent oracle on all 25 ordered pairs. -/
theorem bracket_eq_oracle : ∀ i j : Basis5, bracket (e i) (e j) = oracleFull i j := by
  intro i j
  rw [bracket_e_e]
  cases i <;> cases j <;>
    (funext k; fin_cases k <;>
      simp [bracketBasis, bracketLL, bracketLF, bracketFF, oracleFull, Lof, Fof, Jmat, e,
        neg_smul, smul_smul] <;>
      norm_num)

/-- Parity sign helper for the graded formulas, `(-1)^(p·q)`, valued in `Coeff`. -/
noncomputable def gsign (p q : ZMod 2) : Coeff := if p = 1 ∧ q = 1 then crat (-1) else crat 1

/-- `T2` (degree0): each basis-pair bracket output is supported at the summed parity. -/
theorem bracket_degree0 (i j k : Basis5) (h : bracket (e i) (e j) k ≠ 0) :
    Basis5.parity k = Basis5.parity i + Basis5.parity j := by
  rw [bracket_eq_oracle] at h
  cases i <;> cases j <;> cases k <;>
    simp_all [oracleFull, Basis5.parity] <;> decide

set_option maxHeartbeats 800000 in
/-- `T2` (super-skew) on basis vectors: `[X,Y] = -(-1)^{p(X)p(Y)}[Y,X]`. -/
theorem bracket_super_skew (i j : Basis5) :
    bracket (e i) (e j) = -(gsign (Basis5.parity i) (Basis5.parity j)) • bracket (e j) (e i) := by
  rw [bracket_eq_oracle, bracket_eq_oracle]
  cases i <;> cases j <;> funext k <;> fin_cases k <;>
    simp [oracleFull, gsign, Basis5.parity, neg_smul, smul_smul, smul_eq_mul] <;> norm_num

set_option maxHeartbeats 800000 in
/-- `T3`: the graded Jacobi identity on all 125 ordered n=1 basis triples. -/
theorem jacobi125 (X Y Z : Basis5) :
    (gsign (Basis5.parity X) (Basis5.parity Z)) • bracket (e X) (bracket (e Y) (e Z))
      + (gsign (Basis5.parity Y) (Basis5.parity X)) • bracket (e Y) (bracket (e Z) (e X))
      + (gsign (Basis5.parity Z) (Basis5.parity Y)) • bracket (e Z) (bracket (e X) (e Y))
      = 0 := by
  cases X <;> cases Y <;> cases Z <;>
    simp only [bracket_eq_oracle, oracleFull, bracket_smul_right] <;>
    (funext k; fin_cases k) <;>
    simp [gsign, Basis5.parity, crat_smul_e, smul_smul, smul_add, add_smul, neg_smul,
      smul_eq_mul] <;>
    ring_nf <;>
    (try simp [crat_add, crat_mul, crat_two_mul, crat_mul_two, crat_natCast, crat_zero, crat_one,
      crat_neg, crat_neg_one, crat_eq_iff]) <;>
    norm_num

/-- `bracket 0 y = 0` and `bracket x 0 = 0`: the finite double sum vanishes term
by term since `(0 : Mod) i = 0`. -/
@[simp] lemma bracket_zero_left (y : Mod) : bracket 0 y = 0 := by
  unfold bracket; simp
@[simp] lemma bracket_zero_right (x : Mod) : bracket x 0 = 0 := by
  unfold bracket; simp

/-- `bracket` distributes over an arbitrary finite sum in either argument. -/
lemma bracket_sum_left (s : Finset Basis5) (f : Basis5 → Mod) (y : Mod) :
    bracket (∑ i ∈ s, f i) y = ∑ i ∈ s, bracket (f i) y := by
  induction s using Finset.induction with
  | empty => simp
  | @insert a s ha ih =>
    rw [Finset.sum_insert ha, bracket_add_left, ih, Finset.sum_insert ha]

lemma bracket_sum_right (s : Finset Basis5) (f : Basis5 → Mod) (x : Mod) :
    bracket x (∑ j ∈ s, f j) = ∑ j ∈ s, bracket x (f j) := by
  induction s using Finset.induction with
  | empty => simp
  | @insert a s ha ih =>
    rw [Finset.sum_insert ha, bracket_add_right, ih, Finset.sum_insert ha]

/-- Every module element is its own basis expansion. -/
lemma expand_basis (x : Mod) : x = ∑ i : Basis5, x i • e i := by
  funext k
  rw [Finset.sum_apply]
  rw [Finset.sum_eq_single k]
  · simp
  · intro b _ hb; simp [e_apply_ne (Ne.symm hb)]
  · intro h; exact absurd (Finset.mem_univ k) h

/-- `bracket` on general elements, unfolded to a double sum of basis-pair brackets
(via `bracket_e_e`); the reusable form for the `T4` bilinear-extension arguments. -/
lemma bracket_bilinear_expand (x y : Mod) :
    bracket x y = ∑ i : Basis5, ∑ j : Basis5, (x i * y j) • bracket (e i) (e j) := by
  simp_rw [bracket_e_e]
  rfl

/-- Expand `bracket` in its first argument's basis coordinates. -/
lemma bracket_expand_left (v w : Mod) : bracket v w = ∑ i : Basis5, (v i) • bracket (e i) w := by
  conv_lhs => rw [expand_basis v]
  rw [bracket_sum_left]
  apply Finset.sum_congr rfl; intro i _
  rw [bracket_smul_left]

/-- `T4` (super-skew, general homogeneous elements). -/
theorem bracket_super_skew_homog (x y : Mod) (dx dy : ZMod 2)
    (hx : IsHomog x dx) (hy : IsHomog y dy) :
    bracket x y = -(gsign dx dy) • bracket y x := by
  have hswap : bracket y x
      = ∑ i : Basis5, ∑ j : Basis5, (x i * y j) • bracket (e j) (e i) := by
    rw [bracket_bilinear_expand y x, Finset.sum_comm]
    apply Finset.sum_congr rfl; intro j _
    apply Finset.sum_congr rfl; intro i _
    rw [mul_comm (y i) (x j)]
  rw [bracket_bilinear_expand x y, hswap, Finset.smul_sum]
  apply Finset.sum_congr rfl; intro i _
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl; intro j _
  by_cases hxy : x i * y j = 0
  · simp [hxy]
  · have hxi : x i ≠ 0 := fun h0 => hxy (by rw [h0]; ring)
    have hyj : y j ≠ 0 := fun h0 => hxy (by rw [h0]; ring)
    rw [bracket_super_skew i j, hx i hxi, hy j hyj, smul_smul, smul_smul, neg_mul]
    congr 1
    ring

/-- `T4` (graded Jacobi identity, general homogeneous elements). Each cyclic
term expands to a basis triple-sum via `bracket_expand_left`/
`bracket_bilinear_expand`; `expand2`/`expand3`'s naturally-peeled sum order is
reconciled to the canonical `i,j,k` order by two `Finset.sum_comm` steps each,
after which the whole identity reduces termwise to `jacobi125` under
homogeneity. -/
theorem jacobi_homog (x y z : Mod) (dx dy dz : ZMod 2)
    (hx : IsHomog x dx) (hy : IsHomog y dy) (hz : IsHomog z dz) :
    gsign dx dz • bracket x (bracket y z) + gsign dy dx • bracket y (bracket z x)
      + gsign dz dy • bracket z (bracket x y) = 0 := by
  have key : ∀ i j k : Basis5, x i ≠ 0 → y j ≠ 0 → z k ≠ 0 →
      gsign dx dz • bracket (e i) (bracket (e j) (e k))
        + gsign dy dx • bracket (e j) (bracket (e k) (e i))
        + gsign dz dy • bracket (e k) (bracket (e i) (e j)) = 0 := by
    intro i j k hi hj hk
    rw [← hx i hi, ← hy j hj, ← hz k hk]
    exact jacobi125 i j k
  have expand1 : bracket x (bracket y z)
      = ∑ i, ∑ j, ∑ k, (x i * y j * z k) • bracket (e i) (bracket (e j) (e k)) := by
    rw [bracket_expand_left x (bracket y z)]
    apply Finset.sum_congr rfl; intro i _
    rw [bracket_bilinear_expand y z, bracket_sum_right, Finset.smul_sum]
    apply Finset.sum_congr rfl; intro j _
    rw [bracket_sum_right, Finset.smul_sum]
    apply Finset.sum_congr rfl; intro k _
    rw [bracket_smul_right, smul_smul]
    congr 1; ring
  have expand2 : bracket y (bracket z x)
      = ∑ i, ∑ j, ∑ k, (x i * y j * z k) • bracket (e j) (bracket (e k) (e i)) := by
    have natural : bracket y (bracket z x)
        = ∑ j, ∑ k, ∑ i, (x i * y j * z k) • bracket (e j) (bracket (e k) (e i)) := by
      rw [bracket_expand_left y (bracket z x)]
      apply Finset.sum_congr rfl; intro j _
      rw [bracket_bilinear_expand z x, bracket_sum_right, Finset.smul_sum]
      apply Finset.sum_congr rfl; intro k _
      rw [bracket_sum_right, Finset.smul_sum]
      apply Finset.sum_congr rfl; intro i _
      rw [bracket_smul_right, smul_smul]
      congr 1; ring
    rw [natural]
    rw [show (∑ j, ∑ k, ∑ i, (x i * y j * z k) • bracket (e j) (bracket (e k) (e i)))
          = ∑ j, ∑ i, ∑ k, (x i * y j * z k) • bracket (e j) (bracket (e k) (e i)) from by
        apply Finset.sum_congr rfl; intro j _; rw [Finset.sum_comm]]
    rw [Finset.sum_comm]
  have expand3 : bracket z (bracket x y)
      = ∑ i, ∑ j, ∑ k, (x i * y j * z k) • bracket (e k) (bracket (e i) (e j)) := by
    have natural : bracket z (bracket x y)
        = ∑ k, ∑ i, ∑ j, (x i * y j * z k) • bracket (e k) (bracket (e i) (e j)) := by
      rw [bracket_expand_left z (bracket x y)]
      apply Finset.sum_congr rfl; intro k _
      rw [bracket_bilinear_expand x y, bracket_sum_right, Finset.smul_sum]
      apply Finset.sum_congr rfl; intro i _
      rw [bracket_sum_right, Finset.smul_sum]
      apply Finset.sum_congr rfl; intro j _
      rw [bracket_smul_right, smul_smul]
      congr 1; ring
    rw [natural, Finset.sum_comm]
    apply Finset.sum_congr rfl; intro i _
    rw [Finset.sum_comm]
  rw [expand1, expand2, expand3]
  simp_rw [Finset.smul_sum]
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_eq_zero; intro i _
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_eq_zero; intro j _
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_eq_zero; intro k _
  by_cases h : x i * y j * z k = 0
  · simp [h]
  · have hxi : x i ≠ 0 := fun h0 => h (by rw [h0]; ring)
    have hyj : y j ≠ 0 := fun h0 => h (by rw [h0]; ring)
    have hzk : z k ≠ 0 := fun h0 => h (by rw [h0]; ring)
    have hjac := key i j k hxi hyj hzk
    have hcomb : gsign dx dz • (x i * y j * z k) • bracket (e i) (bracket (e j) (e k))
        + gsign dy dx • (x i * y j * z k) • bracket (e j) (bracket (e k) (e i))
        + gsign dz dy • (x i * y j * z k) • bracket (e k) (bracket (e i) (e j))
        = (x i * y j * z k) • (gsign dx dz • bracket (e i) (bracket (e j) (e k))
            + gsign dy dx • bracket (e j) (bracket (e k) (e i))
            + gsign dz dy • bracket (e k) (bracket (e i) (e j))) := by
      simp_rw [smul_add, smul_smul, mul_comm (x i * y j * z k)]
    rw [hcomb, hjac, smul_zero]

end InhomogeneousDeformations
