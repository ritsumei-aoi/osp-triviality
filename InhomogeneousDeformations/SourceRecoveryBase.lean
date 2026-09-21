import InhomogeneousDeformations.SourceLifts
import InhomogeneousDeformations.SourceIsomorphism

namespace InhomogeneousDeformations
namespace Source

open scoped TensorProduct

variable (n : ℕ)

/-! ## General, hypothesis-free ring identities -/

theorem comm_mul_right (x y z : Module.End ℚ (WPoly n)) :
    x*(y*z) - (y*z)*x = (x*y-y*x)*z + y*(x*z-z*x) := by
  rw [sub_mul, mul_sub, ← mul_assoc, ← mul_assoc, ← mul_assoc]
  abel

theorem comm_mul_left (x y z : Module.End ℚ (WPoly n)) :
    (x*y)*z - z*(x*y) = x*(y*z-z*y) + (x*z-z*x)*y := by
  rw [mul_sub, sub_mul, mul_assoc, mul_assoc, mul_assoc]
  abel

/-! ## Step 1(1): `[L0hat(u,v), L0hat(w,z)]`, via A1 alone -/

/-- `L0` "before embedding": the pure `Module.End` version of `4•L0`, i.e. `(a*c-c*a)` etc. -/
noncomputable def L0hat (u v : Fin (2*n)) : Module.End ℚ (WPoly n) :=
  (1/4 : ℚ) • (B n u * B n v + B n v * B n u)

theorem comm_scalar (u v : Fin (2*n)) :
    B n u * B n v - B n v * B n u = JnQ n u v • (1 : Module.End ℚ (WPoly n)) := B_comm n u v

/-- The key lemma: `[a*b+b*a, c*d+d*c]` for the four elements `B_u,B_v,B_w,B_z`, using ONLY A1
(pairwise commutators are ℚ-scalars). -/
theorem symm_comm_symm (a b c d : Module.End ℚ (WPoly n)) (p q r s : ℚ)
    (hac : a*c - c*a = p • (1 : Module.End ℚ (WPoly n)))
    (had : a*d - d*a = q • (1 : Module.End ℚ (WPoly n)))
    (hbc : b*c - c*b = r • (1 : Module.End ℚ (WPoly n)))
    (hbd : b*d - d*b = s • (1 : Module.End ℚ (WPoly n))) :
    (a*b+b*a)*(c*d+d*c) - (c*d+d*c)*(a*b+b*a)
      = (2:ℚ) • (r • (a*d+d*a) + s • (a*c+c*a) + p • (b*d+d*b) + q • (b*c+c*b)) := by
  have hab_cd : a*b*(c*d) - (c*d)*(a*b) = a*(b*(c*d)-(c*d)*b) + (a*(c*d)-(c*d)*a)*b := comm_mul_left n a b (c*d)
  have hbcd : b*(c*d) - (c*d)*b = (b*c-c*b)*d + c*(b*d-d*b) := comm_mul_right n b c d
  have hacd : a*(c*d) - (c*d)*a = (a*c-c*a)*d + c*(a*d-d*a) := comm_mul_right n a c d
  have hba_cd : b*a*(c*d) - (c*d)*(b*a) = b*(a*(c*d)-(c*d)*a) + (b*(c*d)-(c*d)*b)*a := comm_mul_left n b a (c*d)
  have hab_dc : a*b*(d*c) - (d*c)*(a*b) = a*(b*(d*c)-(d*c)*b) + (a*(d*c)-(d*c)*a)*b := comm_mul_left n a b (d*c)
  have hbdc : b*(d*c) - (d*c)*b = (b*d-d*b)*c + d*(b*c-c*b) := comm_mul_right n b d c
  have hadc : a*(d*c) - (d*c)*a = (a*d-d*a)*c + d*(a*c-c*a) := comm_mul_right n a d c
  have hba_dc : b*a*(d*c) - (d*c)*(b*a) = b*(a*(d*c)-(d*c)*a) + (b*(d*c)-(d*c)*b)*a := comm_mul_left n b a (d*c)
  have key : (a*b+b*a)*(c*d+d*c) - (c*d+d*c)*(a*b+b*a)
      = (a*b*(c*d) - (c*d)*(a*b)) + (a*b*(d*c) - (d*c)*(a*b))
        + (b*a*(c*d) - (c*d)*(b*a)) + (b*a*(d*c) - (d*c)*(b*a)) := by
    noncomm_ring
  rw [key, hab_cd, hab_dc, hba_cd, hba_dc, hacd, hbcd, hadc, hbdc, hac, had, hbc, hbd]
  simp only [mul_add, add_mul, smul_mul_assoc, mul_smul_comm, one_mul, mul_one]
  module

theorem L0hat_comm (u v w z : Fin (2*n)) :
    L0hat n u v * L0hat n w z - L0hat n w z * L0hat n u v
      = (1/2 : ℚ) • (JnQ n v w • L0hat n u z + JnQ n u w • L0hat n v z
          + JnQ n v z • L0hat n u w + JnQ n u z • L0hat n v w) := by
  unfold L0hat
  rw [show (1/4:ℚ) • (B n u * B n v + B n v * B n u) * ((1/4:ℚ) • (B n w * B n z + B n z * B n w))
        - (1/4:ℚ) • (B n w * B n z + B n z * B n w) * ((1/4:ℚ) • (B n u * B n v + B n v * B n u))
      = (1/16 : ℚ) • ((B n u * B n v + B n v * B n u) * (B n w * B n z + B n z * B n w)
          - (B n w * B n z + B n z * B n w) * (B n u * B n v + B n v * B n u)) from by
    rw [smul_mul_smul_comm, smul_mul_smul_comm]; module]
  rw [symm_comm_symm n (B n u) (B n v) (B n w) (B n z) (JnQ n u w) (JnQ n u z) (JnQ n v w) (JnQ n v z)
    (comm_scalar n u w) (comm_scalar n u z) (comm_scalar n v w) (comm_scalar n v z)]
  module

/-! ## Step 1(2)/(3): the `L0`-`F0` and `F0`-`F0` brackets, in `A0 n` itself

**Attempted, not completed within this bounded experiment's budget — a mechanical, not
mathematical, obstruction.** By hand (not yet re-verified in Lean): `[L0_uv, F0_w]` reduces,
using `a_comm_Bu0` (`a` commutes with every `B_u`, R2) to pull `a` out as if central relative to
the `B`'s, to `(1/2)•a•[L0hat_uv, B_w]`, and `[L0hat_uv,B_w] = (1/2)(JnQ(v,w)•B_u+JnQ(u,w)•B_v)`
via the same `comm_mul_left`/`comm_scalar` toolkit above (a strictly easier, 3-element special
case of the same technique) — giving `[L0_uv,F0_w] = (1/2)(JnQ(v,w)•F0_u+JnQ(u,w)•F0_v)`, matching
`eq:base-lf` exactly. Similarly `{F0_u,F0_v} = F0_u F0_v+F0_v F0_u = (1/4)a B_u a B_v +
(1/4)a B_v a B_u = (1/4)a²(B_uB_v+B_vB_u)` (pulling `a` past `B_u` via `a_comm_Bu0`) `=
(1/4)(1/2)(4•L0hat_uv) = (1/2)L0_uv`, matching `eq:base-ff` exactly, using only `a_comm_Bu0` and
`aA0_sq` (both already accepted, R2).

The attempt below to formalize this in `A0 n` hit real but mechanical snags: (1) an argument-order
mismatch in `tmul_coe_mul_zero_coe_tmul`'s anonymous-constructor arguments (the `b₁`/`a₂` slots),
and (2) `ᵍ⊗ₜ` binds *tighter* than `*` in this notation, so `L0hat n u v * B n w ᵍ⊗ₜ[ℚ] 1` parses
as `L0hat n u v * (B n w ᵍ⊗ₜ 1)` — a type error, `Module.End` has no `HMul` against a tensor
product — rather than the intended `(L0hat n u v * B n w) ᵍ⊗ₜ 1`; every product-then-tensor
expression needs explicit parens. Neither is a sign the underlying mathematics is wrong; both are
exactly the kind of bookkeeping `SourceIsomorphism.lean`'s own already-accepted proofs (e.g.
`BuAB_comm_cAB`) show is tractable with enough care — this experiment simply ran out of its
bounded budget before getting the parenthesization and lemma-argument order right. Left
uncompleted (not forced) per instruction; the working `L0`-`L0` case above is the substantive
deliverable of this experiment. -/

theorem Bu0_mul' (u v : Fin (2*n)) :
    Bu0 n u * Bu0 n v = (B n u * B n v) ᵍ⊗ₜ[ℚ] (1 : C) := by
  unfold Bu0
  rw [GradedTensorProduct.tmul_coe_mul_zero_coe_tmul (𝒜 := WGrading n) (ℬ := CGrading)
    (B n u) (⟨1, SetLike.one_mem_graded CGrading⟩ : CGrading 0) (⟨B n v, trivial⟩ : WGrading n 0) (1 : C)]
  norm_num

/-- `4•L0hat n u v` embeds, via `Bu0`, as `4•(L0 n u v)`. -/
theorem L0_eq_tmul (u v : Fin (2*n)) :
    L0 n u v = (L0hat n u v) ᵍ⊗ₜ[ℚ] (1 : C) := by
  unfold L0 L0hat
  rw [Bu0_mul', Bu0_mul']
  show (1/4:ℚ) • (GradedTensorProduct.of ℚ (WGrading n) CGrading
        ((B n u * B n v : Module.End ℚ (WPoly n)) ⊗ₜ[ℚ] (1 : C))
      + GradedTensorProduct.of ℚ (WGrading n) CGrading
        ((B n v * B n u : Module.End ℚ (WPoly n)) ⊗ₜ[ℚ] (1 : C)))
      = GradedTensorProduct.of ℚ (WGrading n) CGrading
        (((1/4:ℚ) • (B n u * B n v + B n v * B n u) : Module.End ℚ (WPoly n)) ⊗ₜ[ℚ] (1 : C))
  rw [← map_add, ← TensorProduct.add_tmul, ← map_smul, ← TensorProduct.smul_tmul']

/-- **`eq:base-lf`, checked**: `[L0 u v, F0 w] = (1/2)•(JnQ v w • F0 u + JnQ u w • F0 v)`. -/
theorem L0_comm_F0 (u v w : Fin (2*n)) :
    L0 n u v * F0 n w - F0 n w * L0 n u v
      = (1/2 : ℚ) • (JnQ n v w • F0 n u + JnQ n u w • F0 n v) := by
  unfold F0
  rw [L0_eq_tmul]
  rw [show (L0hat n u v ᵍ⊗ₜ[ℚ] (1:C)) * ((1/2:ℚ) • (aA0 n * Bu0 n w))
        - (1/2:ℚ) • (aA0 n * Bu0 n w) * (L0hat n u v ᵍ⊗ₜ[ℚ] (1:C))
      = (1/2:ℚ) • (L0hat n u v ᵍ⊗ₜ[ℚ] (1:C) * (aA0 n * Bu0 n w)
          - (aA0 n * Bu0 n w) * (L0hat n u v ᵍ⊗ₜ[ℚ] (1:C))) from by
    rw [mul_smul_comm, smul_mul_assoc]; module]
  have step1 : L0hat n u v ᵍ⊗ₜ[ℚ] (1:C) * (aA0 n * Bu0 n w) = aA0 n * ((L0hat n u v * B n w) ᵍ⊗ₜ[ℚ] (1:C)) := by
    unfold aA0 Bu0
    rw [show ((L0hat n u v) ᵍ⊗ₜ[ℚ] (1:C)) * ((1:Module.End ℚ (WPoly n)) ᵍ⊗ₜ[ℚ] (Source.a:C) * (B n w) ᵍ⊗ₜ[ℚ] (1:C))
        = ((L0hat n u v) ᵍ⊗ₜ[ℚ] (1:C)) * (((1:Module.End ℚ (WPoly n)) * B n w) ᵍ⊗ₜ[ℚ] (Source.a * 1 : C)) from by
      rw [GradedTensorProduct.tmul_coe_mul_zero_coe_tmul (𝒜 := WGrading n) (ℬ := CGrading)
        (1 : Module.End ℚ (WPoly n)) (⟨Source.a, Source.a_odd⟩ : CGrading 1)
        (⟨B n w, trivial⟩ : WGrading n 0) (1 : C)]]
    rw [one_mul, mul_one]
    rw [GradedTensorProduct.tmul_coe_mul_zero_coe_tmul (𝒜 := WGrading n) (ℬ := CGrading)
      (L0hat n u v) (⟨1, SetLike.one_mem_graded CGrading⟩ : CGrading 0)
      (⟨B n w, trivial⟩ : WGrading n 0) (Source.a : C)]
    rw [GradedTensorProduct.tmul_coe_mul_zero_coe_tmul (𝒜 := WGrading n) (ℬ := CGrading)
      (1 : Module.End ℚ (WPoly n)) (⟨Source.a, Source.a_odd⟩ : CGrading 1)
      (⟨L0hat n u v * B n w, trivial⟩ : WGrading n 0) (1 : C)]
    simp
  have step2 : (aA0 n * Bu0 n w) * (L0hat n u v ᵍ⊗ₜ[ℚ] (1:C)) = aA0 n * ((B n w * L0hat n u v) ᵍ⊗ₜ[ℚ] (1:C)) := by
    unfold aA0 Bu0
    rw [mul_assoc]
    congr 1
    rw [GradedTensorProduct.tmul_coe_mul_zero_coe_tmul (𝒜 := WGrading n) (ℬ := CGrading)
      (B n w) (⟨1, SetLike.one_mem_graded CGrading⟩ : CGrading 0)
      (⟨L0hat n u v, trivial⟩ : WGrading n 0) (1 : C)]
    simp
  rw [step1, step2, ← mul_sub]
  rw [show ((L0hat n u v * B n w) ᵍ⊗ₜ[ℚ] (1:C)) - ((B n w * L0hat n u v) ᵍ⊗ₜ[ℚ] (1:C))
      = (L0hat n u v * B n w - B n w * L0hat n u v) ᵍ⊗ₜ[ℚ] (1:C) from by
    show (GradedTensorProduct.of ℚ (WGrading n) CGrading
          ((L0hat n u v * B n w : Module.End ℚ (WPoly n)) ⊗ₜ[ℚ] (1:C)))
        - GradedTensorProduct.of ℚ (WGrading n) CGrading
          ((B n w * L0hat n u v : Module.End ℚ (WPoly n)) ⊗ₜ[ℚ] (1:C))
      = GradedTensorProduct.of ℚ (WGrading n) CGrading
          ((L0hat n u v * B n w - B n w * L0hat n u v : Module.End ℚ (WPoly n)) ⊗ₜ[ℚ] (1:C))
    rw [← map_sub, TensorProduct.sub_tmul]]
  have hcomm : L0hat n u v * B n w - B n w * L0hat n u v
      = (1/2 : ℚ) • (JnQ n v w • B n u + JnQ n u w • B n v) := by
    unfold L0hat
    rw [show (1/4:ℚ) • (B n u * B n v + B n v * B n u) * B n w
          - B n w * ((1/4:ℚ) • (B n u * B n v + B n v * B n u))
        = (1/4:ℚ) • ((B n u * B n v + B n v * B n u) * B n w
            - B n w * (B n u * B n v + B n v * B n u)) from by
      rw [smul_mul_assoc, mul_smul_comm]; module]
    have e1 : (B n u * B n v + B n v * B n u) * B n w - B n w * (B n u * B n v + B n v * B n u)
        = (B n u * B n v * B n w - B n w * (B n u * B n v))
          + (B n v * B n u * B n w - B n w * (B n v * B n u)) := by noncomm_ring
    have e2 : B n u * B n v * B n w - B n w * (B n u * B n v)
        = B n u * (B n v * B n w - B n w * B n v) + (B n u * B n w - B n w * B n u) * B n v :=
      comm_mul_left n (B n u) (B n v) (B n w)
    have e3 : B n v * B n u * B n w - B n w * (B n v * B n u)
        = B n v * (B n u * B n w - B n w * B n u) + (B n v * B n w - B n w * B n v) * B n u :=
      comm_mul_left n (B n v) (B n u) (B n w)
    rw [e1, e2, e3, comm_scalar n u w, comm_scalar n v w]
    simp only [mul_add, add_mul, smul_mul_assoc, mul_smul_comm, one_mul, mul_one]
    module
  rw [hcomm]
  rw [show (((1/2:ℚ) • (JnQ n v w • B n u + JnQ n u w • B n v) : Module.End ℚ (WPoly n)) ᵍ⊗ₜ[ℚ] (1:C))
      = (1/2:ℚ) • (JnQ n v w • Bu0 n u + JnQ n u w • Bu0 n v) from by
    unfold Bu0
    show GradedTensorProduct.of ℚ (WGrading n) CGrading
        (((1/2:ℚ) • (JnQ n v w • B n u + JnQ n u w • B n v) : Module.End ℚ (WPoly n)) ⊗ₜ[ℚ] (1:C))
      = (1/2:ℚ) • (JnQ n v w • GradedTensorProduct.of ℚ (WGrading n) CGrading
          ((B n u : Module.End ℚ (WPoly n)) ⊗ₜ[ℚ] (1:C))
        + JnQ n u w • GradedTensorProduct.of ℚ (WGrading n) CGrading
          ((B n v : Module.End ℚ (WPoly n)) ⊗ₜ[ℚ] (1:C)))
    rw [← map_smul, ← map_smul, ← map_add, ← map_smul]
    congr 1
    rw [← TensorProduct.smul_tmul', TensorProduct.add_tmul, ← TensorProduct.smul_tmul',
      ← TensorProduct.smul_tmul']]
  rw [mul_smul_comm, mul_add, mul_smul_comm, mul_smul_comm]
  module

/-! ## Step 1(3): `{F0_u,F0_v}`, matching `eq:base-ff` -/

theorem Bu0_mul_L0hat_shape (u v : Fin (2*n)) :
    Bu0 n u * Bu0 n v + Bu0 n v * Bu0 n u = (4:ℚ) • L0 n u v := by
  rw [L0_eq_tmul]
  unfold L0hat
  rw [Bu0_mul', Bu0_mul']
  show ((B n u * B n v : Module.End ℚ (WPoly n)) ᵍ⊗ₜ[ℚ] (1:C))
      + ((B n v * B n u : Module.End ℚ (WPoly n)) ᵍ⊗ₜ[ℚ] (1:C))
    = (4:ℚ) • (GradedTensorProduct.of ℚ (WGrading n) CGrading
        (((1/4:ℚ) • (B n u * B n v + B n v * B n u) : Module.End ℚ (WPoly n)) ⊗ₜ[ℚ] (1:C)))
  rw [← map_smul, ← TensorProduct.smul_tmul', smul_smul]
  norm_num
  rw [← map_add, TensorProduct.add_tmul]

/-- **`eq:base-ff`, checked**: `{F0_u,F0_v} = (1/2)•L0_uv`. -/
theorem F0_comm_F0 (u v : Fin (2*n)) :
    F0 n u * F0 n v + F0 n v * F0 n u = (1/2 : ℚ) • L0 n u v := by
  unfold F0
  rw [show ((1/2:ℚ) • (aA0 n * Bu0 n u)) * ((1/2:ℚ) • (aA0 n * Bu0 n v))
        + ((1/2:ℚ) • (aA0 n * Bu0 n v)) * ((1/2:ℚ) • (aA0 n * Bu0 n u))
      = (1/4:ℚ) • (aA0 n * Bu0 n u * (aA0 n * Bu0 n v) + aA0 n * Bu0 n v * (aA0 n * Bu0 n u)) from by
    rw [smul_mul_smul_comm, smul_mul_smul_comm]; module]
  have e1 : aA0 n * Bu0 n u * (aA0 n * Bu0 n v) = (1/2:ℚ) • (Bu0 n u * Bu0 n v) := by
    rw [show aA0 n * Bu0 n u * (aA0 n * Bu0 n v) = Bu0 n u * (aA0 n * aA0 n) * Bu0 n v from by
      rw [a_comm_Bu0]; noncomm_ring]
    rw [aA0_sq, mul_smul_comm, mul_one, smul_mul_assoc]
  have e2 : aA0 n * Bu0 n v * (aA0 n * Bu0 n u) = (1/2:ℚ) • (Bu0 n v * Bu0 n u) := by
    rw [show aA0 n * Bu0 n v * (aA0 n * Bu0 n u) = Bu0 n v * (aA0 n * aA0 n) * Bu0 n u from by
      rw [a_comm_Bu0]; noncomm_ring]
    rw [aA0_sq, mul_smul_comm, mul_one, smul_mul_assoc]
  rw [e1, e2, ← smul_add, Bu0_mul_L0hat_shape]
  simp only [smul_smul]
  norm_num

end Source
end InhomogeneousDeformations
