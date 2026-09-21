import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import InhomogeneousDeformations.Basis
import InhomogeneousDeformations.Carrier

/-!
# Native n=1 bracket, from the manuscript's rank-indexed formulas

Independent of all wire rows / oracle tables (PILOT_CONTRACT.md §1).
`I_1 = {1,2}`; mathematical index `u` maps to zero-based `u-1`, i.e. `Fin 2`.
`J_{2q-1,2q}=1, J_{2q,2q-1}=-1` for the single `q=1` block, other entries `0`.
`L(u,v)` sorts indices with no sign; the module carrier is the free
`Coeff`-module on `Basis5` (formalized as `Basis5 → Coeff`, a `Coeff`-module
via the standard `Pi` instances since `Basis5` is a `Fintype`).

    [F_u,F_v] = (1/2) L_uv
    [L_uv,F_w] = (1/2)(J_vw F_u + J_uw F_v)
    [L_uv,L_wz] = (1/2)(J_vw L_uz + J_uw L_vz + J_vz L_uw + J_uz L_vw)
    [F_w,L_uv] = -[L_uv,F_w]
-/

namespace InhomogeneousDeformations

open Basis5

/-- The free `Coeff`-module on the five-element basis. -/
abbrev Mod : Type := Basis5 → Coeff

/-- `J` matrix for the single `q=1` block on `I_1={1,2}` (zero-based `Fin 2`),
valued in `Coeff` via `crat` (a parameter-free constant). -/
noncomputable def Jmat : Fin 2 → Fin 2 → Coeff
  | 0, 1 => crat 1
  | 1, 0 => crat (-1)
  | _, _ => crat 0

/-- `L(u,v)`: sorts indices, no sign. -/
def Lof : Fin 2 → Fin 2 → Basis5
  | 0, 0 => .L11
  | 0, 1 => .L12
  | 1, 0 => .L12
  | 1, 1 => .L22

def Fof : Fin 2 → Basis5
  | 0 => .F1
  | 1 => .F2

lemma Lof_ne_F1 : ∀ u v : Fin 2, Lof u v ≠ Basis5.F1 := by decide
lemma Lof_ne_F2 : ∀ u v : Fin 2, Lof u v ≠ Basis5.F2 := by decide
lemma Fof_ne_L11 : ∀ u : Fin 2, Fof u ≠ Basis5.L11 := by decide
lemma Fof_ne_L12 : ∀ u : Fin 2, Fof u ≠ Basis5.L12 := by decide
lemma Fof_ne_L22 : ∀ u : Fin 2, Fof u ≠ Basis5.L22 := by decide

/-- Indicator (single basis) vector. -/
noncomputable def e (b : Basis5) : Mod := fun k => if k = b then 1 else 0

@[simp] lemma e_apply_self (b : Basis5) : e b b = 1 := if_pos rfl
@[simp] lemma e_apply_ne {b k : Basis5} (h : k ≠ b) : e b k = 0 := if_neg h

noncomputable def bracketFF (u v : Fin 2) : Mod := (crat (1 / 2)) • e (Lof u v)

noncomputable def bracketLF (u v w : Fin 2) : Mod :=
  (crat (1 / 2) * Jmat v w) • e (Fof u) + (crat (1 / 2) * Jmat u w) • e (Fof v)

noncomputable def bracketLL (u v w z : Fin 2) : Mod :=
  (crat (1 / 2) * Jmat v w) • e (Lof u z) + (crat (1 / 2) * Jmat u w) • e (Lof v z)
  + (crat (1 / 2) * Jmat v z) • e (Lof u w) + (crat (1 / 2) * Jmat u z) • e (Lof v w)

/-- Structure constants on all 25 ordered basis pairs, from the formulas above.
`[F_w,L_uv] = -[L_uv,F_w]` supplies the `F,L` rows from the `L,F` rows. -/
noncomputable def bracketBasis : Basis5 → Basis5 → Mod
  | .L11, .L11 => bracketLL 0 0 0 0
  | .L11, .L12 => bracketLL 0 0 0 1
  | .L11, .L22 => bracketLL 0 0 1 1
  | .L11, .F1  => bracketLF 0 0 0
  | .L11, .F2  => bracketLF 0 0 1
  | .L12, .L11 => bracketLL 0 1 0 0
  | .L12, .L12 => bracketLL 0 1 0 1
  | .L12, .L22 => bracketLL 0 1 1 1
  | .L12, .F1  => bracketLF 0 1 0
  | .L12, .F2  => bracketLF 0 1 1
  | .L22, .L11 => bracketLL 1 1 0 0
  | .L22, .L12 => bracketLL 1 1 0 1
  | .L22, .L22 => bracketLL 1 1 1 1
  | .L22, .F1  => bracketLF 1 1 0
  | .L22, .F2  => bracketLF 1 1 1
  | .F1,  .L11 => -bracketLF 0 0 0
  | .F1,  .L12 => -bracketLF 0 1 0
  | .F1,  .L22 => -bracketLF 1 1 0
  | .F1,  .F1  => bracketFF 0 0
  | .F1,  .F2  => bracketFF 0 1
  | .F2,  .L11 => -bracketLF 0 0 1
  | .F2,  .L12 => -bracketLF 0 1 1
  | .F2,  .L22 => -bracketLF 1 1 1
  | .F2,  .F1  => bracketFF 1 0
  | .F2,  .F2  => bracketFF 1 1

/-- `T0`: finite-sum `Coeff`-bilinear extension of `bracketBasis` to the whole module. -/
noncomputable def bracket (x y : Mod) : Mod :=
  ∑ i : Basis5, ∑ j : Basis5, (x i * y j) • bracketBasis i j

theorem bracket_add_left (x1 x2 y : Mod) :
    bracket (x1 + x2) y = bracket x1 y + bracket x2 y := by
  funext k
  simp only [bracket, Finset.sum_apply, Pi.add_apply]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro i _
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro j _
  simp only [Pi.smul_apply, smul_eq_mul]
  ring

theorem bracket_add_right (x y1 y2 : Mod) :
    bracket x (y1 + y2) = bracket x y1 + bracket x y2 := by
  funext k
  simp only [bracket, Finset.sum_apply, Pi.add_apply]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro i _
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro j _
  simp only [Pi.smul_apply, smul_eq_mul]
  ring

theorem bracket_smul_left (c : Coeff) (x y : Mod) :
    bracket (c • x) y = c • bracket x y := by
  funext k
  simp only [bracket, Finset.sum_apply, Pi.smul_apply]
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl; intro i _
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl; intro j _
  simp only [Pi.smul_apply, smul_eq_mul]
  ring

theorem bracket_smul_right (c : Coeff) (x y : Mod) :
    bracket x (c • y) = c • bracket x y := by
  funext k
  simp only [bracket, Finset.sum_apply, Pi.smul_apply]
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl; intro i _
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl; intro j _
  simp only [Pi.smul_apply, smul_eq_mul]
  ring

/-- `T0`: homogeneity by basis support with the stated parity. -/
def IsHomog (x : Mod) (d : ZMod 2) : Prop := ∀ i : Basis5, x i ≠ 0 → Basis5.parity i = d

@[simp] theorem e_isHomog (b : Basis5) : IsHomog (e b) (Basis5.parity b) := by
  intro i hi
  by_cases h : i = b
  · rw [h]
  · exact absurd (e_apply_ne h) hi

end InhomogeneousDeformations
