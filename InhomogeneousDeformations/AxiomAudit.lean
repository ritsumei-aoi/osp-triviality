import InhomogeneousDeformations.N1Proofs
import InhomogeneousDeformations.Indexed
import InhomogeneousDeformations.HomogeneousDegree
import InhomogeneousDeformations.N1Specialization
import InhomogeneousDeformations.Wire
import InhomogeneousDeformations.Decode
import InhomogeneousDeformations.FixtureData
import InhomogeneousDeformations.Bridge
import InhomogeneousDeformations.IndexedLaws
import InhomogeneousDeformations.IndexedJacobi
import InhomogeneousDeformations.IndexedCoboundary
import InhomogeneousDeformations.IndexedKappa
import InhomogeneousDeformations.IndexedU
import InhomogeneousDeformations.SourceC
import InhomogeneousDeformations.SourceWeyl
import InhomogeneousDeformations.SourceTensor
import InhomogeneousDeformations.SourceLifts
import InhomogeneousDeformations.SourceIndependence
import InhomogeneousDeformations.SourceQuadraticIndependence
import InhomogeneousDeformations.SourceIsomorphism
import InhomogeneousDeformations.SourceRecoveryLifts
import InhomogeneousDeformations.SourceRecoveryBase
import InhomogeneousDeformations.SourceRecoveryBridge
import InhomogeneousDeformations.SourceRecoveryFinal
import InhomogeneousDeformations.SourceRecoveryClosure
import InhomogeneousDeformations.SourceRecoveryClosedImage
import InhomogeneousDeformations.SourceRecoveryFLConvention

/-!
# Axiom audit

Reports the actual dependency axioms of every named result completed this
pilot, both correction rounds, T5-T7, R2-C-1 (U0-U3: general-rank
interface, degree law, and super-skew), R2-C-2 (U4-U5: general-rank
super-Jacobi, completing P1), R2-D (G0-G5: the coefficient/coboundary
identity `Gamma_beta = delta f_beta`, completing P2), and this round's
R2-E (K0-K5: the `kappa` extension `g_R` and the trivializing map
`T_beta`, completing P3). Following correction002, `Phi_bracket` and
`bracketN_isHomog` (C1.3/C2.2) ARE sorry-free: `basis_case` was rewritten
as 25 separately named basis-pair lemmas plus small bridge lemmas (per
Agent1's recommended route), replacing the prior single 125-goal `<;>`
tactic chain that left an undocumented residual. T5-T7 added
`Bridge.decodedBracket_eq_bracket` (T6) and its supporting lemmas. R2-C-1
added, all at general `n` with no `n=1` specialization:
`Jn_diag`/`Jn_antisymm`/`Lof_comm`/`Lof_eq_iff` (U0), `bracketN_isHomogN`
(U1), `bracketBasisN_super_skew` (U2), and `bracketN_super_skew_homog`
(U3). R2-C-2 added the `bracketN` bilinearity toolkit and basis-shape
reduction toolkit, the four U4 sector theorems
`jacobiN_FFF`/`jacobiN_LFF`/`jacobiN_LLF`/`jacobiN_LLL`, the rotation
lemma `jacobiSum_rot` and its assembly `jacobiN_basis` (U4 on all eight
basis-triple shapes), the finite-sum expansion toolkit, and `U5`'s
`jacobiN_homog` (unconditional, all four U4 sectors proved). R2-D added,
likewise at general `n`: G0's `betaN`/`vN`/`partnerN`/
`contraction_identity`(`_neg`), G1's `fBetaBasis`/`fBetaN` and the
required `eq:primitive`-vs-`-sum v_tL_tu` transcription check
(`fBetaBasis_alt_form`), the generic `bilinearExtend` combinator shared by
G2's `GammaBetaN` and G3's `deltaFN`, the four G4 sector theorems
`G4_LL`/`G4_LF`/`G4_FF`/`G4_FL` and their assembly `G4` (all eight basis
pairs), and G5's `GammaBetaN_eq_deltaFN`(`_homog`) (unconditional, all
four G4 sectors proved). This round adds, likewise at general `n`: K0's
`RBasis`/`RMod`/`parityR`/`iotaR`/`kappaEmbed`/`kappaMulR` model and
`kappaMulR_sq`/`kappaMulR_not_injective` (K0c); K1's
`bracketRBasis`/`bracketR`, `bracketR_iotaR_iotaR` (K1a),
`bracketR_add_left/right`/`bracketR_smul_left/right` (K1b), and
`bracketRBasis_super_skew`/`bracketR_super_skew_homog` (K1c, the
convention-pinning check, proved before K5 is attempted); K2's `fBetaR`
with `fBetaR_iotaR`/`fBetaR_kappaEmbed` (the two extension-rule cases,
proved) and `fBetaR_add`/`fBetaR_smul`; K3's `TBeta`/`TBetaInv` with
`TBetaInv_TBeta`/`TBeta_TBetaInv` (exact inverses),
`TBeta_add`/`TBeta_smul`/`TBeta_kappaMulR` (`R`-linearity), and
`TBeta_isHomogR` (even); K4's `bracketRBetaBasis`/`bracketRBeta` and
`bracketRBeta_iotaR_iotaR` (restricts to `bracketN + kappa GammaBetaN`);
and K5's `intertwining_basis` (every basis pair, citing `G4` by name in
the substantive case) and `intertwining` (arbitrary elements,
unconditional, no homogeneity hypothesis needed). I106 R1
(`IndexedU.lean`, the one new module this round, `Indexed.lean`,
`IndexedLaws.lean`, `IndexedJacobi.lean`, `IndexedCoboundary.lean` and
`IndexedKappa.lean` frozen throughout) adds, likewise at general `n`: V0's
`hBasis`/`hMap` (the manuscript's `h`, transcribed directly, degree-raising
by `1` exactly as `fBetaN` is); V1's `hR`/`UMap`/`UMapInv` (`TBeta`'s own
shape with `hMap` in place of `fBetaN`) with `UMapInv_UMap`/`UMap_UMapInv`
(exact inverses) and `UMap_add`/`UMap_smul`/`UMap_kappaMulR`/
`UMap_isHomogR` (`R`-linearity and evenness); V2's `deltaH_LL` (the one
sector needing Jacobi, via `twoFVecBeta`/`FVecBeta_deriv_Lof_Lof` and the
manuscript's own `f_beta = h - ad(2F(v))` reduction to already-accepted P2,
not a fresh 48-term expansion), `deltaH_LF`/`deltaH_FF`/`deltaH_FL`
(direct, no Jacobi, since `hMap` vanishes on every `F_u`), their assembly
`deltaH` (all four sectors), and `GammaBetaN_eq_deltaHN` (unconditional
extension to arbitrary elements, `delta h = Gamma_beta`); and V3's
`intertwiningU_basis` (citing `deltaH` in the substantive case, exactly
transposing `intertwining_basis`), `intertwiningU` (arbitrary elements,
unconditional), `intertwiningU_inv` (the gate statement
`U^{-1}[Ux,Uy]_0 = [x,y]_beta`), and `intertwiningU_coordinate_corollary`
(`GammaBetaN` appearing literally, via the already-accepted
`bracketRBeta_iotaR_iotaR`). I106 R2 (`SourceC.lean`, `SourceWeyl.lean`,
`SourceTensor.lean`, `SourceLifts.lean`, `SourceIndependence.lean`, all
five new modules this round; the six-layer freeze — `Indexed.lean`,
`IndexedLaws.lean`, `IndexedJacobi.lean`, `IndexedCoboundary.lean`,
`IndexedKappa.lean`, `IndexedU.lean` — untouched throughout) builds the
manuscript's source objects: W0's `C = Q[a]/(a^2-1/2)` via
`CliffordAlgebra`, with `a_sq`/`a_odd` (**A2**); W1's `W_n` as
`Module.End ℚ (WPoly n)`, the even generators `B_u` as differentiation/
multiplication operators, with `B_comm` (**A1**, `[B_u,B_v]=Jn n u v * 1`
against the *frozen* `Indexed.Jn`, proved by direct computation, not a
fresh matrix); W2's `A_0 = W_n ⊗ C` and `A_B = R ⊗ A_0` via
`GradedTensorProduct`, with hand-built `GradedAlgebra` instances for both
(`A0GradedAlgebra`, `ABGradedAlgebra`) since mathlib supplies no general
"tensor product of graded algebras is graded" result, `kappa_sq`/
`kappa_mem_RGrading_one`/`kappa_ne_zero`/`a_ne_zero`/`A3_independence`
(**A3**, `κ` and `a` genuinely anti-commute in `A_B` and their product is
nonzero) and `A4_W_into_A0`/`A4_C_into_A0`/`A4_R_into_AB`/`A4_A0_into_AB`
(**A4**); W3's undeformed lifts `L0`/`F0` (`L^0_{uv}`, `F^0_u`), defined
in `A_0` exactly as `lem:base` states, with their `A0Grading` membership
recorded (`L0_mem_A0Grading_zero`/`F0_mem_A0Grading_one`); and W4's
`IndependenceStatement`, which **splits, provably**
(`independence_of_L0_and_F0`), into `B_linearIndependent`/
`F0_linearIndependent` (proved, by evaluation on test polynomials) and
`L0FamilyIndependent` (the quadratic-lift half, left as an honestly
assessed but unproved `Prop` — no axiom declared for it, per the
axiom-versus-parameter rule; see `SourceIndependence.lean`'s own module
docstring for the full reasoning). `#print axioms` below is expected to
show only the standard three axioms (or a proper subset) for every
declaration audited, with no `sorryAx` anywhere in the endpoint.
-/

namespace InhomogeneousDeformations

-- Original R2-B pilot (T0-T4), retained unchanged.
#print axioms bracket_add_left
#print axioms bracket_add_right
#print axioms bracket_smul_left
#print axioms bracket_smul_right
#print axioms bracket_eq_oracle
#print axioms bracket_degree0
#print axioms bracket_super_skew
#print axioms jacobi125
#print axioms bracket_sum_left
#print axioms bracket_sum_right
#print axioms expand_basis
#print axioms bracket_bilinear_expand
#print axioms bracket_expand_left
#print axioms bracket_super_skew_homog
#print axioms jacobi_homog

-- CORRECTION-001, C1.1 (Indexed.lean): genuinely rank-indexed definitions;
-- the only proved (non-definitional) declarations at this stage are the
-- `crat`-analogue embedding lemmas for the indexed coefficient ring.
#print axioms Indexed.cratN_zero
#print axioms Indexed.cratN_one
#print axioms Indexed.cratN_add
#print axioms Indexed.cratN_mul
#print axioms Indexed.cratN_neg

-- CORRECTION-002, C1 (Indexed.lean): explicit canonical indexed basis
-- order (injective key, pulled-back `LinearOrder`, and the three required
-- general-rank order semantics theorems).
#print axioms Indexed.orderKey_injective
#print axioms Indexed.orderKey_even_lt_odd
#print axioms Indexed.orderKey_even_lex
#print axioms Indexed.orderKey_odd_lt

-- CORRECTION-001, C1.2 (N1Specialization.lean): coefficient identification
-- `rho` and basis equivalence `E`, with parity preservation.
#print axioms Indexed.rho_beta1
#print axioms Indexed.rho_beta2
#print axioms Indexed.rho_add
#print axioms Indexed.rho_mul
#print axioms Indexed.rho_C
#print axioms Indexed.cratN_one_eq_crat
#print axioms Indexed.Eforward_backward
#print axioms Indexed.Ebackward_forward
#print axioms Indexed.E_parity

-- CORRECTION-002, C1 rank-one compatibility (N1Specialization.lean): the
-- canonical n=1 order is exactly L11,L12,L22,F1,F2, and `E` preserves it.
#print axioms Indexed.canonical_order_n1
#print axioms Indexed.E_preserves_order

-- CORRECTION-001, C1.3 (N1Specialization.lean): the induced module map
-- `Phi`, its bijectivity/linearity/homogeneity-equivalence.
#print axioms Indexed.Phi_bijective
#print axioms Indexed.Phi_add
#print axioms Indexed.Phi_smul
#print axioms Indexed.IsHomog_Phi_iff
#print axioms Indexed.Phi_zero
#print axioms Indexed.Phi_neg
#print axioms Indexed.Phi_sum

-- CORRECTION-002, C2 bridge lemmas (N1Specialization.lean), proved before
-- the finite compatibility table per Agent1's recommended route.
#print axioms Indexed.Phi_eN
#print axioms Indexed.rho_Jn
#print axioms Indexed.E_Lof
#print axioms Indexed.E_Fof
#print axioms Indexed.Phi_bracketFFn
#print axioms Indexed.Phi_bracketLFn
#print axioms Indexed.Phi_bracketLLn

-- CORRECTION-002, C2 basis-pair compatibility table: 25 separately named
-- lemmas, one per `(E p, E q)` branch.
#print axioms Indexed.Phi_bracket_basis_L11_L11
#print axioms Indexed.Phi_bracket_basis_L11_L12
#print axioms Indexed.Phi_bracket_basis_L11_L22
#print axioms Indexed.Phi_bracket_basis_L12_L11
#print axioms Indexed.Phi_bracket_basis_L12_L12
#print axioms Indexed.Phi_bracket_basis_L12_L22
#print axioms Indexed.Phi_bracket_basis_L22_L11
#print axioms Indexed.Phi_bracket_basis_L22_L12
#print axioms Indexed.Phi_bracket_basis_L22_L22
#print axioms Indexed.Phi_bracket_basis_L11_F1
#print axioms Indexed.Phi_bracket_basis_L11_F2
#print axioms Indexed.Phi_bracket_basis_L12_F1
#print axioms Indexed.Phi_bracket_basis_L12_F2
#print axioms Indexed.Phi_bracket_basis_L22_F1
#print axioms Indexed.Phi_bracket_basis_L22_F2
#print axioms Indexed.Phi_bracket_basis_F1_L11
#print axioms Indexed.Phi_bracket_basis_F1_L12
#print axioms Indexed.Phi_bracket_basis_F1_L22
#print axioms Indexed.Phi_bracket_basis_F2_L11
#print axioms Indexed.Phi_bracket_basis_F2_L12
#print axioms Indexed.Phi_bracket_basis_F2_L22
#print axioms Indexed.Phi_bracket_basis_F1_F1
#print axioms Indexed.Phi_bracket_basis_F1_F2
#print axioms Indexed.Phi_bracket_basis_F2_F1
#print axioms Indexed.Phi_bracket_basis_F2_F2

-- CORRECTION-002, C2: the general basis_case lemma and the completed,
-- sorry-free `Phi_bracket` bracket-compatibility theorem.
#print axioms Indexed.basis_case
#print axioms Indexed.Phi_bracket

-- CORRECTION-001, C2.1 (HomogeneousDegree.lean): general homogeneous
-- degree-preservation for arbitrary (not just basis) module elements.
#print axioms bracket_isHomog

-- CORRECTION-001/002, C2.2 (N1Specialization.lean): C2.1 transported to
-- the indexed n=1 specialization via `Phi`. Sorry-free following
-- correction002 since `Phi_bracket` itself is now sorry-free.
#print axioms Indexed.bracketN_isHomog

-- T5-T7 (this round), Decode.lean: the decision procedure and its
-- computable sub-checks are `def`s, not proved theorems, so nothing to
-- audit there; the interpretation function `polyAsCoeff`/`termAsCoeff`
-- are likewise plain `def`s.

-- T5 (Bridge.lean): decoding the actual generated fixture datum succeeds,
-- and yields the declared module/basis/degree/operation/coverage/row-count.
#print axioms Bridge.T5_decode_succeeds
#print axioms Bridge.T5_module_id
#print axioms Bridge.T5_basis_ids
#print axioms Bridge.T5_basis_degrees
#print axioms Bridge.T5_op_id
#print axioms Bridge.T5_op_kind
#print axioms Bridge.T5_op_degree
#print axioms Bridge.T5_op_scalar_behavior
#print axioms Bridge.T5_def_kind
#print axioms Bridge.T5_coverage
#print axioms Bridge.T5_row_count

-- T6 (Bridge.lean): the fifteen forward-pair lemmas (against the wire
-- rows alone), the super-skew reverse-pair bridge, the full 25-pair
-- equality, and its extension to the whole module by the same
-- finite-sum shape `bracket` itself already uses.
#print axioms Bridge.forward_L11_L11
#print axioms Bridge.forward_L11_L12
#print axioms Bridge.forward_L11_L22
#print axioms Bridge.forward_L11_F1
#print axioms Bridge.forward_L11_F2
#print axioms Bridge.forward_L12_L12
#print axioms Bridge.forward_L12_L22
#print axioms Bridge.forward_L12_F1
#print axioms Bridge.forward_L12_F2
#print axioms Bridge.forward_L22_L22
#print axioms Bridge.forward_L22_F1
#print axioms Bridge.forward_L22_F2
#print axioms Bridge.forward_F1_F1
#print axioms Bridge.forward_F1_F2
#print axioms Bridge.forward_F2_F2
#print axioms Bridge.gsign_reverse
#print axioms Bridge.decodedBracketBasis_eq
#print axioms Bridge.decodedBracketBasis_eq'
#print axioms Bridge.decodedBracket_eq_bracket

-- T7 (Bridge.lean): positive coefficient examples and negative decoder
-- examples, each reduced to its specified result (Agent1's twelve plus
-- two of Agent2's own adversarial additions).
#print axioms Bridge.posRatExample_valid
#print axioms Bridge.posRatExample_value
#print axioms Bridge.posPolyExample_valid
#print axioms Bridge.posPolyExample_value
#print axioms Bridge.negBadDen_rejected
#print axioms Bridge.negUnreduced_rejected
#print axioms Bridge.negDupExp_rejected
#print axioms Bridge.negUnsorted_rejected
#print axioms Bridge.negWrongLen_rejected
#print axioms Bridge.negDupVecBasis_rejected
#print axioms Bridge.negUnknownBasis_rejected
#print axioms Bridge.negMissingRow_rejected
#print axioms Bridge.negDuplicateRow_rejected
#print axioms Bridge.negReversedRow_rejected
#print axioms Bridge.negPartialInput_rejected
#print axioms Bridge.negSparseInput_rejected
#print axioms Bridge.ownWrongVarOrderInput_rejected
#print axioms Bridge.ownDegreeMismatchInput_rejected

-- R2-C-1, U0 (IndexedLaws.lean): `Jn`/`Lof` interface at general `n`.
#print axioms Indexed.Jn_diag
#print axioms Indexed.Jn_antisymm
#print axioms Indexed.Lof_comm
#print axioms Indexed.Lof_eq_iff
#print axioms Indexed.Lof_ne_inr
#print axioms Indexed.eN_Lof_apply_inr

-- R2-C-1, U1 (IndexedLaws.lean): degree law at basis level and for
-- arbitrary homogeneous elements, general `n`.
#print axioms Indexed.bracketBasisN_degree0
#print axioms Indexed.bracketN_isHomogN

-- R2-C-1, U2 (IndexedLaws.lean): super-skew at basis level, all four
-- sectors, general `n`.
#print axioms Indexed.gsignN_00
#print axioms Indexed.gsignN_01
#print axioms Indexed.gsignN_10
#print axioms Indexed.gsignN_11
#print axioms Indexed.superskew_FF
#print axioms Indexed.superskew_LF
#print axioms Indexed.superskew_FL
#print axioms Indexed.superskew_LL
#print axioms Indexed.bracketBasisN_super_skew

-- R2-C-1, U3 (IndexedLaws.lean): super-skew for arbitrary homogeneous
-- module elements, general `n`.
#print axioms Indexed.bracketN_super_skew_homog

-- R2-C-2 (IndexedJacobi.lean): `bracketN` bilinearity toolkit, general `n`.
#print axioms Indexed.bracketN_add_left
#print axioms Indexed.bracketN_add_right
#print axioms Indexed.bracketN_smul_left
#print axioms Indexed.bracketN_smul_right
#print axioms Indexed.bracketN_neg_right
#print axioms Indexed.bracketN_eN_eN

-- R2-C-2 (IndexedJacobi.lean): basis-shape reduction toolkit, general `n`.
#print axioms Indexed.parity_Lof
#print axioms Indexed.parity_Fof
#print axioms Indexed.bracketBasisN_Fof_Fof
#print axioms Indexed.bracketLFn_comm
#print axioms Indexed.bracketBasisN_Lof_Fof
#print axioms Indexed.bracketBasisN_Fof_Lof
#print axioms Indexed.bracketLLn_comm12
#print axioms Indexed.bracketLLn_comm34
#print axioms Indexed.bracketBasisN_Lof_Lof

-- R2-C-2, U4 (IndexedJacobi.lean): super-Jacobi at basis level, the four
-- sector representatives, general `n`.
#print axioms Indexed.jacobiN_FFF
#print axioms Indexed.jacobiN_LFF
#print axioms Indexed.jacobiN_LLF
#print axioms Indexed.jacobiN_LLL

-- R2-C-2, U4 (IndexedJacobi.lean): the cyclic rotation lemma and the
-- assembly to all eight ordered basis-triple shapes, general `n`.
#print axioms Indexed.jacobiSum_rot
#print axioms Indexed.jacobiN_basis

-- R2-C-2 (IndexedJacobi.lean): finite-sum expansion toolkit for
-- `bracketN`, general `n`, used only by U5's extension.
#print axioms Indexed.bracketN_sum_left
#print axioms Indexed.bracketN_sum_right
#print axioms Indexed.expand_basisN
#print axioms Indexed.bracketN_bilinear_expand
#print axioms Indexed.bracketN_expand_left

-- R2-C-2, U5 (IndexedJacobi.lean): super-Jacobi for arbitrary homogeneous
-- module elements, general `n`, unconditional.
#print axioms Indexed.jacobiN_homog

-- R2-D, G0 (IndexedCoboundary.lean): beta, the vector v, and the
-- contraction identity, general `n`.
#print axioms Indexed.Jn_ne_zero_iff_partner
#print axioms Indexed.contraction_identity
#print axioms Indexed.contraction_identity_neg
#print axioms Indexed.sum_split_even_odd

-- R2-D, G1 (IndexedCoboundary.lean): the primitive f_beta, its two-forms
-- transcription check, its degree law, and its module-level extension.
#print axioms Indexed.fBetaBasis_alt_form
#print axioms Indexed.fBetaBasis_degree
#print axioms Indexed.fBetaN_eN
#print axioms Indexed.fBetaN_add
#print axioms Indexed.fBetaN_smul
#print axioms Indexed.fBetaN_zero
#print axioms Indexed.fBetaN_neg
#print axioms Indexed.fBetaBasis_Lof
#print axioms Indexed.fBetaN_Lof
#print axioms Indexed.fBetaN_isHomogN

-- R2-D (IndexedCoboundary.lean): the generic bilinear-extension combinator
-- shared by G2's GammaBetaN and G3's deltaFN, general `n`.
#print axioms Indexed.bilinearExtend_eN_eN

-- R2-D (IndexedCoboundary.lean): finite-sum bilinearity toolkit for
-- bracketN, general `n`, used only by G4's sector proofs.
#print axioms Indexed.bracketN_zero_left
#print axioms Indexed.bracketN_zero_right
#print axioms Indexed.bracketN_neg_left
#print axioms Indexed.bracketN_sum_right'
#print axioms Indexed.bracketN_sum_left'

-- R2-D, G2 (IndexedCoboundary.lean): the coefficient Gamma_beta, all four
-- sectors, and its map degree, general `n`.
#print axioms Indexed.GammaBetaBasis_degree0

-- R2-D, G4 (IndexedCoboundary.lean): the sum-reduction helpers feeding
-- the LF/FF/FL sectors, general `n`.
#print axioms Indexed.sum_vN_smul_Lof
#print axioms Indexed.sum_vN_mul_Jn_const_smul
#print axioms Indexed.sum_vN_mul_Jn_const_smul'
#print axioms Indexed.sum_vN_const_smul_Lof
#print axioms Indexed.sum_vN_const_smul_Lof'
#print axioms Indexed.sum_vN_const_smul_Fof

-- R2-D, G4 (IndexedCoboundary.lean): Gamma_beta = delta f_beta, the four
-- sector representatives (LL, LF, FF, FL) and their assembly, general `n`.
#print axioms Indexed.GammaBetaBasis_Lof_Lof
#print axioms Indexed.G4_LL
#print axioms Indexed.GammaBetaBasis_Lof_Fof
#print axioms Indexed.G4_LF
#print axioms Indexed.G4_FF
#print axioms Indexed.GammaBetaBasis_Fof_Lof
#print axioms Indexed.G4_FL
#print axioms Indexed.G4

-- R2-D, G5 (IndexedCoboundary.lean): extension to arbitrary elements of
-- IndexedMod n, general `n`, unconditional.
#print axioms Indexed.GammaBetaN_eq_deltaFN
#print axioms Indexed.GammaBetaN_eq_deltaFN_homog

-- R2-E, K0 (IndexedKappa.lean): the model g_R, its embedding/decomposition, and kappa-multiplication; K0c (kappa^2=0, non-injectivity).
#print axioms Indexed.rMod_zero_apply
#print axioms Indexed.rMod_add_apply
#print axioms Indexed.rMod_smul_apply
#print axioms Indexed.rMod_neg_apply
#print axioms Indexed.falsePart_iotaR
#print axioms Indexed.truePart_iotaR
#print axioms Indexed.falsePart_kappaEmbed
#print axioms Indexed.truePart_kappaEmbed
#print axioms Indexed.decompose
#print axioms Indexed.iotaR_add
#print axioms Indexed.iotaR_smul
#print axioms Indexed.iotaR_neg
#print axioms Indexed.kappaEmbed_add
#print axioms Indexed.kappaEmbed_smul
#print axioms Indexed.kappaEmbed_neg
#print axioms Indexed.iotaR_injective
#print axioms Indexed.kappaEmbed_injective
#print axioms Indexed.iotaR_ne_zero_iff
#print axioms Indexed.kappaMulR_eq
#print axioms Indexed.kappaMulR_sq
#print axioms Indexed.kappaMulR_not_injective

-- R2-E, K1 (IndexedKappa.lean): the extended undeformed bracket, eq:scalar-rule, and K1a's restriction to bracketN.
#print axioms Indexed.bracketR_eR_eR
#print axioms Indexed.bracketR_zero_left
#print axioms Indexed.bracketR_zero_right
#print axioms Indexed.bracketR_add_left
#print axioms Indexed.bracketR_add_right
#print axioms Indexed.bracketR_smul_left
#print axioms Indexed.bracketR_smul_right
#print axioms Indexed.bracketR_neg_left
#print axioms Indexed.sum_bool_eq
#print axioms Indexed.iotaR_zero
#print axioms Indexed.iotaR_sum_smul
#print axioms Indexed.iotaR_sum
#print axioms Indexed.bracketR_iotaR_iotaR

-- R2-E, K1c (IndexedKappa.lean): super-skew symmetry on g_R, the convention-pinning check, proved before K5.
#print axioms Indexed.zmod2_cases
#print axioms Indexed.gsignN_comm
#print axioms Indexed.gsignN_mul_one_left
#print axioms Indexed.gsignN_succ_left_mul
#print axioms Indexed.parityR_false
#print axioms Indexed.parityR_true
#print axioms Indexed.bracketRBasis_super_skew
#print axioms Indexed.bracketR_super_skew_homog

-- R2-E, K2 (IndexedKappa.lean): the odd extension (f_beta)_R, with the two extension-rule cases proved.
#print axioms Indexed.falsePart_add
#print axioms Indexed.truePart_add
#print axioms Indexed.falsePart_smul
#print axioms Indexed.truePart_smul
#print axioms Indexed.fBetaR_iotaR
#print axioms Indexed.fBetaR_kappaEmbed
#print axioms Indexed.fBetaR_add
#print axioms Indexed.fBetaR_smul

-- R2-E, K3 (IndexedKappa.lean): T_beta and its inverse, eq:splitting, exact and R-linear and even.
#print axioms Indexed.falsePart_neg
#print axioms Indexed.truePart_neg
#print axioms Indexed.kappaMulR_add
#print axioms Indexed.kappaMulR_iotaR
#print axioms Indexed.kappaMulR_kappaEmbed
#print axioms Indexed.TBeta_eq
#print axioms Indexed.TBetaInv_eq
#print axioms Indexed.falsePart_TBeta
#print axioms Indexed.falsePart_TBetaInv
#print axioms Indexed.TBetaInv_TBeta
#print axioms Indexed.TBeta_TBetaInv
#print axioms Indexed.TBeta_add
#print axioms Indexed.TBeta_smul
#print axioms Indexed.TBeta_kappaMulR
#print axioms Indexed.IsHomogR_falsePart
#print axioms Indexed.TBeta_isHomogR

-- R2-E, K4 (IndexedKappa.lean): the deformed bracket, eq:deformed-bracket, restricting to bracketN + kappa GammaBetaN.
#print axioms Indexed.bracketRBeta_eR_eR
#print axioms Indexed.kappaEmbed_zero
#print axioms Indexed.kappaEmbed_sum
#print axioms Indexed.kappaEmbed_sum_smul
#print axioms Indexed.bracketRBeta_iotaR_iotaR

-- R2-E, K5 (IndexedKappa.lean): the intertwining, eq:intertwining -- basis level (citing G4) and unconditional for arbitrary elements.
#print axioms Indexed.bracketR_eR_left
#print axioms Indexed.bracketR_eR_right
#print axioms Indexed.kappaEmbed_eq_sum
#print axioms Indexed.bracketN_eN_left_eq_sum
#print axioms Indexed.bracketN_eN_right_eq_sum
#print axioms Indexed.bracketR_eR_false_kappaEmbed
#print axioms Indexed.bracketR_sum_left'
#print axioms Indexed.bracketR_sum_right'
#print axioms Indexed.bracketR_kappaEmbed_eR_false
#print axioms Indexed.bracketR_eR_true_kappaEmbed
#print axioms Indexed.bracketR_kappaEmbed_eR_true
#print axioms Indexed.bracketR_kappaEmbed_kappaEmbed
#print axioms Indexed.falsePart_eR_false
#print axioms Indexed.falsePart_eR_true
#print axioms Indexed.TBeta_eR_false
#print axioms Indexed.TBeta_eR_true
#print axioms Indexed.intertwining_basis
#print axioms Indexed.eR_decompose
#print axioms Indexed.TBeta_zero
#print axioms Indexed.TBeta_sum'
#print axioms Indexed.bracketR_sum_left''
#print axioms Indexed.bracketR_sum_right''
#print axioms Indexed.TBeta_sum
#print axioms Indexed.intertwining

-- I106 R1, V0-V3 (IndexedU.lean): the U-conjugation identity, eq:recover-by-u.
#print axioms Indexed.hBasis_Lof
#print axioms Indexed.hBasis_Fof
#print axioms Indexed.hMap_eN
#print axioms Indexed.hMap_add
#print axioms Indexed.hMap_smul
#print axioms Indexed.hMap_zero
#print axioms Indexed.hMap_neg
#print axioms Indexed.hMap_Lof
#print axioms Indexed.hMap_Fof
#print axioms Indexed.hBasis_degree
#print axioms Indexed.hMap_isHomogN
#print axioms Indexed.hR_iotaR
#print axioms Indexed.hR_kappaEmbed
#print axioms Indexed.hR_add
#print axioms Indexed.hR_smul
#print axioms Indexed.UMap_eq
#print axioms Indexed.UMapInv_eq
#print axioms Indexed.falsePart_UMap
#print axioms Indexed.falsePart_UMapInv
#print axioms Indexed.UMapInv_UMap
#print axioms Indexed.UMap_UMapInv
#print axioms Indexed.UMap_add
#print axioms Indexed.UMap_smul
#print axioms Indexed.UMap_kappaMulR
#print axioms Indexed.UMap_isHomogR
#print axioms Indexed.eN_isHomogN
#print axioms Indexed.cratN_two_half
#print axioms Indexed.bracket_FVecBeta_Lof
#print axioms Indexed.bracket_twoFVecBeta_Lof
#print axioms Indexed.bracket_FVecBeta_Fof
#print axioms Indexed.bracket_twoFVecBeta_Fof
#print axioms Indexed.twoFVecBeta_deriv_Lof_Lof_term
#print axioms Indexed.FVecBeta_deriv_Lof_Lof
#print axioms Indexed.twoFVecBeta_deriv_Lof_Lof
#print axioms Indexed.hMap_bracket_Lof_Lof
#print axioms Indexed.deltaH_LL
#print axioms Indexed.deltaH_LF
#print axioms Indexed.deltaH_FF
#print axioms Indexed.deltaH_FL
#print axioms Indexed.deltaH
#print axioms Indexed.GammaBetaN_eq_deltaHN
#print axioms Indexed.UMap_eR_false
#print axioms Indexed.UMap_eR_true
#print axioms Indexed.intertwiningU_basis
#print axioms Indexed.UMap_zero
#print axioms Indexed.UMap_sum'
#print axioms Indexed.UMap_sum
#print axioms Indexed.intertwiningU
#print axioms Indexed.intertwiningU_inv
#print axioms Indexed.intertwiningU_coordinate_corollary

-- I106 R2, W0 (SourceC.lean): C = Q[a]/(a^2-1/2), a odd. A2.
#print axioms Source.Qhalf_apply
#print axioms Source.a_sq
#print axioms Source.a_odd
#print axioms Source.CGrading_gradedAlgebra

-- I106 R2, W1 (SourceWeyl.lean): W_n as operators on WPoly n. A1.
#print axioms Source.wIndex_evenIdx
#print axioms Source.wIndex_oddIdx
#print axioms Source.B_evenIdx
#print axioms Source.B_oddIdx
#print axioms Source.pderiv_mulLeft_comm
#print axioms Source.pderiv_pderiv_X_zero
#print axioms Source.pderiv_pderiv_comm
#print axioms Source.opComm_pderiv_pderiv
#print axioms Source.opComm_mulLeft_mulLeft
#print axioms Source.opComm_swap
#print axioms Source.mulLeft_pderiv_comm
#print axioms Source.u_eq_evenIdx_of_even
#print axioms Source.u_eq_oddIdx_of_odd
#print axioms Source.JnQ_eq_zero_of_same_parity
#print axioms Source.JnQ_evenIdx_oddIdx
#print axioms Source.JnQ_oddIdx_evenIdx
#print axioms Source.B_comm

-- I106 R2, W2 (SourceTensor.lean): A_0 = W_n ⊗ C, A_B = R ⊗ A_0. A3, A4.
#print axioms Source.kappa_sq
#print axioms Source.kappa_mem_RGrading_one
#print axioms Source.kappa_ne_zero
#print axioms Source.RGradedAlgebra
#print axioms Source.a_ne_zero
#print axioms Source.RRing_algebra_rat
#print axioms Source.RRing_isScalarTower
#print axioms Source.RGradedAlgebraQ
#print axioms Source.ZMod2_eq_zero_or_one
#print axioms Source.ZMod2_add_cases
#print axioms Source.ZMod2_total_degree_cases
#print axioms Source.ZMod2_one_add_succ
#print axioms Source.WGrading_setLike
#print axioms Source.WGradedAlgebra
#print axioms Source.A0GradingMap_tmul
#print axioms Source.A0Grading_mem_of_tmul
#print axioms Source.A0Grading_setLike
#print axioms Source.A0_decompose_lin_tmul_coe
#print axioms Source.A0_decompose_lin_tmul
#print axioms Source.A0GradedAlgebra
#print axioms Source.ABGradingMap_tmul
#print axioms Source.ABGradingMap_range_le
#print axioms Source.ABGrading_mem_of_tmul
#print axioms Source.ABGrading_setLike
#print axioms Source.AB_decompose_lin_tmul
#print axioms Source.DirectSum_lof_ABGrading_congr
#print axioms Source.ABGradingMap0_range_le
#print axioms Source.ABGradingMap1_range_le
#print axioms Source.ABGradedAlgebra
#print axioms Source.aA0_mem_A0Grading_one
#print axioms Source.aA0_ne_zero
#print axioms Source.kappa_mem_RGradingQ_one
#print axioms Source.A3_independence
#print axioms Source.A4_W_into_A0
#print axioms Source.A4_C_into_A0
#print axioms Source.A4_R_into_AB
#print axioms Source.A4_A0_into_AB

-- I106 R2, W3 (SourceLifts.lean): the undeformed lifts L^0_uv, F^0_u.
#print axioms Source.Bu0_mem_A0Grading_zero
#print axioms Source.a_comm_Bu0
#print axioms Source.L0_symm
#print axioms Source.L0_mem_A0Grading_zero
#print axioms Source.F0_mem_A0Grading_one

-- I106 R2, W4 (SourceIndependence.lean): the independence statement,
-- written and assessed. L0FamilyIndependent is a bare, deliberately
-- unproved Prop (see the file's own docstring) and so is not audited here.
#print axioms Source.B_linearIndependent
#print axioms Source.F0_eq_smul_tmul
#print axioms Source.F0_linearIndependent
#print axioms Source.A0Grading_disjoint_zero_one
#print axioms Source.independence_of_L0_and_F0

-- I106 R3 (SourceQuadraticIndependence.lean, the one new module this
-- round; the seven-layer freeze -- Indexed.lean, IndexedLaws.lean,
-- IndexedJacobi.lean, IndexedCoboundary.lean, IndexedKappa.lean,
-- IndexedU.lean, and SourceIndependence.lean -- untouched throughout,
-- plus SourceC.lean/SourceWeyl.lean/SourceTensor.lean/SourceLifts.lean
-- also untouched): proves `L0FamilyIndependent` unconditionally, closing
-- the gap W4 left open. Step 0 (`L0FamilyIndependent_of_weyl`) reduces it
-- to independence of the symmetrized `B`-products in `Module.End ℚ
-- (WPoly n)`; the sector identity `B_mul_add_B_mul_swap` (from the
-- already-proved `A1`) turns the four-sector case analysis into one
-- formula. Four independent-sector theorems are proved: `hOO`
-- (both-odd), `hEO_offdiag`/`hEO_diag` (even-odd, off-diagonal via
-- evaluation at `X_m`, diagonal via evaluation at `X_k` combined with the
-- diagonal-sum lemma `hS`), `hOE` (odd-even), and `hEE` (both-even, via
-- evaluation at `X_k * X_l`) -- assembled in `B_sum_linearIndependent`.
-- `L0FamilyIndependent_proved` and `IndependenceStatement_proved`
-- (discharged via the frozen `independence_of_L0_and_F0`, which is not
-- re-proved) are the round's closing results.
#print axioms Source.Bu0_mul
#print axioms Source.B_mul_add_B_mul_swap
#print axioms Source.L0FamilyIndependent_of_weyl
#print axioms Source.oddIdx_le_oddIdx
#print axioms Source.evenIdx_le_evenIdx
#print axioms Source.evenIdx_le_oddIdx
#print axioms Source.oddIdx_le_evenIdx_of_lt
#print axioms Source.coeff_smul_one_eq_zero
#print axioms Source.single_add_single_ne_zero
#print axioms Source.coeff_single_add_single_X_mul_X
#print axioms Source.hval1
#print axioms Source.hOO
#print axioms Source.hval2
#print axioms Source.hval0
#print axioms Source.hS
#print axioms Source.hEO_offdiag
#print axioms Source.hOE
#print axioms Source.hval3
#print axioms Source.hEO_diag
#print axioms Source.hval4
#print axioms Source.hEE
#print axioms Source.B_sum_linearIndependent
#print axioms Source.L0FamilyIndependent_proved
#print axioms Source.IndependenceStatement_proved

-- I106 R4 (SourceIsomorphism.lean, the one new module this round; the
-- eight-layer freeze -- the seven of R3 plus SourceQuadraticIndependence.lean
-- -- untouched throughout): `lem:source-isomorphism`, route (b) --
-- `A_beta` realized concretely inside `AB n`, not via `FreeAlgebra`/
-- `RingQuot`. `b_u := B_u + beta_u*kappa*a` (`bU`) is a specific element
-- of `AB n`; `eq:source`'s three relations are proved as THEOREMS about
-- `bU`, never built into its definition (X1): `bU_comm_bU` (`[b_u,b_v] =
-- J_uv * 1`, against the frozen `Jn` via `JnQ`, exactly as A1 quoted it --
-- proved by expanding into four terms, of which only the lifted A1
-- commutator `BuAB_comm` survives, the three deformation cross-terms
-- `BuAB_comm_betaKappaAB_mul_aAB`/`betaKappaAB_mul_aAB_comm` cancelling),
-- `bU_comm_aAB` (`[b_u,a] = beta_u*kappa`, the one relation that is FALSE
-- for the unshifted `B_u` alone -- `BuAB_comm_aAB` proves `[B_u,a]=0`
-- separately, for contrast), and `aAB_sq` (`a^2=1/2*1`, restating A2).
-- X2, the manuscript's own device `c := kappa*a` (`cAB`): `cAB_sq`
-- (`c^2=0`), `cAB_comm_aAB` (`[c,a]=kappa`), `BuAB_comm_cAB` (`[B_u,c]=0`),
-- `cAB_mem_ABGrading_zero` (`c` even) -- all reduced to the already-
-- accepted A2/A3 facts (`kappa_sq`, `a_sq`, `A3_independence`'s
-- supercentrality `a*kappa=-kappa*a`), none re-derived from scratch. X3,
-- `Phi`/`Psi` mutually inverse: a single inner automorphism (conjugation
-- by the unit `1+c`) was investigated and found NOT to realize the shift
-- (`B_u` already commutes with `c`, so conjugation by `1+c` fixes `B_u`
-- rather than shifting it) -- reported as a disclosed false start, not
-- hidden. Route (b) instead renders the "mutually inverse" content as the
-- direct algebraic cancellation the manuscript's own proof turns on
-- (`bU_sub_eq_BuAB`, `BuAB_add_eq_bU`). X4's disclosure (route (b) shows
-- `AB n` contains a system of generators satisfying `eq:source`, not that
-- the abstractly presented `A_beta` is isomorphic to `AB n`) is recorded
-- in `documents/R4_IMPLEMENTATION_REPORT.md`, written for reuse by a
-- later appendix. No new axiom, no hypothesis parameter, anywhere.
#print axioms Source.BuAB_mul
#print axioms Source.aAB_mul_kappaAB
#print axioms Source.cAB_sq
#print axioms Source.aA0_sq
#print axioms Source.aAB_sq
#print axioms Source.cAB_comm_aAB
#print axioms Source.cAB_eq_tmul
#print axioms Source.BuAB_comm_cAB
#print axioms Source.cAB_mem_ABGrading_zero
#print axioms Source.betaKappa_mem_RGrading_one
#print axioms Source.betaKappa_mem_RGradingQ_one
#print axioms Source.aAB_mul_betaKappaAB
#print axioms Source.BuAB_comm_aAB
#print axioms Source.bU_comm_aAB
#print axioms Source.betaKappa_mul_betaKappa
#print axioms Source.betaKappaAB_mul_aAB
#print axioms Source.BuAB_comm_tmul_aA0
#print axioms Source.betaKappaAB_mul_betaKappaAB
#print axioms Source.betaKappaAB_mul_aAB_comm
#print axioms Source.BuAB_comm
#print axioms Source.bU_sub_eq_BuAB
#print axioms Source.BuAB_add_eq_bU
#print axioms Source.BuAB_comm_betaKappaAB_mul_aAB
#print axioms Source.bU_comm_bU

-- I106 R5 (Y0-Y5): `prop:recovery`, the recovered bracket. Y0 confirmed (an internal
-- second-opinion pass plus Agent2c's own independent computation) that the abstract
-- coordinate model (`Indexed.bracketN`/`Indexed.GammaBetaN`) and the concrete operator
-- model (`L0`/`F0`) satisfy the identical bracket formulas of `lem:base`
-- (`SourceRecoveryBase.lean`, formerly a scratch file, now the base layer: `L0hat_comm`,
-- `L0_comm_F0`, `F0_comm_F0`). Y1/Y2 transcribed `eq:lifts`/`eq:lift-change` under route
-- (b) (`SourceRecoveryLifts.lean`: `L0hatBeta`, `F0hatBeta`, `L0hatBeta_eq`,
-- `F0hatBeta_eq_F0AB`), Y3 proved injectivity of `iota_beta`
-- (`liftsFamilyBeta_linearIndependent`, via a new R-degree-only grading `ABGradingR`). Y4
-- built the bridge `iota0` from the abstract model into `AB n`
-- (`SourceRecoveryBridge.lean`: `iota0_bracketN`, the Stage 1 intertwining theorem) and
-- then the recovery theorem itself (`SourceRecoveryFinal.lean`): unwinding R1's own
-- `intertwiningU_coordinate_corollary` shows the raw concrete bracket carries an extra,
-- forced `hMap`-of-the-output term beyond `GammaBetaN` (not an error -- see the file's
-- header comment); `liftsBracket_general` proves the general-element identity with that
-- term explicit, and `liftsBracket_recovery` rearranges it (pure ring arithmetic on an
-- already-proved identity) into exactly the two-term shape `eq:recover-by-u` and this
-- round's own anti-circularity obligation require, with `Indexed.GammaBetaN` appearing
-- literally and nothing else. Y5 (closed image) is a disclosed, precise partial:
-- `L0AB_eq_L0hatBeta_sub` proves the bare span of `liftsFamilyBeta` is not closed under
-- the bracket (closure needs `kappaAB`-multiples added to the family), not forced into a
-- false closure claim. Full account: `documents/R5_IMPLEMENTATION_REPORT.md`. No new
-- axiom, no hypothesis parameter, anywhere.
-- InhomogeneousDeformations/SourceRecoveryLifts.lean
#print axioms Source.RRing_one_ne_zero
#print axioms Source.L0hatBeta_symm
#print axioms Source.aAB_mul_BuAB
#print axioms Source.F0hatBeta_eq_F0AB
#print axioms Source.BuAB_mul_tmul
#print axioms Source.L0hatBeta_eq
#print axioms Source.ABGradingRMap_tmul
#print axioms Source.ABGradingR_mem_of_tmul
#print axioms Source.AB_decompose_R_lin_tmul_coe
#print axioms Source.AB_decompose_R_lin_tmul
#print axioms Source.ABGradingR_disjoint_zero_one
#print axioms Source.L0AB_mem_ABGradingR_zero
#print axioms Source.correction_mem_ABGradingR_one
#print axioms Source.L0AB_linearIndependent
#print axioms Source.liftsFamilyBeta_linearIndependent
-- InhomogeneousDeformations/SourceRecoveryBase.lean
#print axioms Source.comm_mul_right
#print axioms Source.comm_mul_left
#print axioms Source.comm_scalar
#print axioms Source.symm_comm_symm
#print axioms Source.L0hat_comm
#print axioms Source.Bu0_mul'
#print axioms Source.L0_eq_tmul
#print axioms Source.L0_comm_F0
#print axioms Source.Bu0_mul_L0hat_shape
#print axioms Source.F0_comm_F0
-- InhomogeneousDeformations/SourceRecoveryBridge.lean
#print axioms Source.algebraMap_mem_RGradingQ_zero
#print axioms Source.Jn_eq_C_JnQ
#print axioms Source.algebraMap_Jn
#print axioms Source.algebraMap_cratN
#print axioms Source.iota0_add
#print axioms Source.iota0_smul
#print axioms Source.iota0_eN
#print axioms Source.AB_zero_tmul
#print axioms Source.iota0_zero
#print axioms Source.iota0_neg
#print axioms Source.iota0AB_scale
#print axioms Source.iota0_smul_eN
#print axioms Source.AB_tmul_smul_left
#print axioms Source.algebraMap_cratN_half_mul_Jn
#print axioms Source.A0_tmul_add
#print axioms Source.A0_tmul_smul
#print axioms Source.tmul_one_mul_tmul_one
#print axioms Source.L0_comm
#print axioms Source.iota0Basis_Lof
#print axioms Source.iota0Basis_Fof
#print axioms Source.AB_tmul_add
#print axioms Source.AB_tmul_smul
#print axioms Source.AB_tmul_sub
#print axioms Source.AB_one_tmul_mul_one_tmul
#print axioms Source.iota0AB_comm_LL
#print axioms Source.iota0AB_comm_LF
#print axioms Source.iota0AB_comm_FL
#print axioms Source.iota0AB_anticomm_FF
#print axioms Source.iota0_bracketLLn_sum
#print axioms Source.iota0_bracketLLn
#print axioms Source.iota0_bracketLFn_sum
#print axioms Source.iota0_bracketLFn
#print axioms Source.iota0_bracketFLn
#print axioms Source.iota0_bracketFFn_sum
#print axioms Source.iota0_bracketFFn
#print axioms Source.iota0_bracketBasisN_Lof_Lof
#print axioms Source.iota0_bracketBasisN_Lof_Fof
#print axioms Source.iota0_bracketBasisN_Fof_Lof
#print axioms Source.iota0_bracketBasisN_Fof_Fof
#print axioms Source.iota0_sum
#print axioms Source.iota0_sum'
#print axioms Source.iota0_bracketN
-- InhomogeneousDeformations/SourceRecoveryFinal.lean
#print axioms Source.comm_mul_right_AB
#print axioms Source.comm_mul_left_AB
#print axioms Source.symm_comm_symm_AB
#print axioms Source.L0hatBeta_comm
#print axioms Source.F0AB_eq_aAB_mul_BuAB
#print axioms Source.iota0AB_Fof_eq_F0AB
#print axioms Source.iota0AB_Lof_eq_L0AB
#print axioms Source.F0hatBeta_bracket_FF
#print axioms Source.BuAB_sandwich_aAB
#print axioms Source.BuAB_mul_BuAB_add_swap
#print axioms Source.D_mul_aBuAB
#print axioms Source.aBuAB_mul_D
#print axioms Source.LF_correction
#print axioms Source.kappaAB_mul_algebraMap_tmul
#print axioms Source.kappaAB_mul_algebraMap_tmul_L0
#print axioms Source.iota0_gammaLFn
#print axioms Source.L0hatBeta_bracket_LF
#print axioms Source.F0hatBeta_bracket_FL
#print axioms Indexed.hMap_bracketFFn
#print axioms Indexed.hMap_bracketLFn
#print axioms Indexed.hMap_bracketLLn
#print axioms Source.betaKappaAB_mul_aAB_mul_BuAB
#print axioms Source.Corr_eq
#print axioms Source.kappaAB_mul_algebraMap_tmul_F0
#print axioms Source.iota0_hMap_term
#print axioms Source.iota0_hMap_bracketLLn
#print axioms Source.L0hatBeta_bracket_LL
#print axioms Source.liftsBracket_eq_bridge
#print axioms Indexed.hMap_sum
#print axioms Indexed.hMap_sum'
#print axioms Source.hMap_bracketN_eq
#print axioms Source.GammaBetaN_eq_sum
#print axioms Source.algebraMap_tmul_comm_kappaAB
#print axioms Source.algebraMap_tmul_mul_kappaAB_mul
#print axioms Source.iota0_bilinearExtend
#print axioms Source.liftsBracket_general
#print axioms Source.liftsBracket_recovery
#print axioms Source.betaKappaAB_eq_kappaAB_mul
#print axioms Source.L0AB_eq_L0hatBeta_sub

-- I106 R6 (Z0-Z4): closing P4. Z0 defines `iotaBetaR : RMod n -> AB n`
-- (`SourceRecoveryClosure.lean`), the map R5 lacked, extending `liftsFamilyBeta` the way `iota0`
-- extends `iota0Basis` (false slot to the deformed lift, true slot to `kappaAB *` the deformed
-- lift). Z1 proves `iota0_eq_iotaBetaR_UMapInv_iotaR` -- `UMapInv` (frozen `IndexedU.lean`)
-- appearing literally in a Lean statement for the first time, closing the gap Agent1c's R5 review
-- identified (the identification had lived only in a header comment). Z2 (`prop_recovery`)
-- restates R5's `liftsBracket_general` with no hand subtraction: the raw bracket equals
-- `iotaBetaR` applied to `iotaR (bracketN x y) + kappaEmbed (GammaBetaN x y)`, `GammaBetaN`
-- literal; the cancellation (two independent hand-derivations, then the Lean proof itself) came
-- out exactly clean, no adjustment. Z2b (`prop_recovery_bracketRBeta`) ties this to the frozen
-- `bracketRBeta`, one line. Z3 proves `iotaBetaR` injective (`iotaBetaR_injective`, via a genuine
-- decomposition `iotaBetaR_eq_iota0_add` into `ABGradingR n 0`/`n 1` pieces, and new
-- `TrivSqZeroExt`-based coordinate-retraction infrastructure -- `iota0_injective`,
-- `kappaAB_iota0_eq_zero_imp` -- since R5's existing independence results were stated for
-- `Pn n`-scalar coefficients, not the polynomial coefficients `iota0`'s domain actually carries)
-- and the coefficient's uniqueness (`coefficient_unique`), matching the manuscript's own
-- `g_R = g_P (+) kappa g_P` splitting (lines 270-272) without ever needing kappa-multiplication
-- injective on anything wider than `algebraMap`'s image -- `kappaMulR_not_injective` (K0c) is
-- left completely undisturbed. Z3's closed-image half is NOT completed: reported as a precise,
-- typed gap (would need `liftsBracket_general`'s machinery generalized to brackets of
-- `kappa`-multiplied family members via `bracketR`/`bracketRBasis`'s own structure, a genuinely
-- open-ended extension, not a corollary of what is proved). Full account:
-- `documents/R6_IMPLEMENTATION_REPORT.md`. No new axiom, no hypothesis parameter, anywhere.
-- InhomogeneousDeformations/SourceRecoveryClosure.lean
#print axioms Source.kappaAB_sq_eq_zero
#print axioms Source.falsePart_sub
#print axioms Source.truePart_sub
#print axioms Source.liftBeta_add
#print axioms Source.liftBeta_smul
#print axioms Source.liftBeta_zero
#print axioms Source.liftBeta_eN
#print axioms Source.liftBeta_neg
#print axioms Source.liftBeta_sum'
#print axioms Source.iotaBetaR_add
#print axioms Source.iotaBetaR_smul
#print axioms Source.iota0_eN_sub_eq
#print axioms Source.iota0_eq_liftBeta_sub
#print axioms Source.liftBeta_eq_iota0_add
#print axioms Source.iota0_eq_iotaBetaR_UMapInv_iotaR
#print axioms Source.prop_recovery
#print axioms Source.prop_recovery_bracketRBeta
#print axioms Source.iota0Basis_linearIndependent
#print axioms Source.iota0BasisCombo_injective
#print axioms Source.RRingRetract_algebraMap
#print axioms Source.iota0Coord_apply
#print axioms Source.iota0_eq_zero_imp
#print axioms Source.iota0_injective
#print axioms Source.RRingRetractKappa_kappa_mul_algebraMap
#print axioms Source.kappaCoord_apply
#print axioms Source.kappaAB_iota0_eq_zero_imp
#print axioms Source.iota0_mem_ABGradingR_zero
#print axioms Source.kappa_mul_algebraMap_mem_RGradingQ_one
#print axioms Source.kappaAB_mul_iota0_mem_ABGradingR_one
#print axioms Source.iotaBetaR_zero
#print axioms Source.iotaBetaR_neg
#print axioms Source.iotaBetaR_eq_iota0_add
#print axioms Source.iotaBetaR_injective
#print axioms Source.coefficient_unique

-- I106 R7 (W0-W3): closed image, the last clause of prop:recovery. The naive extension of
-- liftsBracket's commutator/anticommutator dispatch (bare IndexedBasis n parity, unchanged) to
-- kappa-multiplied generators does not close; the correct extension uses parityR (the kappa-slot
-- itself contributes to parity, already used in the frozen bracketRBasis_super_skew) to build a
-- genuine supercommutator, iotaBetaRBracket. W0 confirms the image characterization
-- (RMod n ~= IndexedMod n x IndexedMod n via (falsePart r, hMap(falsePart r) + truePart r)). W1
-- proves all four RBasis-pair rows against the frozen bracketRBetaBasis using this single
-- parityR-based formula; the (false,false) row reduces exactly to the already-frozen
-- liftsBracket_eq_bridge, confirming this is not a new operation. W2 (iotaBetaR_bracket_closed)
-- assembles the strong bilinear closure theorem for all of RMod n, and
-- iotaBetaR_bracket_closed_specializes confirms it reduces exactly to R6's own
-- liftsBracket_general/prop_recovery on iotaR-embedded coordinates -- no drift. W3
-- (prop_recovery_assembled) ties all four clauses of prop:recovery (injectivity, the recovered
-- bracket, the coefficient's uniqueness, closed image) to their carrying theorems in one place.
-- With this round, prop:recovery is complete, clause by clause, without qualification. Full
-- account: documents/R7_IMPLEMENTATION_REPORT.md. No new axiom, no hypothesis parameter, anywhere.
-- InhomogeneousDeformations/SourceRecoveryClosedImage.lean
#print axioms Source.gsignNQ_00
#print axioms Source.gsignNQ_01
#print axioms Source.gsignNQ_10
#print axioms Source.gsignNQ_11
#print axioms Source.L0AB_comm_kappaAB
#print axioms Source.F0AB_comm_kappaAB
#print axioms Source.algebraMap_tmul_F0_comm_kappaAB
#print axioms Source.iota0_hMap_eN_comm_kappaAB
#print axioms Source.truePart_eR_false
#print axioms Source.truePart_eR_true
#print axioms Source.iotaBetaR_eR_false
#print axioms Source.iotaBetaR_eR_true
#print axioms Source.iotaBetaR_image_eq
#print axioms Source.iotaBetaR_iotaR_add_kappaEmbed
#print axioms Source.iotaBetaRBracket_false_false
#print axioms Source.kappaAB_mul_liftsFamilyBeta
#print axioms Source.kappaAB_mul_iota0AB_mul_kappaAB
#print axioms Source.iotaBetaRBracket_true_true
#print axioms Source.iota0AB_mul_kappaAB
#print axioms Source.liftsFamilyBeta_mul_kappaAB
#print axioms Source.kappaAB_mul_iota0AB_mul_liftsFamilyBeta
#print axioms Source.iotaBetaR_kappaEmbed
#print axioms Source.algebraMap_gsignN
#print axioms Source.Lof_eq_inl
#print axioms Source.iota0_bracketBasisN_unified
#print axioms Source.gsignNQ_mul_eq
#print axioms Source.gsignNQ_succ_mul_eq
#print axioms Source.iotaBetaRBracket_true_false
#print axioms Source.iotaBetaRBracket_false_true
#print axioms Source.iotaBetaR_sum
#print axioms Source.iotaBetaR_sum'
#print axioms Source.iotaBetaR_bracket_closed
#print axioms Source.iotaBetaRBracket_false_false_eq_liftsBracket
#print axioms Source.iotaBetaR_bracket_closed_specializes
#print axioms Source.prop_recovery_assembled

-- I106 R8b (U0-U5): the FL sector re-derived without the circular link, correcting R8 (not
-- part of this endpoint; rejected). Agent1c's mechanical dependency-path checker found that R8's
-- chain (GammaBetaBasis_Fof_Lof_forced -> bracketRBetaBasis_super_skew ->
-- iotaBetaR_bracketRBetaBasis -> iotaBetaRBracket_false_false -> liftsBracket_eq_bridge ->
-- GammaBetaBasis_Fof_Lof) consumed the declared FL row one layer down, inside frozen R5's
-- F0hatBeta_bracket_FL. That chain and its three FL-carrying theorems are dropped entirely here,
-- not relabelled. U1 keeps gsignNQ_symm, gsignNQ_sq, iotaBetaRBracket_super_skew verbatim
-- (confirmed FL-free, unaffected). U2 (F0hatBeta_comm_L0hatBeta_FL_free) restates the FL
-- commutator via R5's own LF-sector theorem (L0hatBeta_bracket_LF, itself FL-free) plus a sign
-- flip in A_B, never citing F0hatBeta_bracket_FL or liftsBracket_eq_bridge. U3
-- (GammaBetaBasis_Fof_Lof_forced_FL_free) pins the FL coefficient to -(gammaLFn n u v w) via R6's
-- kappaAB_iota0_eq_zero_imp, never inspecting GammaBetaBasis. Agent1c's own mechanical checker
-- (BFS over compiled proof-term dependencies to InhomogeneousDeformations.Indexed.GammaBetaBasis_Fof_Lof) was
-- run on all five theorems below, independently in this round's implementation and again in
-- Agent2c's own verification pass, and found no dependency path from any of them to the FL row
-- lemma; Agent1c's own independent re-check is part of this round's review. Full account:
-- documents/R8B_IMPLEMENTATION_REPORT.md. No new axiom, no hypothesis parameter, anywhere.
-- InhomogeneousDeformations/SourceRecoveryFLConvention.lean
#print axioms Source.gsignNQ_symm
#print axioms Source.gsignNQ_sq
#print axioms Source.iotaBetaRBracket_super_skew
#print axioms Source.F0hatBeta_comm_L0hatBeta_FL_free
#print axioms Source.GammaBetaBasis_Fof_Lof_forced_FL_free

end InhomogeneousDeformations
