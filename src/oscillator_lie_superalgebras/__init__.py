"""
osp-triviality: Computational verification of triviality for osp(1|2n) deformations.

Core library for oscillator Lie superalgebras B(0,n) = osp(1|2n).
"""

from .triviality_checker import check_triviality
from .cohomology_solver import solve_odd_f_generic, reconstruct_gamma_from_f
from .adjoint_from_json import build_adjoint_from_json
from .gamma_from_B import compute_gamma_from_B
from .algebra_parser import OSpAlgebraParser
from .gamma_parser import GammaStructureParser
from .oscillator_algebra import BmnAlgebra
from .B_generators import build_B0n_basis, build_B0n_structure_constants

__version__ = "0.1.0"
