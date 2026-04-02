"""
Compute gamma (2-cocycle) from matrix B.

This module computes the 2-cocycle gamma that arises from an inhomogeneous
bilinear form specified by matrix B, for the osp(1|2n) Lie superalgebra.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union, List
from .gamma_parser import GammaStructureParser


# 既存コードの修正箇所のみ記載

def compute_gamma_from_B(
    n: int, 
    B: Union[np.ndarray, List[List[float]], List[float]]
) -> Dict[Tuple[str, str], Dict[str, complex]]:  # ← 型ヒント修正
    """
    Compute the 2-cocycle gamma from matrix B for osp(1|2n).
    
    This function uses pre-computed gamma structures stored in JSON files.
    The gamma structure contains symbolic expressions in terms of B matrix
    parameters. This function substitutes concrete B values into those
    expressions to obtain numerical gamma coefficients.
    
    Args:
        n: Size parameter (osp(1|2n))
        B: Matrix B specifying the inhomogeneous deformation.
           Can be provided as:
           - np.ndarray of shape (2n, 1) for column vector (real or complex dtype)
           - List[List[float]] for 2D array
           - List[float] for 1D column vector (will be reshaped)
           
           For osp(1|2n), B should have 2n entries: [B_0, B_1, ..., B_{2n-1}]
           Values can be real or complex.
        
    Returns:
        Dictionary mapping (gen1, gen2) -> {result_gen: coefficient}
        Coefficients are complex numbers (real coefficients have imag=0).
        Keys use code basis (M_i_j, Q_i) for the algebra generators.
        
    Raises:
        FileNotFoundError: If gamma structure JSON for this n is not found
        ValueError: If B has incorrect shape
        
    Example:
        >>> # For osp(1|2) with B_1 matrix from paper
        >>> B = [1.0, 0.0]
        >>> gamma = compute_gamma_from_B(1, B)
        >>> # gamma[("Q_0", "Q_0")] will contain {"Q_0": 2.828...}
        
        >>> # For complex B
        >>> B = [1.0, 1.0j]
        >>> gamma = compute_gamma_from_B(1, B)
    """
    # Convert B to proper format
    B_array = _validate_and_format_B(n, B)
    
    # Load gamma structure from JSON
    gamma_json_path = _get_gamma_structure_path(n)
    parser = GammaStructureParser(str(gamma_json_path))
    
    # Create substitution dictionary
    B_values = _create_B_substitution_dict(n, B_array)
    
    # Substitute B values into gamma structure
    gamma_numeric = parser.substitute_B(B_values)
    
    return gamma_numeric


def _create_B_substitution_dict(n: int, B: np.ndarray) -> Dict[str, Union[float, complex]]:
    """
    Create substitution dictionary from B array.
    
    Args:
        n: Size parameter
        B: B matrix as numpy array of shape (2n, 1)
        
    Returns:
        Dictionary mapping "B_i" to numerical values (real or complex)
        
    Example:
        >>> B = np.array([[1.0], [0.5]])
        >>> _create_B_substitution_dict(1, B)
        {'B_0': 1.0, 'B_1': 0.5}
        
        >>> B = np.array([[1.0+0j], [1.0j]])
        >>> _create_B_substitution_dict(1, B)
        {'B_0': (1+0j), 'B_1': 1j}
    """
    B_dict = {}
    for i in range(2 * n):
        param_name = f"B_{i}"
        value = B[i, 0]
        
        # MODIFIED: Support both real and complex B values
        if isinstance(value, (complex, np.complex128, np.complex64)):
            B_dict[param_name] = complex(value)
        elif isinstance(value, (float, np.float64, np.float32)):
            B_dict[param_name] = float(value)
        else:
            # Fallback: try to convert
            B_dict[param_name] = complex(value)
    
    return B_dict

def _validate_and_format_B(n: int, B: Union[np.ndarray, List, List[List[float]]]) -> np.ndarray:
    """
    Validate and convert B to standard numpy array format.
    
    Args:
        n: Size parameter
        B: Input B matrix in various formats
        
    Returns:
        B as numpy array of shape (2n, 1)
        
    Raises:
        ValueError: If B has incorrect shape or dimensions
    """
    # Convert to numpy array
    if not isinstance(B, np.ndarray):
        B = np.array(B)
    
    # Handle 1D array (column vector)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    
    expected_shape = (2 * n, 1)
    if B.shape != expected_shape:
        raise ValueError(
            f"B matrix has incorrect shape. Expected {expected_shape}, got {B.shape}. "
            f"For osp(1|{2*n}), B should be a column vector with {2*n} entries."
        )
    
    return B


def _get_gamma_structure_path(n: int) -> Path:
    """
    Get the path to the gamma structure JSON file for osp(1|2n).
    
    Args:
        n: Size parameter
        
    Returns:
        Path to gamma structure JSON
        
    Raises:
        FileNotFoundError: If gamma structure file doesn't exist
    """
    # Determine project root (assuming this file is in src/oscillator_lie_superalgebras/)
    module_path = Path(__file__).resolve()
    project_root = module_path.parent.parent.parent
    
    gamma_file = project_root / "data" / "gamma_structures" / f"osp_1_{2*n}_gamma.json"
    
    if not gamma_file.exists():
        raise FileNotFoundError(
            f"Gamma structure file not found: {gamma_file}\n"
            f"Available gamma structures may be limited. "
            f"Please generate gamma structure for n={n} first using "
            f"experiments/generate_gamma_structure_json.py"
        )
    
    return gamma_file


def get_available_gamma_structures() -> List[int]:
    """
    Get list of n values for which gamma structures are available.
    
    Returns:
        List of n values (e.g., [1, 2, 3])
    """
    module_path = Path(__file__).resolve()
    project_root = module_path.parent.parent.parent
    gamma_dir = project_root / "data" / "gamma_structures"
    
    available = []
    if gamma_dir.exists():
        for file in gamma_dir.glob("osp_1_*_gamma.json"):
            # Extract 2n from filename
            two_n = int(file.stem.split('_')[2])
            n = two_n // 2
            available.append(n)
    
    return sorted(available)
