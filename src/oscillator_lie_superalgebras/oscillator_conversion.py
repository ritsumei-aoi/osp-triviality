"""
Conversion functions between algebra basis and oscillator standard form.

This module provides bidirectional conversion:
- expand_algebra_basis_to_standard: M_{i,j}, Q_k -> oscillator standard form
- standard_form_to_algebra_basis: oscillator standard form -> linear combination of M_{i,j}, Q_k

Key principle: The algebra basis {M_{i,j}, Q_k} is symmetrized, so conversion
requires solving linear equations, not naive term-by-term mapping.

References:
- handover/oscillator_standard_form.md
- Bakalov & Sullivan (arXiv:1612.09400), Eq. (4.3)
"""

import sympy as sp
from typing import Dict


def expand_algebra_basis_to_standard(
    basis_name: str,
    a_std: sp.Symbol,
    b_std: Dict[int, sp.Symbol]
) -> sp.Expr:
    """
    Expand a single algebra basis element into oscillator standard form.
    
    Parameters
    ----------
    basis_name : str
        Name of the algebra basis element (e.g., "M_0_1", "Q_2")
    a_std : sp.Symbol
        Commutative symbol for the fermionic oscillator a
    b_std : Dict[int, sp.Symbol]
        Dictionary mapping index to commutative symbol for bosonic oscillator b_i
    
    Returns
    -------
    sp.Expr
        Oscillator expression in standard form (commutative symbols)
    
    Notes
    -----
    Expansion rules (Bakalov & Sullivan, Eq. 4.3):
    - M_{i,j} = (1/2) * (b_i * b_j + b_j * b_i)  [symmetrized]
    - Q_k = (1/sqrt(2)) * a * b_k
    
    The output is already in standard order since we use commutative symbols.
    """
    if basis_name.startswith("M_"):
        # Parse M_{i,j}
        parts = basis_name.split("_")
        if len(parts) != 3:
            raise ValueError(f"Invalid M basis name: {basis_name}")
        i, j = int(parts[1]), int(parts[2])
        
        if i > j:
            raise ValueError(f"M basis must have i <= j, got {basis_name}")
        
        # M_{i,j} = (1/2) * (b_i * b_j + b_j * b_i)
        return sp.Rational(1, 2) * (b_std[i] * b_std[j] + b_std[j] * b_std[i])
    
    elif basis_name.startswith("Q_"):
        # Parse Q_k
        parts = basis_name.split("_")
        if len(parts) != 2:
            raise ValueError(f"Invalid Q basis name: {basis_name}")
        k = int(parts[1])
        
        # Q_k = (1/sqrt(2)) * a * b_k
        return sp.Rational(1, 1) / sp.sqrt(2) * a_std * b_std[k]
    
    else:
        raise ValueError(f"Unknown basis element: {basis_name}")

def standard_form_to_algebra_basis(
    std_expr: sp.Expr,
    a_std: sp.Symbol,
    b_std: Dict[int, sp.Symbol],
    max_index: int
) -> Dict[str, sp.Expr]:
    """
    Convert oscillator standard form to algebra basis via linear equation solving.
    
    Parameters
    ----------
    std_expr : sp.Expr
        Expression in oscillator standard form (commutative symbols a_std, b_std[k])
    a_std : sp.Symbol
        Commutative symbol for fermionic oscillator
    b_std : Dict[int, sp.Symbol]
        Dictionary mapping index to commutative symbol for bosonic oscillator
    max_index : int
        Maximum index for b_std (determines dimension = 2n)
    
    Returns
    -------
    Dict[str, sp.Expr]
        Dictionary mapping basis element name to coefficient.
        Does NOT include "I" (identity/constant term).
    
    Notes
    -----
    This function inverts the expansion defined in expand_algebra_basis_to_standard.
    Key points:
    - Constant terms (no oscillators) are discarded (not part of algebra basis)
    - b_i * b_j terms -> M_{i,j} (with i <= j)
    - a * b_k terms -> Q_k
    
    The conversion is exact because:
    1. M_{i,j} with i < j has unique b_i * b_j and b_j * b_i terms
    2. M_{i,i} corresponds to b_i^2
    3. Q_k corresponds to a * b_k
    """
    expanded = sp.expand(std_expr)
    
    if expanded.is_Add:
        terms = expanded.args
    else:
        terms = [expanded]
    
    basis_dict = {}
    
    for term in terms:
        # Separate coefficient and oscillator variables
        coeff = sp.Integer(1)
        osc_vars = []  # List of tuples (type, index/None)
        
        if term.is_Mul:
            for arg in term.args:
                if arg == a_std:
                    osc_vars.append(('a', None))
                elif arg in b_std.values():
                    # Find index
                    idx = [k for k, v in b_std.items() if v == arg][0]
                    osc_vars.append(('b', idx))
                elif arg.is_Pow:
                    # Handle b_i^2, a^2, etc.
                    base, exp = arg.args
                    if base == a_std:
                        osc_vars.extend([('a', None)] * int(exp))
                    elif base in b_std.values():
                        idx = [k for k, v in b_std.items() if v == base][0]
                        osc_vars.extend([('b', idx)] * int(exp))
                    else:
                        coeff *= arg
                elif arg.is_commutative and arg.is_number:
                    coeff *= arg
                elif arg.is_commutative:
                    coeff *= arg
                else:
                    raise ValueError(f"Unexpected argument in term: {arg}")
        elif term.is_Pow:
            # Single power term (e.g., b_0^2)
            base, exp = term.args
            if base == a_std:
                osc_vars.extend([('a', None)] * int(exp))
            elif base in b_std.values():
                idx = [k for k, v in b_std.items() if v == base][0]
                osc_vars.extend([('b', idx)] * int(exp))
            else:
                coeff = term
        else:
            # Single symbol or constant
            if term == a_std:
                osc_vars.append(('a', None))
            elif term in b_std.values():
                idx = [k for k, v in b_std.items() if v == term][0]
                osc_vars.append(('b', idx))
            elif term.is_commutative:
                coeff = term
            else:
                raise ValueError(f"Unexpected term: {term}")
        
        # Extract b indices and a presence
        b_indices = [v[1] for v in osc_vars if v[0] == 'b']
        has_a = any(v[0] == 'a' for v in osc_vars)
        
        # Classify and convert to basis
        if len(b_indices) == 0 and not has_a:
            # Pure constant -> Skip (not part of algebra basis)
            continue
        
        elif len(b_indices) == 2 and not has_a:
            # b_i * b_j -> Solve for M basis
            i, j = sorted(b_indices)  # Ensure i <= j
            
            if i == j:
                # b_i^2 -> M_{i,i}
                # M_{i,i} = (1/2) * (b_i * b_i + b_i * b_i) = b_i^2
                # So: b_i^2 = M_{i,i}
                target = f"M_{i}_{i}"
                basis_dict[target] = basis_dict.get(target, 0) + coeff
            else:
                # b_i * b_j (i < j) -> M_{i,j}
                # M_{i,j} = (1/2) * (b_i * b_j + b_j * b_i)
                # In commutative symbols: b_i * b_j = b_j * b_i
                # So: M_{i,j} = (1/2) * 2 * b_i * b_j = b_i * b_j
                # Therefore: b_i * b_j = M_{i,j}
                target = f"M_{i}_{j}"
                basis_dict[target] = basis_dict.get(target, 0) + coeff
        
        elif len(b_indices) == 1 and has_a:
            # a * b_k -> Q_k
            # Q_k = (1/sqrt(2)) * a * b_k
            # So: a * b_k = sqrt(2) * Q_k
            k = b_indices[0]
            target = f"Q_{k}"
            basis_dict[target] = basis_dict.get(target, 0) + coeff * sp.sqrt(2)
        
        else:
            raise ValueError(
                f"Unexpected oscillator structure in standard form: "
                f"a={has_a}, b_indices={b_indices}"
            )
    
    # Clean up zero entries
    cleaned = {k: v for k, v in basis_dict.items() if sp.simplify(v) != 0}
    
    return cleaned
