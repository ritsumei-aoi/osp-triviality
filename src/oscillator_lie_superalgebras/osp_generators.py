from typing import Dict, List, Tuple
import numpy as np

def standard_symplectic_matrix(n: int) -> np.ndarray:
    """
    Returns the standard 2n x 2n symplectic matrix J:
      J = [ 0  I ]
          [-I  0 ]
    where I is the n x n identity matrix.
    """
    J = np.zeros((2 * n, 2 * n), dtype=int)
    I = np.eye(n, dtype=int)
    J[:n, n:] = I
    J[n:, :n] = -I
    return J

def build_osp12n_structure_constants(n: int) -> Tuple[List[str], Dict[str, int], Dict[Tuple[str, str], Dict[str, complex]]]:
    """
    Builds the structure constants for osp(1|2n) using the oscillator realization basis.
    
    Generators:
      - Even: M_{i,j} = (1/2)(b_i b_j + b_j b_i)  for 0 <= i <= j < 2n
      - Odd:  Q_{i}   = (1/2)(a b_i + b_i a)      for 0 <= i < 2n
      
    Parity:
      - p(M_{i,j}) = 0
      - p(Q_i) = 1
      
    Brackets:
      Let J be the standard 2n x 2n symplectic matrix.
      [M_{i,j}, M_{k,l}] = J_{j,k} M_{i,l} + J_{i,k} M_{j,l} + J_{j,l} M_{i,k} + J_{i,l} M_{j,k}
      [M_{i,j}, Q_k]     = J_{j,k} Q_i + J_{i,k} Q_j
      {Q_i, Q_j}         = (1/2) M_{i,j}
      
    Returns:
      basis: List of generator names (e.g., "M_0_0", "Q_0")
      parity: Dictionary mapping generator name to its parity (0 or 1)
      br: Dictionary mapping (gen1, gen2) to a dictionary of {result_gen: coeff}
    """
    J = standard_symplectic_matrix(n)
    dim = 2 * n
    
    basis = []
    parity: Dict[str, int] = {}
    
    def m_name(i: int, j: int) -> str:
        if i > j:
            i, j = j, i
        return f"M_{i}_{j}"
        
    for i in range(dim):
        for j in range(i, dim):
            name = m_name(i, j)
            if name not in basis:
                basis.append(name)
                parity[name] = 0
            
    def q_name(i: int) -> str:
        return f"Q_{i}"
        
    for i in range(dim):
        name = q_name(i)
        basis.append(name)
        parity[name] = 1
        
    br: Dict[Tuple[str, str], Dict[str, complex]] = {}
    
    def add_term(out_dict: Dict[str, complex], gen: str, coeff: complex) -> None:
        if abs(coeff) > 1e-12:
            out_dict[gen] = out_dict.get(gen, 0j) + coeff
            if abs(out_dict[gen]) <= 1e-12:
                del out_dict[gen]

    # [M_{i,j}, M_{k,l}]
    for i in range(dim):
        for j in range(i, dim):
            m_ij = m_name(i, j)
            for k in range(dim):
                for l in range(k, dim):
                    m_kl = m_name(k, l)
                    res: Dict[str, complex] = {}
                    add_term(res, m_name(i, l), J[j, k])
                    add_term(res, m_name(j, l), J[i, k])
                    add_term(res, m_name(i, k), J[j, l])
                    add_term(res, m_name(j, k), J[i, l])
                    if res:
                        br[(m_ij, m_kl)] = res

    # [M_{i,j}, Q_k]
    for i in range(dim):
        for j in range(i, dim):
            m_ij = m_name(i, j)
            for k in range(dim):
                q_k = q_name(k)
                res: Dict[str, complex] = {}
                add_term(res, q_name(i), J[j, k])
                add_term(res, q_name(j), J[i, k])
                if res:
                    br[(m_ij, q_k)] = res
                    # Anti-symmetry: [Q_k, M_{i,j}] = - [M_{i,j}, Q_k]
                    br[(q_k, m_ij)] = {g: -c for g, c in res.items()}

    # {Q_i, Q_j}
    for i in range(dim):
        q_i = q_name(i)
        for j in range(dim):
            q_j = q_name(j)
            res: Dict[str, complex] = {}
            add_term(res, m_name(i, j), 0.5)
            if res:
                br[(q_i, q_j)] = res

    return basis, parity, br
