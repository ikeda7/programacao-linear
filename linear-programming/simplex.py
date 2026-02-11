import numpy as np
from numpy import linalg

def simplex_min(A, b, c, Xn, Xb, max_iter=50, verbose=True):
    for it in range(max_iter):
        B = A[:, Xb]
        B_inv = linalg.inv(B)
        cb = c[Xb]
        cn = c[Xn]

        xBarra = B_inv @ b
        zBarra = cb.T @ xBarra

        zj_cj = cb.T @ B_inv @ A[:, Xn] - cn.T
        zj_cj = zj_cj.flatten()
        imax = np.argmax(zj_cj)
        if zj_cj[imax] <= 0:
            if verbose:
                print("\n***Optimal solution found!***")
                print(f"Optimal value: {zBarra.flatten()[0]}")
                print(f"Basic variables: {Xb}, Non-basic variables: {Xn}")
                print(f"Solution: {xBarra.flatten()}")
            return Xb, xBarra, zBarra

        yi = B_inv @ A[:, Xn[imax]]
        with np.errstate(divide='ignore', invalid='ignore'):
            theta = np.where(yi > 0, xBarra.flatten() / yi.flatten(), np.inf)
        if np.all(theta == np.inf):
            if verbose:
                print("Unbounded solution")
            return None
        imin = np.argmin(theta)
        Xb[imin], Xn[imax] = Xn[imax], Xb[imin]
        if verbose:
            print(f"Iteration {it + 1}: Xb = {Xb}, xBarra = {xBarra.flatten()}, zBarra = {zBarra.flatten()}")
    if verbose:
        print("Maximum iterations reached without finding optimal solution")
    return None

def simplex_duas_fases(A, b, c, Xn, Xb, artificiais, verbose=True):
    # Fase 1: Minimizar soma das artificiais
    c_fase1 = np.zeros_like(c)
    for idx in artificiais:
        c_fase1[idx] = 1
    if verbose:
        print("\n--- Fase 1: Removendo variáveis artificiais ---")
    res = simplex_min(A, b, c_fase1, Xn.copy(), Xb.copy(), verbose=verbose)
    if res is None:
        print("Problema inviável na Fase 1.")
        return None
    Xb1, xBarra1, zBarra1 = res
    # Checa se todas artificiais saíram da base
    for idx in artificiais:
        if idx in Xb1 and abs(xBarra1[Xb1.tolist().index(idx)]) > 1e-8:
            print("Problema inviável: variáveis artificiais permanecem na base.")
            return None
    # Remove colunas artificiais de A e c
    A2 = np.delete(A, artificiais, axis=1)
    c2 = np.delete(c, artificiais, axis=0)
    # Atualiza índices das variáveis básicas e não básicas
    Xb2 = np.array([i for i in Xb1 if i not in artificiais])
    Xn2 = np.array([i for i in Xn if i not in artificiais])
    # Corrige índices após remoção
    def corrige_indices(indices, removidos):
        for r in sorted(removidos):
            indices = np.where(indices > r, indices - 1, indices)
        return indices
    Xb2 = corrige_indices(Xb2, artificiais)
    Xn2 = corrige_indices(Xn2, artificiais)
    if verbose:
        print("\n--- Fase 2: Resolvendo problema original ---")
    return simplex_min(A2, b, c2, Xn2, Xb2, verbose=verbose)

def handle_target(c, target):
    if target == 'max':
        return -c
    elif target == 'min':
        return c
    else:
        raise ValueError("Target must be 'max' or 'min'")

def parse(problem, z, target='max'):
    num_constraints, num_vars = problem.shape[0], problem.shape[1] - 2
    tipos = problem[:, -2]
    # Conta quantas de cada tipo
    n_folga = np.sum((tipos == '<='))
    n_excesso = np.sum((tipos == '>=') | (tipos == '='))
    n_artificiais = np.sum((tipos == '>=') | (tipos == '='))
    total_vars = num_vars + n_folga + n_excesso
    total_vars_art = num_vars + n_folga + n_excesso + n_artificiais
    A = np.zeros((num_constraints, total_vars_art))
    b = np.zeros((num_constraints, 1))
    c = np.zeros((total_vars_art, 1))
    artificiais = []
    folga_idx = num_vars
    excesso_idx = folga_idx + n_folga
    art_idx = excesso_idx + n_excesso
    for i in range(num_constraints):
        A[i, :num_vars] = problem[i, :num_vars]
        b[i] = problem[i, -1]
        if problem[i, -2] == '<=':
            A[i, folga_idx] = 1
            folga_idx += 1
        elif problem[i, -2] == '>=':
            A[i, excesso_idx] = -1
            A[i, art_idx] = 1
            artificiais.append(art_idx)
            excesso_idx += 1
            art_idx += 1
        elif problem[i, -2] == '=':
            A[i, art_idx] = 1
            artificiais.append(art_idx)
            art_idx += 1
        else:
            raise ValueError("Inequality must be '<=', '>=', or '='")
    c[:num_vars] = z[:, None]
    c = handle_target(c, target)
    # Variáveis básicas: folga e artificiais
    Xb = []
    folga_idx = num_vars
    excesso_idx = num_vars + n_folga
    art_idx = num_vars + n_folga + n_excesso
    for i in range(num_constraints):
        if problem[i, -2] == '<=':
            Xb.append(folga_idx)
            folga_idx += 1
        elif problem[i, -2] == '>=':
            Xb.append(art_idx)
            excesso_idx += 1
            art_idx += 1
        elif problem[i, -2] == '=':
            Xb.append(art_idx)
            art_idx += 1
    Xb = np.array(Xb)
    Xn = np.array([i for i in range(total_vars_art) if i not in Xb])
    return A, b, c, Xn, Xb, artificiais

def solve(problem, z, target='max', verbose=True):
    A, b, c, Xn, Xb, artificiais = parse(problem, z, target)
    if len(artificiais) == 0:
        return simplex_min(A, b, c, Xn, Xb, verbose=verbose)
    else:
        return simplex_duas_fases(A, b, c, Xn, Xb, artificiais, verbose=verbose)

# Exemplo de uso:
if __name__ == "__main__":
    # Problema com restrições <=, >=, =
    # Max z = 5x1 + 2x2
    # s.a. x1 + x2 <= 4
    #      x1      >= 3
    #           x2 = 2
    target = 'max'
    problem = np.array([
        [1, 1, '<=', 4],
        [1, 0, '>=', 3],
        [0, 1, '=', 2]
    ], dtype=object)
    z = np.array([5, 2])
    solve(problem, z, target, verbose=True)

def simplex_min(A, b, c, Xn, Xb):
    max_iter = 10

    for _ in range(max_iter):
        B = A[:, Xb]
        B_inv = linalg.inv(B)
        cb = c[Xb]
        cn = c[Xn]

        xBarra = B_inv @ b
        zBarra = cb.T @ xBarra

        zj_cj = cb.T @ B_inv @ A[:, Xn] - cn.T
        zj_cj = zj_cj.flatten()
        imax = np.argmax(zj_cj)
        if zj_cj[imax] <= 0:
            print("\n***Optimal solution found!***")
            print(f"Optimal value: {zBarra.flatten()[0]}")
            print(f"Basic variables: {Xb}, Non-basic variables: {Xn}")
            print(f"Solution: {xBarra.flatten()}")
            return Xb, xBarra, zBarra
        
        yi = B_inv @ A[:, Xn[imax]]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            theta = np.where(yi > 0, xBarra.flatten() / yi.flatten(), np.inf)
        
        if np.all(theta == np.inf):
            print("Unbounded solution")
            return None
        
        imin = np.argmin(theta)

        Xb[imin], Xn[imax] = Xn[imax], Xb[imin]
        print(f"Iteration {_ + 1}: Xb = {Xb}, xBarra = {xBarra.flatten()}, zBarra = {zBarra.flatten()}")
        


    print("Maximum iterations reached without finding optimal solution")
    
    return None

def handle_target(c, target):
    if target == 'max':
        return -c
    elif target == 'min':
        return c
    else:
        raise ValueError("Target must be 'max' or 'min'")

def parse(problem, z, target='max'):
    num_constraints, num_vars = problem.shape[0], problem.shape[1] - 2

    A = np.zeros((num_constraints, num_vars + num_constraints))
    b = np.zeros((num_constraints, 1))
    c = np.zeros((num_vars + num_constraints, 1))

    for i in range(num_constraints):
        A[i, :num_vars] = problem[i, :num_vars]
        b[i] = problem[i, -1]
        if problem[i, -2] == '<=':
            A[i, num_vars + i] = 1
        elif problem[i, -2] == '>=':
            A[i, num_vars + i] = -1
        else:
            raise ValueError("Inequality must be '<=' or '>='")

    c[:num_vars] = z[:, None]
    c = handle_target(c, target)


    Xb = np.array(range(num_vars, num_vars + num_constraints))
    Xn = np.array(range(num_vars))

    return A, b, c, Xn, Xb

def simplex(z, target, constraints):
    A, b, c, Xn, Xb = parse(constraints, z, target)
    
    two_phase_needed = np.any(constraints[:, -2] == '>=') or np.any(b < 0)
    if two_phase_needed:
        print("Two-phase method needed, but not implemented.")
        return None
    

    return simplex_min(A, b, c, Xn, Xb)

target = 'max'
constraints = np.array([
    [1, 1, '<=', 4],
    [1, 0, '<=', 3],
    [0, 1, '<=', 2]
])
z = np.array([5, 2])

simplex(z, target, constraints)

