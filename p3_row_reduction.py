import numpy as np

rng = np.random.default_rng(0)
def rref(A, tol=1e-10):
    R = np.array(A, dtype=float) 
    r = 0
    m, n = R.shape
    pivots = []
    for c in range(n):
        if r == m: 
            break
        p = r + np.argmax(np.abs(R[r:, c]))
        if abs(R[p, c])< tol:
            continue
        else:
            R[[r, p]] = R[[p, r]]
        R[r] = R[r]/ R[r, c]
        for i in range(m):
            if i == r:
                continue 
            R[i] = R[i] - R[i, c] * R[r]
        pivots.append(c)
        r+=1
    return R, pivots
def rank(A):
    R, pivots = rref(A, tol = 1e-10)
    return len(pivots)
def col_space_basis(A, tol = 1e-10):
    A = np.array(A, dtype=float)  
    R, pivots = rref(A, tol=tol)           
    basis = []                     
    for j in pivots:               
        basis.append(A[:,j ])       
    return basis
def null_space_basis(A, tol=1e-10):
    R, pivots = rref(A, tol=tol)
    n = R.shape[1]                       
    free = [j for j in range(n) if j not in pivots]   
    basis = []
    for f in free:                       
        v = np.zeros(n)                 
        v[f] = 1                  
        for row, p in enumerate(pivots):
            v[p] = -R[row, f]
        basis.append(v)
    return basis

if __name__ == "__main__":
    A = np.array([[1, 2, -1], [3, -1, 4]])
    R, pivots = rref(A)
    print("RREF of A:\n", R, "\npivots:", pivots)
    assert np.allclose(R, [[1, 0, 1], [0, 1, -1]])
    assert pivots == [0, 1]

    F = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    R, pivots = rref(F)
    print("RREF of F:\n", R, "\npivots:", pivots)
    assert np.allclose(R, [[1, 0, -1], [0, 1, 2], [0, 0, 0]])
    assert pivots == [0, 1]
    print("null space basis of F:", null_space_basis(F))
    print("column space basis of F:", col_space_basis(F))

    for _ in range(20):
        A = rng.normal(size=(5, 7))
        N = null_space_basis(A)
        for v in N:
            assert np.allclose(A @ v, 0)              
        assert rank(A) + len(N) == 7                  
        assert rank(A) == np.linalg.matrix_rank(A)    
    print("20 random 5x7 matrices: OK")

    A = rng.normal(size=(6, 4)) @ rng.normal(size=(4, 8))
    N = null_space_basis(A)
    print("6x8 matrix: rank =", rank(A), ", dim of null space =", len(N))
    assert rank(A) == 4
    assert len(N) == 4
    for v in N:
        assert np.allclose(A @ v, 0)

    print("All tests passed.")
    
