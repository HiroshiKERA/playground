'''
Target problem:
    min_{u, v}  || A*(uv') - B ||, 
where, 
    - All elements in the matries A and B are posiotive
    - u, v are column vectors
    - '*' denotes the Hadamard product (element-wise product)
    - '|| ||' denotes the Frobenius norm
    - v' denots the transposed v

Motivation:
    http://math.stackexchange.com/questions/2157191/approximating-a-given-matrix-with-a-rank-1-matrix-hadamard-product-with-another

'''

import numpy as np
import theano as th
import theano.tensor as T

## proposed method by greg in the link above. 
def hd(A, B):
    u, d, vt = np.linalg.svd(B / A)
    u = u[:, :1]
    v = vt[0].reshape(-1,1)
    d = d[0]
    
    r = np.trace((u@v.T * A).T @ B) / np.trace((u@v.T*A).T @ (u@v.T*A))
    
    cost = lambda u_, v_: np.linalg.norm(u_ @ v_.T * A - B)**2
    res = cost(r*u, v)

    return r*u, v, res, cost

## numerical approach by me
def hd_gd(A, B, max_iter=1000, tol=10**(-6), r=0.01):
    n, m = A.shape
    
    u, v = T.dmatrices(['u', 'v'])
    uv = T.dot(u, v.T)
    s = T.sum((B-A*uv)**2)
    gu = th.function([u, v], T.grad(s, u))
    gv = th.function([u, v], T.grad(s, v))
    cost = th.function([u, v], s)
    
    u0 = np.random.randn(n, 1)
    v0 = np.random.randn(m, 1)
    
    ut, vt = u0, v0
    res = 100.
    for i in range(max_iter):
        ut -= r*gu(ut, vt)
        vt -= r*gv(ut, vt)
        
        tmp_res = cost(ut, vt)
        if np.abs(res-tmp_res) < tol:
            res = tmp_res
            break
        
        res = tmp_res

    return ut, vt, res, cost
    

## Compare the results of hd() and hd_gd() at the counterexample shown in the link.
def main1():
    print('Comparison in the counterexample')
    A = np.array([[2,1],[1,2.]])
    B = np.ones((2,2))

    u1, v1, res1, cost1 = hd(A, B)
    u2, v2, res2, cost2 = hd_gd(A, B)

    print(cost1(u1, v1))
    print(cost2(u2, v2))
    print('residual of hd = ', res1)
    print('residual of hd_gde = ', res2)

    print('')

## Varidate if cost function defined in hd and hd_gd is the same
def _main():
    A = np.array([[2,1],[1,2.]])
    B = np.ones((2,2))

    u1, v1, res1, cost1 = hd(A, B)
    u2, v2, res2, cost2 = hd_gd(A, B)
    
    n_ok = 0
    n_trial = 100
    for i in range(n_trial):
        u = np.random.randn(2, 1)
        v = np.random.randn(2, 1)
        
        n_ok += 1 if np.abs(cost1(u, v) - cost2(u, v)) < 10**(-3) else 0
    
    print('succeed %d in %d trials' % (n_ok, n_trial))
    print('')

## Compare the results of hd() and hd_gd() at many random cases.
# if hd() provides optimal, than res1 is always smallter than res2
def main2():
    print('Comparison in the random cases')

    n_optimal = 0
    n_trial = 10
    for i in range(n_trial):
        A = np.abs(np.random.randn(2, 2)) + 0.1  # be positive
        B = np.abs(np.random.randn(2, 2)) + 0.3  # be positive

        _, _, res1, _ = hd(A, B)
        _, _, res2, _ = hd_gd(A, B)
    
        # res1-res2 are expected to be negative or almost the same as 0 if res1 is optimal
        n_optimal += 1 if res1 - res2 < 10**(-5) else 0

    print('succeed %d in %d trial' % (n_optimal, n_trial))
    print('')

if __name__ == '__main__':
    main1()
    main2()
    # _main()
