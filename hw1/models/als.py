import numpy as np


class IALS:

    def __init__(self, factors=64, lam=0.1, iters=10, seed=42):
        self.f = int(factors)   
        self.lam = lam          
        self.iters = iters       
        self.seed = seed     

    def _half(self, C, P, Y):
        YtY = Y.T @ Y                                       
        reg = self.lam * np.eye(self.f, dtype=np.float32)   
        out = np.zeros((C.shape[0], self.f), dtype=np.float32)
        for u in range(C.shape[0]):                         
            a, b = C.indptr[u], C.indptr[u + 1]            
            if a == b:                                      
                continue
            Yu = Y[C.indices[a:b]]                         
            c = C.data[a:b]                 
            p = P.data[a:b]
            A = YtY + (Yu.T * (c - 1)) @ Yu + reg
            out[u] = np.linalg.solve(A, Yu.T @ (c * p))   
        return out

    def fit_cp(self, C, P):
        C = C.tocsr(); P = P.tocsr()
        C.sort_indices(); P.sort_indices()                 
        rng = np.random.default_rng(self.seed)      
        self.Y = rng.normal(0, 0.01, (C.shape[1], self.f)).astype(np.float32) 
        Ct, Pt = C.T.tocsr(), P.T.tocsr()        
        Ct.sort_indices(); Pt.sort_indices()
        for _ in range(self.iters):
            self.X = self._half(C, P, self.Y)          
            self.Y = self._half(Ct, Pt, self.X)     
        return self

    def fit(self, X, alpha=20.0):
        C = X.tocsr().astype(np.float32).copy()
        P = C.copy()
        C.data[:] = 1.0 + alpha                         
        P.data[:] = 1.0                      
        return self.fit_cp(C, P)

    def score(self, users):
        return self.X[users] @ self.Y.T     
