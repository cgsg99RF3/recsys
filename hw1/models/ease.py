import numpy as np


class EASE:

    def __init__(self, lam=500.0):
        self.lam = lam         

    def fit(self, X):
        self.X = X.tocsr().astype(np.float32)            
        G = (self.X.T @ self.X).toarray().astype(np.float32)  
        G[np.diag_indices_from(G)] += self.lam            
        P = np.linalg.inv(G)                             
        B = -P / np.diag(P)[None, :]                    
        np.fill_diagonal(B, 0.0)                      
        self.B = B.astype(np.float32)
        return self

    def score(self, users):
        return self.X[users] @ self.B               
