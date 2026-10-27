import numpy as np                     
import scipy.sparse as sp              


class ItemKNN:
    def __init__(self, k=50, shrink=10.0, alpha=0.5):
        self.k = int(k)        
        self.shrink = shrink    
        self.alpha = alpha     

    def fit(self, X):
        X = X.tocsr().astype(np.float32)           
        self.X = X                                  
        norms = np.asarray(X.sum(0)).ravel() + 1e-9 
        co = (X.T @ X).toarray()                    
       
        S = co / ((norms[:, None] ** self.alpha) * (norms[None, :] ** (1 - self.alpha)) + self.shrink)
        np.fill_diagonal(S, 0)                     
        if self.k < S.shape[0]:
           
            idx = np.argpartition(-S, self.k, axis=1)[:, self.k:]
            np.put_along_axis(S, idx, 0, axis=1)    
        self.S = sp.csr_matrix(S)                 
        return self

    def score(self, users):
        return (self.X[users] @ self.S.T).toarray()
