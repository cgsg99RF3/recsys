import numpy as np
import scipy.sparse as sp
from models.als import IALS


def _csr(users, items, n_users, n_items, data):
    return sp.csr_matrix((np.asarray(data, dtype=np.float32), (users, items)), shape=(n_users, n_items))


def build_custom_signal(users, items, ratings, ts, n_users, n_items, t_max,
                        alpha_pos=20.0, alpha_neg=10.0, tau_days=365.0, gamma=0.5):
    days = (t_max - ts) / 86400000.0                        
    decay = np.exp(-days / tau_days)                      
    pos = ratings >= 4                                 
    g = np.where(ratings >= 5, 1.0, 0.5)           
    p = np.where(pos, g * (gamma + (1 - gamma) * decay), 0.0) 
    c = np.where(pos, 1 + alpha_pos * (ratings - 3) / 2 * decay,
                 1 + alpha_neg * (4 - ratings) / 3) 
    return _csr(users, items, n_users, n_items, c), _csr(users, items, n_users, n_items, p)


class CustomIALS(IALS):

    def fit_signal(self, users, items, ratings, ts, n_users, n_items, t_max, **kw):
        C, P = build_custom_signal(users, items, ratings, ts, n_users, n_items, t_max, **kw)
        return self.fit_cp(C, P)
