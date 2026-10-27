import numpy as np
import scipy.sparse as sp
from models.als import IALS


def _csr(users, items, n_users, n_items, data):
    # собираем разреженную матрицу из троек (пользователь, объект, значение)
    return sp.csr_matrix((np.asarray(data, dtype=np.float32), (users, items)), shape=(n_users, n_items))


def build_custom_signal(users, items, ratings, ts, n_users, n_items, t_max,
                        alpha_pos=20.0, alpha_neg=10.0, tau_days=365.0, gamma=0.5):
    """Собственный целевой сигнал по ВСЕМ отзывам обучающего периода.
    r>=4 (позитив): p = g(r) * (gamma + (1-gamma)*exp(-dt/tau)),  g(4)=0.5, g(5)=1
                    c = 1 + alpha_pos * (r-3)/2 * exp(-dt/tau)
    r<=3 (негатив): p = 0,  c = 1 + alpha_neg * (4-r)/3   (чем ниже оценка, тем увереннее негатив)
    Зачем: 5 звёзд сильнее 4, свежее важнее старого, а плохая оценка — явный минус, а не «не знаю»."""
    days = (t_max - ts) / 86400000.0                         # возраст отзыва в днях (timestamp в миллисекундах)
    decay = np.exp(-days / tau_days)                         # 1 для свежих, -> 0 для старых
    pos = ratings >= 4                                       # маска положительных отзывов
    g = np.where(ratings >= 5, 1.0, 0.5)                     # сила одобрения: 5 звёзд = 1, 4 звезды = 0.5
    p = np.where(pos, g * (gamma + (1 - gamma) * decay), 0.0)   # предпочтение; gamma — «пол» затухания
    c = np.where(pos, 1 + alpha_pos * (ratings - 3) / 2 * decay,  # уверенность позитива
                 1 + alpha_neg * (4 - ratings) / 3)               # уверенность негатива
    return _csr(users, items, n_users, n_items, c), _csr(users, items, n_users, n_items, p)


class CustomIALS(IALS):
    """Тот же ALS-решатель, но с другими матрицами C и P (то есть другим обучающим таргетом)."""

    def fit_signal(self, users, items, ratings, ts, n_users, n_items, t_max, **kw):
        C, P = build_custom_signal(users, items, ratings, ts, n_users, n_items, t_max, **kw)
        return self.fit_cp(C, P)
