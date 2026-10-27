import numpy as np


class IALS:
    """Implicit ALS (Hu, Koren, Volinsky).
    Минимизируем sum_{u,i} c_ui * (p_ui - x_u·y_i)^2 + lam*(|X|^2 + |Y|^2).
    p_ui — предпочтение (1 если было взаимодействие), c_ui — уверенность (насколько верим этому p).
    Для всех ненаблюдаемых пар p=0, c=1. Фиксируя Y, для каждого пользователя получаем
    линейную задачу наименьших квадратов (и наоборот) — поэтому «чередующиеся» шаги."""

    def __init__(self, factors=64, lam=0.1, iters=10, seed=42):
        self.f = int(factors)    # размерность латентного пространства
        self.lam = lam           # L2-регуляризация
        self.iters = iters       # число полных проходов (обновить X, затем Y)
        self.seed = seed         # для воспроизводимой инициализации

    def _half(self, C, P, Y):
        """Обновить факторы строк матрицы C при зафиксированных факторах Y."""
        YtY = Y.T @ Y                                        # общий для всех пользователей член (вклад c=1 для всех пар)
        reg = self.lam * np.eye(self.f, dtype=np.float32)    # lam*I
        out = np.zeros((C.shape[0], self.f), dtype=np.float32)
        for u in range(C.shape[0]):                          # по каждому пользователю (или объекту) отдельно
            a, b = C.indptr[u], C.indptr[u + 1]              # границы наблюдаемых элементов строки в CSR
            if a == b:                                       # нет наблюдений -> фактор остаётся нулевым
                continue
            Yu = Y[C.indices[a:b]]                           # факторы объектов, с которыми есть взаимодействие
            c = C.data[a:b]                                  # уверенности c_ui этих пар
            p = P.data[a:b]                                  # предпочтения p_ui этих пар
            # A = Y^T Y + sum_obs (c-1) y y^T + lam I  — экономия: считаем только по наблюдаемым
            A = YtY + (Yu.T * (c - 1)) @ Yu + reg
            # правая часть: sum_obs c*p*y (ненаблюдаемые дают 0, так как p=0)
            out[u] = np.linalg.solve(A, Yu.T @ (c * p))      # решаем A x = rhs (быстрее и стабильнее, чем inv)
        return out

    def fit_cp(self, C, P):
        """Общий вариант: на вход матрицы уверенности C и предпочтения P одинаковой структуры."""
        C = C.tocsr(); P = P.tocsr()
        C.sort_indices(); P.sort_indices()                   # одинаковый порядок индексов, чтобы data[] совпадали
        rng = np.random.default_rng(self.seed)               # генератор случайных чисел с фиксированным seed
        self.Y = rng.normal(0, 0.01, (C.shape[1], self.f)).astype(np.float32)  # малая случайная инициализация
        Ct, Pt = C.T.tocsr(), P.T.tocsr()                    # транспонированные: строки = объекты
        Ct.sort_indices(); Pt.sort_indices()
        for _ in range(self.iters):
            self.X = self._half(C, P, self.Y)                # фиксируем Y -> обновляем X
            self.Y = self._half(Ct, Pt, self.X)              # фиксируем X -> обновляем Y
        return self

    def fit(self, X, alpha=20.0):
        """Базовый iALS: на положительных парах p=1, c=1+alpha."""
        C = X.tocsr().astype(np.float32).copy()
        P = C.copy()
        C.data[:] = 1.0 + alpha                              # уверенность растёт с alpha
        P.data[:] = 1.0                                      # предпочтение бинарное
        return self.fit_cp(C, P)

    def score(self, users):
        return self.X[users] @ self.Y.T                      # оценка = скалярное произведение факторов
