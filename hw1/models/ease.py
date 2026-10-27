import numpy as np


class EASE:
    """EASE (Embarrassingly Shallow Autoencoder).
    Учим линейную модель X ≈ X·B, где B — матрица объект x объект с нулевой диагональю.
    Ограничение diag(B)=0 запрещает объекту «предсказывать» самого себя.
    Решение закрытое (без итераций): B = I - P / diag(P), P = (X^T X + lam I)^-1."""

    def __init__(self, lam=500.0):
        self.lam = lam           # сила L2-регуляризации; больше lam -> проще модель

    def fit(self, X):
        self.X = X.tocsr().astype(np.float32)               # сохраняем историю для предсказания
        G = (self.X.T @ self.X).toarray().astype(np.float32)  # матрица Грама X^T X (совстречаемость объектов)
        G[np.diag_indices_from(G)] += self.lam              # G + lam*I: регуляризация и обратимость
        P = np.linalg.inv(G)                                # P = (X^T X + lam I)^-1
        B = -P / np.diag(P)[None, :]                        # делим каждый столбец j на P_jj и меняем знак
        np.fill_diagonal(B, 0.0)                            # диагональ B = 0 (условие модели)
        self.B = B.astype(np.float32)
        return self

    def score(self, users):
        return self.X[users] @ self.B                       # предсказание = история пользователя, умноженная на B
