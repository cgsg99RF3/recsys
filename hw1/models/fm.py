import numpy as np
import pandas as pd
import torch                             # тензоры и автоград (автоматические градиенты) — разрешены заданием
import torch.nn as nn


def _codes(s, top=None):
    """Категориальный столбец -> целые коды 1..K (0 зарезервирован под padding)."""
    s = s.astype(object).where(s.notna(), "NA").astype(str)      # пропуски -> отдельная категория "NA"
    if top:                                                      # оставляем только top самых частых значений
        keep = set(s.value_counts().index[:top])
        s = s.where(s.isin(keep), "OTHER")                       # редкие -> "OTHER" (борьба с разреженностью)
    return pd.factorize(s)[0] + 1                                # +1: код 0 — это padding


def build_fm_features(meta, tr_u, tr_i, n_users, n_items):
    """Строит таблицы признаков пользователей (UF) и объектов (IF).
    Только то, что известно в момент рекомендации: метаданные каталога + статистика по train.
    Каждая строка — список индексов активных (=1) признаков; индексы глобально уникальны; 0 = padding."""
    cols, sizes = [], []                                         # столбцы признаков и размеры их словарей

    def add(codes):                                              # зарегистрировать очередной признак
        codes = np.asarray(codes)
        cols.append(codes)
        sizes.append(int(codes.max()) + 1)

    add(np.arange(n_items) + 1)                                  # [0] ID объекта
    add(_codes(meta["main_category"], 50))                       # [1] главная категория
    add(_codes(meta["store"], 500))                              # [2] магазин / издатель (топ-500)
    price = pd.to_numeric(meta["price"], errors="coerce")        # цена как число (мусор -> NaN)
    q = pd.qcut(price.rank(method="first"), 10, labels=False, duplicates="drop")   # децили цены
    pb = q.fillna(-1).astype(int).values + 2                     # [3] бин цены; пропуск -> 1
    add(pb)
    pop = np.bincount(tr_i, minlength=n_items)                   # популярность объекта В TRAIN (без утечки)
    add(np.minimum(np.log2(pop + 1).astype(int), 15) + 1)        # [4] лог-бин популярности
    cats = meta["categories"].apply(                             # первые 3 уровня иерархии категорий
        lambda c: list(c)[:3] if isinstance(c, (list, np.ndarray)) else [])
    vc = pd.Series([t for l in cats for t in l]).value_counts().index[:300]   # 300 самых частых токенов
    cmap = {t: j + 1 for j, t in enumerate(vc)}
    for j in range(3):                                           # [5,6,7] токены категорий (0 = нет токена)
        add(np.array([cmap.get(l[j], 0) if len(l) > j else 0 for l in cats]) + 1)
    item_cols = len(cols)                                        # сколько столбцов относятся к объекту

    add(np.arange(n_users) + 1)                                  # ID пользователя
    act = np.bincount(tr_u, minlength=n_users)                   # активность пользователя в train
    add(np.minimum(np.log2(act + 1).astype(int), 12) + 1)        # лог-бин активности
    d = pd.DataFrame({"u": tr_u, "cat": cols[1][tr_i], "pb": pb[tr_i]})
    for name in ["cat", "pb"]:                                   # любимая категория и типичный ценовой бин
        m = d.groupby("u")[name].agg(lambda x: x.value_counts().index[0])   # мода по истории пользователя
        v = np.zeros(n_users, int)
        v[m.index.values] = m.values
        add(v + 1)

    offs = np.concatenate([[1], 1 + np.cumsum(sizes)[:-1]])      # сдвиги, чтобы индексы разных признаков не пересекались
    shifted = [c + o - 1 for c, o in zip(cols, offs)]            # глобальные индексы каждого признака
    IFg = np.stack(shifted[:item_cols], 1)                       # признаки объектов: (n_items, число_признаков)
    UFg = np.stack(shifted[item_cols:], 1)                       # признаки пользователей: (n_users, число_признаков)
    for j in (5, 6, 7):                                          # «нет токена категории» -> padding 0
        IFg[:, j] = np.where(cols[j] == 1, 0, IFg[:, j])
    return (torch.tensor(UFg, dtype=torch.long),
            torch.tensor(IFg, dtype=torch.long),
            int(1 + sum(sizes)))                                 # размер общего словаря признаков

class FM(nn.Module):
    """Factorization Machine: y(x) = w0 + sum_i w_i x_i + sum_{i<j} <v_i, v_j> x_i x_j.
    Признаки бинарные, поэтому x_i ∈ {0,1}: суммируем только по активным признакам.
    Быстрая формула парных взаимодействий: 0.5 * (|sum v|^2 - sum |v|^2).
    Обучение — BPR: хотим, чтобы y(u, положительный) > y(u, случайный объект)."""

    def __init__(self, n_feat, dim=32, l2=1e-5, lr=1e-2, epochs=15, bs=8192, neg=1, seed=42, device=None):
        super().__init__()
        self.dim, self.l2, self.lr = int(dim), l2, lr             # размерность v, L2, шаг оптимизатора
        self.epochs, self.bs, self.neg, self.seed = int(epochs), bs, neg, seed
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")   # GPU если есть
        torch.manual_seed(seed)                                   # воспроизводимая инициализация
        self.emb = nn.Embedding(n_feat, self.dim, padding_idx=0)  # векторы v_i; padding_idx=0 не обучается
        self.lin = nn.Embedding(n_feat, 1, padding_idx=0)         # линейные веса w_i
        nn.init.normal_(self.emb.weight, 0, 0.01)                 # малая инициализация v
        nn.init.zeros_(self.lin.weight)                           # w = 0
        with torch.no_grad():
            self.emb.weight[0] = 0                                # padding-вектор строго нулевой
        self.to(self.dev)

    def _side(self, F):
        """Вклад одной «стороны» (пользователя или объекта). F: (B, число_признаков) индексы."""
        E = self.emb(F)                                           # (B, F, dim) векторы активных признаков
        S = E.sum(1)                                              # (B, dim) сумма векторов стороны
        # линейная часть + парные взаимодействия ВНУТРИ стороны
        return S, self.lin(F).sum((1, 2)) + 0.5 * ((S ** 2).sum(-1) - (E ** 2).sum((1, 2)))

    def pair(self, u, i):
        """y(u,i) без константы w0 и без постоянного для пользователя слагаемого — для обучения и ранжирования."""
        Su, cu = self._side(self.UF[u])                           # сторона пользователя
        Si, ci = self._side(self.IF[i])                           # сторона объекта
        return (Su * Si).sum(-1) + cu + ci                        # взаимодействия между сторонами + внутренние части

    def fit(self, X, UF, IF, verbose=False):
        self.UF, self.IF = UF.to(self.dev), IF.to(self.dev)       # таблицы признаков на устройство
        co = X.tocoo()                                            # список положительных пар (u, i)
        U = torch.tensor(co.row, dtype=torch.long, device=self.dev)
        I = torch.tensor(co.col, dtype=torch.long, device=self.dev)
        n_items = IF.shape[0]
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)     # оптимизатор (разрешён)
        g = torch.Generator(device=self.dev); g.manual_seed(self.seed)   # детерминированный сэмплинг
        for ep in range(self.epochs):
            perm = torch.randperm(len(U), device=self.dev, generator=g)  # перемешиваем пары каждую эпоху
            tot = 0
            for s in range(0, len(U), self.bs):                   # мини-батчи
                b = perm[s:s + self.bs]
                u, ip = U[b], I[b]                                # пользователь и его положительный объект
                for _ in range(self.neg):
                    ineg = torch.randint(0, n_items, ip.shape, device=self.dev, generator=g)  # случайный негатив
                    x = self.pair(u, ip) - self.pair(u, ineg)     # разность оценок позитив - негатив
                    reg = self.l2 * (self.emb(self.UF[u]).pow(2).sum()
                                     + self.emb(self.IF[ip]).pow(2).sum()
                                     + self.emb(self.IF[ineg]).pow(2).sum()) / len(b)   # L2 на задействованные векторы
                    loss = -torch.nn.functional.logsigmoid(x).mean() + reg   # BPR: -log σ(x) + L2
                    opt.zero_grad()                               # обнуляем старые градиенты
                    loss.backward()                               # автоград считает градиенты
                    opt.step()                                    # шаг Adam
                    tot += loss.item()
            if verbose:
                print(ep, tot)
        with torch.no_grad():
            self.Si, self.ci = self._side(self.IF)                # кэш стороны объектов для быстрого score
        return self

    @torch.no_grad()
    def score(self, users):
        Su, _ = self._side(self.UF[torch.as_tensor(users, device=self.dev)])   # сторона пользователей
        # слагаемые, зависящие только от пользователя, не меняют порядок объектов -> опущены
        return (Su @ self.Si.T + self.ci[None]).cpu().numpy()     # (польз. x объекты)
