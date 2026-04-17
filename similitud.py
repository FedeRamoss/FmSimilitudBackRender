"""
FM DataLab v3 - Motor de Similitud V3
=======================================
Comparador percentil/categórico con métricas híbridas.
Sin dependencias de Streamlit — puro numpy/scipy.
Testeable de forma aislada con pytest.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean


# ── Umbrales por percentil ───────────────────────────────────────────────────

@dataclass
class PercentileThresholds:
    p10: float; p25: float; p40: float; p50: float
    p60: float; p75: float; p90: float
    min: float; max: float; mean: float; std: float


# ── Comparador principal ─────────────────────────────────────────────────────

class SimilitudComparatorV3:

    def __init__(self):
        self.percentile_thresholds: dict = {}
        self.stats: list = []
        self.fitted = False

    def fit(self, df_pool: pd.DataFrame, stats: list):
        """
        Ajusta el comparador al pool de jugadores.
        Calcula y almacena los umbrales de percentil por stat.
        """
        self.stats = stats
        self.percentile_thresholds = {}

        for stat in stats:
            if stat not in df_pool.columns:
                continue
            col = df_pool[stat].dropna()
            if len(col) == 0:
                continue
            self.percentile_thresholds[stat] = PercentileThresholds(
                p10=np.percentile(col, 10), p25=np.percentile(col, 25),
                p40=np.percentile(col, 40), p50=np.percentile(col, 50),
                p60=np.percentile(col, 60), p75=np.percentile(col, 75),
                p90=np.percentile(col, 90), min=float(np.min(col)),
                max=float(np.max(col)),     mean=float(np.mean(col)),
                std=float(np.std(col)),
            )

        self.fitted = True

    # ── Categorización ────────────────────────────────────────────────────────

    def _cat(self, value, stat) -> int:
        """Categoriza un valor en 7 niveles (-3 a 3) según percentiles."""
        if pd.isna(value) or stat not in self.percentile_thresholds:
            return 0
        t = self.percentile_thresholds[stat]
        if   value <= t.p10: return -3
        elif value <= t.p25: return -2
        elif value <= t.p40: return -1
        elif value <= t.p60: return  0
        elif value <= t.p75: return  1
        elif value <= t.p90: return  2
        else:                return  3

    def categorize_player(self, player_stats: dict) -> np.ndarray:
        return np.array([self._cat(player_stats.get(s, np.nan), s) for s in self.stats])

    def categorize_dataframe(self, df_stats: pd.DataFrame) -> np.ndarray:
        result = np.zeros((len(df_stats), len(self.stats)))
        for i, stat in enumerate(self.stats):
            if stat in df_stats.columns:
                result[:, i] = df_stats[stat].apply(lambda x: self._cat(x, stat))
        return result

    # ── Normalización z-score ─────────────────────────────────────────────────

    def normalize_player(self, player_stats: dict) -> np.ndarray:
        out = []
        for stat in self.stats:
            v = player_stats.get(stat, np.nan)
            if pd.isna(v) or stat not in self.percentile_thresholds:
                out.append(0.0)
                continue
            t = self.percentile_thresholds[stat]
            out.append(float(np.clip((v - t.mean) / t.std, -3, 3)) if t.std != 0 else 0.0)
        return np.array(out)

    def normalize_dataframe(self, df_stats: pd.DataFrame) -> np.ndarray:
        result = np.zeros((len(df_stats), len(self.stats)))
        for i, stat in enumerate(self.stats):
            if stat not in df_stats.columns or stat not in self.percentile_thresholds:
                continue
            t = self.percentile_thresholds[stat]
            if t.std == 0:
                continue
            result[:, i] = np.clip(
                (df_stats[stat].fillna(t.mean) - t.mean) / t.std, -3, 3
            )
        return result

    # ── Métricas individuales ─────────────────────────────────────────────────

    def mae_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return 1 - np.mean(np.abs(a - b)) / 3

    def euc_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        d = euclidean(a, b)
        return 1 - d / np.sqrt(len(a) * 36)

    def pear_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.5
        try:
            c, _ = pearsonr(a, b)
            return (c + 1) / 2
        except Exception:
            return 0.5

    def ord_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return 1 - np.mean(np.abs(a - b)) / 6

    def hybrid(self, cat1: np.ndarray, cat2: np.ndarray,
               norm1: np.ndarray, norm2: np.ndarray) -> float:
        """Score híbrido: 40% MAE + 25% Euclidiana + 20% Pearson + 15% Ordinal."""
        return (
            0.40 * self.mae_sim(norm1, norm2) +
            0.25 * self.euc_sim(norm1, norm2) +
            0.20 * self.pear_sim(norm1, norm2) +
            0.15 * self.ord_sim(cat1, cat2)
        )


# ── Funciones de alto nivel ──────────────────────────────────────────────────

def compute_similarity_v3(
    comparator: SimilitudComparatorV3,
    cat_pool: np.ndarray, cat_query: np.ndarray,
    norm_pool: np.ndarray, norm_query: np.ndarray,
) -> tuple:
    """
    Calcula todas las métricas de similitud para cada jugador del pool
    contra el jugador de consulta.
    Retorna (mae_s, euc_s, pear_s, ord_s, hyb_s) como arrays numpy.
    """
    n = cat_pool.shape[0]
    mae_s  = np.zeros(n)
    euc_s  = np.zeros(n)
    pear_s = np.zeros(n)
    ord_s  = np.zeros(n)
    hyb_s  = np.zeros(n)

    for i in range(n):
        mae_s[i]  = comparator.mae_sim(norm_pool[i],  norm_query)
        euc_s[i]  = comparator.euc_sim(norm_pool[i],  norm_query)
        pear_s[i] = comparator.pear_sim(norm_pool[i], norm_query)
        ord_s[i]  = comparator.ord_sim(cat_pool[i],   cat_query)
        hyb_s[i]  = comparator.hybrid(cat_pool[i], cat_query, norm_pool[i], norm_query)

    return mae_s, euc_s, pear_s, ord_s, hyb_s


def ranking_jugadores(
    df_pool: pd.DataFrame,
    mae_s: np.ndarray, euc_s: np.ndarray,
    pear_s: np.ndarray, ord_s: np.ndarray, hyb_s: np.ndarray,
    query_nombre: str,
) -> pd.DataFrame:
    """
    Construye el DataFrame de ranking ordenado por similitud híbrida.
    Excluye al propio jugador de consulta del resultado.
    """
    result = df_pool.copy()
    result["sim_mae"]       = (mae_s  * 100).round(1)
    result["sim_euclidiana"]= (euc_s  * 100).round(1)
    result["sim_pearson"]   = (pear_s * 100).round(1)
    result["sim_ordinal"]   = (ord_s  * 100).round(1)
    result["similitud"]     = (hyb_s  * 100).round(1)

    result = result[result["jugador"].str.lower() != query_nombre.lower()]
    return result.sort_values("similitud", ascending=False).reset_index(drop=True)