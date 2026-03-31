from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy.stats import kendalltau

app = FastAPI(title="FM Similarity API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODELOS
# =========================

class SimilitudRequest(BaseModel):
    pool:     list[dict]
    query:    dict
    stats:    list[str]
    pesos:    dict[str, float]
    posicion: str | None = None
    pct_min:  float | None = None

class ResultadoJugador(BaseModel):
    jugador:            str
    posicion:           str
    similitudCombinada: float
    similitudOrdinal:   float
    similitudKendall:   float
    statsResultado:     dict[str, float]

class RankingRequest(BaseModel):
    pool:            list[dict]
    stats_positivas: list[str]
    stats_negativas: list[str]
    posicion:        str | None = None
    min_minutos:     int | None = None
    max_minutos:     int | None = None

class ResultadoRanking(BaseModel):
    jugador:  str
    posicion: str
    minutos:  float | None
    puntaje:  float
    stats:    dict[str, float]

# =========================
# COMPARADOR CATEGÓRICO
# =========================

@dataclass
class PercentileThresholds:
    p20: float
    p40: float
    p60: float
    p80: float


class SimilitudComparator:
    def __init__(self):
        self.percentile_thresholds: dict[str, PercentileThresholds] = {}
        self.stats: list[str] = []

    def fit(self, df_pool: pd.DataFrame, stats: list[str]):
        self.stats = stats
        self.percentile_thresholds = {}
        for stat in stats:
            if stat not in df_pool.columns:
                continue
            col = pd.to_numeric(df_pool[stat], errors='coerce').dropna()
            if len(col) == 0:
                continue
            self.percentile_thresholds[stat] = PercentileThresholds(
                p20=float(np.percentile(col, 20)),
                p40=float(np.percentile(col, 40)),
                p60=float(np.percentile(col, 60)),
                p80=float(np.percentile(col, 80)),
            )

    def _categorize(self, value, stat: str) -> int:
        # nan → 0 (categoría media), igual que el CLI
        if pd.isna(value) or stat not in self.percentile_thresholds:
            return 0
        value = float(value)
        t = self.percentile_thresholds[stat]
        if   value <= t.p20: return -2
        elif value <= t.p40: return -1
        elif value <= t.p60: return  0
        elif value <= t.p80: return  1
        else:                return  2

    def categorize_player(self, player_stats: dict) -> np.ndarray:
        # nan para stats faltantes, igual que el CLI
        return np.array([
            self._categorize(player_stats.get(s, np.nan), s)
            for s in self.stats
        ], dtype=int)

    def categorize_df(self, df: pd.DataFrame) -> np.ndarray:
        result = np.zeros((len(df), len(self.stats)), dtype=int)
        for i, stat in enumerate(self.stats):
            if stat in df.columns:
                for j in range(len(df)):
                    result[j, i] = self._categorize(df[stat].iloc[j], stat)
        return result

    def ordinal_sim(self, c1: np.ndarray, c2: np.ndarray) -> float:
        max_dist = len(c1) * 4
        return 1.0 - (float(np.sum(np.abs(c1 - c2))) / max_dist)

    def kendall_sim(self, c1: np.ndarray, c2: np.ndarray) -> float:
        tau, _ = kendalltau(c1, c2)
        return 0.5 if np.isnan(tau) else float((tau + 1) / 2)

    def cosine_sim(self, c1: np.ndarray, c2: np.ndarray) -> float:
        n1, n2 = np.linalg.norm(c1), np.linalg.norm(c2)
        if n1 == 0 or n2 == 0:
            return 0.5
        return float((np.dot(c1, c2) / (n1 * n2) + 1) / 2)

    def shape_score(self, c1: np.ndarray, c2: np.ndarray) -> float:
        return (
            0.45 * self.ordinal_sim(c1, c2) +
            0.30 * self.kendall_sim(c1, c2) +
            0.25 * self.cosine_sim(c1, c2)
        )

# =========================
# HELPERS
# =========================

def filtrar_posicion(df: pd.DataFrame, posicion: str) -> pd.DataFrame:
    def tiene_posicion(pos_str):
        if not pos_str or pd.isna(pos_str): return False
        buscar = posicion.upper().strip()
        if '(' in buscar:
            base_buscar = buscar.split('(')[0].strip()
            rol_buscar  = buscar.split('(')[1].replace(')', '').strip()
            for grupo in str(pos_str).split(','):
                grupo      = grupo.strip()
                parte_base = grupo.split('(')[0].strip()
                roles      = list(grupo.split('(')[1].replace(')', '').strip()) if '(' in grupo else []
                for token in parte_base.split('/'):
                    if token.strip().upper() == base_buscar and rol_buscar in roles:
                        return True
            return False
        else:
            bases = [t.split('(')[0].strip().upper()
                     for grupo in str(pos_str).split(',')
                     for t in grupo.split('/')]
            return buscar in bases

    col_pos = next((c for c in df.columns if c in ['posición','posicion','pos','position']), None)
    if col_pos:
        df = df[df[col_pos].apply(tiene_posicion)]
    return df

# =========================
# ENDPOINTS
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/similitud", response_model=list[ResultadoJugador])
def calcular_similitud(req: SimilitudRequest):
    if not req.pool:
        raise HTTPException(status_code=400, detail="El pool está vacío.")
    if not req.stats:
        raise HTTPException(status_code=400, detail="No se enviaron stats.")
    if req.query.get("jugador") is None:
        raise HTTPException(status_code=400, detail="Falta el jugador query.")

    df_pool = pd.DataFrame(req.pool)

    if req.posicion:
        df_pool = filtrar_posicion(df_pool, req.posicion)

    if req.pct_min is not None:
        query_min = req.query.get("minutos")
        if query_min is not None:
            umbral  = float(query_min) * (req.pct_min / 100)
            col_min = next((c for c in df_pool.columns if c in ['minutos','min','minutes']), None)
            if col_min:
                df_pool = df_pool[pd.to_numeric(df_pool[col_min], errors='coerce').fillna(0) >= umbral]

    if df_pool.empty:
        raise HTTPException(status_code=400, detail="El pool quedó vacío después de filtrar.")

    df_pool = df_pool.reset_index(drop=True)
    query   = pd.Series(req.query)

    stats_validas = [s for s in req.stats if s in df_pool.columns]
    if not stats_validas:
        raise HTTPException(status_code=400, detail="Ninguna stat coincide con las columnas del pool.")

    pesos = np.array([req.pesos.get(s, 1.0) for s in stats_validas])

    comp = SimilitudComparator()
    comp.fit(df_pool, stats_validas)

    cat_pool  = comp.categorize_df(df_pool[stats_validas])
    # nan para stats faltantes, igual que el CLI
    cat_query = comp.categorize_player({s: query.get(s, np.nan) for s in stats_validas})

    cat_pool_w  = cat_pool  * pesos
    cat_query_w = cat_query * pesos

    shape_sim   = np.array([comp.shape_score(cat_pool_w[i],   cat_query_w) for i in range(len(df_pool))])
    ordinal_sim = np.array([comp.ordinal_sim(cat_pool_w[i],   cat_query_w) for i in range(len(df_pool))])
    kendall_sim = np.array([comp.kendall_sim(cat_pool_w[i],   cat_query_w) for i in range(len(df_pool))])

    query_nombre = str(req.query.get("jugador", "")).lower()
    resultados   = []

    for i, row in df_pool.iterrows():
        nombre = str(row.get("jugador", ""))
        if nombre.lower() == query_nombre:
            continue

        posicion       = str(row.get("posición", "") or row.get("posicion", "") or "")
        statsResultado = {s: round(float(cat_pool_w[i][j]), 4) for j, s in enumerate(stats_validas)}

        resultados.append(ResultadoJugador(
            jugador            = nombre,
            posicion           = posicion,
            similitudCombinada = round(float(shape_sim[i])   * 100, 1),
            similitudOrdinal   = round(float(ordinal_sim[i]) * 100, 1),
            similitudKendall   = round(float(kendall_sim[i]) * 100, 1),
            statsResultado     = statsResultado,
        ))

    resultados.sort(key=lambda x: x.similitudCombinada, reverse=True)
    return resultados


@app.post("/ranking", response_model=list[ResultadoRanking])
def calcular_ranking(req: RankingRequest):
    if not req.pool:
        raise HTTPException(status_code=400, detail="El pool está vacío.")

    todas_stats = req.stats_positivas + req.stats_negativas
    if not todas_stats:
        raise HTTPException(status_code=400, detail="No se enviaron stats.")

    df = pd.DataFrame(req.pool)

    if req.posicion:
        df = filtrar_posicion(df, req.posicion)

    col_min = next((c for c in df.columns if c in ['minutos','min','minutes']), None)
    if col_min:
        df[col_min] = pd.to_numeric(df[col_min], errors='coerce')
        if req.min_minutos is not None:
            df = df[df[col_min].fillna(0) >= req.min_minutos]
        if req.max_minutos is not None:
            df = df[df[col_min].fillna(0) <= req.max_minutos]

    if df.empty:
        raise HTTPException(status_code=400, detail="El pool quedó vacío después de filtrar.")

    df = df.reset_index(drop=True)

    stats_validas_pos = [s for s in req.stats_positivas if s in df.columns]
    stats_validas_neg = [s for s in req.stats_negativas  if s in df.columns]
    todas_validas     = stats_validas_pos + stats_validas_neg

    for s in todas_validas:
        df[s] = pd.to_numeric(df[s], errors='coerce').fillna(0)

    columnas_puntaje = []

    for s in stats_validas_pos:
        col = f"puntaje_{s}"
        df[col] = df[s].rank(pct=True)
        columnas_puntaje.append(col)

    for s in stats_validas_neg:
        col = f"puntaje_{s}"
        df[col] = 1.0 - df[s].rank(pct=True)
        columnas_puntaje.append(col)

    if columnas_puntaje:
        df["puntaje_crudo"]  = df[columnas_puntaje].sum(axis=1)
        df["PUNTAJE_GLOBAL"] = (df["puntaje_crudo"] / len(columnas_puntaje)) * 100
    else:
        df["PUNTAJE_GLOBAL"] = 0.0

    df = df.sort_values("PUNTAJE_GLOBAL", ascending=False).reset_index(drop=True)

    col_pos = next((c for c in df.columns if c in ['posición','posicion','pos','position']), None)
    resultados = []

    for _, row in df.iterrows():
        nombre   = str(row.get("jugador", ""))
        posicion = str(row.get(col_pos, "") if col_pos else "")
        minutos  = float(row[col_min]) if col_min and not pd.isna(row.get(col_min)) else None
        stats    = {s: round(float(row[s]), 2) for s in todas_validas}
        puntaje  = round(float(row["PUNTAJE_GLOBAL"]), 1)

        resultados.append(ResultadoRanking(
            jugador  = nombre,
            posicion = posicion,
            minutos  = minutos,
            puntaje  = puntaje,
            stats    = stats,
        ))

    return resultados