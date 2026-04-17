from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

from similitud import SimilitudComparatorV3, compute_similarity_v3, ranking_jugadores
from filtros  import filtrar_por_posicion, filtrar_minutos
from loader   import limpiar_data

app = FastAPI(title="FM Similarity API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    jugador:             str
    posicion:            str
    similitudCombinada:  float
    similitudMAE:        float
    similitudEuclidiana: float
    similitudPearson:    float
    similitudOrdinal:    float
    statsResultado:      dict[str, float]

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

    # limpiar_data del loader.py
    df_pool = limpiar_data(pd.DataFrame(req.pool))

    # filtros.py
    if req.posicion:
        df_pool = filtrar_por_posicion(df_pool, req.posicion)

    if req.pct_min is not None:
        query_min = req.query.get("minutos")
        if query_min is not None:
            query_row = pd.Series({"minutos": float(query_min)})
            df_pool, _ = filtrar_minutos(df_pool, query_row, req.pct_min)

    if df_pool.empty:
        raise HTTPException(status_code=400, detail="El pool quedó vacío después de filtrar.")

    df_pool = df_pool.reset_index(drop=True)

    # Query con valores numéricos
    query = pd.Series(req.query)
    for stat in req.stats:
        if stat in query.index:
            try:    query[stat] = float(query[stat])
            except: query[stat] = np.nan

    stats_validas = [s for s in req.stats if s in df_pool.columns]
    if not stats_validas:
        raise HTTPException(status_code=400, detail="Ninguna stat coincide con las columnas del pool.")

    pesos = np.array([req.pesos.get(s, 1.0) for s in stats_validas])

    # similitud.py — idéntico a similitud_vm.py
    comp = SimilitudComparatorV3()
    comp.fit(df_pool, stats_validas)

    q_stats   = {s: query.get(s, np.nan) for s in stats_validas}
    cat_pool  = comp.categorize_dataframe(df_pool[stats_validas]) * pesos
    norm_pool = comp.normalize_dataframe(df_pool[stats_validas])  * pesos
    cat_q     = comp.categorize_player(q_stats).astype(float)     * pesos
    norm_q    = comp.normalize_player(q_stats)                    * pesos

    mae_s, euc_s, pear_s, ord_s, hyb_s = compute_similarity_v3(
        comp, cat_pool, cat_q, norm_pool, norm_q
    )

    df_ranking = ranking_jugadores(
        df_pool, mae_s, euc_s, pear_s, ord_s, hyb_s,
        str(req.query.get("jugador", ""))
    )

    resultados = []
    for _, row in df_ranking.iterrows():
        posicion       = str(row.get("posición", "") or row.get("posicion", "") or "")
        statsResultado = {s: round(float(row.get(s, 0) or 0), 4) for s in stats_validas}
        resultados.append(ResultadoJugador(
            jugador             = str(row["jugador"]),
            posicion            = posicion,
            similitudCombinada  = float(row["similitud"]),
            similitudMAE        = float(row["sim_mae"]),
            similitudEuclidiana = float(row["sim_euclidiana"]),
            similitudPearson    = float(row["sim_pearson"]),
            similitudOrdinal    = float(row["sim_ordinal"]),
            statsResultado = {s: float(row.get(s, 0) or 0) for s in stats_validas}
        ))

    return resultados


@app.post("/ranking", response_model=list[ResultadoRanking])
def calcular_ranking(req: RankingRequest):
    if not req.pool:
        raise HTTPException(status_code=400, detail="El pool está vacío.")

    todas_stats = req.stats_positivas + req.stats_negativas
    if not todas_stats:
        raise HTTPException(status_code=400, detail="No se enviaron stats.")

    df = limpiar_data(pd.DataFrame(req.pool))

    if req.posicion:
        df = filtrar_por_posicion(df, req.posicion)

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