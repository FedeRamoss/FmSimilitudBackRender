"""
FM DataLab v3 - Filtros
========================
Parser de posiciones FM y filtros de DataFrame.
Sin dependencias de Streamlit — puro pandas/numpy/re.
"""

import re
import pandas as pd
import numpy as np


# ── Columnas de posición reconocidas ────────────────────────────────────────

_COLS_POSICION = ["posición", "posicion", "pos", "position"]


def _col_posicion(df: pd.DataFrame) -> str | None:
    """Retorna el nombre de la columna de posición, o None si no existe."""
    return next((c for c in df.columns if c in _COLS_POSICION), None)


# ── Parser de posiciones FM ─────────────────────────────────────────────────

def extraer_posiciones_jugador(posicion_str: str) -> list:
    """
    Extrae todas las posiciones con rol de un string de posición FM.
    Ej: 'MP (DI). DL (C)' → ['MP(D)', 'MP(I)', 'DL(C)']
    """
    if not posicion_str or pd.isna(posicion_str):
        return []

    posiciones_limpias = []
    texto = str(posicion_str).upper().replace(".", ",").replace(";", ",")

    for grupo in texto.split(","):
        grupo = grupo.strip()
        if not grupo:
            continue

        if "(" in grupo:
            base, roles_str = grupo.split("(", 1)
            bases = [b.strip() for b in base.split("/") if b.strip()]
            roles = list(dict.fromkeys(re.findall(r"[DCI]", roles_str)))
            for b in bases:
                for r in roles:
                    posiciones_limpias.append(f"{b}({r})")
                if not roles:
                    posiciones_limpias.append(b)
        else:
            posiciones_limpias.extend([b.strip() for b in grupo.split("/") if b.strip()])

    return list(dict.fromkeys(posiciones_limpias))


def _jugador_tiene_posicion(pos_str: str, buscar: str) -> bool:
    """
    Comprueba si un jugador tiene una posición específica.
    Soporta búsqueda con rol exacto (ej: 'MD(D)') o solo base (ej: 'MD').
    """
    if not pos_str or pd.isna(pos_str):
        return False

    buscar = buscar.upper().strip()
    texto = str(pos_str).upper().replace(".", ",").replace(";", ",")

    if "(" in buscar:
        base_buscar = buscar.split("(")[0].strip()
        rol_buscar  = buscar.split("(")[1].replace(")", "").strip()
        for grupo in texto.split(","):
            grupo = grupo.strip()
            parte_base = grupo.split("(")[0].strip() if "(" in grupo else grupo
            roles = list(grupo.split("(")[1].replace(")", "")) if "(" in grupo else []
            for token in parte_base.split("/"):
                if token.strip() == base_buscar and rol_buscar in roles:
                    return True
        return False
    else:
        bases = []
        for grupo in texto.split(","):
            grupo = grupo.strip()
            parte_base = grupo.split("(")[0].strip() if "(" in grupo else grupo
            bases.extend([t.strip() for t in parte_base.split("/")])
        return buscar in bases


def filtrar_por_posicion(df: pd.DataFrame, posicion: str) -> pd.DataFrame:
    """Filtra el DataFrame usando el parser avanzado de posición FM."""
    col_pos = _col_posicion(df)
    if col_pos is None:
        return df
    return df[df[col_pos].apply(lambda x: _jugador_tiene_posicion(x, posicion))].copy()


def obtener_posiciones_unicas(df: pd.DataFrame) -> list:
    """
    Extrae todas las posiciones únicas con rol del dataset completo.
    Usa el parser avanzado para consistencia con el módulo de Similitud.
    """
    col_pos = _col_posicion(df)
    if col_pos is None:
        return []
    todas = []
    for ps in df[col_pos].dropna().unique():
        todas.extend(extraer_posiciones_jugador(ps))
    return sorted(set(todas))


# ── Filtro de minutos ────────────────────────────────────────────────────────

def filtrar_minutos(df: pd.DataFrame, query_jugador: dict,
                    porcentaje: float, min_col: str = "minutos") -> tuple:
    """
    Filtra jugadores por minutos relativos al jugador de referencia.
    Retorna (df_filtrado, info_dict).
    """
    info = {}

    if min_col not in df.columns:
        info["warning"] = f"Columna '{min_col}' no encontrada"
        return df, info

    qmin = query_jugador.get(min_col, None)
    if qmin is None or pd.isna(qmin):
        info["warning"] = "El jugador no tiene datos de minutos"
        return df, info

    qmin   = float(qmin)
    umbral = qmin * (porcentaje / 100)
    antes  = len(df)

    filtrado = df[df[min_col] >= umbral].reset_index(drop=True)
    info.update({
        "query_minutos": qmin,
        "porcentaje":    porcentaje,
        "min_minutos":   umbral,
        "excluidos":     antes - len(filtrado),
        "pool_final":    len(filtrado),
    })

    if len(filtrado) < 30:
        info["warning"] = (
            f"Pool con menos de 30 jugadores ({len(filtrado)}). "
            "Resultados menos precisos."
        )

    return filtrado, info