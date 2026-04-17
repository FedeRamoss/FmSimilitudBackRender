"""
FM DataLab v3 - Loader
=======================
Carga, detección de separador y limpieza del CSV.
Sin dependencias de Streamlit — puro pandas/numpy.
"""

import pandas as pd
import numpy as np


def _detectar_separador(contenido: bytes) -> str:
    """Detecta el separador del CSV probando ';' y ','."""
    muestra = contenido[:4096].decode("utf-8-sig", errors="ignore")
    return ";" if muestra.count(";") >= muestra.count(",") else ","


def limpiar_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y normaliza el DataFrame.
    - Normaliza nombres de columnas a minúsculas sin espacios
    - Reemplaza comas por puntos en strings numéricos
    - Convierte a numérico donde sea posible
    - Reemplaza valores vacíos/guiones con NaN
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col].astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", ".", regex=False)
                .str.strip()
            )
            df[col] = df[col].replace(["-", "nan", ""], np.nan)
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


def cargar_csv(uploaded_file) -> tuple:
    """
    Carga un CSV con detección automática de separador.
    Retorna (df, separador_usado, error_msg).
    El error_msg es None si la carga fue exitosa.
    """
    try:
        contenido = uploaded_file.read()
        uploaded_file.seek(0)

        sep = _detectar_separador(contenido)
        df = pd.read_csv(uploaded_file, sep=sep, encoding="utf-8-sig")
        df = limpiar_data(df)

        cols_req = {"jugador"}
        faltantes = cols_req - set(df.columns)
        if faltantes:
            return None, sep, f"Faltan columnas requeridas: {', '.join(faltantes)}"

        return df, sep, None

    except Exception as e:
        return None, None, str(e)