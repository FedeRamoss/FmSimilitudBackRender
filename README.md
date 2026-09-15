<h1 align="center">FM Similitud — Backend</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-2E5339?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-2E5339?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Vercel-2E5339?style=for-the-badge&logo=vercel&logoColor=white" alt="Vercel"/>
</p>

API en FastAPI que calcula similitud entre jugadores de fútbol (similitud coseno + distancia euclidiana ponderada, con `scikit-learn`). Es el backend de **FM Similitud**, una herramienta para comparar jugadores por estadísticas.

🔗 Deploy: **fm-similitud-back-render.vercel.app**

## Estructura

```
api/
├── index.py       # Endpoints
├── similitud.py    # Cálculo de similitud
├── filtros.py      # Filtros (posición, % minutos, etc.)
└── loader.py       # Carga y normalización de datos
```

## Cómo correrlo local

```bash
pip install -r requirements.txt
uvicorn api.index:app --reload
```

## Frontend

El frontend en Angular que consume esta API vive en un repo aparte (en proceso de organizar).

## Autor

Federico Ramos
