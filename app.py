"""
=============================================================================
MICROSERVICIO DE PREDICCIÓN - FastAPI
=============================================================================
Integración del modelo Python con el sistema Spring Boot / Angular.

Endpoints:
  POST /predecir        — Realiza predicción de rendimiento estudiantil
  GET  /metricas        — Retorna métricas del modelo entrenado
  POST /entrenar        — Re-entrena el modelo con el dataset actual
  GET  /health          — Health check del servicio
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging
import train as trainer

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Inicialización de FastAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Servicio de Predicción Estudiantil",
    description="Microservicio ML para predicción de rendimiento académico - Tecnológico Comfenalco",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # En producción especificar dominios exactos
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Rutas de artefactos
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
MODEL_PATH   = os.path.join(BASE_DIR, "modelo_estudiante.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoders.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metricas.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# Carga de modelo en arranque
# ─────────────────────────────────────────────────────────────────────────────
modelo   = None
encoders = None
metricas = None

def cargar_artefactos():
    """Carga modelo, encoders y métricas desde disco. Los entrena si no existen."""
    global modelo, encoders, metricas
    if not os.path.exists(MODEL_PATH):
        logger.info("Modelo no encontrado. Iniciando entrenamiento automático...")
        trainer.entrenar()
    modelo   = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    metricas = joblib.load(METRICS_PATH)
    logger.info("Modelo cargado correctamente.")

cargar_artefactos()


# ─────────────────────────────────────────────────────────────────────────────
# Esquema de entrada (compatible con DatosEstudiante de Spring Boot)
# ─────────────────────────────────────────────────────────────────────────────
class DatosEstudianteRequest(BaseModel):
    documento:             Optional[int]   = Field(None, description="Documento de identidad del estudiante")
    edad:                  int             = Field(...,  description="Edad en años")
    genero:                str             = Field(...,  description="Masculino | Femenino | Otro")
    horasEstudioSemanal:   int             = Field(...,  description="Horas de estudio por semana")
    asistencia:            float           = Field(...,  description="Porcentaje de asistencia (0-100)")
    promedioParciales:     float           = Field(...,  description="Promedio de notas parciales")
    participacionClases:   str             = Field(...,  description="Baja | Media | Alta")
    usoPlataformaVirtual:  str             = Field(...,  description="Bajo | Medio | Alto")
    antecedentesPerdida:   str             = Field(...,  description="Sí | No")
    apoyoFamiliar:         str             = Field(...,  description="Bajo | Medio | Alto")
    cargaAcademica:        int             = Field(...,  description="Número de materias cursadas")
    problemasPersonales:   str             = Field(...,  description="Ninguno | Leves | Graves")

    class Config:
        json_schema_extra = {
            "example": {
                "documento": 91503778,
                "edad": 17,
                "genero": "Masculino",
                "horasEstudioSemanal": 5,
                "asistencia": 78.5,
                "promedioParciales": 2.8,
                "participacionClases": "Baja",
                "usoPlataformaVirtual": "Bajo",
                "antecedentesPerdida": "Sí",
                "apoyoFamiliar": "Bajo",
                "cargaAcademica": 5,
                "problemasPersonales": "Leves"
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# Función de preprocesamiento para una sola instancia
# ─────────────────────────────────────────────────────────────────────────────
def preprocesar_instancia(datos: DatosEstudianteRequest) -> pd.DataFrame:
    """
    Transforma los datos de entrada al formato esperado por el modelo.
    Aplica los mismos mapeos usados durante el entrenamiento.
    """
    mapeo_genero       = encoders.get("Genero", {"Masculino": 0, "Femenino": 1, "Otro": 2})
    mapeo_participacion = encoders.get("Participacion_Clases", {"Baja": 0, "Media": 1, "Alta": 2})
    mapeo_plataforma   = encoders.get("Uso_Plataforma_Virtual", {"Bajo": 0, "Medio": 1, "Alto": 2})
    mapeo_antecedentes = encoders.get("Antecedentes_Perdida", {"No": 0, "Si": 1, "Sí": 1})
    mapeo_apoyo        = encoders.get("Apoyo_Familiar", {"Bajo": 0, "Medio": 1, "Alto": 2})
    mapeo_problemas    = encoders.get("Problemas_Personales", {"Ninguno": 0, "Leves": 1, "Graves": 2})

    def mapear(valor: str, mapa: dict, campo: str) -> int:
        if valor not in mapa:
            raise HTTPException(
                status_code=422,
                detail=f"Valor inválido para '{campo}': '{valor}'. Valores permitidos: {list(mapa.keys())}"
            )
        return mapa[valor]

    fila = {
        "Edad":                  datos.edad,
        "Genero":                mapear(datos.genero, mapeo_genero, "genero"),
        "Horas_Estudio_Semanal": datos.horasEstudioSemanal,
        "Asistencia":            datos.asistencia,
        "Promedio_Parciales":    datos.promedioParciales,
        "Participacion_Clases":  mapear(datos.participacionClases, mapeo_participacion, "participacionClases"),
        "Uso_Plataforma_Virtual":mapear(datos.usoPlataformaVirtual, mapeo_plataforma, "usoPlataformaVirtual"),
        "Antecedentes_Perdida":  mapear(datos.antecedentesPerdida, mapeo_antecedentes, "antecedentesPerdida"),
        "Apoyo_Familiar":        mapear(datos.apoyoFamiliar, mapeo_apoyo, "apoyoFamiliar"),
        "Carga_Academica":       datos.cargaAcademica,
        "Problemas_Personales":  mapear(datos.problemasPersonales, mapeo_problemas, "problemasPersonales"),
    }

    return pd.DataFrame([fila])


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Verifica que el servicio y el modelo estén disponibles."""
    return {
        "status": "ok",
        "modelo_cargado": modelo is not None,
        "version": "2.0.0 (Python/scikit-learn)"
    }


@app.post("/predecir")
def predecir(datos: DatosEstudianteRequest):
    """
    Realiza la predicción de rendimiento estudiantil.

    Retorna:
      - prediccion: 'tested_positive' (perderá) | 'tested_negative' (no perderá)
      - confianza:  porcentaje de confianza del modelo (e.g. '87.3%')
      - probabilidades: distribución completa de probabilidades
      - factores_riesgo: variables con mayor impacto (informativo)
    """
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no inicializado")

    try:
        X = preprocesar_instancia(datos)
        prediccion_idx  = modelo.predict(X)[0]
        probabilidades  = modelo.predict_proba(X)[0]

        etiqueta = "tested_positive" if prediccion_idx == 1 else "tested_negative"
        confianza = probabilidades[prediccion_idx] * 100

        # Importancia de variables del modelo (orientación para el docente)
        clf = modelo.named_steps["clf"]
        feature_names = metricas.get("feature_names", list(X.columns))
        top_factores = []
        if hasattr(clf, "feature_importances_"):
            importancias = dict(zip(feature_names, clf.feature_importances_.tolist()))
            top_factores = sorted(importancias.items(), key=lambda x: x[1], reverse=True)[:3]

        logger.info(
            f"Predicción: {etiqueta} | Confianza: {confianza:.1f}% | "
            f"Documento: {datos.documento}"
        )

        return {
            "prediccion":    etiqueta,
            "confianza":     f"{confianza:.1f}%",
            "probabilidades": {
                "no_perdera": round(float(probabilidades[0]) * 100, 2),
                "perdera":    round(float(probabilidades[1]) * 100, 2),
            },
            "factores_riesgo_principales": [
                {"variable": k, "importancia": round(v, 4)} for k, v in top_factores
            ],
            "modelo": f"{metricas.get('modelo', 'ML')} (Python/scikit-learn)"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.get("/metricas")
def obtener_metricas():
    """
    Retorna las métricas de desempeño del modelo entrenado:
    accuracy, precision, recall, F1-score, AUC-ROC,
    matriz de confusión y comparación con otros modelos.
    """
    if metricas is None:
        raise HTTPException(status_code=503, detail="Métricas no disponibles")
    return metricas


@app.post("/entrenar")
def reentrenar():
    """
    Re-entrena el modelo con el dataset actual y recarga los artefactos.
    Útil cuando se agregan nuevos datos al ARFF.
    """
    global modelo, encoders, metricas
    try:
        logger.info("Iniciando re-entrenamiento del modelo...")
        nuevas_metricas = trainer.entrenar()
        cargar_artefactos()
        return {
            "status": "ok",
            "mensaje": "Modelo re-entrenado exitosamente",
            "metricas": nuevas_metricas
        }
    except Exception as e:
        logger.error(f"Error en re-entrenamiento: {e}")
        raise HTTPException(status_code=500, detail=f"Error al entrenar: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=False)
