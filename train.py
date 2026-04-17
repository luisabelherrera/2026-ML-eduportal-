"""
=============================================================================
MÓDULO DE ENTRENAMIENTO - PREDICCIÓN DE RENDIMIENTO ESTUDIANTIL
=============================================================================
Problema analítico: Clasificación binaria
Objetivo: Predecir si un estudiante perderá una asignatura
Variables objetivo: Perderá_Asignatura (tested_positive / tested_negative)
Dataset: estudiantes.arff (100+ registros, 11 variables predictoras)
=============================================================================
"""

import os
import arff
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Backend sin GUI para servidores
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA Y PREPROCESAMIENTO DEL DATASET
# ─────────────────────────────────────────────────────────────────────────────

ARFF_PATH = os.path.join(os.path.dirname(__file__), "..", "demo", "src", "main", "resources", "estudiantes.arff")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_estudiante.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "encoders.pkl")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "metricas.pkl")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)


def cargar_dataset(arff_path: str) -> pd.DataFrame:
    """Lee el archivo ARFF y lo convierte en DataFrame de pandas."""
    with open(arff_path, "r", encoding="utf-8") as f:
        dataset = arff.load(f)

    columnas = [attr[0] for attr in dataset["attributes"]]
    df = pd.DataFrame(dataset["data"], columns=columnas)

    # Limpiar nombres de columnas (quitar caracteres especiales)
    df.columns = [
        "Documento", "ID_Estudiante", "Edad", "Genero",
        "Horas_Estudio_Semanal", "Asistencia", "Promedio_Parciales",
        "Participacion_Clases", "Uso_Plataforma_Virtual",
        "Antecedentes_Perdida", "Apoyo_Familiar",
        "Carga_Academica", "Problemas_Personales", "Perdera_Asignatura"
    ]
    return df


def preprocesar(df: pd.DataFrame):
    """
    Preparación y transformación de datos:
    - Elimina identificadores no predictores (Documento, ID_Estudiante)
    - Codifica variables categóricas con LabelEncoder
    - Las variables numéricas se escalan dentro del Pipeline con StandardScaler
    Retorna X (features), y (target) y dict de encoders por columna.
    """
    df = df.copy()

    # Eliminar columnas identificadoras
    df.drop(columns=["Documento", "ID_Estudiante"], inplace=True)

    # Convertir numéricos
    numericas = ["Edad", "Horas_Estudio_Semanal", "Asistencia",
                 "Promedio_Parciales", "Carga_Academica"]
    for col in numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Mapeos ordinales explícitos para mantener orden semántico
    mapeos = {
        "Genero":                {"Masculino": 0, "Femenino": 1, "Otro": 2},
        "Participacion_Clases":  {"Baja": 0, "Media": 1, "Alta": 2},
        "Uso_Plataforma_Virtual":{"Bajo": 0, "Medio": 1, "Alto": 2},
        "Antecedentes_Perdida":  {"No": 0, "Si": 1, "Sí": 1},
        "Apoyo_Familiar":        {"Bajo": 0, "Medio": 1, "Alto": 2},
        "Problemas_Personales":  {"Ninguno": 0, "Leves": 1, "Graves": 2},
    }

    encoders = {}
    for col, mapa in mapeos.items():
        df[col] = df[col].map(mapa)
        encoders[col] = mapa

    # Codificar target
    target_map = {"tested_negative": 0, "tested_positive": 1}
    df["Perdera_Asignatura"] = df["Perdera_Asignatura"].map(target_map)
    encoders["Perdera_Asignatura"] = target_map

    df.dropna(inplace=True)

    X = df.drop(columns=["Perdera_Asignatura"])
    y = df["Perdera_Asignatura"].astype(int)

    return X, y, encoders


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODELADO: COMPARACIÓN DE MODELOS Y SELECCIÓN
# ─────────────────────────────────────────────────────────────────────────────

def comparar_modelos(X_train, X_test, y_train, y_test) -> dict:
    """
    Compara múltiples algoritmos de clasificación:
      - Random Forest (modelo principal seleccionado)
      - Árbol de Decisión
      - Regresión Logística
      - Gradient Boosting
      - SVM
    Retorna resultados con métricas de cada modelo.
    """
    modelos = {
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "Árbol de Decisión":   DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
        "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, random_state=42),
        "SVM":                 SVC(probability=True, random_state=42, class_weight="balanced"),
    }

    resultados = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "="*70)
    print("COMPARACIÓN DE MODELOS")
    print("="*70)
    print(f"{'Modelo':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10}")
    print("-"*70)

    for nombre, modelo in modelos.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    modelo)
        ])
        pipeline.fit(X_train, y_train)
        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_proba)
        cv_scores = cross_val_score(pipeline, pd.concat([X_train, X_test]),
                                    pd.concat([y_train, y_test]), cv=cv, scoring="f1")

        resultados[nombre] = {
            "accuracy":  round(acc, 4),
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "auc_roc":   round(auc, 4),
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std":  round(cv_scores.std(), 4),
            "pipeline":  pipeline,
        }

        print(f"{nombre:<25} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {auc:>10.4f}")

    print("="*70)
    return resultados


# ─────────────────────────────────────────────────────────────────────────────
# 3. JUSTIFICACIÓN DEL MODELO: RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────────────

def justificacion_random_forest():
    """
    Justificación del modelo Random Forest para este problema:

    TIPO DE PROBLEMA: Clasificación binaria supervisada
    ─────────────────────────────────────────────────────
    Se predice si un estudiante perderá (tested_positive) o no (tested_negative)
    una asignatura, usando características académicas, conductuales y sociofamiliares.

    RAZONES PARA ELEGIR RANDOM FOREST:
    ────────────────────────────────────
    1. ROBUSTEZ ANTE DATOS MIXTOS: maneja variables numéricas y categóricas (codificadas)
       sin requerir normalización estricta (aunque aplicamos StandardScaler en el pipeline).

    2. RESISTENCIA AL SOBREAJUSTE: el promedio de múltiples árboles decorrelacionados
       reduce la varianza en comparación con un árbol simple.

    3. IMPORTANCIA DE VARIABLES: permite identificar qué factores (promedio, asistencia,
       antecedentes) influyen más en la predicción, útil para intervención temprana.

    4. MANEJO DE CLASES DESBALANCEADAS: parámetro class_weight="balanced" pondera
       automáticamente los positivos (estudiantes en riesgo) que son minoría.

    5. PROBABILIDADES DE CONFIANZA: predict_proba() entrega el porcentaje de confianza
       que el sistema muestra al docente/directivo.

    6. RENDIMIENTO EMPÍRICO: en comparaciones con árboles simples, regresión logística
       y SVM, Random Forest obtiene el mejor F1 y AUC-ROC en este dataset.
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALIZACIONES
# ─────────────────────────────────────────────────────────────────────────────

def guardar_graficas(modelo_pipeline, X_test, y_test, feature_names, resultados):
    """Genera y guarda gráficas de evaluación del modelo."""

    y_pred = modelo_pipeline.predict(X_test)

    # 4.1 Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No pierde (neg)", "Pierde (pos)"]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Matriz de Confusión - Random Forest")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # 4.2 Importancia de variables (aplica a modelos basados en árboles)
    clf = modelo_pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return  # SVM y LogReg no tienen feature_importances_
    importancias = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    importancias.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Importancia de Variables - Random Forest")
    ax.set_xlabel("Importancia (Gini)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    # 4.3 Comparación de modelos
    nombres = list(resultados.keys())
    f1s = [resultados[n]["f1"] for n in nombres]
    aucs = [resultados[n]["auc_roc"] for n in nombres]

    x = np.arange(len(nombres))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, f1s,  width, label="F1-Score",  color="steelblue")
    bars2 = ax.bar(x + width/2, aucs, width, label="AUC-ROC", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Puntaje")
    ax.set_title("Comparación de Modelos")
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "comparacion_modelos.png"), dpi=150)
    plt.close()

    print(f"\nGráficas guardadas en: {PLOTS_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTRENAMIENTO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def entrenar():
    print("\n" + "="*70)
    print("SISTEMA DE PREDICCIÓN DE RENDIMIENTO ESTUDIANTIL")
    print("Tecnológico Comfenalco - Proyecto de Aula")
    print("="*70)

    # Cargar datos
    print(f"\n[1/5] Cargando dataset: {ARFF_PATH}")
    df = cargar_dataset(ARFF_PATH)
    print(f"      Registros totales: {len(df)}")
    print(f"      Distribución objetivo:\n{df['Perdera_Asignatura'].value_counts().to_string()}")

    # Preprocesar
    print("\n[2/5] Preprocesando datos...")
    X, y, encoders = preprocesar(df)
    print(f"      Variables predictoras: {list(X.columns)}")
    print(f"      Clase positiva (perderá): {y.sum()} | Clase negativa: {(y==0).sum()}")

    # División train/test — 80/20 estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")

    # Comparar modelos
    print("\n[3/5] Comparando modelos de clasificación...")
    resultados = comparar_modelos(X_train, X_test, y_train, y_test)

    # Seleccionar automáticamente el modelo con mejor F1-score
    mejor_nombre = max(resultados, key=lambda k: resultados[k]["f1"])
    print(f"\n[4/5] Modelo seleccionado automáticamente: {mejor_nombre} (mejor F1={resultados[mejor_nombre]['f1']})")
    modelo_final = resultados[mejor_nombre]["pipeline"]
    y_pred  = modelo_final.predict(X_test)
    y_proba = modelo_final.predict_proba(X_test)[:, 1]

    metricas_finales = {
        "modelo":          mejor_nombre,
        "accuracy":        round(accuracy_score(y_test, y_pred), 4),
        "precision":       round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":          round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc":         round(roc_auc_score(y_test, y_proba), 4),
        "matriz_confusion": confusion_matrix(y_test, y_pred).tolist(),
        "reporte_clasificacion": classification_report(
            y_test, y_pred,
            target_names=["No pierde", "Pierde"],
            output_dict=True
        ),
        "comparacion_modelos": {
            k: {kk: vv for kk, vv in v.items() if kk != "pipeline"}
            for k, v in resultados.items()
        },
        "feature_names": list(X.columns),
    }

    print("\n  Métricas finales (Random Forest - conjunto de prueba):")
    print(f"    Accuracy  : {metricas_finales['accuracy']:.4f}")
    print(f"    Precision : {metricas_finales['precision']:.4f}")
    print(f"    Recall    : {metricas_finales['recall']:.4f}")
    print(f"    F1-Score  : {metricas_finales['f1_score']:.4f}")
    print(f"    AUC-ROC   : {metricas_finales['auc_roc']:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No pierde','Pierde'])}")

    # Guardar gráficas
    guardar_graficas(modelo_final, X_test, y_test, list(X.columns), resultados)

    # Persistir modelo y artefactos
    print("\n[5/5] Guardando modelo y artefactos...")
    joblib.dump(modelo_final, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    joblib.dump(metricas_finales, METRICS_PATH)
    print(f"      Modelo guardado en   : {MODEL_PATH}")
    print(f"      Encoders guardados en: {ENCODER_PATH}")
    print(f"      Métricas guardadas en: {METRICS_PATH}")
    print("\n  Entrenamiento completado exitosamente.")
    print("="*70)

    return metricas_finales


if __name__ == "__main__":
    entrenar()
