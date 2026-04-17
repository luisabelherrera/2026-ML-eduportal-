@echo off
echo ============================================================
echo  SERVICIO ML - Prediccion Estudiantil (Python + scikit-learn)
echo  Tecnologico Comfenalco - Proyecto de Aula
echo ============================================================

REM Crear entorno virtual si no existe
if not exist "venv" (
    echo [1/3] Creando entorno virtual Python...
    python -m venv venv
)

REM Activar entorno virtual
echo [2/3] Activando entorno virtual...
call venv\Scripts\activate.bat

REM Instalar dependencias si faltan
echo [3/3] Instalando dependencias...
pip install -r requirements.txt --quiet

REM Entrenar modelo si no existe
if not exist "modelo_estudiante.pkl" (
    echo [ENTRENAMIENTO] Entrenando modelo por primera vez...
    python train.py
)

echo.
echo Iniciando servidor FastAPI en http://localhost:5000
echo Documentacion Swagger: http://localhost:5000/docs
echo.
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
