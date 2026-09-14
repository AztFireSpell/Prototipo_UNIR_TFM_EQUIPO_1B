from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
MODELOS_DIR = BASE_DIR / "modelos"

XGB_MODEL_PATH = MODELOS_DIR / "xgboost_model.pkl"
XGB_PREPROCESSOR_PATH = MODELOS_DIR / "preprocesador_xgboost.pkl"

SEQUENTIAL_MODEL_PATH = (
    MODELOS_DIR / "mejor_modelo.keras"
)

THRESHOLDS_PATH = MODELOS_DIR / "umbrales.json"


def cargar_umbrales():
    """
    Carga los umbrales validados para los modelos.
    """

    valores_por_defecto = {
        "xgboost": 0.5,
        "secuencial": 0.5
    }

    if not THRESHOLDS_PATH.exists():
        return valores_por_defecto

    with THRESHOLDS_PATH.open(
        "r",
        encoding="utf-8"
    ) as archivo:
        guardados = json.load(archivo)

    return {
        **valores_por_defecto,
        **guardados
    }


def cargar_modelo_xgboost():
    """
    Carga y verifica el modelo XGBoost serializado.
    """

    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo XGBoost: {XGB_MODEL_PATH}"
        )

    print("Cargando XGBoost desde:", XGB_MODEL_PATH)

    modelo = joblib.load(XGB_MODEL_PATH)

    print("Tipo cargado:", type(modelo))

    if not hasattr(modelo, "predict_proba"):
        raise TypeError(
            "El objeto cargado no tiene el método predict_proba. "
            f"Tipo encontrado: {type(modelo)}"
        )

    return modelo


def cargar_preprocesador_xgboost():
    """
    Carga el preprocesador utilizado durante entrenamiento.
    """

    if not XGB_PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el preprocesador de XGBoost: "
            f"{XGB_PREPROCESSOR_PATH}"
        )

    return joblib.load(XGB_PREPROCESSOR_PATH)


def cargar_modelo_secuencial():
    """
    Carga el modelo GRU/LSTM sin configuración de entrenamiento.
    """

    if not SEQUENTIAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el modelo secuencial: "
            f"{SEQUENTIAL_MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        SEQUENTIAL_MODEL_PATH,
        compile=False
    )


def predecir_xgboost(
    modelo,
    preprocesador,
    transaccion
):
    """
    Produce una probabilidad de fraude para una transacción.

    transaccion debe ser un diccionario con los nombres
    originales utilizados durante el entrenamiento.
    """

    df_entrada = pd.DataFrame([transaccion])

    entrada_transformada = preprocesador.transform(
        df_entrada
    )

    probabilidad = modelo.predict_proba(
        entrada_transformada
    )[0, 1]

    return float(probabilidad)


def predecir_secuencia(
    modelo,
    secuencia
):
    """
    Produce una probabilidad para una secuencia ya transformada.

    Forma requerida:
    (1, 5, 9)
    """

    secuencia = np.asarray(
        secuencia,
        dtype=np.float32
    )

    if secuencia.shape != (1, 5, 9):
        raise ValueError(
            "El modelo secuencial espera una entrada "
            f"con forma (1, 5, 9), pero recibió "
            f"{secuencia.shape}."
        )

    probabilidad = modelo.predict(
        secuencia,
        verbose=0
    )[0, 0]

    return float(probabilidad)


def obtener_decision_hibrida(
    prob_xgboost,
    umbral_xgboost,
    prob_secuencial=None,
    umbral_secuencial=None
):
    """
    Política híbrida inicial del prototipo.

    No determina jurídicamente un fraude. Genera una
    recomendación operativa para aprobación o revisión.
    """

    alerta_xgb = prob_xgboost >= umbral_xgboost

    if prob_secuencial is None:
        if alerta_xgb:
            return {
                "estado": "REVISIÓN",
                "nivel": "Medio",
                "motivo": (
                    "XGBoost superó su umbral, pero no existe "
                    "historial suficiente para evaluación secuencial."
                )
            }

        return {
            "estado": "APROBACIÓN SUGERIDA",
            "nivel": "Bajo",
            "motivo": (
                "XGBoost no superó el umbral operativo y no existe "
                "historial suficiente para evaluación secuencial."
            )
        }

    alerta_secuencial = (
        prob_secuencial >= umbral_secuencial
    )

    if alerta_xgb and alerta_secuencial:
        return {
            "estado": "RETENER",
            "nivel": "Alto",
            "motivo": (
                "Los modelos transaccional y secuencial "
                "emitieron una alerta."
            )
        }

    if alerta_xgb or alerta_secuencial:
        return {
            "estado": "REVISIÓN",
            "nivel": "Medio",
            "motivo": (
                "Solo uno de los dos modelos superó su umbral."
            )
        }

    return {
        "estado": "APROBACIÓN SUGERIDA",
        "nivel": "Bajo",
        "motivo": (
            "Ninguno de los modelos superó su umbral."
        )
    }