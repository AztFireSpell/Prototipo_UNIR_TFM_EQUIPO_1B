# MODULO GLOBAL (Monitoreo Ejecutivo)
from pathlib import Path
import json
import sqlite3
from datetime import datetime

from motor_inferencia import (
    cargar_umbrales,
    cargar_modelo_xgboost,
    cargar_preprocesador_xgboost,
    cargar_modelo_secuencial,
    predecir_xgboost,
    predecir_secuencia,
    obtener_decision_hibrida,
)

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from xgboost import XGBClassifier


@st.cache_resource(show_spinner="Cargando modelo XGBoost...")
def obtener_xgboost():
    modelo = cargar_modelo_xgboost()
    preprocesador = cargar_preprocesador_xgboost()
    return modelo, preprocesador


@st.cache_resource(show_spinner="Cargando modelo secuencial...")
def obtener_modelo_secuencial():
    return cargar_modelo_secuencial()


@st.cache_data(show_spinner=False)
def obtener_umbrales():
    return cargar_umbrales()


# Configuracion inicial de la pagina
st.set_page_config(page_title="Deteccion de Fraude Bancario", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
RESULTADOS_DIR = BASE_DIR / "resultados"

# Estructura esperada:
# resultados/random_forest/metricas.json
# resultados/xgboost/metricas.json
# resultados/transformer/metricas.json
# resultados/lstm/metricas.json
ARCHIVOS_METRICAS = {
    "Random Forest": RESULTADOS_DIR / "random_forest" / "metricas.json",
    "XGBoost": RESULTADOS_DIR / "xgboost" / "metricas.json",
    "Transformer": RESULTADOS_DIR / "transformer" / "metricas.json",
    "LSTM": RESULTADOS_DIR / "lstm" / "metricas.json",
}

TIPOS_MODELO = {
    "Random Forest": "Transaccional",
    "XGBoost": "Transaccional",
    "Transformer": "Secuencial",
    "LSTM": "Secuencial",
}

ARCHIVOS_PREDICCIONES = {
    "Random Forest": RESULTADOS_DIR / "random_forest" / "predicciones_test.csv",
    "XGBoost": RESULTADOS_DIR / "xgboost" / "predicciones_test.csv",
    "Transformer": RESULTADOS_DIR / "transformer" / "predicciones_test.csv",
    "LSTM": RESULTADOS_DIR / "lstm" / "predicciones_test.csv",
}

st.title("🛡️ Sistema de Deteccion Temprana de Fraudes Bancarios")
st.caption("Prototipo de Monitoreo Analitico y Despliegue de Modelos (TFM)")


def _primer_bloque_disponible(resultado, nombres):
    """Devuelve el primer diccionario de metricas disponible."""
    for nombre in nombres:
        bloque = resultado.get(nombre)
        if isinstance(bloque, dict) and bloque:
            return bloque, nombre
    return {}, None


@st.cache_data(show_spinner=False)
def cargar_resultados_modelos(firmas_archivos):
    """Carga metricas reales exportadas por los cuatro notebooks.

    firmas_archivos se usa para invalidar automaticamente la cache cuando
    cambia un metricas.json.
    """
    del firmas_archivos
    filas = []
    matrices = {}
    errores = []

    for nombre_modelo, ruta_json in ARCHIVOS_METRICAS.items():
        if not ruta_json.exists():
            errores.append(f"No se encontro el archivo de {nombre_modelo}: {ruta_json}")
            continue

        try:
            with ruta_json.open("r", encoding="utf-8") as archivo:
                resultado = json.load(archivo)

            metricas, origen_metricas = _primer_bloque_disponible(
                resultado,
                (
                    "metrics_threshold_validation",
                    "metricas_umbral_validacion",
                    "metrics_validation_threshold",
                    "metrics_threshold_0_5",
                    "metricas_umbral_0_5",
                ),
            )

            if not metricas:
                errores.append(
                    f"{nombre_modelo}: no se encontro un bloque de metricas "
                    "compatible en metricas.json."
                )
                continue

            def numero(diccionario, clave, defecto=np.nan):
                valor = diccionario.get(clave, defecto)
                if valor is None:
                    return defecto
                return float(valor)

            tiempo_entrenamiento = resultado.get("training_seconds", np.nan)
            tiempo_inferencia = resultado.get("inference_seconds", np.nan)
            n_test = resultado.get("n_test", np.nan)

            fila = {
                "Modelo": nombre_modelo,
                "Tipo": TIPOS_MODELO[nombre_modelo],
                "PR-AUC": numero(metricas, "pr_auc"),
                "ROC-AUC": numero(metricas, "roc_auc"),
                "Precision": numero(metricas, "precision"),
                "Recall": numero(metricas, "recall"),
                "F1-Score": numero(metricas, "f1"),
                "Umbral": numero(metricas, "threshold", 0.5),
                "Tiempo Entrenamiento (s)": (
                    float(tiempo_entrenamiento)
                    if tiempo_entrenamiento is not None
                    else np.nan
                ),
                "Tiempo Inferencia (s)": (
                    float(tiempo_inferencia)
                    if tiempo_inferencia is not None
                    else np.nan
                ),
                "N Test": int(n_test) if pd.notna(n_test) else np.nan,
                "Bloque de metricas": origen_metricas,
            }

            if (
                pd.notna(fila["Tiempo Inferencia (s)"])
                and pd.notna(fila["N Test"])
                and fila["N Test"] > 0
            ):
                fila["Inferencia por muestra (ms)"] = (
                    fila["Tiempo Inferencia (s)"] * 1000 / fila["N Test"]
                )
            else:
                fila["Inferencia por muestra (ms)"] = np.nan

            filas.append(fila)

            if all(clave in metricas for clave in ("tn", "fp", "fn", "tp")):
                matrices[nombre_modelo] = {
                    "TN": int(metricas["tn"]),
                    "FP": int(metricas["fp"]),
                    "FN": int(metricas["fn"]),
                    "TP": int(metricas["tp"]),
                }
            else:
                errores.append(
                    f"{nombre_modelo}: faltan TN, FP, FN o TP; "
                    "no se mostrara su matriz de confusion."
                )

        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            errores.append(f"{nombre_modelo}: {error}")

    datos = pd.DataFrame(filas)
    if not datos.empty:
        orden = ["Random Forest", "XGBoost", "Transformer", "LSTM"]
        datos["Modelo"] = pd.Categorical(
            datos["Modelo"], categories=orden, ordered=True
        )
        datos = datos.sort_values("Modelo").reset_index(drop=True)
        datos["Modelo"] = datos["Modelo"].astype(str)

    return datos, matrices, errores


def construir_firmas_archivos():
    """Permite refrescar la cache al reemplazar algun JSON."""
    firmas = []
    for nombre, ruta in ARCHIVOS_METRICAS.items():
        if ruta.exists():
            estado = ruta.stat()
            firmas.append((nombre, str(ruta), estado.st_mtime_ns, estado.st_size))
        else:
            firmas.append((nombre, str(ruta), None, None))
    return tuple(firmas)


datos_modelos, matrices, errores_carga = cargar_resultados_modelos(
    construir_firmas_archivos()
)

# Organizacion por pestanas
tab_global, tab_whatif, tab_simulador = st.tabs(
    [
        "📊 Monitoreo Global Ejecutivo",
        "🎛️ Optimizador What-If",
        "🧪 Simulador en Vivo",
    ]
)

with tab_global:
    st.subheader("Rendimiento Comparativo de Modelos")
    st.markdown(
        "Evaluacion de las arquitecturas transaccionales "
        "(Random Forest, XGBoost) y secuenciales "
        "(LSTM, Transformer) con las metricas exportadas por sus notebooks."
    )

    if errores_carga:
        with st.expander(
            f"Advertencias de carga ({len(errores_carga)})",
            expanded=datos_modelos.empty,
        ):
            for error in errores_carga:
                st.warning(error)

    if datos_modelos.empty:
        st.error(
            "No se pudo cargar ningun resultado. Verifica que las carpetas "
            "esten dentro de 'resultados' y que cada una contenga metricas.json."
        )
        st.code(
            "resultados/\n"
            "  random_forest/metricas.json\n"
            "  xgboost/metricas.json\n"
            "  transformer/metricas.json\n"
            "  lstm/metricas.json",
            language="text",
        )
    else:
        # KPIs calculados unicamente con filas que contienen el valor requerido.
        validos_pr = datos_modelos.dropna(subset=["PR-AUC"])
        validos_recall = datos_modelos.dropna(subset=["Recall"])
        validos_f1 = datos_modelos.dropna(subset=["F1-Score"])
        validos_tiempo = datos_modelos.dropna(subset=["Tiempo Inferencia (s)"])

        mejor_pr = (
            validos_pr.loc[validos_pr["PR-AUC"].idxmax()]
            if not validos_pr.empty
            else None
        )
        mejor_recall = (
            validos_recall.loc[validos_recall["Recall"].idxmax()]
            if not validos_recall.empty
            else None
        )
        mejor_f1 = (
            validos_f1.loc[validos_f1["F1-Score"].idxmax()]
            if not validos_f1.empty
            else None
        )
        modelo_rapido = (
            validos_tiempo.loc[validos_tiempo["Tiempo Inferencia (s)"].idxmin()]
            if not validos_tiempo.empty
            else None
        )

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(
            "🏆 Mejor PR-AUC",
            f"{mejor_pr['PR-AUC']:.4f}" if mejor_pr is not None else "N/D",
            mejor_pr["Modelo"] if mejor_pr is not None else None,
        )
        kpi2.metric(
            "⚡ Inferencia mas rapida",
            (
                f"{modelo_rapido['Tiempo Inferencia (s)']:.2f} s"
                if modelo_rapido is not None
                else "N/D"
            ),
            modelo_rapido["Modelo"] if modelo_rapido is not None else None,
            delta_color="off",
        )
        kpi3.metric(
            "🎯 Recall maximo",
            f"{mejor_recall['Recall']:.2%}" if mejor_recall is not None else "N/D",
            mejor_recall["Modelo"] if mejor_recall is not None else None,
        )
        kpi4.metric(
            "🛡️ Mejor F1-Score",
            f"{mejor_f1['F1-Score']:.4f}" if mejor_f1 is not None else "N/D",
            mejor_f1["Modelo"] if mejor_f1 is not None else None,
        )

        st.divider()
        col_izq, col_der = st.columns([3, 2])

        with col_izq:
            st.write("##### PR-AUC vs tiempo total de inferencia")
            datos_grafica = datos_modelos.dropna(
                subset=["PR-AUC", "Tiempo Inferencia (s)"]
            ).copy()
            datos_grafica = datos_grafica[datos_grafica["Tiempo Inferencia (s)"] > 0]

            if datos_grafica.empty:
                st.info(
                    "No hay tiempos de inferencia validos para construir la grafica."
                )
            else:
                fig = px.scatter(
                    datos_grafica,
                    x="Tiempo Inferencia (s)",
                    y="PR-AUC",
                    color="Tipo",
                    text="Modelo",
                    size="F1-Score",
                    size_max=28,
                    log_x=True,
                    hover_data={
                        "ROC-AUC": ":.4f",
                        "Precision": ":.4f",
                        "Recall": ":.4f",
                        "F1-Score": ":.4f",
                        "Umbral": ":.4f",
                        "N Test": True,
                    },
                    title="Trade-off: capacidad predictiva y latencia operativa",
                )
                fig.update_traces(textposition="top center")
                st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "El tiempo total solo es directamente comparable cuando los modelos "
                "se evaluaron sobre volumenes equivalentes. Revisa N Test y la latencia "
                "por muestra para una comparacion mas justa."
            )

        with col_der:
            st.write("##### Tabla resumen de metricas")
            columnas_tabla = [
                "Modelo",
                "Tipo",
                "PR-AUC",
                "ROC-AUC",
                "Precision",
                "Recall",
                "F1-Score",
                "Umbral",
                "Tiempo Inferencia (s)",
                "N Test",
                "Inferencia por muestra (ms)",
            ]
            tabla = datos_modelos[
                [c for c in columnas_tabla if c in datos_modelos.columns]
            ]
            formato = {
                "PR-AUC": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-Score": "{:.4f}",
                "Umbral": "{:.4f}",
                "Tiempo Inferencia (s)": "{:.2f}",
                "Inferencia por muestra (ms)": "{:.6f}",
            }
            estilo = tabla.style.format(formato, na_rep="N/D")
            estilo = estilo.highlight_max(
                subset=["PR-AUC", "ROC-AUC", "Recall", "F1-Score"],
                color="#d4edda",
            )
            estilo = estilo.highlight_min(
                subset=["Tiempo Inferencia (s)"],
                color="#d4edda",
            )
            st.dataframe(estilo, use_container_width=True, hide_index=True)

        st.divider()
        st.write("##### Diagnostico de errores por modelo")

        modelos_con_matriz = [
            modelo for modelo in datos_modelos["Modelo"].tolist() if modelo in matrices
        ]

        if not modelos_con_matriz:
            st.warning("Ningun archivo de metricas contiene TN, FP, FN y TP.")
        else:
            indice_lstm = (
                modelos_con_matriz.index("LSTM") if "LSTM" in modelos_con_matriz else 0
            )
            modelo_sel = st.selectbox(
                "Selecciona un modelo para auditar sus predicciones:",
                modelos_con_matriz,
                index=indice_lstm,
            )
            res = matrices[modelo_sel]
            m_col1, m_col2 = st.columns([1, 2])

            with m_col1:
                df_matriz = pd.DataFrame(
                    [[res["TN"], res["FP"]], [res["FN"], res["TP"]]],
                    columns=["Pred: Legitimo", "Pred: Fraude"],
                    index=["Real: Legitimo", "Real: Fraude"],
                )
                st.table(df_matriz.style.format("{:,}"))

            with m_col2:
                total_fraudes = res["FN"] + res["TP"]
                total_legitimas = res["FP"] + res["TN"]
                fn_ratio = res["FN"] / total_fraudes if total_fraudes else 0.0
                fp_ratio = res["FP"] / total_legitimas if total_legitimas else 0.0

                st.metric(
                    "Fraude no detectado (FN)",
                    f"{res['FN']:,} casos",
                    f"{fn_ratio:.2%} de los fraudes",
                    delta_color="inverse",
                )
                st.metric(
                    "Alertas erroneas (FP)",
                    f"{res['FP']:,} transacciones",
                    f"{fp_ratio:.2%} de las legitimas",
                    delta_color="inverse",
                )

with tab_whatif:
    st.subheader("🎛️ Optimizador de Umbral de Decision (Analisis What-If)")
    st.markdown(
        "Este modulo usa las probabilidades reales del conjunto de prueba para "
        "simular distintas politicas de decision. Al modificar el umbral, se "
        "actualizan la matriz de confusion, Precision, Recall, F1, la friccion "
        "con clientes y el costo operativo estimado."
    )

    st.info(
        "El modelo no cambia ni se reentrena en esta pantalla. Solo cambia la "
        "regla que transforma su probabilidad de fraude en una alerta operativa."
    )

    @st.cache_data(show_spinner="Cargando predicciones del modelo...")
    def cargar_predicciones_csv(ruta_texto, firma_archivo):
        del firma_archivo
        ruta = Path(ruta_texto)
        df = pd.read_csv(ruta)

        # Nombres compatibles con las exportaciones de los notebooks.
        candidatos_y = ("y_true", "y_test", "real", "etiqueta", "isFraud")
        candidatos_p = (
            "probabilidad_fraude",
            "y_prob",
            "proba",
            "probabilidad",
            "score",
            "prediction_probability",
        )
        col_y = next((c for c in candidatos_y if c in df.columns), None)
        col_p = next((c for c in candidatos_p if c in df.columns), None)

        if col_y is None or col_p is None:
            raise ValueError(
                "El CSV debe contener una columna real (por ejemplo, y_true) y "
                "una de probabilidad (por ejemplo, probabilidad_fraude). "
                f"Columnas encontradas: {list(df.columns)}"
            )

        limpio = df[[col_y, col_p]].copy()
        limpio.columns = ["y_true", "probabilidad_fraude"]
        limpio["y_true"] = pd.to_numeric(limpio["y_true"], errors="coerce")
        limpio["probabilidad_fraude"] = pd.to_numeric(
            limpio["probabilidad_fraude"], errors="coerce"
        )
        limpio = limpio.dropna()
        limpio = limpio[limpio["y_true"].isin([0, 1])]
        limpio["y_true"] = limpio["y_true"].astype(np.int8)
        limpio["probabilidad_fraude"] = (
            limpio["probabilidad_fraude"].clip(0.0, 1.0).astype(np.float32)
        )

        if limpio.empty:
            raise ValueError("No quedaron filas validas despues de limpiar el CSV.")
        if limpio["y_true"].nunique() < 2:
            raise ValueError("El CSV debe contener casos legitimos y fraudulentos.")
        return limpio

    modelos_predicciones = [
        modelo for modelo, ruta in ARCHIVOS_PREDICCIONES.items() if ruta.exists()
    ]

    if not modelos_predicciones:
        st.error(
            "No se encontro ningun archivo predicciones_test.csv. La segunda "
            "pestana no usa datos simulados, necesita las probabilidades reales."
        )
        st.code(
            "resultados/\n"
            "  random_forest/predicciones_test.csv\n"
            "  xgboost/predicciones_test.csv\n"
            "  transformer/predicciones_test.csv\n"
            "  lstm/predicciones_test.csv",
            language="text",
        )
        st.caption("Cada CSV debe incluir, como minimo, y_true y probabilidad_fraude.")
    else:
        modelo_whatif = st.selectbox(
            "Modelo para analizar:",
            modelos_predicciones,
            index=(
                modelos_predicciones.index("XGBoost")
                if "XGBoost" in modelos_predicciones
                else 0
            ),
            key="modelo_whatif",
            help=(
                "XGBoost representa la primera linea transaccional. LSTM y "
                "Transformer analizan secuencias, por lo que sus conteos pueden "
                "corresponder a otra unidad de analisis."
            ),
        )

        ruta_pred = ARCHIVOS_PREDICCIONES[modelo_whatif]
        estado_pred = ruta_pred.stat()

        try:
            df_pred = cargar_predicciones_csv(
                str(ruta_pred),
                (estado_pred.st_mtime_ns, estado_pred.st_size),
            )
        except Exception as error:
            st.error(f"No fue posible cargar {ruta_pred}: {error}")
            df_pred = None

        if df_pred is not None:
            # El umbral recomendado se toma del JSON del mismo modelo.
            fila_modelo = datos_modelos[datos_modelos["Modelo"] == modelo_whatif]
            umbral_recomendado = 0.5
            if not fila_modelo.empty and pd.notna(fila_modelo.iloc[0]["Umbral"]):
                umbral_recomendado = float(fila_modelo.iloc[0]["Umbral"])
            umbral_recomendado = min(max(umbral_recomendado, 0.0), 1.0)

            col_ctrl, col_contexto = st.columns([2, 3])
            with col_ctrl:
                usar_recomendado = st.toggle(
                    "Usar umbral optimizado en validacion",
                    value=True,
                    help=(
                        "Activa el umbral elegido previamente con validacion. "
                        "Desactivalo para comenzar desde 0.5."
                    ),
                )
                valor_inicial = umbral_recomendado if usar_recomendado else 0.5
                umbral_seleccionado = st.slider(
                    "Umbral de decision:",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(round(valor_inicial, 4)),
                    step=0.001,
                    format="%.3f",
                    help=(
                        "Una probabilidad igual o superior al umbral genera "
                        "una alerta de fraude."
                    ),
                )
                st.caption(
                    f"Umbral recomendado para {modelo_whatif}: "
                    f"**{umbral_recomendado:.4f}**"
                )
                st.caption(f"Fuente: {len(df_pred):,} predicciones reales de prueba.")

            y_true = df_pred["y_true"].to_numpy()
            probabilidades = df_pred["probabilidad_fraude"].to_numpy()
            y_pred = (probabilidades >= umbral_seleccionado).astype(np.int8)

            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            fnr = fn / (fn + tp) if (fn + tp) else 0.0
            tasa_alertas = (tp + fp) / len(y_true) if len(y_true) else 0.0

            with col_contexto:
                m1, m2, m3 = st.columns(3)
                m1.metric("Precision", f"{precision:.3f}")
                m2.metric("Recall", f"{recall:.3f}")
                m3.metric("F1-Score", f"{f1:.3f}")

                m4, m5, m6 = st.columns(3)
                m4.metric("FPR / Friccion", f"{fpr:.3%}")
                m5.metric("FNR / Fraude escapado", f"{fnr:.3%}")
                m6.metric("Tasa de alertas", f"{tasa_alertas:.3%}")

            st.divider()
            c_matriz, c_negocio = st.columns([2, 3])

            with c_matriz:
                st.write("##### Matriz de confusion dinamica")
                matriz = pd.DataFrame(
                    [[tn, fp], [fn, tp]],
                    columns=["Pred: Legitimo", "Pred: Fraude"],
                    index=["Real: Legitimo", "Real: Fraude"],
                )
                st.table(matriz.style.format("{:,}"))
                st.success(f"Fraudes detectados: **{tp:,}**")
                st.warning(
                    f"Friccion operativa: **{fp:,}** transacciones legitimas alertadas."
                )
                st.error(f"Fraude escapado: **{fn:,}** casos no detectados.")

            with c_negocio:
                st.write("##### Impacto de negocio")
                b1, b2 = st.columns(2)
                with b1:
                    costo_fraude = st.number_input(
                        "Costo promedio por fraude no detectado ($)",
                        min_value=0.0,
                        value=2500.0,
                        step=100.0,
                    )
                with b2:
                    costo_revision = st.number_input(
                        "Costo por revision de una alerta falsa ($)",
                        min_value=0.0,
                        value=35.0,
                        step=5.0,
                    )

                perdida_fn = fn * costo_fraude
                costo_fp = fp * costo_revision
                costo_total = perdida_fn + costo_fp

                df_costos = pd.DataFrame(
                    {
                        "Concepto": ["Fraude no detectado (FN)", "Revision falsa (FP)"],
                        "Costo estimado ($)": [perdida_fn, costo_fp],
                    }
                )
                fig_costos = px.bar(
                    df_costos,
                    x="Concepto",
                    y="Costo estimado ($)",
                    color="Concepto",
                    text="Costo estimado ($)",
                    color_discrete_sequence=["#d62728", "#ff9f1c"],
                )
                fig_costos.update_traces(
                    texttemplate="$%{text:,.0f}", textposition="outside"
                )
                fig_costos.update_layout(showlegend=False)
                st.plotly_chart(fig_costos, use_container_width=True)
                st.metric("Costo operativo total estimado", f"${costo_total:,.2f}")

            st.divider()
            st.write("##### Curva de sensibilidad de la politica de umbral")

            # 101 puntos proporcionan respuesta fluida incluso con millones de filas.
            umbrales = np.linspace(0.0, 1.0, 101)
            filas_sensibilidad = []
            positivos = int(np.sum(y_true == 1))
            negativos = int(np.sum(y_true == 0))

            for u in umbrales:
                pred_u = probabilidades >= u
                tp_u = int(np.sum(pred_u & (y_true == 1)))
                fp_u = int(np.sum(pred_u & (y_true == 0)))
                fn_u = positivos - tp_u
                tn_u = negativos - fp_u
                recall_u = tp_u / positivos if positivos else 0.0
                precision_u = tp_u / (tp_u + fp_u) if (tp_u + fp_u) else 0.0
                f1_u = (
                    2 * precision_u * recall_u / (precision_u + recall_u)
                    if (precision_u + recall_u)
                    else 0.0
                )
                fpr_u = fp_u / negativos if negativos else 0.0
                fnr_u = fn_u / positivos if positivos else 0.0
                costo_u = fn_u * costo_fraude + fp_u * costo_revision
                filas_sensibilidad.append(
                    {
                        "Umbral": u,
                        "Precision": precision_u,
                        "Recall": recall_u,
                        "F1": f1_u,
                        "FPR": fpr_u,
                        "FNR": fnr_u,
                        "Costo estimado ($)": costo_u,
                        "TN": tn_u,
                    }
                )

            sensibilidad = pd.DataFrame(filas_sensibilidad)
            metricas_largas = sensibilidad.melt(
                id_vars="Umbral",
                value_vars=["Precision", "Recall", "F1", "FPR", "FNR"],
                var_name="Metrica",
                value_name="Valor",
            )
            fig_sensibilidad = px.line(
                metricas_largas,
                x="Umbral",
                y="Valor",
                color="Metrica",
                title="Efecto del umbral sobre deteccion y friccion",
            )
            fig_sensibilidad.add_vline(
                x=umbral_seleccionado,
                line_dash="dash",
                line_color="black",
                annotation_text="Umbral seleccionado",
            )
            fig_sensibilidad.update_yaxes(tickformat=".0%", range=[0, 1])
            st.plotly_chart(fig_sensibilidad, use_container_width=True)

            mejor_costo = sensibilidad.loc[sensibilidad["Costo estimado ($)"].idxmin()]
            st.caption(
                "Escenario de referencia por costos ingresados: el menor costo "
                f"estimado aparece cerca del umbral {mejor_costo['Umbral']:.2f}. "
                "Este valor es exploratorio y no sustituye el umbral seleccionado "
                "con el conjunto de validacion."
            )

with tab_simulador:
    st.subheader("🧪 Simulador Transaccional con Persistencia")
    st.markdown(
        "Selecciona una cuenta existente de PaySim o crea un cliente simulado, "
        "registra una nueva operacion y evaluala con el modelo XGBoost. "
        "Las simulaciones se almacenan por separado y no modifican el dataset original."
    )

    BASE_DIR_SIM = Path(__file__).resolve().parent
    MODEL_PATH_SIM = BASE_DIR_SIM / "modelos" / "xgboost_paysim.json"
    METRICS_PATH_SIM = BASE_DIR_SIM / "resultados" / "xgboost" / "metricas.json"
    DB_PATH_SIM = BASE_DIR_SIM / "database" / "database" /"paysim.db"

    FEATURES_SIM = [
        "step", "amount", "hour", "day", "dest_is_merchant",
        "type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
        "type_PAYMENT", "type_TRANSFER",
    ]

    @st.cache_resource(show_spinner="Cargando XGBoost...")
    def cargar_modelo_simulador(ruta_texto, firma):
        del firma
        modelo = XGBClassifier()
        modelo.load_model(ruta_texto)
        return modelo

    @st.cache_data(show_spinner=False)
    def cargar_umbral_simulador(ruta_texto, firma):
        del firma
        ruta = Path(ruta_texto)
        if not ruta.exists():
            return 0.5
        with ruta.open("r", encoding="utf-8") as archivo:
            resultado = json.load(archivo)
        metricas = resultado.get(
            "metrics_threshold_validation",
            resultado.get("metricas_umbral_validacion", {}),
        )
        return float(metricas.get("threshold", 0.5))

    def conectar_db():
        conexion = sqlite3.connect(DB_PATH_SIM, timeout=30)
        conexion.execute("PRAGMA foreign_keys = ON")
        return conexion

    def preparar_tablas_simulacion():
        with conectar_db() as conexion:
            conexion.execute("""
                CREATE TABLE IF NOT EXISTS simulated_customers (
                    customer_id TEXT PRIMARY KEY,
                    display_name TEXT,
                    opening_balance REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1
                )
            """)
            conexion.execute("""
                CREATE TABLE IF NOT EXISTS simulated_transactions (
                    simulation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    origin_account TEXT NOT NULL,
                    destination_account TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    oldbalanceOrg REAL,
                    newbalanceOrig REAL,
                    oldbalanceDest REAL,
                    newbalanceDest REAL,
                    fraud_probability REAL NOT NULL,
                    operating_threshold REAL NOT NULL,
                    decision TEXT NOT NULL,
                    model_name TEXT NOT NULL DEFAULT 'XGBoost',
                    analyst_status TEXT NOT NULL DEFAULT 'PENDIENTE',
                    analyst_comment TEXT
                )
            """)
            conexion.execute(
                "CREATE INDEX IF NOT EXISTS idx_sim_origin "
                "ON simulated_transactions(origin_account)"
            )
            conexion.execute(
                "CREATE INDEX IF NOT EXISTS idx_sim_created "
                "ON simulated_transactions(created_at)"
            )

            try:
                conexion.execute("ALTER TABLE simulated_transactions ADD COLUMN analyst_name TEXT")
                conexion.execute("ALTER TABLE simulated_transactions ADD COLUMN resolution_date TEXT")
            except sqlite3.OperationalError:
                pass # Las columnas ya existen

            conexion.commit()

    def registrar_resolucion(sim_id, estado, comentario, responsable):
        with conectar_db() as conexion:
            conexion.execute(
                """
                UPDATE simulated_transactions 
                SET analyst_status = ?, 
                    analyst_comment = ?, 
                    analyst_name = ?, 
                    resolution_date = ?
                WHERE simulation_id = ?
                """,
                (estado, comentario, responsable, datetime.now().isoformat(timespec="seconds"), sim_id)
            )
            conexion.commit()

    def transformar_operacion(step, tipo, monto, cuenta_destino):
        fila = {
            "step": int(step),
            "amount": float(monto),
            "hour": int((step - 1) % 24),
            "day": int((step - 1) // 24),
            "dest_is_merchant": int(str(cuenta_destino).upper().startswith("M")),
            "type_CASH_IN": int(tipo == "CASH_IN"),
            "type_CASH_OUT": int(tipo == "CASH_OUT"),
            "type_DEBIT": int(tipo == "DEBIT"),
            "type_PAYMENT": int(tipo == "PAYMENT"),
            "type_TRANSFER": int(tipo == "TRANSFER"),
        }
        return pd.DataFrame([fila], columns=FEATURES_SIM)

    def buscar_cuentas_paysim(texto, limite=40):
        patron = f"%{texto.strip()}%"
        with conectar_db() as conexion:
            return pd.read_sql_query(
                """
                SELECT nameOrig AS account_id,
                       COUNT(*) AS operations,
                       MAX(step) AS last_step
                FROM transactions
                WHERE nameOrig LIKE ?
                GROUP BY nameOrig
                ORDER BY operations DESC, account_id
                LIMIT ?
                """,
                conexion,
                params=(patron, limite),
            )

    def listar_clientes_simulados():
        with conectar_db() as conexion:
            return pd.read_sql_query(
                """
                SELECT customer_id, display_name, opening_balance, created_at
                FROM simulated_customers
                WHERE active = 1
                ORDER BY created_at DESC
                """,
                conexion,
            )

    def ultimo_estado_cuenta(cuenta):
        cuenta = str(cuenta).strip()
        with conectar_db() as conexion:
            sim = pd.read_sql_query(
                """
                SELECT newbalanceOrig AS balance, step
                FROM simulated_transactions
                WHERE origin_account = ?
                ORDER BY simulation_id DESC
                LIMIT 1
                """,
                conexion,
                params=(cuenta,),
            )
            if not sim.empty:
                return float(sim.iloc[0]["balance"] or 0), int(sim.iloc[0]["step"])

            historico = pd.read_sql_query(
                """
                SELECT newbalanceOrig AS balance, step
                FROM transactions
                WHERE nameOrig = ?
                ORDER BY step DESC, rowid DESC
                LIMIT 1
                """,
                conexion,
                params=(cuenta,),
            )
            if not historico.empty:
                return float(historico.iloc[0]["balance"] or 0), int(historico.iloc[0]["step"])

            cliente = pd.read_sql_query(
                """
                SELECT opening_balance AS balance
                FROM simulated_customers
                WHERE customer_id = ?
                """,
                conexion,
                params=(cuenta,),
            )
            if not cliente.empty:
                return float(cliente.iloc[0]["balance"] or 0), 0
            return 0.0, 0

    def historial_cuenta(cuenta, limite=20):
        cuenta = str(cuenta).strip()
        with conectar_db() as conexion:
            historico = pd.read_sql_query(
                """
                SELECT 'PaySim' AS source, rowid AS operation_id, step, type, amount,
                       nameDest AS destination, oldbalanceOrg, newbalanceOrig,
                       NULL AS fraud_probability, NULL AS decision, NULL AS created_at
                FROM transactions
                WHERE nameOrig = ?
                ORDER BY step DESC, rowid DESC
                LIMIT ?
                """,
                conexion,
                params=(cuenta, limite),
            )
            simuladas = pd.read_sql_query(
                """
                SELECT 'Simulada' AS source, simulation_id AS operation_id, step,
                       type, amount, destination_account AS destination,
                       oldbalanceOrg, newbalanceOrig, fraud_probability,
                       decision, created_at
                FROM simulated_transactions
                WHERE origin_account = ?
                ORDER BY simulation_id DESC
                LIMIT ?
                """,
                conexion,
                params=(cuenta, limite),
            )
        combinado = pd.concat([simuladas, historico], ignore_index=True)
        return combinado.head(limite)

    def saldo_actual_destino(cuenta):
        """Obtiene el ultimo saldo conocido del destino, sea PaySim o simulado."""
        cuenta = str(cuenta).strip().upper()
        if not cuenta:
            return 0.0
        with conectar_db() as conexion:
            sim_destino = pd.read_sql_query(
                """
                SELECT newbalanceDest AS balance
                FROM simulated_transactions
                WHERE destination_account = ?
                ORDER BY simulation_id DESC
                LIMIT 1
                """,
                conexion,
                params=(cuenta,),
            )
            if not sim_destino.empty:
                return float(sim_destino.iloc[0]["balance"] or 0)

            sim_origen = pd.read_sql_query(
                """
                SELECT newbalanceOrig AS balance
                FROM simulated_transactions
                WHERE origin_account = ?
                ORDER BY simulation_id DESC
                LIMIT 1
                """,
                conexion,
                params=(cuenta,),
            )
            if not sim_origen.empty:
                return float(sim_origen.iloc[0]["balance"] or 0)

            historico_destino = pd.read_sql_query(
                """
                SELECT newbalanceDest AS balance
                FROM transactions
                WHERE nameDest = ?
                ORDER BY step DESC, rowid DESC
                LIMIT 1
                """,
                conexion,
                params=(cuenta,),
            )
            if not historico_destino.empty:
                return float(historico_destino.iloc[0]["balance"] or 0)

            historico_origen = pd.read_sql_query(
                """
                SELECT newbalanceOrig AS balance
                FROM transactions
                WHERE nameOrig = ?
                ORDER BY step DESC, rowid DESC
                LIMIT 1
                """,
                conexion,
                params=(cuenta,),
            )
            if not historico_origen.empty:
                return float(historico_origen.iloc[0]["balance"] or 0)

            cliente = pd.read_sql_query(
                """
                SELECT opening_balance AS balance
                FROM simulated_customers
                WHERE customer_id = ?
                """,
                conexion,
                params=(cuenta,),
            )
            if not cliente.empty:
                return float(cliente.iloc[0]["balance"] or 0)
        return 0.0

    def guardar_cliente(cliente_id, nombre, saldo):
        cliente_id = cliente_id.strip().upper()
        if not cliente_id:
            raise ValueError("El identificador del cliente es obligatorio.")
        if not cliente_id.startswith("C"):
            raise ValueError("Usa un identificador que comience con C.")
        with conectar_db() as conexion:
            conexion.execute(
                """
                INSERT INTO simulated_customers(
                    customer_id, display_name, opening_balance, created_at, active
                ) VALUES (?, ?, ?, ?, 1)
                """,
                (
                    cliente_id,
                    nombre.strip() or None,
                    float(saldo),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            conexion.commit()

    def guardar_simulacion(datos):
        with conectar_db() as conexion:
            cursor = conexion.execute(
                """
                INSERT INTO simulated_transactions(
                    created_at, origin_account, destination_account, step, type,
                    amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest,
                    newbalanceDest, fraud_probability, operating_threshold,
                    decision, model_name, analyst_status, analyst_comment
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'XGBoost',
                          'PENDIENTE', NULL)
                """,
                datos,
            )
            conexion.commit()
            return int(cursor.lastrowid)

    def eliminar_ultima_simulacion():
        with conectar_db() as conexion:
            fila = conexion.execute(
                """
                SELECT simulation_id FROM simulated_transactions
                ORDER BY simulation_id DESC LIMIT 1
                """
            ).fetchone()
            if fila is None:
                return None
            conexion.execute(
                "DELETE FROM simulated_transactions WHERE simulation_id = ?",
                (fila[0],),
            )
            conexion.commit()
            return int(fila[0])

    def ultimas_simulaciones(limite=20):
        with conectar_db() as conexion:
            return pd.read_sql_query(
                """
                SELECT simulation_id, created_at, origin_account,
                       destination_account, step, type, amount,
                       fraud_probability, operating_threshold, decision,
                       analyst_status
                FROM simulated_transactions
                ORDER BY simulation_id DESC
                LIMIT ?
                """,
                conexion,
                params=(limite,),
            )

    if not DB_PATH_SIM.exists():
        st.error(
            "No existe database/paysim.db. Ejecuta crear_db_paysim.py antes "
            "de usar el simulador persistente."
        )
    elif not MODEL_PATH_SIM.exists():
        st.error("No existe modelos/xgboost_paysim.json.")
    else:
        try:
            preparar_tablas_simulacion()
            estado_modelo = MODEL_PATH_SIM.stat()
            modelo_sim = cargar_modelo_simulador(
                str(MODEL_PATH_SIM),
                (estado_modelo.st_mtime_ns, estado_modelo.st_size),
            )
            if METRICS_PATH_SIM.exists():
                estado_metricas = METRICS_PATH_SIM.stat()
                umbral_validado_sim = cargar_umbral_simulador(
                    str(METRICS_PATH_SIM),
                    (estado_metricas.st_mtime_ns, estado_metricas.st_size),
                )
            else:
                umbral_validado_sim = 0.5
        except Exception as error:
            st.error(f"No fue posible preparar el simulador: {error}")
            modelo_sim = None

        if modelo_sim is not None:
            seleccion_tab, nuevo_tab = st.tabs([
                "Seleccionar cuenta", "Añadir cliente simulado"
            ])

            with nuevo_tab:
                st.write("##### Alta de cliente simulado")
                st.caption(
                    "El cliente nuevo se guarda en una tabla separada y no altera PaySim."
                )
                with st.form("form_nuevo_cliente", clear_on_submit=True):
                    nc1, nc2, nc3 = st.columns(3)
                    with nc1:
                        nuevo_id = st.text_input(
                            "Identificador de cuenta",
                            placeholder="C_SIM_0001",
                        )
                    with nc2:
                        nuevo_nombre = st.text_input(
                            "Alias o nombre descriptivo",
                            placeholder="Cliente de prueba",
                        )
                    with nc3:
                        nuevo_saldo = st.number_input(
                            "Saldo inicial ($)", min_value=0.0, value=100000.0
                        )
                    crear_cliente = st.form_submit_button("Añadir cliente")
                if crear_cliente:
                    try:
                        guardar_cliente(nuevo_id, nuevo_nombre, nuevo_saldo)
                        st.success(f"Cliente {nuevo_id.strip().upper()} creado.")
                        st.cache_data.clear()
                    except sqlite3.IntegrityError:
                        st.error("Ese identificador ya existe.")
                    except Exception as error:
                        st.error(str(error))

                clientes_sim = listar_clientes_simulados()
                if not clientes_sim.empty:
                    st.dataframe(clientes_sim, use_container_width=True, hide_index=True)

            with seleccion_tab:
                origen_cliente = st.radio(
                    "Origen de la cuenta:",
                    ["Cuenta existente de PaySim", "Cliente simulado"],
                    horizontal=True,
                )

                cuenta_seleccionada = None
                if origen_cliente == "Cuenta existente de PaySim":
                    texto_busqueda = st.text_input(
                        "Buscar cuenta de origen",
                        placeholder="Escribe parte de un identificador, por ejemplo C123",
                    )
                    if len(texto_busqueda.strip()) >= 2:
                        coincidencias = buscar_cuentas_paysim(texto_busqueda)
                        if coincidencias.empty:
                            st.warning("No se encontraron cuentas.")
                        else:
                            opciones = coincidencias["account_id"].tolist()
                            cuenta_seleccionada = st.selectbox(
                                "Cuenta encontrada:", opciones
                            )
                            st.dataframe(
                                coincidencias,
                                use_container_width=True,
                                hide_index=True,
                            )
                    else:
                        st.caption("Escribe al menos dos caracteres para buscar.")
                else:
                    clientes = listar_clientes_simulados()
                    if clientes.empty:
                        st.warning("Primero añade un cliente simulado.")
                    else:
                        etiquetas = {
                            row["customer_id"]: (
                                f"{row['customer_id']} | "
                                f"{row['display_name'] or 'Sin alias'}"
                            )
                            for _, row in clientes.iterrows()
                        }
                        cuenta_seleccionada = st.selectbox(
                            "Cliente simulado:",
                            list(etiquetas),
                            format_func=lambda x: etiquetas[x],
                        )

                if cuenta_seleccionada:
                    saldo_actual, ultimo_step = ultimo_estado_cuenta(
                        cuenta_seleccionada
                    )
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Cliente / cuenta", cuenta_seleccionada)
                    k2.metric("Saldo disponible", f"${saldo_actual:,.2f}")
                    k3.metric("Operaciones simuladas", f"{int((historial_cuenta(cuenta_seleccionada)['source'] == 'Simulada').sum()) if not historial_cuenta(cuenta_seleccionada).empty else 0}")

                    with st.expander("Historial reciente de la cuenta", expanded=True):
                        historial = historial_cuenta(cuenta_seleccionada)
                        if historial.empty:
                            st.info("La cuenta aun no tiene operaciones.")
                        else:
                            st.dataframe(
                                historial,
                                use_container_width=True,
                                hide_index=True,
                            )

                    st.divider()
                    st.write("##### Nueva transaccion")
                    st.caption(
                        "El sistema obtiene automaticamente el saldo disponible, "
                        "el saldo conocido del destinatario y el momento temporal interno."
                    )
                    with st.form("form_operacion_persistente"):
                        f1, f2 = st.columns(2)
                        with f1:
                            tipo_nuevo = st.selectbox(
                                "Tipo de operacion",
                                ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
                                format_func=lambda x: {
                                    "TRANSFER": "Transferencia",
                                    "CASH_OUT": "Retiro de efectivo",
                                    "PAYMENT": "Pago",
                                    "CASH_IN": "Deposito / ingreso",
                                    "DEBIT": "Cargo por debito",
                                }[x],
                            )
                            monto_nuevo = st.number_input(
                                "Monto de la operacion ($)",
                                min_value=0.01,
                                value=1000.0,
                                step=100.0,
                            )
                        with f2:
                            destino_nuevo = st.text_input(
                                "Cuenta beneficiaria o destino",
                                value="C_DEST_SIM_001",
                                help=(
                                    "Es la cuenta que recibe el dinero. Puede ser una "
                                    "cuenta PaySim, un cliente simulado o un comercio. "
                                    "Los comercios de PaySim comienzan con M."
                                ),
                            )
                            usar_umbral_validado_sim = st.checkbox(
                                "Usar politica de riesgo validada",
                                value=True,
                                help=(
                                    "Aplica automaticamente el umbral seleccionado "
                                    "durante la validacion del modelo."
                                ),
                            )
                            umbral_operativo_sim = (
                                float(umbral_validado_sim)
                                if usar_umbral_validado_sim else 0.5
                            )

                        st.info(
                            f"Saldo disponible de {cuenta_seleccionada}: "
                            f"${saldo_actual:,.2f}"
                        )
                        confirmar_guardado = st.checkbox(
                            "Confirmo los datos de la transaccion"
                        )
                        evaluar_guardar = st.form_submit_button(
                            "Evaluar y registrar transaccion",
                            use_container_width=True,
                        )

                    if evaluar_guardar:
                        destino_limpio = destino_nuevo.strip().upper()
                        step_nuevo = max(int(ultimo_step) + 1, 1)
                        saldo_destino_anterior = saldo_actual_destino(destino_limpio)

                        if not confirmar_guardado:
                            st.warning("Confirma los datos antes de registrar.")
                        elif not destino_limpio:
                            st.error("La cuenta beneficiaria es obligatoria.")
                        elif destino_limpio == cuenta_seleccionada.upper():
                            st.error("La cuenta origen y la cuenta destino no pueden ser iguales.")
                        elif tipo_nuevo != "CASH_IN" and monto_nuevo > saldo_actual:
                            st.error(
                                "Fondos insuficientes. El monto supera el saldo disponible."
                            )
                        else:
                            if tipo_nuevo == "CASH_IN":
                                saldo_origen_nuevo = saldo_actual + monto_nuevo
                                saldo_destino_nuevo = saldo_destino_anterior
                            else:
                                saldo_origen_nuevo = max(saldo_actual - monto_nuevo, 0.0)
                                saldo_destino_nuevo = (
                                    saldo_destino_anterior + monto_nuevo
                                    if tipo_nuevo in ("TRANSFER", "CASH_OUT", "PAYMENT")
                                    else saldo_destino_anterior
                                )

                            entrada_modelo = transformar_operacion(
                                step_nuevo,
                                tipo_nuevo,
                                monto_nuevo,
                                destino_limpio,
                            )
                            
                            score = float(modelo_sim.predict_proba(entrada_modelo)[0, 1])
                            
                            # --- LÓGICA DE TRES BANDAS DE RIESGO ---
                            umbral_bloqueo = 0.85
                            
                            if score >= umbral_bloqueo:
                                decision = "BLOQUEO_AUTOMATICO"
                            elif score >= umbral_operativo_sim:
                                decision = "REVISION"
                            else:
                                decision = "SIN_ALERTA"
                                
                            registro_id = guardar_simulacion((
                                datetime.now().isoformat(timespec="seconds"),
                                cuenta_seleccionada,
                                destino_limpio,
                                int(step_nuevo),
                                tipo_nuevo,
                                float(monto_nuevo),
                                float(saldo_actual),
                                float(saldo_origen_nuevo),
                                float(saldo_destino_anterior),
                                float(saldo_destino_nuevo),
                                float(score),
                                float(umbral_operativo_sim),
                                decision,
                            ))
                            
                            st.session_state["ultima_simulacion_id"] = registro_id
                            st.session_state["ultimo_score"] = score
                            st.session_state["ultima_decision"] = decision
                            st.session_state["ultimo_saldo_antes"] = saldo_actual
                            st.session_state["ultimo_saldo_despues"] = saldo_origen_nuevo
                            st.session_state["ultimo_destino"] = destino_limpio
                            st.cache_data.clear()
                            
                    if "ultima_simulacion_id" in st.session_state:
                        score = st.session_state["ultimo_score"]
                        decision = st.session_state["ultima_decision"]
                        
                        # --- FEEDBACK VISUAL DINÁMICO ---
                        if decision == "BLOQUEO_AUTOMATICO":
                            st.error(
                                f"🚨 Transaccion #{st.session_state['ultima_simulacion_id']} "
                                f"RECHAZADA AUTOMATICAMENTE. Riesgo extremo: {score:.2%}"
                            )
                        elif decision == "REVISION":
                            st.warning(
                                f"⚠️ Transaccion #{st.session_state['ultima_simulacion_id']} "
                                f"RETENIDA PARA AUDITORIA. Riesgo moderado: {score:.2%}"
                            )
                        else:
                            st.success(
                                f"✅ Transaccion #{st.session_state['ultima_simulacion_id']} "
                                f"registrada sin alerta. Riesgo: {score:.2%}"
                            )
                            
                        r1, r2, r3 = st.columns(3)
                        r1.metric(
                            "Saldo anterior",
                            f"${st.session_state.get('ultimo_saldo_antes', 0):,.2f}",
                        )
                        r2.metric(
                            "Saldo posterior",
                            f"${st.session_state.get('ultimo_saldo_despues', 0):,.2f}",
                        )
                        r3.metric(
                            "Cuenta destino",
                            st.session_state.get("ultimo_destino", "N/D"),
                        )

            st.divider()
            st.write("##### Bitacora de simulaciones")
            simulaciones = ultimas_simulaciones()
            if simulaciones.empty:
                st.info("Todavia no hay transacciones simuladas.")
            else:
                st.dataframe(simulaciones, use_container_width=True, hide_index=True)
                st.warning(
                    "Eliminar solo afecta la ultima simulacion. El historico original "
                    "de PaySim nunca se modifica."
                )
                confirmar_eliminar = st.checkbox(
                    "Confirmo que deseo eliminar la ultima simulacion",
                    key="confirmar_eliminar_ultima",
                )
                if st.button(
                    "🗑️ Eliminar ultima simulacion",
                    disabled=not confirmar_eliminar,
                ):
                    eliminada = eliminar_ultima_simulacion()
                    if eliminada is None:
                        st.info("No habia simulaciones para eliminar.")
                    else:
                        for clave in (
                            "ultima_simulacion_id", "ultimo_score", "ultima_decision",
                            "ultimo_saldo_antes", "ultimo_saldo_despues", "ultimo_destino"
                        ):
                            st.session_state.pop(clave, None)
                        st.cache_data.clear()
                        st.success(f"Simulacion #{eliminada} eliminada.")
                        st.rerun()

            st.divider()
            st.write("##### 🧑‍💻 Auditoría Manual de Alertas (Human-in-the-Loop)")
            st.markdown("Gestión de operaciones retenidas por el modelo XGBoost para retroalimentación y auditoría.")

            with conectar_db() as conexion:
                alertas = pd.read_sql_query(
                    """
                    SELECT simulation_id, origin_account, destination_account, amount, fraud_probability 
                    FROM simulated_transactions 
                    WHERE analyst_status IN ('PENDIENTE', 'REVISIÓN') 
                      AND decision = 'REVISION' 
                    ORDER BY simulation_id DESC
                    """, 
                    conexion
                )

            if alertas.empty:
                st.success("🎉 Bandeja limpia: No hay alertas pendientes de resolución manual.")
            else:
                # Formatear opciones para el selector
                opciones_alerta = alertas.apply(
                    lambda row: f"OP-{int(row['simulation_id'])} | Riesgo: {row['fraud_probability']:.1%} | Monto: ${row['amount']:,.2f}", 
                    axis=1
                ).tolist()
                
                alerta_seleccionada = st.selectbox("Selecciona una alerta para auditar:", opciones_alerta)
                sim_id_str = int(alerta_seleccionada.split(" |")[0].replace("OP-", ""))
                
                with st.form("form_resolucion_humana", clear_on_submit=True):
                    c_res1, c_res2 = st.columns(2)
                    with c_res1:
                        nuevo_estado = st.selectbox(
                            "Veredicto del Analista:", 
                            [
                                "CONFIRMADO FRAUDE", 
                                "OPERACIÓN LEGÍTIMA", 
                                "FALSO POSITIVO (Bloqueo Erróneo)", 
                                "FALSO NEGATIVO (Fraude Escapado)", 
                                "MANTENER EN REVISIÓN"
                            ]
                        )
                        responsable = st.text_input("Analista Responsable (Matrícula/Nombre)")
                    with c_res2:
                        comentario = st.text_area("Justificación de la auditoría", height=110)
                    
                    if st.form_submit_button("💾 Guardar Resolución en Bitácora"):
                        if not responsable.strip():
                            st.error("Es obligatorio registrar el nombre o ID del responsable.")
                        else:
                            registrar_resolucion(sim_id_str, nuevo_estado, comentario, responsable)
                            st.success(f"Resolución registrada exitosamente para la operación #{sim_id_str}")
                            st.cache_data.clear()
                            st.rerun()

