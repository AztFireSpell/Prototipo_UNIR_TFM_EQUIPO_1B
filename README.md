# Sistema de Detección Temprana de Fraudes Bancarios

Prototipo interactivo desarrollado con **Python** y **Streamlit** para visualizar el desempeño de modelos de detección de fraude, analizar distintos umbrales de decisión y simular nuevas transacciones sobre un histórico basado en PaySim.

El archivo principal de la aplicación es:

```text
MODULO_GLOBAL_INTERFAZ_BANCO.py
```

## Funcionalidades principales

La aplicación contiene tres módulos:

1. **Monitoreo Global Ejecutivo**
   - Compara Random Forest, XGBoost, Transformer y LSTM.
   - Presenta PR-AUC, ROC-AUC, Precision, Recall, F1 y tiempos de inferencia.
   - Muestra matrices de confusión y errores por modelo.

2. **Optimizador What-If**
   - Utiliza las probabilidades reales exportadas por los modelos.
   - Permite modificar el umbral de decisión.
   - Recalcula Precision, Recall, F1, falsos positivos y falsos negativos.
   - Estima el impacto operativo y financiero de cada política de umbral.

3. **Simulador en Vivo**
   - Permite buscar cuentas existentes en PaySim.
   - Permite crear clientes simulados.
   - Registra nuevas transacciones sin modificar el conjunto PaySim original.
   - Ejecuta el modelo XGBoost entrenado como primera línea de evaluación.
   - Guarda el score, el umbral aplicado y la recomendación operativa.
   - Permite corregir una captura eliminando la última simulación.

> La aplicación es un prototipo académico. Sus resultados no deben utilizarse para tomar decisiones financieras reales.

---

## 1. Requisitos previos

Antes de comenzar, se necesita:

- Windows, Linux o macOS.
- Python 3.10 recomendado.
- pip o Conda.
- Espacio suficiente para el CSV de PaySim y la base SQLite generada.
- Los artefactos de modelos y resultados incluidos en el proyecto.

Para comprobar Python:

```bash
python --version
```

La salida debería ser similar a:

```text
Python 3.10.x
```

---

## 2. Descargar el conjunto de datos PaySim

Descargue el conjunto **Synthetic Financial Datasets For Fraud Detection** desde Kaggle:

https://www.kaggle.com/datasets/mtalaltariq/paysim-data

Kaggle puede solicitar iniciar sesión antes de permitir la descarga.

Después de descargar y descomprimir el archivo, identifique el CSV de PaySim. Dependiendo de la versión descargada, el archivo puede tener un nombre como:

```text
PS_20174392719_1491204439457_log.csv
```

Puede conservar ese nombre o renombrarlo como:

```text
paysim.csv
```

Coloque el CSV en la carpeta database despues ejecute el script crear_db_paysim.py, debera tener estos archivos:

```text
database/
├── paysim.csv
├── crear_db_paysim.py
```

---

## 3. Estructura esperada del proyecto

```text
carpeta_donde_hizo_el_git_clone/
├── MODULO_GLOBAL_INTERFAZ_BANCO.py
├── motor_inferencia.py
├── requirements.txt
├── paysim.csv
│
├── modelos/
│   ├── xgboost_paysim.json
│   └── mejor_modelo_lstm.keras
│
├── resultados/
│   ├── random_forest/
│   │   ├── metricas.json
│   │   └── predicciones_test.csv
│   ├── xgboost/
│   │   ├── metricas.json
│   │   ├── predicciones_test.csv
│   │   └── importancia.csv
│   ├── transformer/
│   │   ├── metricas.json
│   │   └── predicciones_test.csv
│   └── lstm/
│       ├── metricas.json
│       └── predicciones_test.csv
│
└── database/
    ├── paysim.db
    ├── crear_db_paysim.py
    └── paysim.csv
```
---

## 4. Crear y activar el entorno virtual

### Opción A: Conda

```bash
conda create -n prototipo_fraude python=3.10 -y
conda activate prototipo_fraude
```

### Opción B: venv

En Windows:

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
```

En Linux o macOS:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Actualice pip:

```bash
python -m pip install --upgrade pip
```

---

## 5. Instalar las dependencias

Con el entorno virtual activado, instale los paquetes indicados en `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Compruebe que no existan conflictos:

```bash
python -m pip check
```

Si el proyecto utiliza TensorFlow 2.10, se recomienda mantener versiones compatibles en `requirements.txt`, especialmente para TensorFlow, protobuf y NumPy.

Una configuración compatible de referencia es:

```text
tensorflow==2.21.0
protobuf==3.19.6
numpy==1.26.4
streamlit
pandas
plotly
scikit-learn
xgboost
joblib
```

> Si Streamlit y TensorFlow presentan restricciones incompatibles de `protobuf`, utilice entornos separados para entrenamiento y aplicación, o conserve las versiones que ya hayan sido verificadas en el equipo de presentación.

---

## 6. Convertir PaySim a una base SQLite

La aplicación consulta PaySim mediante SQLite para evitar cargar millones de filas completas en la memoria RAM.

Con el entorno activado y desde la raíz del proyecto, ejecute:

```bash
python crear_db_paysim.py --csv paysim.csv
```

Si el CSV conserva su nombre original:

```bash
python crear_db_paysim.py --csv "PS_20174392719_1491204439457_log.csv"
```

El resultado se guardará en:

```text
database/paysim.db
```

Durante la importación se mostrarán mensajes similares a:

```text
Bloque 1: 100,000 filas
Bloque 2: 200,000 filas
...
Base creada: database/paysim.db
```

La importación se ejecuta por bloques para limitar el consumo de memoria.

### Recrear la base

Si `database/paysim.db` ya existe y desea generarla nuevamente:

```bash
python crear_db_paysim.py --csv paysim.csv --reemplazar
```

> No interrumpa el proceso mientras se están creando los índices finales de SQLite.

---

## 7. Ejecutar la aplicación

Desde la carpeta raíz del proyecto y con el entorno activado:

```bash
python -m streamlit run MODULO_GLOBAL_INTERFAZ_BANCO.py
```

Streamlit mostrará una dirección local, normalmente:

```text
http://localhost:8501
```

Si el navegador no se abre automáticamente, copie esa dirección y ábrala manualmente.

Para detener la aplicación, presione:

```text
Ctrl + C
```

---

## Autoría

Proyecto académico de detección temprana de fraudes bancarios desarrollado como prototipo analítico con Python, Streamlit, SQLite y modelos de aprendizaje automático.
