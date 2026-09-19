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

Antes de comenzar, se necesita contar con lo siguiente:

- Windows, Linux o macOS.
- Python 3.10 recomendado.
- pip o Conda.
- Espacio suficiente para el CSV de PaySim y la base SQLite generada.
- Los artefactos de modelos y resultados incluidos en el proyecto.

1. Descargar el proyecto

Puedes obtener el proyecto de cualquiera de las siguientes formas:

Opción 1: Clonar el repositorio con Git

Si tienes Git instalado, ejecuta:

```bash
git clone https://github.com/AztFireSpell/Prototipo_UNIR_TFM_EQUIPO_1B.git
```

Después, entra a la carpeta del proyecto:

```bash
cd Prototipo_UNIR_TFM_EQUIPO_1B
```


Opción 2: Descargar el proyecto

También puedes descargar directamente el proyecto utilizando el botón verde Code que se encuentra en la parte superior del repositorio y seleccionar Download ZIP.

Una vez descargado, descomprime el archivo y abre una terminal dentro de la carpeta del proyecto.


2. Comprobar la instalación de Python

Para comprobar que Python está instalado y verificar su versión, ejecuta:


```bash
python --version
```

La salida debería ser similar a:

```text
Python 3.10.x
```

---

3. Crear un entorno virtual

Para evitar conflictos entre las dependencias del proyecto y otras instalaciones de Python, se recomienda utilizar un entorno virtual.

Opción A: utilizando venv

Crea el entorno virtual:

```bash
python -m venv .venv
```

Activa el entorno:

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Una vez activado, puedes comprobar que el entorno está funcionando correctamente:

python --version

Una vez activado el entorno recuerda instalar la dependencia de librerias con el comando:


```bash
pip install -r requerimentos.txt
```


Opción B: utilizando Conda

Si utilizas Conda, se recomienda crear un entorno nuevo para este proyecto:

```bash
conda create -n paysim python=3.10
```

Activa el entorno con:

```bash
conda activate paysim
```

Después, comprueba la versión de Python:

```bash
python --version
```
La salida debería ser similar a:

```text
Python 3.10.x
```

Una vez activado el entorno recuerda instalar la dependencia de librerias con el comando:


```bash
pip install -r requerimentos.txt
```

## 4. Descargar el conjunto de datos PaySim

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

---

## 5. Convertir PaySim a una base SQLite

La aplicación consulta PaySim mediante SQLite para evitar cargar millones de filas completas en la memoria RAM.

Con el entorno activado y desde la carpeta database del proyecto, conjunto al archivo de paysim.csv ejecute:

```bash
python crear_db_paysim.py --csv paysim.csv
```

Si el CSV conserva su nombre original:

```bash
python crear_db_paysim.py --csv "PS_20174392719_1491204439457_log.csv"
```

En dado caso de que el nombre sea diferente a estos 2, deberas usar el nombre de tu CSV

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

## 6. Ejecutar la aplicación

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
