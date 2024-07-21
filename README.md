# Artificial Intelligence based models to improve quality classification of steel coils in automotive industry

## Proyecto de Fin de Grado de Alejandro Álvarez Castro

Este repositorio contiene el código realizado para el proyecto de fin de grado titulado "Artificial Intelligence based models to improve quality classification of steel coils in automotive industry", donde se estudian varias aproximaciones de machine learning para resolver un problema concreto en el sector industrial. El proyecto está dividido en tres apartados principales:

## Estructura del Repositorio

- **A) Deep Learning**
- **B) Deep Transfer Learning**
- **C) Contrastive Learning**

### A) Deep Learning

En esta sección se prueban y construyen varios modelos de deep learning y posteriormente se estudian sus resultados.

#### Archivos Principales:

- `Builder.py`: Clase encargada de construir los modelos de deep learning en base a especificaciones concretas.
- `CreateModel.py`: Clase para crear los modelos (llamando al constructor) con los tags y metadata incluidos.
- `Data.py`: Clase para cargar los datos necesarios.
- `DataProcessor.py`: Código para procesar los datos leídos en el yaml para adaptarlos antes de pasarselos a `CreateModel.py`.
- `SIM_DL.py`: Script para lanzar las simulaciones con los parámetros indicados.
- `Simulator.py`: Clase que contiene el Simulador de experimentos.
- `TrainConfig.py`: Configuraciones de entrenamiento para los modelos.
- `configs.yaml`, `configs10convs.yaml`, `configs3convs.yaml`, `configs5convs.yaml`, `configs7convs.yaml`: Archivos de configuraciones y arquitecturas en función del número de capas convolucionales.
- `justification.ipynb`: Cuaderno Jupyter explicando la justificación de la reducción de configuraciones.
- `final_results.ipynb`: Cuaderno Jupyter que estudia los resultados finales de los experimentos en el conjunto de assessment.
- `results_assessment.csv`: Archivo CSV con los datos de desempeño de los diferentes modelos.

### B) Deep Transfer Learning

En esta sección se explora el uso de técnicas de deep transfer learning para mejorar el rendimiento del modelo. Para ello se utiliza el modelo YOLOv8 de la librería ultralytics (https://docs.ultralytics.com/)

#### Archivos Principales:

- `ConfigGenerator.py`: Generador de configuraciones para los experimentos.
- `DQ-Zn-Coating-CNN_models.ipynb`: Cuaderno Jupyter donde se procesan los datos (imágenes) para poder utilizar YOLOv8.
- `Img_preprocessor.py`: Clase que contiene el procesador de las imágenes para adaptarlas a YOLOv8.
- `SIM_DTL.py`:  Script para lanzar las simulaciones con los parámetros indicados.
- `Simulator_B.py`: Clase que contiene el Simulador de experimentos.
- `configurationsL.yaml`, `configurationsM.yaml`, `configurationsN.yaml`, `configurationsS.yaml`: Archivos de modelos a simular en función del número del modelo preentrenado escogido.
- `final_results_B.ipynb`: Cuaderno Jupyter que estudia los resultados finales de los experimentos en el conjunto de assessment.
- `results_ass.csv`: Archivo CSV con los datos de desempeño de los diferentes modelos.

### C) Contrastive Learning

Esta sección se centra en el aprendizaje contrastivo.

#### Archivos Principales:

- `Builder.py`: Clase para construir los modelos de Deep Learning que se utilizarán como codificadores.
- `CreateModel.py`: Clase para crear los modelos (llamando al constructor) con los tags y metadata incluidos.
- `Data.py`: Clase para cargar los datos necesarios (imágenes originales).
- `DataProcessor.py`: Código para procesar los datos leídos en el yaml para adaptarlos antes de pasarselos a `CreateModel.py`.
- `NoSquare_results.ipynb`: Cuaderno Jupyter que estudia los resultados finales de los experimentos en el conjunto de assessment con las imágenes originales.
- `SIM_CL.py`: Script para lanzar las simulaciones con los parámetros indicados.
- `Simulator_C.py`: Clase que contiene el Simulador de experimentos.
- `Square_results.ipynb`: Cuaderno Jupyter que estudia los resultados finales de los experimentos en el conjunto de assessment con las imágenes procesadas (cuadradas).
- `func.py`:  Script para almacenar funciones útiles para llevar a cabo el CL. Se define la función de pérdida, la distancia euclidea para medir distancia entre caracteristicas en el espacio latente y se definen las funciones para realizar la clasificación a partir del cálculo de similaridad.
- `reference_images.ipynb`: Cuaderno Jupyter en el que se obtinen las imágenes de referencia para poder realizar la clasificación
## Resumen del Proyecto

El control de calidad de las bobinas de acero galvanizado es crucial debido a su amplia gama de aplicaciones y la necesidad de mantener altos estándares en sectores industriales exigentes. Para abordar este problema, se han desarrollado modelos basados en inteligencia artificial, utilizando Redes Neuronales Convolucionales (CNN), Aprendizaje Profundo por Transferencia (DTL) y Aprendizaje Contrastivo (CL) para clasificar las bobinas de acero según el patrón de espesor del recubrimiento de zinc. Este proyecto tiene como objetivo mejorar significativamente el proceso de toma de decisiones en el control de calidad, ejemplificando una colaboración efectiva entre humanos y máquinas en la Industria 5.0.

## Cómo Empezar

1. Clona este repositorio: `git clone https://github.com/jordieres/DQ_ACA_2024.git`
2. Instala las dependencias necesarias desde el archivo requirements.txt. (pip install -r requirements.txt
)
3. Explora las carpetas `A)`, `B)`, y `C`, junto al documento del proyecto para entender y ejecutar los diferentes experimentos.

## Autor

Alejandro Álvarez Castro
