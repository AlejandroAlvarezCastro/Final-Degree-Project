# Artificial Intelligence based models to improve quality classification of steel coils in automotive industry

## Proyecto de Fin de Grado de Alejandro Álvarez Castro

Este repositorio contiene el código y los experimentos realizados para el proyecto de fin de grado titulado "Artificial Intelligence based models to improve quality classification of steel coils in automotive industry", donde se estudian varias aproximaciones de machine learning para resolver un problema concreto en el sector industrial. El proyecto está dividido en tres apartados principales:

## Estructura del Repositorio

- **A) Deep Learning**
- **B) Deep Transfer Learning**
- **C) Contrastive Learning**

### A) Deep Learning

En esta sección se prueban y construyen varios modelos de deep learning para estudiar sus resultados.

#### Archivos Principales:

- `Builder.py`: Código para construir los modelos de deep learning.
- `CreateModel.py`: Script para crear diferentes arquitecturas de modelos.
- `DQ-Zn-Coating-CNN_models.ipynb`: Cuaderno Jupyter con los experimentos y resultados de los modelos de CNN aplicados al problema de recubrimiento de zinc.
- `Data.py`: Script para manejar los datos.
- `DataProcessor.py`: Código para procesar los datos antes de entrenar los modelos.
- `SIM_DL.py`: Simulador específico para los experimentos de deep learning.
- `Simulator.py`: Script para ejecutar simulaciones.
- `TrainConfig.py`: Configuraciones de entrenamiento para los modelos.
- `configs.yaml`, `configs10convs.yaml`, `configs3convs.yaml`, `configs5convs.yaml`, `configs7convs.yaml`: Archivos de configuración con diferentes arquitecturas de redes.
- `final_results.ipynb`: Cuaderno Jupyter con los resultados finales de los experimentos.
- `justification.ipynb`: Cuaderno Jupyter explicando la justificación de los experimentos.
- `results_assessment.csv`: Archivo CSV con la evaluación de los resultados.

### B) Deep Transfer Learning

En esta sección se investiga el uso de técnicas de deep transfer learning para mejorar el rendimiento del modelo.

#### Archivos Principales:

- `ConfigGenerator.py`: Generador de configuraciones para los experimentos.
- `DQ-Zn-Coating-CNN_models.ipynb`: Cuaderno Jupyter con los experimentos y resultados utilizando transfer learning.
- `Img_preprocessor.py`: Script para el preprocesamiento de imágenes.
- `SIM_DTL.py`: Simulador específico para los experimentos de transfer learning.
- `Simulator_B.py`: Script para ejecutar simulaciones en esta sección.
- `configurationsL.yaml`, `configurationsM.yaml`, `configurationsN.yaml`, `configurationsS.yaml`: Archivos de configuración con diferentes variaciones de modelos de transfer learning.
- `final_results_B.ipynb`: Cuaderno Jupyter con los resultados finales de los experimentos en esta sección.
- `results_ass.csv`: Archivo CSV con la evaluación de los resultados.

### C) Contrastive Learning

Esta sección se centra en el aprendizaje contrastivo y sus aplicaciones para el problema estudiado.

#### Archivos Principales:

- `Builder.py`: Código para construir los modelos de aprendizaje contrastivo.
- `CL_square_images.ipynb`: Cuaderno Jupyter con los experimentos usando imágenes cuadradas.
- `CreateModel.py`: Script para crear diferentes arquitecturas de modelos de aprendizaje contrastivo.
- `Data.py`: Script para manejar los datos.
- `DataProcessor.py`: Código para procesar los datos antes de entrenar los modelos de aprendizaje contrastivo.
- `Explanation.md`: Documento explicativo sobre los experimentos de aprendizaje contrastivo.
- `NoSquare_results.ipynb`: Cuaderno Jupyter con los resultados de los experimentos usando imágenes no cuadradas.
- `SIM_CL.py`: Simulador específico para los experimentos de aprendizaje contrastivo.
- `Simulator_C.py`: Script para ejecutar simulaciones en esta sección.
- `Square_results.ipynb`: Cuaderno Jupyter con los resultados de los experimentos usando imágenes cuadradas.
- `func.py`: Script con funciones auxiliares.
- `reference_images.ipynb`: Cuaderno Jupyter con imágenes de referencia utilizadas en los experimentos.

## Cómo Empezar

1. Clona este repositorio: `git clone <URL del repositorio>`
2. Instala las dependencias necesarias.
3. Explora las carpetas `A)`, `B)`, y `C` para entender y ejecutar los diferentes experimentos.
