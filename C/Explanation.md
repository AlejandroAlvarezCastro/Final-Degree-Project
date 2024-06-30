Los ficheros de la carpeta de este apartado tienen el siguiente propósito:

Cosas ya usadas en A):
- Builder.py: Construir los modelos en base a especificaicones de arquitecturas y configuraciones.
- CreateModel.py: Crea tuplas para cada modelo con el modelo.keras en si, el nombre del modelo, sus tags y metadata y los humbrales para los test suites
- Data.py: Carga y procesa los ficheros -npz para generar el conjunto de datos grande y el gold standard
- DataProcessor.py: En base a la arquitectura y valores de configuracion, adecua a los valores necesario paar que le llegen al constructor. También genera los tags y la metadata para cada modelo en base a las especificaciones.
  
Cosas Nuevas:
- functions.py: Script para almacenar funciones útiles para llevar a cabo el CL. Se define la función de pérdida, la distancia euclidea para medir distancia entre caracteristicas en el espacio latente, se define la función para generar las parejas de imagenes que se emplearán en train y test, y se definen las funciones para realizar la clasificación a partir del cálculo de similaridad.
- Simulator.ipynb: Cuaderno con el codigo para realizar la simulación completa. Se ha dejado en un cuaderno aun para tener más facilidad de manejo de variables y de qué esta pasando en el código. En cuanto se arregle lo de las redes siamesas, solo habría que meterlo dentro de un script en una clase y ya. Hay un ejemplo del problema que sucedería si se empezase ahora la simulación (la red no aprende y se obtienen todas las predicciones del gold standard en una clase u otra)
- CL_Classifier_v2.ipynb: Este cuaderno es exactamente el mismo codigo del que se subio hace tiempo llamado CL_CLassifier, que obtuvo tan buenos resultados. EN este cuaderno se tiene el mismo problema, la red, pese a que si ha aprendido, tiene unos resultados en los que todas las predicciones se van a una clase.
- CL_new.ipynb: En este cuaderno he tratado de realizar algunas pruebas y hay comentarios y celdas markdown comentando lo que se va haciendo. Se ha tratado de explicar y vidualizar más en profundidad el problema.
