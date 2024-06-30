Los ficheros de la carpeta de este apartado tienen el siguiente propósito:

Cosas ya usadas en A):
- Builder.py: Construir los modelos en base a especificaicones de arquitecturas y configuraciones.
- CreateModel.py: Crea tuplas para cada modelo con el modelo.keras en si, el nombre del modelo, sus tags y metadata y los humbrales para los test suites
- Data.py: Carga y procesa los ficheros -npz para generar el conjunto de datos grande y el gold standard
- DataProcessor.py: En base a la arquotectura y valores de configuracion, adecua a los valores necesario paar que le llegen al constructor. También genera los tags y la metadata para cada modelo en base a las especificaciones.
  
Cosas Nuevas:
- functions.py: Script para almacenar funciones útiles para llevar a cabo el CL. Se define la función de pérdida, la distancia euclidea para medir distancia entre caracteristiacs en el expacio latente, se define la función para generar las parejas de imagenes que se emplearán en train y test, y se definen las funciones para realizar la clasificación a partir del cá´lculo de similaridad
