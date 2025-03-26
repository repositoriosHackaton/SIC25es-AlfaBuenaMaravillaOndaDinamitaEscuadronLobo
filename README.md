
* Nombre del proyecto

Neuroseñas


El proyecto trata de reconocer el lenguaje de señas salvadoreño (LESSA) e interpretarlo a plabaras para la comprensión de quienes no lo saben.

El programa captura en tiempo real fram a frame de la cámara y por medio de la dataset compara los que ve devolviendo en palabras la seña que analiza.

![image alt](https://github.com/repositoriosHackaton/SIC25es-AlfaBuenaMaravillaOndaDinamitaEscuadronLobo/blob/main/video_eurose%C3%B1as.gif?raw=true)

* Arquitectura del proyecto 

Creación del Dataset:

Para crear la base de datos se tiene un script por medio de opencv toma un video y procesa las señas de las manos para cada una y se guarda en un formato .npy

Para procesar los datos se utiliza mediapipe para leer los puntos de la posición de cada mano, toma 21 características por mano, 3 coordenadas por característica, en la misma se normaliza los datos entre -1 a 1

![image alt](https://github.com/repositoriosHackaton/SIC25es-AlfaBuenaMaravillaOndaDinamitaEscuadronLobo/blob/main/creaci%C3%B3n_dataset_neurose%C3%B1a.jpg?raw=true)

Entrenamiento del modelo:

Para entrenar el modelo se utiliza redes neuronales convolucionales (CNN) para detectar las manos, el modelo accede al dataset y compara, se entrena a 75 epocas

Por medio de mediapipe se procesa en tiempo real las 126 características, en caso de solo haber una mano se rrellena con 0 a la mano que no se detecta o ambas si se llega a no mostrar ninguna y espera a detectar alguna. Normaliza los datos con los mismos valores que el procesado

![image alt](https://github.com/repositoriosHackaton/SIC25es-AlfaBuenaMaravillaOndaDinamitaEscuadronLobo/blob/main/entrenamiento_CNN_neurose%C3%B1a.jpg?raw=true)


Modelo trabajando:

El modelo CNN va detectando frame a frame al momento, comparando en base al ajuste de pesos que se realizó en el entrenamiento, muestra una predicción con un alto índice de confianza.

![image alt](https://github.com/repositoriosHackaton/SIC25es-AlfaBuenaMaravillaOndaDinamitaEscuadronLobo/blob/main/Diagrama_funcionando_modelo_neurose%C3%B1as.jpg?raw=true)

Web para interpretacion de señas en reuniones virtuales:

La web accede a un servidor local en el cual está alojado todo el programa de Neuroseñas, una vez se inicia la extensión el usuario elige la pantalla
en la cual quiere que se ejecute, una vez seleccionada, la web interpretara en dicha ventana, presentando unos subtitulos con la interpretacion escrita de las señas realizadas en LESSA.

![image alt[]()](https://github.com/repositoriosHackaton/SIC25es-AlfaBuenaMaravillaOndaDinamitaEscuadronLobo/blob/main/Diagrama_web.png?raw=true)


* Proceso de desarrollo:

-Fuente del dataset

https://github.com/Alexander1251/RedNeuronalLenguajeSe-as/tree/main/datosABC
https://github.com/Alexander1251/RedNeuronalLenguajeSe-as/tree/main/datos/procesados


-Limpieza de datos 
Al nosotros crear el data set, el script normaliza los datos y solo procesa los datos de los puntos de las manos

-Manejo excepciones/control errores

Al solo mostrar una mano se rrellena con los puntos de la mano y con respecto a la otra se rrellena con 0 para evitar errores, y al momento de no detectar alguna mano se queda esperando.


* Funcionalidades:

1. Lectura de la posicion de la mano
El programa habilita la camara y mediante mediapipe se realiza  la deteccion de los puntos donde se ubica la mano en tiempo real.

2. Analisis y reconocimiento de señas
El modelo reconoce las señas basados en los puntos de la mano tomados en tiempo real, comparandolo con los videos con los que fue entrenado y en base a su analisis muestra en pantalla la senia que reconoce junto con el % de confianza 

* Tecnología/Herramientas usadas 

Para elaborar el Dataset:
- Python 3 
- Pandas (Limpieza de datos)
- Scikitlearning (Normalizacion de los datos)
- Mediapipe (Analisis de la posicion de las manos)
- Open CV (Procesamiento de imagenes)

Modelo lectura de señas
- Python 3
- Open CV (Procesamiento de imagenes)
- Mediapipe (Analisis de la posicion de las manos)
- Scikitlearning (Entrenamiento del modelo)
- Pytorch (Entrenamiento del modelo)

- Arquitectura (img)

* Estado del proyecto:

El proyecto se encuentra en un estado avanzado, mas no definitivo,  el modelo ha sido entrenado con el abecedario y algunas frases
simples de LESSA , mostrando resultados y una fiablidad del 96.54% en cuanto a el reconocimiento de las señas

Se ha estipulado agregar frases mas complejas en un futuro proximo, para poder tener un Datas set mas completo que permita  un uso  real , practico y confiable de este proyecto

* Agradecimientos

- Agradecer al equipo de SIC por toda su formacion y orientacion para realizar el proyecto
- A todo el equipo involucrado en el desarrollo 
- A los equipos que desarrollaron las tecnologias utilizadas en la elaboracion de este proyecto
- Tanto a Tatiana de Morataya por proveer material didactico de LESSA como a sus autores (Ministerio de Educacion)
- A la creadora de contenido Becky Lessa, del canal Aprende LESSA con Becky Soundy

