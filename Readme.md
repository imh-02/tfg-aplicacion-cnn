# Aplicación para la automatización de la verificación de recogida de firmas para proposiciones de Ley mediante Inteligencia Artificial - CNN

**PENDIENTE DE ACTUALIZAR


Ismael Martín Herrera *alu0101397375@ull.edu.es*

## Índice

1. [Introducción](#introducción)
2. [Requerimientos](#requerimientos)
3. [Instrucciones de uso](#instrucciones-de-uso)

## Introducción

La aplicación tiene como objetivo automatizar el proceso de verificación de los datos de los firmantes para proposiciones de Ley, y ha sido desarrollada basado en el caso de uso del Parlamento de Canarias. 

Los datos de los firmantes son recogidos en una plantilla concreta proporcionada por el Parlamento de Canarias, y por tanto son manuscritos. Por lo que mediante la aplicación se pretende extraer los citados datos manuscritos haciendo OCR sobre los mismo, en este caso mediante Inteligencia Artificial, concretamente utilizando redes neuronales de tipo CNN. 

## Requerimientos

_Software_

- Versión de Python: +3.12
- Drivers CUDA 11.8 de NVIDIA
- Drivers cuDNN de NVIDIA

Los drivers de NVIDIA son necesario puesto que el programa está preparado para ejecutarse en una GPU NVIDIA, de lo contrario la aplicación se ejecutará en la CPU y se podría demorar más. 


Para la instalación de las dependencias se puede hacer uso del fichero ```requirements.txt```.

_Modelos de Transformers_
Para el uso de la aplicación es necesario tener descargados los siguientes modelos en local, existen dos opciones para descargarlos. Por un lado indicando el ID del modelo de la plataforma HuggingFace. O por otro lado, clonando los repositorios de los modelos e indicando las rutas en local. En ambos casos es necesario modificar en el fichero ```routes.py``` las variables:  ```signature_model_path``` y ```table_model_path```, indicando la ruta en local o el ID. 

*Modelo de detección de firmas*

- ID: google/siglip-base-patch16-256-multilingual
- Enlace: https://huggingface.co/google/siglip-base-patch16-256-multilingual

*Modelo de extracción de características de la tabla*

- ID: bilguun/table-transformer-structure-recognition
- Enlace: https://huggingface.co/bilguun/table-transformer-structure-recognition

_Modelos de red CNN_

Habría que modificar las siguientes variables, en el fichero ```routes.py```:

- cnn_letters_model_path: ruta del modelo con la red CNN entrenada con letras
- cnn_numbers_model_path: ruta del modelo con la red CNN entrenada con números

## Instrucciones de uso

Para el uso de la aplicación es necesario especificar en la variable ```reports_path_root``` en el fichero ```routes.py``` el directorio en el que la aplicación devolverá los informes tras los análisis. Además de modificar las siguientes variables también en el fichero ```routes.py```: 

- pdf_file_path: se corresponde con la ruta del fichero PDF de ejemplo con los datos de entrada, siguiendo la plantilla correspondiente. 
- output_file: nombre que tendrá el informe final después del análisis. 
- true_file_path: se corresponde con la ruta de datos reales en un fichero .txt para que se pueden realizar la métricas y ver el desempeño de la aplicación. 

Una vez modificadas las variables ejecutar ```python main.py```

