"""
Nombres o rutas de los modelos de transformers que se usan en el análisis
"""
trocr_model_path = "models\\trocr_finetuned_modelCompleteDNI_135_local"
signature_model_path = "models\\siglip-base-patch16-256-multilingual"
table_model_path = "models\\table-transformer-structure-recognition"

"""
Ruta de las dos redes CNN separadas para caracteres individuales
"""
cnn_letters_model_path = "model_training\\cnn_emnist_letters_tutorial40Epochs.pth"
cnn_numbers_model_path = "models\\cnn_mnist_tutorial.pth"



"""
Ruta de la carpeta donde se guardarán los informes
"""
reports_path_root = "reports\\"

"""
Flag para indicar si usar dos modeles de caracteres individuales o uno
"""
use_2_models = True

"""
Rutas ficheros de entrada y salida
"""
pdf_file_path = "ejemplos\\ejemplo10PS.pdf"
output_file = "informe10PS"
true_file_path = "datos_reales\\ejemplo1PS.txt"
