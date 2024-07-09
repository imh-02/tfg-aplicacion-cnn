"""
Nombres o rutas de los modelos de transformers que se usan en el análisis
"""
signature_model_path = "models\\siglip-base-patch16-256-multilingual"
table_model_path = "models\\table-transformer-structure-recognition"

"""
Ruta de las dos redes CNN 
"""
cnn_letters_model_path = "models\\cnn_emnist_letters_tutorial70Epochs.pth"
cnn_numbers_model_path = "models\\cnn_mnist_tutorial.pth"
crnn_complete_model_path = "models\\cnn_emnistCompleto_blank_60_epochs.pth"

"""
Ruta de la carpeta donde se guardarán los informes
"""
reports_path_root = "reports\\"

"""
Flag para indicar si hay GPU disponible o no

False: CPU
True: GPU
"""
device = False

"""
Rutas ficheros de entrada y salida
"""
pdf_file_path = "ejemplos\\ejemplo20PS.pdf"
output_file = "informe20PS-CRNN"
true_file_path = "datos_reales\\ejemplo20PS.txt"
