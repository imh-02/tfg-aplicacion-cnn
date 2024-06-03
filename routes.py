"""
Nombres o rutas de los modelos de transformers que se usan en el análisis
"""
trocr_model_path = "models\\trocr_finetuned_modelCompleteDNI_135_local"
signature_model_path = "models\\siglip-base-patch16-256-multilingual"
table_model_path = "models\\table-transformer-structure-recognition"

"""
Ruta de la red CNN que se usa en el análisis
"""
cnn_model_path = "models\\cnn_ocr_10.000_95x20Binary60Epocas256Bacth.pth"

"""
Ruta de la carpeta donde se guardarán los informes
"""
reports_path_root = "reports\\"

"""
Rutas ficheros de entrada y salida
"""
pdf_file_path = "ejemplos\\ejemplo10PS.pdf"
output_file = "informe10PS-cnn10.000_95x20Binary60Epocas256Batch"
true_file_path = "datos_reales\\ejemplo10PS.txt"
