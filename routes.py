"""
Nombres o rutas de los modelos de transformers que se usan en el análisis
"""
trocr_model_path = "models\\trocr_finetuned_modelCompleteDNI_135_local"
signature_model_path = "models\\siglip-base-patch16-256-multilingual"
table_model_path = "models\\table-transformer-structure-recognition"

"""
Ruta de la red CNN que se usa en el análisis
"""
cnn_model_path = "models\\cnn_ocr_1000_completo95x20_40000.pth"

"""
Ruta de la carpeta donde se guardarán los informes
"""
reports_path_root = "reports\\"

"""
Rutas ficheros de entrada y salida
"""
pdf_file_path = "ejemplos\\ejemplo1PS.pdf"
output_file = "informe1PS-cnn40.000-2"
true_file_path = "datos_reales\\ejemplo1PS.txt"