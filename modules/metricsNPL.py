from jiwer import wer, cer


"""
Función que lee un fichero de texto con los DNI reales y los almacena en un array.
"""
def readTrueFile(file_path):

    trueDNIArray = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() != "":
                trueDNIArray.append(str(line.strip()))
    return trueDNIArray

"""
Clase para calcular las métricas desde el punto de vista de un problema NPL (OCR).
"""
class MetricsNPL:

    dni_pred_before_postprocessing = []
    dni_pred_deleting_special_chars = []
    dni_pred_postprocessed = []


    def __init__(self, y_true, total_number_of_dni = 0):
        self.y_true = y_true
        self.total_number_of_dni = total_number_of_dni

    """
    Función para calcular los parámetros necesarios para las métricas NPL haciendo uso del algoritmo de distancia de edición (Levenshtein).
    TP: True Positives - caracteres correctamente reconocidos
    FP: False Positives - caracteres reconocidos incorrectamente (porque se han añadido caracteres de más)
    FN: False Negatives - caracteres no reconocidos
    """
    def get_metrics_params(self, true_dni, pred_dni):
        m = len(true_dni)
        n = len(pred_dni)
        ground_truth = list(true_dni)
        predictions = list(pred_dni)
        
        # Crear una matriz para almacenar las operaciones de edición
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Llenar la primera fila y columna
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Calcular el número mínimo de operaciones de edición
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ground_truth[i - 1] == predictions[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j],        # Eliminación
                                       dp[i][j - 1],        # Inserción
                                       dp[i - 1][j - 1])   # Sustitución
        
        # El valor en la esquina inferior derecha de la matriz es la distancia de edición
        edit_distance = dp[m][n]
        
        # Calculamos el número de sustituciones, inserciones y eliminaciones
        i, j = m, n
        substitutions, insertions, deletions = 0, 0, 0
        while i > 0 and j > 0:
            if ground_truth[i - 1] == predictions[j - 1]:
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j - 1] + 1:
                substitutions += 1
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j] + 1:
                deletions += 1
                i -= 1
            elif dp[i][j] == dp[i][j - 1] + 1:
                insertions += 1
                j -= 1
        
        # Las operaciones restantes
        insertions += j
        deletions += i

        # tp, fp, fn considerando las sustituciones
        tp = m - edit_distance
        fp = insertions + substitutions
        fn = deletions

        return tp, fp, fn

    """
    Función para calcular la proporción de caracteres correctamente reconocidos respecto al total de caracteres.
    """
    def accuracy(self, true_dnis, pred_dnis):
        total_tp = []
        total_number_characters = 0
        for i in range(len(true_dnis)):
            tp, _, _ = self.get_metrics_params(true_dnis[i], pred_dnis[i])
            total_number_characters += len(true_dnis[i])
            total_tp.append(tp)
        return sum(total_tp) / total_number_characters
    
    """
    Función para calcular la proporción de caracteres correctamente reconocidos respecto al total de caracteres correctamente reconocidos.
    """
    def precision(self, true_dnis, pred_dnis):
        total_tp = []
        total_fp = []

        for i in range(len(true_dnis)):
            tp, fp, _ = self.get_metrics_params(true_dnis[i], pred_dnis[i])
            total_tp.append(tp)
            total_fp.append(fp)
        return sum(total_tp) / (sum(total_tp) + sum(total_fp))
    
    """
    Función para calcular la proporción de caracteres correctamente reconocidos respecto al total de caracteres que deberían haber sido reconocidos.
    """
    def recall(self, true_dnis, pred_dnis):
        total_tp = []
        total_fn = []
        for i in range(len(true_dnis)):
            tp, _, fn = self.get_metrics_params(true_dnis[i], pred_dnis[i])
            total_tp.append(tp)
            total_fn.append(fn)
        return sum(total_tp) / (sum(total_tp) + sum(total_fn))
    
    """
    Función para calcular la media armónica entre la precisión y la sensibilidad.
    """
    def f1(self, true_dnis, pred_dnis):
        precision = self.precision(true_dnis, pred_dnis)
        recall = self.recall(true_dnis, pred_dnis)
        return 2 * (precision * recall) / (precision + recall)
    
    def calculateBeforePostProcessingDNIMetrics(self):
        accuracy = self.accuracy(self.y_true, self.dni_pred_before_postprocessing)*100
        precision = self.precision(self.y_true, self.dni_pred_before_postprocessing)*100
        recall = self.recall(self.y_true, self.dni_pred_before_postprocessing)*100
        f1 = self.f1(self.y_true, self.dni_pred_before_postprocessing)*100
        cer_result = cer(self.y_true, self.dni_pred_before_postprocessing) * 100
        wer_result = wer(self.y_true, self.dni_pred_before_postprocessing) * 100
        result = "Accuracy: " + str(round(accuracy, 2)) + "% \n" + "Precision: " + str(round(precision, 2)) + "% \n" + "Recall: " + str(round(recall, 2)) + "% \n" + "F1: " + str(round(f1, 2)) + "% \n" + "CER: " + str(round(cer_result, 2)) + "% \n" + "WER: " + str(round(wer_result, 2)) + "% \n" 
        return result
        
    def calculateDeletingSpecialCharsDNIMetrics(self):
        accuracy = self.accuracy(self.y_true, self.dni_pred_deleting_special_chars) * 100
        precision = self.precision(self.y_true, self.dni_pred_deleting_special_chars) * 100
        recall = self.recall(self.y_true, self.dni_pred_deleting_special_chars) * 100
        f1 = self.f1(self.y_true, self.dni_pred_deleting_special_chars) * 100
        cer_result = cer(self.y_true, self.dni_pred_deleting_special_chars) * 100
        wer_result = wer(self.y_true, self.dni_pred_deleting_special_chars) * 100
        result = "Accuracy: " + str(round(accuracy, 2)) + "% \n" + "Precision: " + str(round(precision, 2))+ "% \n" + "Recall: " + str(round(recall, 2)) + "% \n" + "F1: " + str(round(f1, 2)) + "% \n" + "CER: "+ str(round(cer_result, 2)) + "% \n" + "WER: " + str(round(wer_result, 2)) + "% \n" 
        return result
    
    def calculatePostProccesingDNIMetrics(self):
        accuracy = self.accuracy(self.y_true, self.dni_pred_postprocessed) * 100
        precision = self.precision(self.y_true, self.dni_pred_postprocessed) * 100
        recall = self.recall(self.y_true, self.dni_pred_postprocessed) * 100
        f1 = self.f1(self.y_true, self.dni_pred_postprocessed) * 100
        cer_result = cer(self.y_true, self.dni_pred_postprocessed) * 100
        wer_result = wer(self.y_true, self.dni_pred_postprocessed) * 100
        result = "Accuracy: " + str(round(accuracy, 2)) + "% \n" + "Precision: " + str(round(precision, 2)) + "% \n" + "Recall: " + str(round(recall, 2)) + "% \n" + "F1: " + str(round(f1, 2)) + "% \n" + "CER: " + str(round(cer_result, 2)) + "% \n" + "WER: " + str(round(wer_result, 2)) + "% \n" 
        return result


    def get_allMetricsParams(self, true_dnis, pred_dnis):
        total_tp = []
        total_fp = []
        total_fn = []
        for i in range(len(true_dnis)):
            tp, fp, fn = self.get_metrics_params(true_dnis[i], pred_dnis[i])
            total_tp.append(tp)
            total_fp.append(fp)
            total_fn.append(fn)
        return sum(total_tp), sum(total_fp), sum(total_fn)

    """
    Método que devuelve las métricas a nivel de DNI obtenidas en el análisis en forma de cadena de texto.
    """
    def get_metrics(self):
        result = "Métricas desde el punto de vista de un problema NPL (OCR): \n"
        result += "\n"
        result += "Métricas antes del postprocesado: \n"
        result += "Parámetros de las métricas: \n"
        tp_before, fp_before, fn_before = self.get_allMetricsParams(self.y_true, self.dni_pred_before_postprocessing)
        result += "TP: " + str(tp_before) + "\n" + "FP: " + str(fp_before) + " (inserciones + sustituciones)" + "\n" + "FN: " + str(fn_before) + " (eliminaciones)" + "\n"
        result += self.calculateBeforePostProcessingDNIMetrics()
        result += "\n"
        result += "Métricas tras eliminar caracteres especiales: \n"
        result += "Parámetros de las métricas: \n"
        tp_special_char, fp_special_char, fn_special_char = self.get_allMetricsParams(self.y_true, self.dni_pred_deleting_special_chars)
        result += "TP: " + str(tp_special_char) + "\n" + "FP: " + str(fp_special_char) + " (inserciones + sustituciones)" + "\n" + "FN: " + str(fn_special_char) + " (eliminaciones)" + "\n"
        result += self.calculateDeletingSpecialCharsDNIMetrics()
        result += "\n"
        result += "Métricas tras el postprocesado: \n"
        result += "Parámetros de las métricas: \n"
        tp_post, fp_post, fn_post = self.get_allMetricsParams(self.y_true, self.dni_pred_postprocessed)
        result += "TP: " + str(tp_post) + "\n" + "FP: " + str(fp_post) + " (inserciones + sustituciones)" + "\n" + "FN: " + str(fn_post) + " (eliminaciones)" + "\n"
        result += self.calculatePostProccesingDNIMetrics()  
        result += "\n"

        result += "*CER: Character Error Rate \n"
        result += "*WER: Word Error Rate \n"

        result += "\n"
        return result

    """
    Método que imprime por consola las métricas a nivel de DNI obtenidas en el análisis.
    """
    def print_metrics(self):
        print(self.get_metrics())

    """
    Método que añade una predicción a las métricas a nivel de DNI tras el postprocesado.
    """
    def addPredictionPostProcessed(self, prediction):
        self.dni_pred_postprocessed.append(str(prediction))


    def addPredictionBeforePostProcessed(self, prediction):
        self.dni_pred_before_postprocessing.append(str(prediction))

    def addPredictionDeletingSpecialChars(self, prediction):
        self.dni_pred_deleting_special_chars.append(str(prediction))



