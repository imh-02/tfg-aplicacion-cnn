import sklearn.metrics as metrics
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
    Función para calcular los parámetros necesarios para las métricas NPL 
    TP: True Positives - caracteres correctamente reconocidos
    FP: False Positives - caracteres reconocidos incorrectamente (porque se han añadido caracteres de más)
    FN: False Negatives - caracteres no reconocidos
    """
    def get_metrics_params(self, true_dni, pred_dni):
        dni = list(true_dni)
        pred = list(pred_dni)

        print("DNI real: ", dni)
        print("DNI predicho: ", pred)

        tp = 0
        fp = 0
        fn = 0

        if len(dni) == len(pred):
            for i in range(len(dni)):
                if dni[i] == pred[i]:
                    tp += 1
                else:
                    fp += 1
                
        elif len(pred) < len(dni): # El DNI predicho es más corto que el real
            pred_index = 0
            true_index = 0
            for i in range(len(pred)):
                if pred[pred_index] == dni[true_index]:
                    tp += 1
                    true_index += 1
                    pred_index += 1
                else:
                    print("Pred_index: ", pred_index)
                    print("True_index + 1: ", true_index + 1)
                    if true_index + 1 < len(dni):
                        if pred[pred_index] == dni[true_index + 1]:
                            fn += 1
                            tp += 1
                            true_index += 2
                            pred_index += 1
                        else:
                            true_index += 1
                            pred_index += 1
                            fn += 1
                    else: # si ya no hay más caracteres en el DNI real que comparar y no coincide con el predicho es una sustitución
                        fp += 1

        else: # El DNI predicho es más largo que el real
            pred_index = 0
            true_index = 0
            for i in range(len(dni)):
                if dni[true_index] == pred[pred_index]:
                    tp += 1
                    true_index += 1
                    pred_index += 1
                else:
                    if pred_index + 1 < len(pred):
                        if dni[true_index] == pred[pred_index + 1]:
                            fp += 1
                            tp += 1
                            true_index += 1
                            pred_index += 2
                        else: 
                            true_index += 1
                            pred_index += 1
                            fp += 1
                    else: # si ya no hay más caracteres en el DNI predicho que comparar y no coincide con el real es una eliminación
                        fn += 1
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


    """
    Método que devuelve las métricas a nivel de DNI obtenidas en el análisis en forma de cadena de texto.
    """
    def get_metrics(self):
        result = "Métricas desde el punto de vista de un problema NPL (OCR): \n"
        result += "\n"
        result += "Métricas antes del postprocesado: \n"
        result += self.calculateBeforePostProcessingDNIMetrics()
        result += "\n"
        result += "Métricas tras eliminar caracteres especiales: \n"
        result += self.calculateDeletingSpecialCharsDNIMetrics()
        result += "\n"
        result += "Métricas tras el postprocesado: \n"
        result += self.calculatePostProccesingDNIMetrics()  
        result += "\n"

        result += "*CER: Character Error Rate \n"
        result += "*WER: Word Error Rate \n"
        result += "*TER: Test Error Rate \n"

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



