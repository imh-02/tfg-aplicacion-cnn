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
Clase que almacena las métricas obtenidas en el análisis de un documento y las funciones necesarias para calcularlas.
"""
class Metrics:

    characteres_true = []
    characteres_pred = []

    characteres_pred_dif_size = []
    characteres_true_dif_size = []


    dni_pred_before_postprocessing = []
    dni_pred_deleting_special_chars = []
    dni_pred_postprocessed = []

    number_of_correct_size_dni = 0


    def __init__(self, y_true, total_number_of_dni = 0):
        self.y_true = y_true
        self.total_number_of_dni = total_number_of_dni

    
    def calculateBeforePostProcessingDNIMetrics(self):
        accuracy = metrics.accuracy_score(self.y_true, self.dni_pred_before_postprocessing) * 100
        precision = metrics.precision_score(self.y_true, self.dni_pred_before_postprocessing, average='weighted', zero_division=0) * 100
        recall = metrics.recall_score(self.y_true, self.dni_pred_before_postprocessing, average='weighted',  zero_division=0) *100
        f1 = metrics.f1_score(self.y_true, self.dni_pred_before_postprocessing, average='weighted',  zero_division=0) *100
        cer_result = cer(self.y_true, self.dni_pred_before_postprocessing) * 100
        wer_result = wer(self.y_true, self.dni_pred_before_postprocessing) * 100
        result = "Accuracy: " + str(round(accuracy, 2)) + "% \n" + "Precision: " + str(round(precision, 2)) + "% \n" + "Recall: " + str(round(recall, 2)) + "% \n" + "F1: " + str(round(f1, 2)) + "% \n" + "CER: " + str(round(cer_result, 2)) + "% \n" + "WER: " + str(round(wer_result, 2)) + "% \n"
        return result
        
    
    def calculateDeletingSpecialCharsDNIMetrics(self):
        accuracy = metrics.accuracy_score(self.y_true, self.dni_pred_deleting_special_chars) * 100
        precision = metrics.precision_score(self.y_true, self.dni_pred_deleting_special_chars, average='weighted', zero_division=0) * 100
        recall = metrics.recall_score(self.y_true, self.dni_pred_deleting_special_chars, average='weighted',  zero_division=0) *100
        f1 = metrics.f1_score(self.y_true, self.dni_pred_deleting_special_chars, average='weighted',  zero_division=0) *100
        cer_result = cer(self.y_true, self.dni_pred_deleting_special_chars) * 100
        wer_result = wer(self.y_true, self.dni_pred_deleting_special_chars) * 100
        result = "Accuracy: " + str(round(accuracy, 2)) + "% \n" + "Precision: " + str(round(precision, 2))+ "% \n" + "Recall: " + str(round(recall, 2)) + "% \n" + "F1: " + str(round(f1, 2)) + "% \n" + "CER: "+ str(round(cer_result, 2)) + "% \n" + "WER: " + str(round(wer_result, 2)) + "% \n"
        return result
    
    def calculatePostProccesingDNIMetrics(self):
        accuracy = metrics.accuracy_score(self.y_true, self.dni_pred_postprocessed) * 100
        precision = metrics.precision_score(self.y_true, self.dni_pred_postprocessed, average='weighted', zero_division=0) * 100
        recall = metrics.recall_score(self.y_true, self.dni_pred_postprocessed, average='weighted',  zero_division=0) *100
        f1 = metrics.f1_score(self.y_true, self.dni_pred_postprocessed, average='weighted',  zero_division=0) *100
        cer_result = cer(self.y_true, self.dni_pred_postprocessed) * 100
        wer_result = wer(self.y_true, self.dni_pred_postprocessed) * 100
        result = "Accuracy: " + str(round(accuracy, 2)) + "% \n" + "Precision: " + str(round(precision, 2)) + "% \n" + "Recall: " + str(round(recall, 2)) + "% \n" + "F1: " + str(round(f1, 2)) + "% \n" + "CER: " + str(round(cer_result, 2)) + "% \n" + "WER: " + str(round(wer_result, 2)) + "% \n"
        return result


    """
    Método que devuelve las métricas a nivel de DNI obtenidas en el análisis en forma de cadena de texto.
    """
    def get_metrics(self):
        result = ""
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



