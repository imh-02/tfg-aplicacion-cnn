from modules.data_proccessing import checkDNIFormat, validSpanishDNI


"""
Clase para el cálculo de las métricas desde el punto de vista de un problema de clasificación de Machine Learning.
"""
class MetricsML:

    dni_pred_before_postprocessing = []
    dni_pred_deleting_special_chars = []
    dni_pred_postprocessed = []

    def __init__(self, y_true, total_number_of_dni = 0):
        self.y_true = y_true
        self.total_number_of_dni = total_number_of_dni
    
    """
    Función para calcular los verdaderos positivos, que se corresponden con los DNI predichos
    de formato correcto y que coinciden con el DNI real.
    """
    def get_metricsTP(self, true_dnis, pred_dnis):
        tp = 0
        for i in range(len(pred_dnis)):
            if checkDNIFormat(pred_dnis[i]):
                if pred_dnis[i] == true_dnis[i]:
                    tp += 1
        return tp
    
    """
    Función para calcular los falsos positivos, que se corresponden con los DNI predichos de formato correcto
    pero que no coinciden con el DNI real.
    """
    def get_metricsFP(self, true_dnis, pred_dnis):
        fp = 0
        for i in range(len(pred_dnis)):
            if checkDNIFormat(pred_dnis[i]):
                if pred_dnis[i] != true_dnis[i]:
                    fp += 1
        return fp

    """
    Función para calcular los falsos negativos, que se corresponden con los DNI predichos de formato incorrecto
    pero cuyo DNI real es correcto y válido. Por lo que se comprueba el formato del DNI predicho y si el DNI real 
    es válido.
    """
    def get_metricsFN(self, true_dnis, pred_dnis):
        fn = 0
        for i in range(len(pred_dnis)):
            if checkDNIFormat(pred_dnis[i]) == False:
                if (checkDNIFormat(true_dnis[i]) and validSpanishDNI(true_dnis[i])):
                    fn += 1
        return fn
    

    """
    Función para calcular los verdaderos negativos, que se corresponden con los DNI predichos de formato incorrecto
    y cuyo DNI real también es incorrecto. Por lo que se comprueba el formato del DNI predicho y si el DNI real 
    es válido.
    """
    def get_metricsTN(self, true_dnis, pred_dnis):
        tn = 0
        for i in range(len(pred_dnis)):
            if checkDNIFormat(pred_dnis[i]) == False:
                if ((checkDNIFormat(true_dnis[i]) == False) and (validSpanishDNI(true_dnis[i]) == False)):
                    tn += 1
        return tn
    

    """
    Función que mide la proporción de todas las predicciones correctas respecto al total de predicciones
    """
    def accuracy(self, true_dnis, pred_dnis):
        tp = self.get_metricsTP(true_dnis, pred_dnis)
        tn = self.get_metricsTN(true_dnis, pred_dnis)
        fp = self.get_metricsFP(true_dnis, pred_dnis)
        fn = self.get_metricsFN(true_dnis, pred_dnis)
        return (tp + tn) / (tp + tn + fp + fn)

    """
    Mide la proporción de DNIs correctamente reconocidos entre todos los DNIs que el sistema identificó como correctos
    """
    def precision(self, true_dnis, pred_dnis):
        tp = self.get_metricsTP(true_dnis, pred_dnis)
        fp = self.get_metricsFP(true_dnis, pred_dnis)
        if tp + fp == 0:
            return 0
        return tp / (tp + fp)
    

    
    def recall(self, true_dnis, pred_dnis):
        tp = self.get_metricsTP(true_dnis, pred_dnis)
        fn = self.get_metricsFN(true_dnis, pred_dnis)
        if tp + fn == 0:
            return 0
        return tp / (tp + fn)
    
    def f1(self, true_dnis, pred_dnis):
        precision = self.precision(true_dnis, pred_dnis)
        recall = self.recall(true_dnis, pred_dnis)
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    


    def calculateBeforePostProcessingDNIMetrics(self):
        accuracy = self.accuracy(self.y_true, self.dni_pred_before_postprocessing)*100
        precision = self.precision(self.y_true, self.dni_pred_before_postprocessing)*100
        recall = self.recall(self.y_true, self.dni_pred_before_postprocessing)*100
        f1 = self.f1(self.y_true, self.dni_pred_before_postprocessing)*100
        result = "Accuracy: " + str(round(accuracy, 2)) + "% \n" + "Precision: " + str(round(precision, 2)) + "% \n" + "Recall: " + str(round(recall, 2)) + "% \n" + "F1: " + str(round(f1, 2)) + "% \n"
        return result
        
    
    def calculateDeletingSpecialCharsDNIMetrics(self):
        accuracy = self.accuracy(self.y_true, self.dni_pred_deleting_special_chars) * 100
        precision = self.precision(self.y_true, self.dni_pred_deleting_special_chars) * 100
        recall = self.recall(self.y_true, self.dni_pred_deleting_special_chars) * 100
        f1 = self.f1(self.y_true, self.dni_pred_deleting_special_chars) * 100
        result = "Accuracy: " + str(round(accuracy, 2)) + "% \n" + "Precision: " + str(round(precision, 2)) + "% \n" + "Recall: " + str(round(recall, 2)) + "% \n" + "F1: " + str(round(f1, 2)) + "% \n"
        return result
    
    def calculatePostProccesingDNIMetrics(self):
        accuracy = self.accuracy(self.y_true, self.dni_pred_postprocessed) * 100
        precision = self.precision(self.y_true, self.dni_pred_postprocessed) * 100
        recall = self.recall(self.y_true, self.dni_pred_postprocessed) * 100
        f1 = self.f1(self.y_true, self.dni_pred_postprocessed) * 100
        result = "Accuracy: " + str(round(accuracy, 2)) + "% \n" + "Precision: " + str(round(precision, 2)) + "% \n" + "Recall: " + str(round(recall, 2)) + "% \n" + "F1: " + str(round(f1, 2)) + "% \n"
        return result


    """
    Método que devuelve las métricas a nivel de DNI obtenidas en el análisis en forma de cadena de texto.
    """
    def get_metrics(self):
        result = "Métricas desde el punto de vista de un problema de clasificación de Machine Learning: \n"
        result += "\n"
        result += "Métricas antes del postprocesado: \n"
        result += "Parámetros de las métricas: \n"
        tp_before_postprocessing = self.get_metricsTP(self.y_true, self.dni_pred_before_postprocessing)
        fp_before_postprocessing = self.get_metricsFP(self.y_true, self.dni_pred_before_postprocessing)
        fn_before_postprocessing = self.get_metricsFN(self.y_true, self.dni_pred_before_postprocessing)
        tn_before_postprocessing = self.get_metricsTN(self.y_true, self.dni_pred_before_postprocessing)
        result += "TP: " + str(tp_before_postprocessing) + "\n" + "FP: " + str(fp_before_postprocessing) + "\n" + "FN: " + str(fn_before_postprocessing) + "\n" + "TN: " + str(tn_before_postprocessing) + "\n"
        result += self.calculateBeforePostProcessingDNIMetrics()
        result += "\n"
        result += "Métricas tras eliminar caracteres especiales: \n"
        result += "Parámetros de las métricas: \n"
        tp_deleting_special_chars = self.get_metricsTP(self.y_true, self.dni_pred_deleting_special_chars)
        fp_deleting_special_chars = self.get_metricsFP(self.y_true, self.dni_pred_deleting_special_chars)
        fn_deleting_special_chars = self.get_metricsFN(self.y_true, self.dni_pred_deleting_special_chars)
        tn_deleting_special_chars = self.get_metricsTN(self.y_true, self.dni_pred_deleting_special_chars)
        result += "TP: " + str(tp_deleting_special_chars) + "\n" + "FP: " + str(fp_deleting_special_chars) + "\n" + "FN: " + str(fn_deleting_special_chars) + "\n" + "TN: " + str(tn_deleting_special_chars) + "\n"
        result += self.calculateDeletingSpecialCharsDNIMetrics()
        result += "\n"
        result += "Métricas tras el postprocesado: \n"
        result += "Parámetros de las métricas: \n"
        tp_postprocessed = self.get_metricsTP(self.y_true, self.dni_pred_postprocessed)
        fp_postprocessed = self.get_metricsFP(self.y_true, self.dni_pred_postprocessed)
        fn_postprocessed = self.get_metricsFN(self.y_true, self.dni_pred_postprocessed)
        tn_postprocessed = self.get_metricsTN(self.y_true, self.dni_pred_postprocessed)
        result += "TP: " + str(tp_postprocessed) + "\n" + "FP: " + str(fp_postprocessed) + "\n" + "FN: " + str(fn_postprocessed) + "\n" + "TN: " + str(tn_postprocessed) + "\n"
        result += self.calculatePostProccesingDNIMetrics()  
        result += "\n"

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



