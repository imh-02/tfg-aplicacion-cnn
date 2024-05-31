from report_module.sign import Sign

"""
Clase encargada de almacenar toda la información que se genera durante el análisis, para posteriormente generar un informe a partir de esa información.
"""
class report: 

    """
    Atributos de la clase report, que almacenan la información generada durante el análisis.
    """
    number_rows_analyzed = 0

    number_dni_analyzed = 0
    number_correct_format_dni = 0
    number_incorrect_format_dni = 0
    number_valid_dni = 0
    number_not_valid_dni = 0
    repeated_dni = 0

    number_signatures_analyzed = 0
    number_signatures_detected = 0
    number_signatures_not_detected = 0

    number_pages_analyzed = 0

    total_validated_citizen_signatures = 0

    processed_dni: Sign = [] # Objeto Sign

    analysis_time = 0

    metrics = ""
    metrics_characters_eq_size = ""
    metrics_characters_dif_size = ""

    def __init__(self, number_rows_analyzed, number_dni_analyzed, number_correct_format_dni, number_incorrect_format_dni, number_valid_dni, number_not_valid_dni, number_signatures_analyzed, number_signatures_detected, number_signatures_not_detected, number_pages_analyzed):
        self.number_rows_analyzed = number_rows_analyzed
        self.number_dni_analyzed = number_dni_analyzed
        self.number_correct_format_dni = number_correct_format_dni
        self.number_incorrect_format_dni = number_incorrect_format_dni
        self.number_valid_dni = number_valid_dni
        self.number_not_valid_dni = number_not_valid_dni
        self.number_signatures_analyzed = number_signatures_analyzed
        self.number_signatures_detected = number_signatures_detected
        self.number_signatures_not_detected = number_signatures_not_detected
        self.number_pages_analyzed = number_pages_analyzed

    """
    Método que devuelve un resumen del análisis realizado en forma de cadena de texto, que permite añadir la información con formato en el fichero PDF
    del informe.
    """
    def getSummary(self):
        summary = ""
        summary += "Número de filas analizadas: " + str(self.number_rows_analyzed) + "\n"
        summary += "Número de DNI analizados: " + str(self.number_dni_analyzed) + "\n"
        summary += "Número de DNI con formato correcto: " + str(self.number_correct_format_dni) + "\n"
        summary += "Número de DNI con formato incorrecto: " + str(self.number_incorrect_format_dni) + "\n"
        summary += "Número de DNI válidos: " + str(self.number_valid_dni) + "\n"
        summary += "Número de DNI no válidos: " + str(self.number_not_valid_dni) + "\n"
        summary += "Número de DNI repetidos: " + str(self.repeated_dni) + "\n"
        summary += "Número de firmas analizadas: " + str(self.number_signatures_analyzed) + "\n"
        summary += "Número de firmas detectadas: " + str(self.number_signatures_detected) + "\n"
        summary += "Número de firmas no detectadas: " + str(self.number_signatures_not_detected) + "\n"
        summary += "Total de firmas ciudadanas validadas: " + str(self.total_validated_citizen_signatures) + "\n"
        summary += "Número de páginas analizadas: " + str(self.number_pages_analyzed) + "\n"
        summary += "Tiempo de análisis: " + str(self.analysis_time) + "\n\n"
        summary += "------------------ MÉTRICAS -----------------------------------------------------\n" + self.metrics + "\n"
        return summary
    
    """
    Método que imprime por consola el resumen del análisis realizado.
    """
    def print_report(self):
        print("Número de filas analizadas: " + str(self.number_rows_analyzed))
        print("Número de DNI analizados: " + str(self.number_dni_analyzed))
        print("Número de DNI con formato correcto: " + str(self.number_correct_format_dni))
        print("Número de DNI con formato incorrecto: " + str(self.number_incorrect_format_dni))
        print("Número de DNI válidos: " + str(self.number_valid_dni))
        print("Número de DNI no válidos: " + str(self.number_not_valid_dni))
        print("Número de DNI repetidos: " + str(self.repeated_dni))
        print("Número de firmas analizadas: " + str(self.number_signatures_analyzed))
        print("Número de firmas detectadas: " + str(self.number_signatures_detected))
        print("Número de firmas no detectadas: " + str(self.number_signatures_not_detected))
        print("Total de firmas ciudadanas validadas: " + str(self.total_validated_citizen_signatures))
        print("Número de páginas analizadas: " + str(self.number_pages_analyzed))
        print("Tiempo de análisis: " + str(self.analysis_time))
        print("Métricas: " + "\n" + str(self.metrics))

    """
    Método para buscar si un DNI está repetido y ya ha sido procesado anteriormente.
    """
    def findDNI(self, dni):
        for i in range(len(self.processed_dni)):
            if self.processed_dni[i].dni == dni:
                return True
        return False
    
    """
    Método para obtener todas las firmas ciudadanas no validadas, comparando el atributo validatedCitizenSignature de cada objeto Sign.
    """
    def getNotValidatedCitizenSignatures(self):
        result = []
        for i in range(len(self.processed_dni)):
            if self.processed_dni[i].validatedCitizenSignature == False:
                result.append(self.processed_dni[i])
        return result
    
    """
    Método para obtener todas las firmas ciudadanas validadas, comparando el atributo validatedCitizenSignature de cada objeto Sign.
    """
    def getValidatedCitizenSignatures(self):
        result = []
        for i in range(len(self.processed_dni)):
            if self.processed_dni[i].validatedCitizenSignature == True:
                result.append(self.processed_dni[i])
        return result
    

