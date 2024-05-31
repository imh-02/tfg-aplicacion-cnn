"""
Clase encargada de almacenar la información de cada firmante tras el análisis.
"""
class Sign:

    def __init__(self, dni, dniFormat = False, dniValid = False, signatureValid = False, validatedCitizenSignature = False, pageNumber = 0, rowNumber = 0):
        self.dni = dni
        self.dniFormat = dniFormat
        self.dniValid = dniValid
        self.signatureValid = signatureValid
        self.validatedCitizenSignature = validatedCitizenSignature
        self.pageNumber = pageNumber
        self.rowNumber = rowNumber
    
    """
    Método que devuelve la información de cada firmante en forma de diccionario.	
    """
    def get_sign(self):
        return {
            "DNI": self.dni,
            "DNI Format": self.dniFormat,
            "DNI Valid": self.dniValid,
            "Signature Valid": self.signatureValid,
            "Validated Citizen Signature": self.validatedCitizenSignature,
            "Page Number": self.pageNumber,
            "Row Number": self.rowNumber
                    }

