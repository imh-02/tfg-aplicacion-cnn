import re 

"""
Función encargada de eliminar algunos caracteres especiales y espacios en blanco de un DNI, y convertirlo a mayúsculas. (Postprocesado)
"""
def formatDNI(dni):
    # Eliminamos los espacios
    dni = dni.replace(" ", "")
    # Eliminamos los puntos
    dni = dni.replace(".", "")
    # Eliminamos los guiones
    dni = dni.replace("-", "")
    # Eliminación de comas
    dni = dni.replace(",", "")
    # Pasamos a mayúsculas
    dni = dni.upper()
    return dni

"""
Función que comprueba si un DNI tiene el formato correcto, es decir, 8 dígitos y una letra al final.
"""
def checkDNIFormat(dni):
    # Patrón DNI
    patron = re.compile (r'^\d{8}[A-z]$')

    if len(dni) != 9:
        return False
    if not patron.match(dni):
        return False
    
    return True

"""
Función que comprueba si un DNI es válido respecto al algoritmo de validación de DNI español.
"""
def validSpanishDNI(dni):
    numero = int(dni[:-1])
    letra = dni[-1]
    resto = numero % 23
    letras = "TRWAGMYFPDXBNJZSQVHLCKE"
    if letra == letras[resto]:
        return True
    else:
        return False

"""
Función que corrige algunos de los errores comunes de predicción en los dígitos de un DNI.
"""
def fixDNIdigits(digit):
    # Mapa de correcciones
    correcciones = {
        'O': '0',   
        'I': '1',
        'Z': '2',
        'S': '5',
        'B': '3',
        '/': '1',
        '€': '3',
        '(': '1',
        ')': '1',
        'L': '1',
    }

    # Aplicar correcciones
    for error, correccion in correcciones.items():
        if digit == error:
            return correccion
    return digit

"""
Función que corrige algunos de los errores comunes de predicción en las letras de un DNI.
"""
def fixDNIlettersConfusions(digit):
    # Mapa de correcciones
    correcciones = {
        '1': 'T',   
        '2': 'Z',
        '3': 'B',
        '4': 'A',
        '6': 'G',
        '5': 'S',
        '€': 'E',
        '0': 'D'
    }
    
    # Aplicar correcciones
    for error, correccion in correcciones.items():
        if digit == error:
            return correccion
    return digit

"""
Función encargada de gestionar la corrección de errores de predicción en un DNI.
"""
def fixPredictionErrorsDNI(dni):
    if len(dni) < 9 or len(dni) > 9:
        return dni

    dniResult = list(dni)
    for i in range(0, 8):
        if dni[i].isnumeric() == False:
            dniResult[i] = fixDNIdigits(dni[i])

    if dni[8].isalpha() == False:
        dniResult[8] = fixDNIlettersConfusions(dni[8])
    
    return "".join(dniResult)
    
