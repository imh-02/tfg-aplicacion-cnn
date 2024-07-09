#!/usr/bin/env python3
import numpy as np
import time
from modules.cnn_model import CNN, CNN_letters, BidirectionalLSTM, CRNN

from modules.preprocess import convert_page_to_image
from modules.table_features import cell_detection
from modules.extract_data import getOrderedRow, getOrderedCol, processRow, extractDNI, detectSignature
from modules.data_proccessing import formatDNI, checkDNIFormat, validSpanishDNI, fixPredictionErrorsDNI
from report_module.report import report
from report_module.sign import Sign
from report_module.report_pdf import reportPDF
import fitz

from modules.metricsNPL import MetricsNPL, readTrueFile
from modules.metricsML import MetricsML
from routes import pdf_file_path, output_file, true_file_path

"""
Objeto de la clase report que se encargará de almacenar los datos del análisis para el informe final
"""
final_report = report(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

"""
Función principal que se encarga de realizar el análisis de las páginas de un documento pdf llamando a las funciones necesarias
"""
def main():
    trueDNIArray = readTrueFile(true_file_path)
    pdf_document = fitz.open(pdf_file_path)

    # Comprobación que hay tantos DNI reales como debería haber importante para las métricas
    if len(trueDNIArray) != len(pdf_document) * 10:
        print("El número de DNI reales no coincide con el número de DNI que debería haber en el documento.")
        return

    metricsNPL = MetricsNPL(trueDNIArray, len(trueDNIArray))
    metricsML = MetricsML(trueDNIArray, len(trueDNIArray))

    print("Analizando...")

    inicio = time.time()

    pageNumber = 1
    true_dni_index = 0

    for page in pdf_document:

        print("Analizando página ", pageNumber)
        
        image = convert_page_to_image(page)

        final_report.number_pages_analyzed += 1

        results = cell_detection(image)
        ordered_row = getOrderedRow(results)
        ordered_col = getOrderedCol(results)

        ordered_row = np.array(ordered_row)
        ordered_col = np.array(ordered_col)

        # Puede haber casos en los que no se detecten las 10 filas esperadas
        if len(ordered_row) < 11:
            print("El número de filas detectadas no es el esperado en la página ", pageNumber)
            pageNumber += 1
            for i in range(true_dni_index, true_dni_index + 10):
                # Extraer DNI real y eliminarlo del array
                trueDNIArray.pop(true_dni_index)
                metricsNPL.total_number_of_dni -= 1
                metricsML.total_number_of_dni -= 1
            metricsNPL.y_true = trueDNIArray
            metricsML.y_true = trueDNIArray
        else:

            for i in range(1, 11):

                newSign = Sign("", "", "", "", "", pageNumber, i) # Objeto Sign para almacenar los datos de la firma en análisis actual

                signatureFlag = False 

                # Sumar filas al report
                final_report.number_rows_analyzed += 1

                dniCoordenates = (ordered_col[2][0]+1.5,ordered_row[i][1]+3.0, ordered_col[2][2]-5.0, ordered_row[i][3]-3.0)
                signatureCoordenates = (ordered_col[3][0]+1.0,ordered_row[i][1]+3, ordered_col[3][2]-11.0, ordered_row[i][3]-4.0)
                result = processRow(image, dniCoordenates, signatureCoordenates)

                extracted_dni = extractDNI(result[0])
            
                
                final_report.number_signatures_analyzed += 1

                if detectSignature(result[1]) == True:
                    signatureFlag = True

                    newSign.signatureValid = True

                    final_report.number_signatures_detected += 1
                else:
                    newSign.signatureValid = False
                    final_report.number_signatures_not_detected += 1


                # Sumar dni al report
                final_report.number_dni_analyzed += 1

                # Métricas antes del postprocesado
                metricsNPL.addPredictionBeforePostProcessed(extracted_dni)
                metricsML.addPredictionBeforePostProcessed(extracted_dni)

                #Procesado del DNI
                formated_dni = formatDNI(extracted_dni)

                # Métricas después de eliminar caracteres especiales
                metricsNPL.addPredictionDeletingSpecialChars(formated_dni)
                metricsML.addPredictionDeletingSpecialChars(formated_dni)

                formated_dni = fixPredictionErrorsDNI(formated_dni)

                newSign.dni = formated_dni

                # Métricas después del postprocesado
                metricsNPL.addPredictionPostProcessed(formated_dni)     
                metricsML.addPredictionPostProcessed(formated_dni)          

                if checkDNIFormat(formated_dni):
                    newSign.dniFormat = True
                    if final_report.findDNI(formated_dni) == False:
                        final_report.number_correct_format_dni += 1
                        if validSpanishDNI(formated_dni):
                            final_report.number_valid_dni += 1

                            newSign.dniValid = True

                            if signatureFlag == True:
                                newSign.validatedCitizenSignature = True
                                final_report.total_validated_citizen_signatures += 1
                            else:
                                newSign.validatedCitizenSignature = False 
                            final_report.processed_dni.append(newSign)
                        else:
                            newSign.dniValid = False
                            newSign.validatedCitizenSignature = False
                            final_report.processed_dni.append(newSign)
                            final_report.number_not_valid_dni += 1
                    else:
                        final_report.repeated_dni += 1                   

                else:
                    newSign.dniFormat = False
                    newSign.dniValid = False
                    newSign.validatedCitizenSignature = False
                    if final_report.findDNI(formated_dni) == False:
                        final_report.number_incorrect_format_dni += 1
                        final_report.processed_dni.append(newSign)
                    else:
                        final_report.repeated_dni += 1
                true_dni_index += 1
            pageNumber += 1



    fin = time.time()
    final_report.analysis_time = time.strftime('%H:%M:%S', time.gmtime(fin-inicio))



    final_report.metrics = metricsNPL.get_metrics() + '\n' + metricsML.get_metrics()
    final_report.print_report()
    
    print("Generando informe...")

    finalPdfReport = reportPDF()
    finalPdfReport.set_title("Informe de análisis de firmas ciudadanas")
    finalPdfReport.save_report( "reports\\" + output_file + ".pdf", 
                               final_report.getSummary(), 
                               final_report.getNotValidatedCitizenSignatures(), 
                               final_report.getValidatedCitizenSignatures())	


    pdf_document.close()

main()
