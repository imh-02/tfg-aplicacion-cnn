from fpdf import FPDF
import os
from routes import reports_path_root

"""
Clase encargada de generar el informe en formato PDF a partir de la información generada durante el análisis.
"""
class reportPDF(FPDF):

    summaryContent = " " # Resumen del análisis
    notValidCitizenSings = [] # Objetos del tipo Sign que almacenan las firmas ciudadanas no válidas
    validCitizenSigns = [] # Objetos del tipo Sign que almacenan las firmas ciudadanas válidas
    
    """
    Método para establecer toda la configuración del Header del informe PDF que se generará.
    """
    def header(self):
        # Setting font: helvetica bold 15
        self.image("img\\marca-universidad-de-la-laguna-original.png", 10, 8, 33)
        self.set_font("helvetica", "B", 15)
        # Calculating width of title and setting cursor position:
        width = self.get_string_width(self.title) + 6
        self.set_x((210 - width) / 2)
        # Setting colors for frame, background and text:
        self.set_draw_color(0, 0, 0)
        self.set_fill_color(255,255,255)
        self.set_text_color(0,0,0)
        # Setting thickness of the frame (1 mm)
        self.set_line_width(1)
        # Printing title:
        self.cell(
            width,
            9,
            self.title,
            border=1,
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
            fill=True,
        )
        # Performing a line break:
        self.ln(10)

    """
    Método para establecer toda la configuración del Footer del informe PDF que se generará.
    """
    def footer(self):
        # Setting position at 1.5 cm from bottom:
        self.set_y(-15)
        # Setting font: helvetica italic 8
        self.set_font("helvetica", "I", 8)
        # Setting text color to gray:
        self.set_text_color(128)
        # Printing page number
        self.cell(0, 10, f"Página {self.page_no()}", align="C")

    """
    Método para establecer toda la configuración del título de cada capítulo del informe PDF que se generará.
    """
    def chapter_title(self, num, label):
        # Setting font: helvetica 12
        self.set_font("helvetica", "", 12)
        self.set_text_color(255,255,255)
        # Setting background color
        self.set_fill_color(87,6,140)
        # Printing chapter name:
        self.cell(
            0,
            6,
            f"{num} : {label}",
            align="L",
            new_x="LMARGIN",
            new_y="NEXT",
            fill=True,
        )
        # Performing a line break:
        self.ln(4)
    
    """
    Método para establecer toda la configuración del capítulo inicial que muestra un resumen de todo el análisis realizado.
    """
    def summaryChapter(self, num, title):

        self.chapter_title(num, title)

        self.set_text_color(0,0,0)

        # Setting font: Times 12
        self.set_font("helvetica", size=12)
        # Printing justified text:
        self.multi_cell(0, 5, self.summaryContent)
        # Performing a line break:
        self.ln()
        # Final mention in italics:
        self.set_font("helvetica", style="I")
    
    """
    Método para establecer toda la configuración de los capítulos con información tabular, tanto de firmas validadas como no validadas.
    """
    def tabularChapter(self, num, title, signatures):

        self.chapter_title(num, title)

        self.set_text_color(0,0,0)

        self.set_fill_color(255,255,255)

        first_row = ["DNI", "Formato DNI", "Validez DNI", "Validez Firma", "Firma ciudadana validada", "Número de página", "Fila de la tabla"]

        with self.table(align="C") as table:
            row = table.row()
            for cell in first_row:
                row.cell(cell)

            for i in range(len(signatures)):
                row = table.row()
                row.cell(str(signatures[i].dni))

                if signatures[i].dniFormat == True:
                    row.cell("Sí")
                else:
                    row.cell("No")
                
                if signatures[i].dniValid == True:
                    row.cell("Sí")
                else:
                    row.cell("No")

                if signatures[i].signatureValid == True:
                    row.cell("Sí")
                else:
                    row.cell("No")

                if signatures[i].validatedCitizenSignature == True:
                    row.cell("Sí")
                else:
                    row.cell("No")

                row.cell(str(signatures[i].pageNumber))
                row.cell(str(signatures[i].rowNumber))
    
    """
    Método para guardar el informe en formato PDF en la ruta especificada.
    """
    def save_report(self, file_path, resume, notValidCitizenSings, validCitizenSigns):
        if os.path.exists(file_path):
            print(f"Ya existe un informe con ese nombre {file_path}, se procederá a eliminarlo. ¿Desea continuar? (s/n)")
            response = input()
            if response == "S" or response == "s":
                os.remove(file_path)
                self.output(file_path)
            elif response == "N" or response == "n":
                print("Introduzca un nuevo nombre para el informe: ")
                file_path = input()
                self.save_report(reports_path_root + file_path + ".pdf", resume, notValidCitizenSings, validCitizenSigns)
            else:
                print("Respuesta no válida")
                self.save_report(file_path, resume, notValidCitizenSings, validCitizenSigns)
        else:
            self.add_page()
            self.summaryContent = resume
            self.summaryChapter(1, "Resumen del análisis")
            self.tabularChapter(2, "Firmas ciudadanas no validadas", notValidCitizenSings)
            ## Separación en blanco entre capítulos
            self.ln(10)
            self.tabularChapter(3, "Firmas ciudadanas validadas", validCitizenSigns)
            self.output(file_path)
