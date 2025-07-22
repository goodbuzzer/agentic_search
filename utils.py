from PyPDF2 import PdfReader
import os

def count_pdf_pages(folder):
    total_pages = 0
    for f in os.listdir(folder):
            file_path = os.path.join(folder, f)
            if os.path.isfile(file_path) and f.lower().endswith(".pdf"):
                try:
                    reader = PdfReader(file_path)
                    total_pages += len(reader.pages)
                except Exception:
                    pass
    return total_pages

