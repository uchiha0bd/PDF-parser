import os
import shutil
from pdf2image import pdfinfo_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError

# The path to your PDF file
pdf_file = 'sample_document.pdf'

print("--- Starting Poppler Debug Script ---")

# --- Test 1: Check the system PATH from within Python ---
print("\n[1] Checking the system PATH that Python sees...")
try:
    path_variable = os.environ.get('PATH', 'PATH variable not found!')
    print(f"PATH = {path_variable}")
except Exception as e:
    print(f"Error getting PATH: {e}")

# --- Test 2: Use Python's own tool to find 'pdfinfo' ---
print("\n[2] Trying to locate 'pdfinfo' executable with shutil.which()...")
try:
    pdfinfo_location = shutil.which('pdfinfo')
    if pdfinfo_location:
        print(f"SUCCESS: Found 'pdfinfo' at: {pdfinfo_location}")
    else:
        print("FAILURE: shutil.which('pdfinfo') could NOT find the executable.")
except Exception as e:
    print(f"Error running shutil.which(): {e}")

# --- Test 3: Attempt the failing operation and catch the error ---
print("\n[3] Attempting to run pdfinfo_from_path() from pdf2image...")
try:
    info = pdfinfo_from_path(pdf_file)
    print("SUCCESS: pdf2image was able to get info from the PDF!")
    print(f"PDF Info: {info}")
except PDFInfoNotInstalledError:
    print("FAILURE: Caught the 'PDFInfoNotInstalledError' exception, as expected.")
except Exception as e:
    print(f"UNEXPECTED ERROR: An error other than the expected one occurred: {e}")

print("\n--- Debug Script Finished ---")
