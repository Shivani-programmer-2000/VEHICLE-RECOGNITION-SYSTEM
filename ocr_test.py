import pytesseract

# Set the path explicitly in case it is not detected automatically
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Test if OCR works
from PIL import Image
text = pytesseract.image_to_string(Image.open(r"C:\Users\Windows\vrs-web-app\th.jpeg"))
print("Extracted Text:", text)
