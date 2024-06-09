from PIL import Image
from pytesseract import pytesseract
import enum

class Language(enum.Enum):
    ENG = 'eng'
    FRA = 'fra'

class ImageReader:

    def __init__(self):
        self.windows_path = r'C:\Users\reyam\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
        pytesseract.tesseract_cmd = self.windows_path

    def extract_text(self, image: str, lang: [Language]) -> str:
        img = Image.open(image)
        lang_str = '+'.join([language.value for language in lang])
        extracted_text = pytesseract.image_to_string(img, lang=lang_str)
        return extracted_text


if __name__ == '__main__':
    ir = ImageReader()
    text = ir.extract_text(r'C:\Users\reyam\PycharmProjects\pythonProject13\test_images\img_1.png', lang=[Language.ENG, Language.FRA])
    print(text)

