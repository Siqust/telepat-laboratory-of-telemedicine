import cv2, pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'E:\python\tesseract\tesseract.exe'


def recognize(image_name):
    data = pytesseract.image_to_data(image_name + '.jpg', output_type=Output.DICT, lang="eng+rus")
    words = []
    for i in range(len(data["text"])):
        if int(data["conf"][i]) > 0:  # Filter out low-confidence detections
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            text = data["text"][i]
            if set(text) != {' '} and text != '':
                words.append((text, (x, y, x + w, y + h)))

    return words
