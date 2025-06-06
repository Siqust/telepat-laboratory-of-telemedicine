from recognize import recognize
from cleaner import cleaner
from filler import filler
from pdftoimg import pdftoimg
import os
from tqdm import tqdm

for index, file in tqdm(enumerate(os.listdir('documents'))):
    images = pdftoimg('./documents/'+file, index)

    for image in images:
        data = recognize(image)
        text = [i[0] for i in data]

        clean_text, cleaned = cleaner(text)
        clean_text = [i for i in clean_text if i!='' and i!=' ']
        print(len(text), len(clean_text))
        print(text)
        print(clean_text)
        print(cleaned)

        coordinates_list = []
        for cords, word in zip(data, clean_text):
            if chr(1243) in word:
                coordinates_list.append(cords[1])
        filler(image, coordinates_list)
