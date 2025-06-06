from pdf2image import convert_from_path


def pdftoimg(pdf, index):
    images = convert_from_path(pdf, poppler_path=r'E:\python\cito_rotcbb\poppler\Library\bin')

    l = []
    for i in range(len(images)):
        images[i].save(f'file_{index}_page{i}.jpg', 'JPEG')
        l.append(f'file_{index}_page{i}')
    return l
