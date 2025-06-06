from PIL import Image, ImageDraw


def filler(image_name, coordinates_list):
    image = Image.open(image_name + '.jpg')
    width, height = image.size  # Get dimensions

    # Flip y-coordinates (convert to top-based y=0)
    flipped_coords = []
    for box in coordinates_list:
        x1, y1, x2, y2 = box
        new_y1 = height - y2  # Original bottom becomes new top
        new_y2 = height - y1  # Original top becomes new bottom
        flipped_coords.append([x1, y1, x2, y2])

    # Draw black rectangles
    draw = ImageDraw.Draw(image)
    for box in flipped_coords:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], fill="black")

    # Save or show
    image.save(image_name + '_formatted.jpg')
    print(image_name + '_formatted.jpg')
