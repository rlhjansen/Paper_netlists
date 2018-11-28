

from PIL import Image, ImageDraw


xdif = 7
ydif = 18


def to_border(f, fromedge):
    img = Image.open(f)
    img.convert('RGBA')
    x, y = img.size
    print("x", x, "\ny", y)

    d = ImageDraw.Draw(img)
    d.rectangle((fromedge,fromedge,x-fromedge,y-fromedge),fill=(255,255,255,255))
    img.save("bordered_image.png")


to_border("achtergrond_cropped.png", 20)