

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.misc import toimage



#chars = np.asarray(list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,^`'. "))
chars = np.asarray(list("@B%8&WM#ZO0QCLJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!;:,^`'. "))
xdif = 7
ydif = 18

GCF = 0.1






def to_border(f, fromedge):
    img = Image.open(f)
    img.convert('RGBA')
    x, y = img.size
    print("x", x, "\ny", y)

    d = ImageDraw.Draw(img)
    d.rectangle((fromedge,fromedge,x-fromedge,y-fromedge),fill=(255,255,255,255))
    img.save("bordered_image.png")


to_border("background_image.jpg", 50)