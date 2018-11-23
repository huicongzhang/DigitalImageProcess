import PIL.Image
import os, sys
def convert(dir):
    path = dir
    PIL.Image.open(path).save("Marshmello"+".bmp")
if __name__ == "__main__":
    dir = "/Users/zhanghuicong/CODE/DigitalImageProcess/EXP1/109951163607348960.jpg"
    convert(dir)