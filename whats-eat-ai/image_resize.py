#!/usr/bin/python
from PIL import Image
import os, sys

path = "whats-eat-ai/img/"
dirs = os.listdir( path )

def resize():
    for direc in dirs:
        directory = os.listdir(path + direc)
        path2 = path + direc + "/"
        print(path2)
        for item in directory:
            print(path2 + item)
            if os.path.isfile(path2+item):
                im = Image.open(path2+item)
                f, e = os.path.splitext(path2+item)
                imResize = im.resize((256,256), Image.ANTIALIAS)
                imResize.save(f + '.jpg', 'JPEG', quality=100)

resize()