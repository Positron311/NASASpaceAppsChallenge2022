
# RGB
# 000 Noire
# 001 RED
# 010 BLUE
# 011 GREY
# 100 GREEN
# 101 YELLOW
# 110 ORANGE
# 111 WHITE

from PIL import Image
import numpy as np

def scale_rgb(x):
    return x * 1.0 / 255

nb_inp = 3
nb_out = 3
nb_hidden = 3


def sigmoid(x, deriv=False):
    if (deriv == True):
        return (x) * (1 - x)
    return 1 / (1 + np.exp(-x))

def trainrgb():
    X = np.array(
        [[scale_rgb(250), scale_rgb(181), scale_rgb(100)], [1, scale_rgb(165), 0], [1, scale_rgb(165), 0],  # Orange
         [scale_rgb(207), scale_rgb(75), scale_rgb(65)], [scale_rgb(173), scale_rgb(216), scale_rgb(230)],
         [0, 0, scale_rgb(255)],  # Blue
         [scale_rgb(144), scale_rgb(238), scale_rgb(144)], [0, scale_rgb(255), 0],  # Green
         [scale_rgb(239), scale_rgb(239), scale_rgb(74)], [scale_rgb(242), scale_rgb(242), scale_rgb(62)],  # Yellow
         [scale_rgb(211), scale_rgb(211), scale_rgb(211)], [scale_rgb(100), scale_rgb(100), scale_rgb(100)],  # Grey
         [scale_rgb(4), scale_rgb(11), scale_rgb(19)], [scale_rgb(0), scale_rgb(0), scale_rgb(7)],
         [scale_rgb(8), scale_rgb(11), scale_rgb(28)],  # Black
         [scale_rgb(255), scale_rgb(255), scale_rgb(255)], [scale_rgb(254), scale_rgb(254), scale_rgb(254)]  # White
         ]);
# REAL CALIBRATION
    y = [[1, 1, 0], [1, 1, 0],
         [0, 0, 1], [0, 0, 1],
         [0, 1, 0], [0, 1, 0],
         [1, 0, 0], [1, 0, 0],
         [1, 0, 1], [1, 0, 1],
         [0, 1, 1], [0, 1, 1],
         [0, 0, 0], [0, 0, 0], [0, 0, 0],
         [1, 1, 1], [1, 1, 1]
         ]

    np.random.seed(1)
    syn0 = 2 * np.random.random((nb_inp, nb_hidden)) - 1
    syn1 = 2 * np.random.random((nb_hidden, nb_out)) - 1
    for i in range(10000):
        l0 = X
        l1 = sigmoid(np.dot(l0, syn0))
        l2 = sigmoid(np.dot(l1, syn1))
        l2_error = y - l2
        l2_delta = l2_error * sigmoid(l2, deriv=True)
        l1_error = l2_error.dot(syn1.T)
        l1_delta = l1_error * sigmoid(l1, deriv=True)
        syn1 += np.dot(l1.T, l2_delta)
        syn0 += np.dot(l0.T, l1_delta)
    rgb_values = (254, 216, 177)
    r, g, b = rgb_values
    xtest = [scale_rgb(r), scale_rgb(g), scale_rgb(b)]
    l0 = xtest
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    l3=l2*2
    return (syn0, syn1)


def recognize_color(jpg):
    syn0, syn1 = trainrgb()
    img = Image.open(jpg)
    pix = img.load()
    lamp = pix
    width, height = img.size
    tot = (width * height)
    black = 0
    purple = 0
    blue = 0
    red = 0
    orange = 0
    yellow = 0
    white = 0
    green = 0
    for y in range(0, height):
        for x in range(width):
            rgb_values = img.getpixel((x, y))
            r, g, b = rgb_values
            Xtest = [scale_rgb(r), scale_rgb(g), scale_rgb(b)]
            l0 = Xtest
            l1 = sigmoid(np.dot(l0, syn0))
            l2 = sigmoid(np.dot(l1, syn1))

            if (round(l2[0], 0) == 0 and round(l2[1], 0) == 0 and round(l2[2], 0) == 0):
                black = black + 1;

            if (round(l2[0], 0) == 0 and round(l2[1], 0) == 0 and round(l2[2], 0) == 1):
                red = red + 1;

            if (round(l2[0], 0) == 0 and round(l2[1], 0) == 1 and round(l2[2], 0) == 0):
                blue = blue + 1;

            if (round(l2[0], 0) == 0 and round(l2[1], 0) == 1 and round(l2[2], 0) == 1):
                purple = purple + 1

            if (round(l2[0], 0) == 1 and round(l2[1], 0) == 0 and round(l2[2], 0) == 0):
                green = green + 1

            if (round(l2[0], 0) == 1 and round(l2[1], 0) == 0 and round(l2[2], 0) == 1):
                yellow = yellow + 1

            if (round(l2[0], 0) == 1 and round(l2[1], 0) == 1 and round(l2[2], 0) == 0):
                orange = orange + 1

            if (round(l2[0], 0) == 1 and round(l2[1], 0) == 1 and round(l2[2], 0) == 1):
                white = white + 1


    print('trace elements  : ' + str(((float(purple) / tot) * 0.5) + ((float(yellow) / tot) * 100)))
    print('ammonia and/or methane: ' + str(((float(green) / tot) * 100) + ((float(blue) / tot) * 100) + ((float(purple) / tot) * 14.5)))
    print('sulfur and phosphorous : ' + str((float(red) / tot) * 100))
    print('hydrogen and helium: ' + str(((float(white) / tot) * 100)+ ((float(orange) / tot) * 100)+ ((float(purple) / tot) * 85)))
    print('outer space :' + str((float(black) / tot) * 100))
    print('=================')
    print("100")


file_name = r'C:\Users\yousu\Desktop\JNCE_2022012_39C00014_V01-mapprojected.jpg'

analyzed = recognize_color(file_name)

print("analyzed")

