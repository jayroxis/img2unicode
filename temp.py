import numpy as np
from math import sqrt
import cv2
import os


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path, 0)
        if img is not None:
            if filename[0] == 'u':
                symbol = filename.split('.')[0][1::]
                symbol = chr(int(symbol, 16))
            else:
                symbol = chr(int(filename.split('.')[0]))
            images.append([img, symbol])
    return images


def naive_transform(path, res_y, show=False):
    # Load an color image in gray scale
    img = cv2.imread(path, 0)

    # Globals
    signs = [chr(32), '\u2591', '\u2592', '\u2593', '\u2588']

    res_x = res_y * 2
    div_x = int(np.shape(img)[0] / res_x)
    div_y = int(np.shape(img)[1] / res_y)

    f = open('out_naive.txt', 'w', encoding='utf-8')
    for i in range(div_x):
        line = ''
        for j in range(div_y):
            fraction = img[i * res_x:(i + 1) * res_x, j * res_y:(j + 1) * res_y]
            index = int(
                np.sum(fraction) / np.size(fraction) / (255 / 5)
            )
            line = signs[4 - index] + line
        print(line)
        line = line + '\n'
        f.write(line)

    f.close()

    if show:
        # draw
        title = "HaHa"
        scaling = img.shape[0] / 540
        cv2.imshow(
            title, cv2.resize(
                img, (int(img.shape[1] / scaling), int(img.shape[0] / scaling))
            )
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class Translator:
    def __init__(self, img, res_y):
        # Load an color image in gray scale
        self.img = cv2.imread(img, 0)
        if self.img.shape[0] < 1920 and self.img.shape[1] < 1920:
            scaling = self.img.shape[1] / 1920
            self.img = cv2.resize(
                self.img,
                (int(self.img.shape[1] / scaling), int(self.img.shape[0] / scaling)),
                interpolation=cv2.INTER_CUBIC
            )
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        self.img = clahe.apply(self.img)

        # self.img = cv2.equalizeHist(self.img)
        # cv2.adaptiveThreshold(src=self.img,
        #                       maxValue=100,
        #                       adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                       thresholdType=cv2.THRESH_BINARY,
        #                       blockSize=3,
        #                       C=5)

        print('max point =', np.max(self.img))
        print('min point =', np.min(self.img))
        self.res_y = res_y
        self.res_x = self.res_y * 2
        self.div_x = int(np.shape(self.img)[0] / self.res_x)
        self.div_y = int(np.shape(self.img)[1] / self.res_y)
        self.templates = load_images_from_folder('templates\\')
        self.f = open('out.txt', 'w', encoding='utf-8')
        # resize to resolution
        for i in range(len(self.templates)):
            self.templates[i][0] = cv2.resize(
                src=self.templates[i][0],
                dsize=(self.res_y * 2, self.res_x * 2),
                interpolation=cv2.INTER_AREA
            )

    def translate(self):
        for i in range(self.div_x):
            line = ''
            for j in range(self.div_y):
                fraction = self.img[
                           i * self.res_x:(i + 1) * self.res_x,
                           j * self.res_y:(j + 1) * self.res_y
                           ]
                symbol, distance = self.euclidean_distance(fraction)
                line = line + symbol
            print(line)
            line = line + '\n'
            self.f.write(line)
        self.f.close()

    def show(self):
        scaling = self.img.shape[0]/540
        cv2.imshow(
            'img', cv2.resize(
                self.img, (int(self.img.shape[1]/scaling), int(self.img.shape[0]/scaling))
            )
        )
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    def euclidean_distance(self, img):
        min_distance = 255 * 255
        best_symbol = ''
        for tmp, symbol in self.templates:
            img = cv2.resize(
                src=img,
                dsize=(self.res_y * 2, self.res_x * 2),
                interpolation=cv2.INTER_AREA
            )
            distance = mse(tmp, img)
            if distance <= min_distance:
                min_distance = distance
                best_symbol = symbol
        return best_symbol, min_distance


def main():
    path = r'images\generatedtext .jpg'
    res = 18
    t = Translator(path, res)
    t.translate()
    t.show()
    naive_transform(path, res)


main()
