import numpy as np
from math import sqrt
import cv2
import os
import sys
from pathlib import Path
from typing import List, Tuple


def mse(imageA: np.ndarray, imageB: np.ndarray) -> float:
    """
    Calculates the Mean Squared Error (MSE) between two images.

    Args:
        imageA (np.ndarray): The first image to compare.
        imageB (np.ndarray): The second image to compare.

    Returns:
        float: The MSE between the two images.
    """
    # Calculate the squared difference between the two images
    diff = (imageA.astype("float") - imageB.astype("float")) ** 2

    # Calculate the mean of the squared difference
    mse = np.mean(diff)

    return mse



def load_images_from_folder(folder: str) -> list:
    """
    Loads all images from a folder and extracts their corresponding 
    symbols.

    Args:
        folder (str): The path to the folder containing the images.

    Returns:
        list: A list of tuples, where each tuple contains an image
        and its corresponding symbol.
    """
    images = []
    for filename in os.listdir(folder):
        if filename.startswith('.'):
            continue

        # Get the full path of the file
        path = os.path.join(folder, filename)

        # Load the image and its corresponding symbol
        if filename.startswith('u'):
            try:
                symbol = chr(int(filename.split('.')[0][1:], 16))
            except ValueError:
                raise ValueError(
                    "Invalid hexadecimal symbol in filename: "
                    f"{filename}")
        else:
            try:
                symbol = chr(int(filename.split('.')[0]))
            except (ValueError, IndexError):
                raise ValueError(
                    "Invalid decimal symbol in filename: "
                    f"{filename}")

        # Load the image in grayscale mode
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Append the image and symbol to the list
        if img is not None:
            images.append([img, symbol])

    return images



def naive_transform(path: str, res_y: int, output_dir: str, out_file: str) -> None:
    """
    Transforms an image into ASCII art using a naive method.

    Args:
        path (str): The path to the image file to be transformed.
        res_y (int): The height of each block in the output image.
        output_dir (str): The path to the directory where the output file should be saved.
        out_file (str): The name of the output file.

    Returns:
        None
    """
    # Normalize the path to the current platform's format
    path = os.path.normpath(path)

    # Load the image in grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to load image at path: " + path)

    # Define the symbols to represent the image
    symbols = [' ', '\u2591', '\u2592', '\u2593', '\u2588']

    # Calculate the width and height of the output image
    res_x = res_y * 2
    height, width = img.shape[:2]

    # Calculate the number of blocks to divide the image into
    div_x = height // res_x
    div_y = width // res_y

    # Determine the path to the output file
    out_path = os.path.join(output_dir, out_file)

    # Open the output file
    with open(out_path, 'w', encoding='utf-8', newline=os.linesep) as f:
        # Loop through each block
        for i in range(0, height, res_x):
            line = ''
            for j in range(0, width, res_y):
                # Extract the current block
                fraction = img[i:i+res_x, j:j+res_y]

                # Calculate the index of the symbol to represent the block
                index = int(np.mean(fraction) / 51)

                # Add the symbol to the line
                line += symbols[4 - index]

            # Add a newline character to the line
            line += '\n'

            # Print the line to the console and write it to the output file
            try:
                print(line, end='')
                f.write(line)
            except UnicodeEncodeError:
                pass




class Translator:
    def __init__(self, img_path: str, res_y: int):
        """
        Initializes the Translator class with the given image path and
        the resolution of y axis.

        Args:
            img_path (str): Path of the input image file.
            res_y (int): The resolution of the y-axis.

        Returns:
            None
        """
        # Read the image in grayscale mode
        self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Load the templates from the folder
        self.templates = load_images_from_folder('templates/')

        # Set the resolution of x-axis and y-axis
        self.res_y = res_y
        self.res_x = res_y * 2

        # Divide the image into segments with respect to resolution
        self.div_x, self.div_y = self.img.shape[:2] \
                                 // np.array([self.res_x, self.res_y])

        # Resize the image if it has a maximum dimension greater than 1920
        max_dim = max(self.img.shape[:2])
        if max_dim > 1920:
            scaling = max_dim / 1920
            new_size = (int(self.img.shape[1] / scaling),
                        int(self.img.shape[0] / scaling))
            self.img = cv2.resize(self.img, new_size,
                                   interpolation=cv2.INTER_CUBIC)

        # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the image
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        self.img = clahe.apply(self.img)

        # Print the maximum and minimum points of the image
        print(f"Max point = {np.max(self.img)}")
        print(f"Min point = {np.min(self.img)}")

        # Resize the templates with respect to the resolution of x-axis and y-axis
        for i, (template, symbol) in enumerate(self.templates):
            new_template = cv2.resize(template, (self.res_y * 2, self.res_x * 2),
                                       interpolation=cv2.INTER_AREA)
            self.templates[i] = (new_template, symbol)

    def translate(self, output_dir: str, out_file: str) -> None:
        """
        Translates the input image and writes the output to a text file.

        Args:
            output_dir (str): The path to the directory where the output file should be saved.
            out_file (str): The name of the output file.

        Returns:
            None
        """
        # Determine the path to the output file
        out_path = os.path.join(output_dir, out_file)

        with open(out_path, 'w', encoding='utf-8') as f:
            for i, row in enumerate(np.array_split(self.img, self.div_x, axis=0)):
                line = ''
                for fraction in np.array_split(row, self.div_y, axis=1):
                    symbol, _ = self.find_best_match(fraction)
                    line += symbol
                print(line)
                line += '\n'
                f.write(line)


    def show(self) -> None:
        """
        Shows the input image in a window.

        Args:
            None

        Returns:
            None
        """
        scaling = self.img.shape[0] / 540
        cv2.imshow('img', cv2.resize(self.img,
                                      (int(self.img.shape[1] / scaling),
                                       int(self.img.shape[0] / scaling))))
        cv2.destroyAllWindows()


    def find_best_match(self, fraction: np.ndarray) -> Tuple[str, float]:
        """
        Finds the best matching symbol from the templates for the
        given fraction of the image.

        Args:
            fraction (numpy.ndarray): A fraction of the input image.

        Returns:
            tuple: A tuple containing the best matching symbol and
            its distance from the fraction
        """
        min_distance = np.inf
        best_symbol = ''

        # Resize the fraction with respect to the resolution of x-axis and y-axis
        fraction = cv2.resize(fraction, (self.res_y * 2, self.res_x * 2),
                              interpolation=cv2.INTER_AREA)

        # Find the best matching symbol from the templates using Mean Squared Error (MSE) method
        for template, symbol in self.templates:
            distance = mse(template, fraction)
            if distance < min_distance:
                min_distance = distance
                best_symbol = symbol

        return best_symbol, min_distance

    
    
def main():
    """
    The main function for the ASCII Art Translator program.

    Args:
        None

    Returns:
        None
    """
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=r'./images/eiffel.jpg')
    parser.add_argument("--output", type=str, default=r'./output/')
    parser.add_argument("--out_file", type=str, default='out.txt')
    args = parser.parse_args()

    # Set the path of the input image and the resolution of the y-axis
    path = args.path
    res = 18

    # Create a Translator object and translate the input image
    t = Translator(path, res)
    out_file = args.out_file
    t.translate(args.output, out_file)

    # Transform the input image into ASCII art using a naive method
    naive_transform(path, res, args.output, 'out_naive.txt')


if __name__ == '__main__':
    main()
