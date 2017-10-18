# -*- coding: utf-8 -*-

from __future__ import print_function

import face_recognition
import argparse
from contextlib import contextmanager
import cv2
import glob
import numpy as np
import os
import sys
from .__version__ import __version__

fixexp = True  # Flag to fix underexposition
INPUT_FILETYPES = ['*.jpg', '*.jpeg', '*.bmp', '*.dib', '*.jp2',
                   '*.png', '*.webp', '*.pbm', '*.pgm', '*.ppm',
                   '*.sr', '*.ras', '*.tiff', '*.tif']
INCREMENT = 0.06
GAMMA_THRES = 0.001
GAMMA = 0.90
FACE_RATIO = 6

# Load XML Resource
cascFile = 'haarcascade_frontalface_default.xml'
d = os.path.dirname(sys.modules['autocrop'].__file__)
cascPath = os.path.join(d, cascFile)
print(d)

# Define directory change within context


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


# Define simple gamma correction fn
def gamma(img, correction):
    img = cv2.pow(img / 255.0, correction)
    return np.uint8(img * 255)


def detect_cv(image, minSize):
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # ====== Detect faces in the image ======
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(minSize, minSize),
        flags=cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH
    )

    if len(faces) == 0:
        # Handle no faces
        return None
    else:
        return faces[-1]


def detect_fr(image):
    faces = face_recognition.face_locations(image)
    if len(faces) == 0:
        # Handle no faces
        return None
    top, right, bottom, left = faces[-1]
    x, y, w, h = left, top, abs(right - left), abs(bottom - top)
    return [x, y, w, h]


def crop(image, fwidth=500, fheight=500):
    """Given a ndarray image with a face, returns cropped array.

    Arguments:
        - image, the numpy array of the image to be processed.
        - fwidth, the final width (px) of the cropped img. Default: 500
        - fheight, the final height (px) of the cropped img. Default: 500
    Returns:
        - image, a cropped numpy array

    ndarray, int, int -> ndarray
    """
    try:
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            print("Gray Scale image with shape: ", image.shape)
            gray = image

        # Histogram Equalization
        gray = cv2.equalizeHist(gray)

        # Scale the image
        height, width = (image.shape[0], image.shape[1])
        minface = int(np.sqrt(height * height + width * width) / 8)

        rectangle = detect_cv(gray, minface)
        # if rectangle is None:
        #     # try another way
        #     rectangle = detect_fr(gray)
        if rectangle is None:
            # try another way with color image
            rectangle = detect_fr(image)
        if rectangle is None:
            # print('rot90')
            image = np.rot90(image)
            rectangle = detect_fr(image)
        if rectangle is None:
            # print('rot270')
            image = np.rot90(image, 2)
            rectangle = detect_fr(image)
        if rectangle is None:
            return None

        x, y, w, h = rectangle
        height, width = (image.shape[0], image.shape[1])

        # Make padding from probable biggest face
        pad = h / FACE_RATIO

        # Make sure padding is contained within picture
        # decreases pad by 6% increments to fit crop into image.
        # Can lead to very small faces.
        count = 0
        while True:
            if ((y - 2 * pad < 0 or y + h + pad > height or
                    int(x - 1.5 * pad) < 0 or x + w + int(1.5 * pad) > width)) and count <= 15000:
                pad = (1 - INCREMENT) * pad
                count += 1
            else:
                break

        # Crop the image from the original
        h1 = int(x - 1.5 * pad)
        h2 = int(x + w + 1.5 * pad)
        v1 = int(y - 2 * pad)
        v2 = int(y + h + pad)
        image = image[v1:v2, h1:h2]

        # Resize the damn thing
        image = cv2.resize(image, (fheight, fwidth), interpolation=cv2.INTER_AREA)

        # ====== Dealing with underexposition ======
        if fixexp:
            # Check if under-exposed
            uexp = cv2.calcHist([gray], [0], None, [256], [0, 256])
            if sum(uexp[-26:]) < GAMMA_THRES * sum(uexp):
                image = gamma(image, GAMMA)
        return image
    except:
        return None


def main(path, fheight, fwidth, output_dir):
    """Given path containing image files to process, will
    1) copy them to `path/bkp`, and
    2) create face-cropped versions and place them in `path/crop`
    """
    errors = 0
    print("Hi crop")
    with cd(path):
        files_grabbed = []
        for files in INPUT_FILETYPES:
            files_grabbed.extend(glob.glob(files))

        for file in files_grabbed:
            # print('processing file {}.'.format(str(file)))

            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            cropfilename = os.path.join(output_dir, str(file))
            # print("Cropfilename: {}".format(cropfilename))

            if os.path.isfile(cropfilename):
                continue

            # Perform the actual crop
            input = cv2.imread(file)

            if input is None:
                continue

            image = crop(input, fwidth, fheight)

            # Make sure there actually was a face in there
            if image is None:
                print('No faces can be detected in file {}.'.format(str(file)))
                if not os.path.exists('./noface'):
                    os.makedirs('./noface')
                os.rename(file, './noface/' + file)
                errors += 1
                continue

            # Write cropfile
            cv2.imwrite(cropfilename, image)

    # Stop and print timer
    print(' {0} files have been cropped'.format(len(files_grabbed) - errors))


def cli():
    help_d = {
        'description': 'Automatically crops faces from batches of pictures',
        'path': 'Folder where images to crop are located. Default=photos/',
        'width': 'Width of cropped files in px. Default=500',
        'height': 'Height of cropped files in px. Default=500',
        'output_dir': 'Folder to save the cropped images'
    }

    parser = argparse.ArgumentParser(description=help_d['description'])
    parser.add_argument('-p', '--path', default='photos', help=help_d['path'])
    parser.add_argument('-w', '--width', type=int,
                        default=500, help=help_d['width'])
    parser.add_argument('-H', '--height',
                        type=int, default=500, help=help_d['height'])
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s version {}'.format(__version__))
    parser.add_argument('-o', '--output_dir', type=str,
                        default=os.path.join(d, 'cropped_images'),
                        help='path to save the output cropped files')

    args = parser.parse_args()
    print('Processing images in folder:', args.path)

    main(args.path, args.height, args.width, args.output_dir)
