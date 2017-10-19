# -*- coding: utf-8 -*-

from __future__ import print_function
# import imutils
from imutils.face_utils import FaceAligner
# from imutils.face_utils import rect_to_bb
import dlib
import face_recognition
import argparse
from contextlib import contextmanager
import cv2
import glob
import numpy as np
import os
import hashlib
from time import clock as now
# import sys

INPUT_FILETYPES = ['*.jpg', '*.jpeg', '*.bmp', '*.dib', '*.jp2',
                   '*.png', '*.webp', '*.pbm', '*.pgm', '*.ppm',
                   '*.sr', '*.ras', '*.tiff', '*.tif']
INCREMENT = 0.06
GAMMA_THRES = 0.001
GAMMA = 0.90
FACE_RATIO = 6

# Load XML Resource
d = './resource'
print(d)
cascFile = 'haarcascade_frontalface_default.xml'
cascFile1 = 'haarcascade_frontalface_alt.xml'
cascFile3 = 'haarcascade_frontalface_alt_tree.xml'
shape_predictor = 'shape_predictor_68_face_landmarks.dat'

cascPath = os.path.join(d, cascFile)
cascPath1 = os.path.join(d, cascFile1)
cascPath3 = os.path.join(d, cascFile3)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
faceCascade1 = cv2.CascadeClassifier(cascPath1)
faceCascade3 = cv2.CascadeClassifier(cascPath3)

# dlib algorithm with align
shape_predictor_path = os.path.join(d, shape_predictor)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
fa = FaceAligner(predictor, desiredFaceWidth=500)


@contextmanager
# Define directory change within context
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


def re_expose(image):
    # Check if under-exposed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    uexp = cv2.calcHist([gray], [0], None, [256], [0, 256])
    if sum(uexp[-26:]) < GAMMA_THRES * sum(uexp):
        image = gamma(image, GAMMA)


def detect_dlib(image, gray):
    rects = detector(gray, 2)
    if len(rects) == 0:
        return None
    else:
        image = fa.align(image, gray, rects[-1])
        return image


def detect_cv(image):
    height, width = (image.shape[0], image.shape[1])
    minSize = int(np.sqrt(height * height + width * width) / 8)
    # ====== Detect faces in the image ======
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(minSize, minSize),
        flags=cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH
    )
    if len(faces) == 0:
        # print('xml1')
        faces = faceCascade1.detectMultiScale(
            image,
            scaleFactor=1.2,
            minNeighbors=3,
            minSize=(minSize, minSize),
            flags=cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH
        )
    if len(faces) == 0:
        # print('xml3')
        faces = faceCascade3.detectMultiScale(
            image,
            scaleFactor=1.2,
            minNeighbors=3,
            minSize=(minSize, minSize),
            flags=cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH
        )
    if len(faces) == 0:
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
        # ===smooth===
        # image = cv2.bilateralFilter(image, 9, 75, 75)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # ===Histogram Equalization===
        gray = cv2.equalizeHist(gray)

        # ===try opencv===
        rectangle = detect_cv(gray)

        # ===try dlib===
        if rectangle is None:
            output = detect_dlib(image, gray)
            if output is not None:
                re_expose(output)
                return output

        # ===try fr===
        if rectangle is None:
            rectangle = detect_fr(image)

        if rectangle is None:
            # print('rot90')
            image = np.rot90(image)
            gray = np.rot90(gray)
            rectangle = detect_cv(gray)

        if rectangle is None:
            # print('rot270')
            image = np.rot90(image, 2)
            gray = np.rot90(gray, 2)
            rectangle = detect_cv(gray)

        # ===no face detected===
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

        return image

    except:
        return None


def crop_wrapper(path, fheight, fwidth, output_dir):
    """Given path containing image files to process, will
    1) copy them to `path/bkp`, and
    2) create face-cropped versions and place them in `path/crop`
    """
    errors = 0
    print("Hi, start cropping...")
    with cd(path):
        files_grabbed = []
        for files in INPUT_FILETYPES:
            files_grabbed.extend(glob.glob(files))

        total_file = len(files_grabbed)
        processed = 0
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
                # print('No faces detected in {}.'.format(str(file)))
                if not os.path.exists('./noface'):
                    os.makedirs('./noface')
                os.rename(file, './noface/' + file)
                errors += 1
                continue

            # Write cropfile
            cv2.imwrite(cropfilename, image)
            processed += 1
            print(str(processed) + ' / ' + str(total_file))

    # Stop and print timer
    print(' {0} files have been cropped'.format(len(files_grabbed) - errors))


def getmd5(filename):
    file_txt = open(filename, 'rb').read()
    m = hashlib.md5()
    m.update(file_txt)
    return m.hexdigest()


def remove_duplicate_files(path):
    print('Removing duplicate files...')
    all_size = {}
    total_file = 0
    total_delete = 0
    start = now()
    for file in os.listdir(path):
        total_file += 1
        real_path = os.path.join(path, file)
        if os.path.isfile(real_path):
            size = os.stat(real_path).st_size
            name_and_md5 = [real_path, '']
            if size in all_size.keys():
                new_md5 = getmd5(real_path)
                if all_size[size][1] == '':
                    all_size[size][1] = getmd5(all_size[size][0])
                if new_md5 in all_size[size]:
                    os.remove(real_path)
                    print('DELETE', file)
                    total_delete += 1
                else:
                    all_size[size].append(new_md5)
            else:
                all_size[size] = name_and_md5
    end = now()
    time_last = end - start
    print('Total files:', total_file)
    print('Deleted files:', total_delete)
    print('Time using:', time_last, 's')
    print()


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
    parser.add_argument('-o', '--output_dir', type=str,
                        default=os.path.join(d, 'cropped_images'),
                        help='path to save the output cropped files')

    args = parser.parse_args()
    print('Processing images in folder:', args.path)

    remove_duplicate_files(args.path)

    crop_wrapper(args.path, args.height, args.width, args.output_dir)


if __name__ == '__main__':
    cli()
    print('Done.')
