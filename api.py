# Product Service

# Import framework
from flask import Flask
from flask_restful import Resource, Api

# USAGE
# python image_diff.py --first images/original_01.png --second images/modified_01.png
# python3 image_diff.py --first images/18.png --second images/19.png
# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
from random import randint
import numpy as np

# Instantiate the app
app = Flask(__name__)
api = Api(app)

class Product(Resource):
    def get(self):
       
        # load the two input images
        imageA = cv2.imread("images/18.png")
        imageB = cv2.imread("images/19.png")

        # convert the images to grayscale
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        print("SSIM: {}".format(score))
        print("Made it to end!")

        return {
            'ssim': "SSIM: {}".format(score)
        }

# Create routes
api.add_resource(Product, '/')

# Run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)