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

class Rect:
	def __init__(self, vals):
	    self.x = vals[0]
	    self.y = vals[1]
	    self.w = vals[2]
	    self.h = vals[3]

	def dist(self,other):
	    #overlaps in x or y:
	    if abs(self.x - other.x) <= (self.w + other.w):
	        dx = 0;
	    else:
	        dx = abs(self.x - other.x) - (self.w + other.w)
	    #
	    if abs(self.y - other.y) <= (self.h + other.h):
	        dy = 0;
	    else:
	        dy = abs(self.y - other.y) - (self.h + other.h)
	    return dx + dy

def union(a,b):
	x = min(a[0], b[0])
	y = min(a[1], b[1])
	w = max(a[0]+a[2], b[0]+b[2]) - x
	h = max(a[1]+a[3], b[1]+b[3]) - y
	return (x, y, w, h)

def randomColor():
    return (randint(0, 255), randint(0, 255), randint(0, 255))

def doRectsIntersect(a,b):
	x = max(a[0], b[0])
	y = max(a[1], b[1])
	w = min(a[0]+a[2], b[0]+b[2]) - x
	h = min(a[1]+a[3], b[1]+b[3]) - y
	return w>0 and h>0

def minDistBetweenRects(first, second):
	a = Rect(first)
	b = Rect(second)

	return a.dist(b)

def minDistBetweenRects2(first, second):
	(x1, y1, w1, h1) = first
	(x2, y2, w2, h2) = second

	horizontalDist = min(abs(x1+w1-x2), abs(x2+w2-x1))
	verticalDist = min(abs(y1+h1-y2), abs(y2+h2-y1))

	return max(horizontalDist, verticalDist)

def mergeIntersections(rects):
	merged = []
	for i in range(0, len(rects) -1):
		didMerge = False
		first = rects[i]
		for j in range(i + 1, len(rects) -1):
			second = rects[j]
			doTheyIntersect = doRectsIntersect(first, second)
			minDist = minDistBetweenRects(first, second)
			if doTheyIntersect or minDist < 20:
				if minDist < 20:
					print("Min dist is ", minDist, " for ", first, " and ", second)
				merge = union(first, second)
				indices = i, j
				mergedList = [i for j, i in enumerate(rects) if j not in indices]
				mergedList.append(merge)
				return mergeIntersections(mergedList)
		
	return rects    			

def padRect(rect, pad, maxW, maxH):
		x = max(rect[0] -pad, 0)
		y = max(rect[1] -pad, 0)
		w = rect[2] + 2*pad
		h = rect[3] + 2*pad

		if x+w > maxW:
			w = maxW-x
		if y+h > maxH:
			h = maxH-y

		return (x, y, w, h)

def rectSizeFilter(rect):
		a = 10 
		b = 20 
		w = rect[2] 
		h = rect[3]
		return (w > b and h > a) or (w > a and h > b)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="first input image")
ap.add_argument("-s", "--second", required=True,
	help="second")
args = vars(ap.parse_args())

# load the two input images
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])

# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 250, 255,
	cv2.THRESH_BINARY)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# loop over the contours
i = 0
pad = 10
rects = list(map(cv2.boundingRect, cnts))
rects = list(filter(rectSizeFilter, rects))
rects = list(map(lambda rect: padRect(rect, pad, grayA.shape[1], grayA.shape[0]), rects))
#rects = mergeIntersections(rects)

blackImage = np.zeros(grayA.shape, np.uint8)

for rect in rects:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = rect
	#crop_img = imageA[y:y+h,x:x+w]
	#cv2.imshow("Crop {}".format(i), crop_img)
	#i = i + 1
	blackImage[y:y+h, x:x+w] = grayA[y:y+h, x:x+w]
	#cv2.imshow("gray", grayA[y:y+h, x:x+w])
	#cv2.rectangle(imageA, (x, y), (x + w, y + h), randomColor(), -1)
	#cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), -1)
		
print("Made it to end!")
# show the output images
#cv2.imshow("Gray", grayA)
#cv2.imshow("Cropped", blackImage)
#cv2.imwrite('cropped.png',blackImage)
#cv2.imshow("Original", imageA)
#cv2.imshow("Modified", imageB)
#cv2.imshow("Diff", diff)
#cv2.imshow("Thresh", thresh)
#cv2.waitKey(0)