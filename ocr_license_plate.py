# python ocr_license_plate.py --input license_plates/group1

from plate_recognition.anpr import PyImageSearchANPR
from imutils import paths
import argparse
import imutils
import cv2

user_info = {"accepted": ["MH15TC584", "MH20EE7601", "HR26DA2330"],
	"banned": ["KL55R2473"]}

def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of images")
ap.add_argument("-c", "--clear-border", type=int, default=-1,
	help="whether or to clear border pixels before OCR'ing")
ap.add_argument("-p", "--psm", type=int, default=7,
	help="default PSM mode for OCR'ing license plates")
ap.add_argument("-d", "--debug", type=int, default=-1,
	help="whether or not to show additional visualizations")
args = vars(ap.parse_args())

# initialize our ANPR class
anpr = PyImageSearchANPR(debug=args["debug"] > 0)

imagePaths = sorted(list(paths.list_images(args["input"])))

for imagePath in imagePaths:
	# load the input image from disk and resize it
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)

	# apply automatic license plate recognition
	(lpText, lpCnt) = anpr.find_and_ocr(image, psm=args["psm"],
		clearBorder=args["clear_border"] > 0)

	# only continue if the license plate was successfully OCR'd
	if lpText is not None and lpCnt is not None:
		# color is based on what status the license plate is
		lpText = cleanup_text(lpText)
		if lpText in user_info["accepted"]:
			color = (0, 255, 0)
		elif lpText in user_info["banned"]:
			color = (255, 0, 0)
		else:
			color = (0, 0, 255)
		
		# fit a rotated bounding box to the license plate contour and
		# draw the bounding box on the license plate
		box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
		box = box.astype("int")
		cv2.drawContours(image, [box], -1, color, 2)

		# compute a normal (unrotated) bounding box for the license
		(x, y, w, h) = cv2.boundingRect(lpCnt)

		# plate and then draw the OCR'd license plate text on the
		# image
		cv2.putText(image, cleanup_text(lpText), (x, y - 15),
			cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

		# show the output image
		print("[INFO] {}".format(lpText))
		cv2.imshow("Output ANPR", image)
		cv2.waitKey(0)
	else:
		print(imagePath + " failed")