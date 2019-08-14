from imutils import paths
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image.")
ap.add_argument("-t", "--threshold", type=float, default=100.0, 
		help="Focus measures falling below the threshold is considered blurry. ")
ap.add_argument("-d", "--display", type=str, default=0, 
		help="Show image with label on top. \
		1 to get a visual represntation and 0 to avoid visualization.")
args = vars(ap.parse_args())

def detect_blurness(img_path, show_=0):
	"""
	Input : Path/to/image/file
	Output : Tuple 
		First item string - either 'Not Blurred' or 'Blurred'
		Second item - variance of laplacian
	Compute the variance and the focus of measure (the variance of laplacian).
	"""

	img = cv2.imread(img_path, 0)
	vol = cv2.Laplacian(img, cv2.CV_64F).var()
	label = "Blurred" if vol < args["threshold"] else 'Not Blurred'
	if show_:
		cv2.putText(img, "{}: {:.2f}".format(label, vol), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
		cv2.imshow("Image", img)
		cv2.waitKey(0)
	return (label, vol)

if __name__ == '__main__':
	detect_blurness(args["image"], int(args["display"]))
