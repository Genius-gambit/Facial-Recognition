import cv2
import matplotlib.pyplot as plt

def	MeanSquareError(img1, img2):
	h, w, l = img1.shape
	# print(h, w)
	h, w, l = img2.shape
	# print(h, w)
	# print(h, w)
	# diff = cv2.subtract(img1, img2)

def	getFaceImage(imagePath: str):
	img = cv2.imread(imagePath)
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	face_classifier = cv2.CascadeClassifier(
	cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
	face = face_classifier.detectMultiScale(
	gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
	for (x, y, w, h) in face:
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return (img_rgb, face)

if __name__ == '__main__':
	imagePath1 = '1.jpg'
	imagePath2 = '2.jpg'
	
	img_rgb1, face1 = getFaceImage(imagePath1)
	img_rgb2, face2 = getFaceImage(imagePath2)
	print(face1[0][2])
	print(face2[0][2])
	# MeanSquareError(img1, img2)
	plt.figure(figsize=(5, 5))
	plt.imshow(img_rgb1)
	plt.figure(figsize=(5, 5))
	plt.imshow(img_rgb2)
	plt.axis('off')
	plt.show()
	# print(gray_image.shape)