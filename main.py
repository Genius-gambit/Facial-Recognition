import face_recognition
import os, sys
import cv2
import numpy as np
import math

def	faceConfidence(face_distance, face_match_threshold=0.6):
	range = (1.0 - face_match_threshold)
	linear_val = (1.0 - face_distance) / (range * 2.0)

	if face_distance > face_match_threshold:
		return str(round(linear_val * 100, 2)) + '%'
	else:
		value = (linear_val + ((1.0 - linear_val) * (((linear_val - 0.5) * 2)
			** 0.2))) * 100
		return str(round(value, 2)) + '%'
	
class FaceRecognition(object):
	def	__init__(self):
		self.encode_faces()
		self.face_locations = []
		self.face_encodings = []
		self.face_names = []
		self.known_face_encodings = []
		self.known_face_names = []
		self.process_current_frame = True
		# encode faces
	
	def	encode_faces(self):
		for image in os.listdir('faces'):
			self.face_image = face_recognition.load_image_file(f'faces/{image}')
			self.face_encoding = face_recognition.face_encodings(self.face_image)[0]

			self.known_face_encodings.append(self.face_encoding)
			self.known_face_names.append(image.replace('.png', ''))

	def	runRecognition(self, frame):
		while True:
				small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
				rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
				# rgb_small_frame = small_frame[:, :, ::-1]

				if self.process_current_frame:
					self.face_locations = face_recognition.face_locations(rgb_small_frame)
					self.face_encodings = face_recognition.face_encodings(rgb_small_frame,
						self.face_locations)

					for face_encoding in self.face_encodings:
						matches = face_recognition.compare_faces(self.known_face_encodings,
							face_encoding)
						name = 'Unknown'

						face_distances = face_recognition.face_distance(
							self.known_face_encodings, face_encoding)
						best_match_index = np.argmin(face_distances)
						if matches[best_match_index]:
							name = self.known_face_names[best_match_index]
							confidence = faceConfidence(face_distances[best_match_index])
							if len(confidence) > 2:
								_str = confidence[0:3]
								if float(_str) < 92.0:
									confidence = ''
									name = 'Unknown'
									res = False
						self.face_names.append(name)
				return self.face_locations, self.face_names

if __name__ == '__main__':
	fr = FaceRecognition()
	fr.runRecognition()
