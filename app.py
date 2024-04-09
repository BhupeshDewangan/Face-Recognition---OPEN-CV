import streamlit as st 
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import os


face_cascade = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Models/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('Models/haarcascade_smile.xml')



def cartonize_image(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Edges
	gray = cv2.medianBlur(gray, 5)
	edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
	#Color
	color = cv2.bilateralFilter(img, 9, 300, 300)
	#Cartoon
	cartoon = cv2.bitwise_and(color, color, mask=edges)

	return cartoon


def cannize_image(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	img = cv2.GaussianBlur(img, (11, 11), 0)
	canny = cv2.Canny(img, 100, 150)
	return canny

def detect_smiles(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Detect Smiles
	smiles = smile_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the Smiles
	for (x, y, w, h) in smiles:
	    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
	return img

def detect_faces(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the faces
	for (x, y, w, h) in faces:
				 cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
	return img,faces 

def detect_eyes(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
	for (ex,ey,ew,eh) in eyes:
	        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	return img

def load_image(image):
    im = Image.open(image)
    return im


def main():
	"""Face Detection App"""

	st.title("Face Detection App")
	st.text("Build with Streamlit and OpenCV")

	activities = ["Detection","About"]
	choice = st.sidebar.selectbox("Select Activty",activities)

	if choice == 'Detection':
		st.subheader("Face Detection")

		image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

		col1, col2 = st.columns(2)


		if image_file is not None:		
			our_image = Image.open(image_file)
			# our_image = load_image(image_file)			
			st.text("Original Image")
			# st.write(type(our_image))
			col1.image(our_image)

		enhance_type = st.sidebar.radio("Enhance Type",["Original",  "Sharpness", "Gray-Scale","Contrast","Brightness","Blurring", "Thresholding", "Morphological Operations","Edge Enhancement"])

		if enhance_type == 'Gray-Scale':
			new_img = np.array(our_image.convert('RGB'))
			img = cv2.cvtColor(new_img,1)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			# st.write(new_img)
			col2.image(gray)

		elif enhance_type == 'Contrast':
			c_rate = st.sidebar.slider("Contrast",0.5,3.5)
			enhancer = ImageEnhance.Contrast(our_image)
			img_output = enhancer.enhance(c_rate)
			col2.image(img_output)
			col2.download_button("Download Image", img_output, 'Image.png', mime = 'png')

		elif enhance_type == 'Brightness':
			c_rate = st.sidebar.slider("Brightness",0.5,3.5)
			enhancer = ImageEnhance.Brightness(our_image)
			img_output = enhancer.enhance(c_rate)
			col2.image(img_output)

		elif enhance_type == 'Blurring':
			new_img = np.array(our_image.convert('RGB'))
			blur_rate = st.sidebar.slider("Brightness",0.5,3.5)
			img = cv2.cvtColor(new_img,1)
			blur_img = cv2.GaussianBlur(img,(11,11),blur_rate)
			col2.image(blur_img)
		
		elif enhance_type == 'Sharpness':
			s_rate = st.sidebar.slider("Sharpness", 0.0, 2.0)
			enhancer = ImageEnhance.Sharpness(our_image)
			img_output = enhancer.enhance(s_rate)
			col2.image(img_output)

		elif enhance_type == 'Thresholding':
			threshold_type = st.sidebar.radio("Threshold Type", ["BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV"])
			threshold_value = st.sidebar.slider("Threshold Value", 0, 255, 128)

			img_array = np.array(our_image)
			gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
			_, img_output = cv2.threshold(gray_image, threshold_value, 255, getattr(cv2, f'THRESH_{threshold_type.upper()}'))
			col2.image(img_output, channels="GRAY")


		elif enhance_type == 'Morphological Operations':
			operation_type = st.sidebar.selectbox("Operation Type", ["Erosion", "Dilation", "Opening", "Closing"])
			kernel_size = st.sidebar.slider("Kernel Size", 3, 11, 3)
			
			# Convert the uploaded image to a numpy array
			img_array = np.array(our_image)

			# Convert the image to grayscale if it's in color
			if len(img_array.shape) == 3:
				img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
			
			# Perform morphological operation based on user selection
			kernel = np.ones((kernel_size, kernel_size), np.uint8)
			if operation_type == "Erosion":
				img_output = cv2.erode(img_array, kernel, iterations=1)
			elif operation_type == "Dilation":
				img_output = cv2.dilate(img_array, kernel, iterations=1)
			elif operation_type == "Opening":
				img_output = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)
			elif operation_type == "Closing":
				img_output = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)

			col2.image(img_output)

		elif enhance_type == 'Edge Enhancement':
			edge_type = st.sidebar.selectbox("Edge Type", ["Laplacian", "Sobel X", "Sobel Y"])
			img_array = np.array(our_image)

			gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
			if edge_type == "Laplacian":
				img_output = cv2.Laplacian(gray_image, cv2.CV_64F)
			elif edge_type == "Sobel X":
				img_output = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
			elif edge_type == "Sobel Y":
				img_output = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
			img_output = cv2.convertScaleAbs(img_output)

			col2.image(img_output, channels="GRAY")

		
		# elif enhance_type == 'Bilateral Filter':
		# 	d = st.sidebar.slider("Diameter", 5, 20, 9)
		# 	sigma_color = st.sidebar.slider("Sigma Color", 10, 200, 75)
		# 	sigma_space = st.sidebar.slider("Sigma Space", 10, 200, 75)

		# 	img_array = np.array(our_image)
		# 	img_output = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)

		# 	col2.image(img_output)

		# elif enhance_type == 'Original':
		# 	new_img = np.array(our_image.convert('RGB'))
		# 	img = cv2.cvtColor(new_img,1)
		# 	st.image(img)
		# else:
		# 	st.image(our_image,width=300)



		# Face Detection
		task = ["Faces","Smiles","Eyes","Cannize","Cartonize"]
		feature_choice = st.sidebar.selectbox("Find Features",task)
		if st.button("Process"):

			if feature_choice == 'Faces':
				result_img,result_faces = detect_faces(our_image)
				col2.image(result_img)

				st.success("Found {} faces".format(len(result_faces)))

			elif feature_choice == 'Smiles':
				result_img = detect_smiles(our_image)
				col2.image(result_img)

			elif feature_choice == 'Eyes':
				result_img = detect_eyes(our_image)
				col2.image(result_img)

			elif feature_choice == 'Cartonize':
				result_img = cartonize_image(our_image)
				col2.image(result_img)

			elif feature_choice == 'Cannize':
				result_canny = cannize_image(our_image)
				col2.image(result_canny)


	elif choice == 'About':
		st.subheader("About Face Detection App")
		st.markdown("Built with Streamlit by [Bhupesh Dewangan](https://www.linkedin.com/in/bhupesh-dewangan-7121851ba/)")
		st.success("Bhupesh Dewangan")
		


if __name__ == '__main__':
		main()