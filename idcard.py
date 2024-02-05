
# importing packages
from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image
import requests



!mkdir id_card
!mkdir id_card/image



# Opening and displaying
original = Image.open(requests.get('', stream=True).raw)
tampered = Image.open(requests.get('', stream=True).raw) 


# The file format of the source file.
print("Original image format : ",original.format) 
print("Tampered image format : ",tampered.format)

#size of the image
print("Original image size : ",original.size) 
print("Tampered image size : ",tampered.size) 


# Resizing and saving the Image
original = original.resize((250, 160))
print(original.size)
original.save('id_card/image/original.png')
tampered = tampered.resize((250,160))
print(tampered.size)
tampered.save('id_card/image/tampered.png')



#changing image format
tampered = Image.open('id_card/image/tampered.png')
tampered.save('id_card/image/tampered.png')


# loading the two input images
original = cv2.imread('id_card/image/original.png')
tampered = cv2.imread('id_card/image/tampered.png')




# Convert the images to grayscale
original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)


# Compute the Structural Similarity Index (SSIM) between the two images, ensuring that the difference image is returned
(score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# Calculating threshold and contours 
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)



# loop over the contours
for c in cnts:
    # applying contours on image
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(tampered, (x, y), (x + w, y + h), (0, 0, 255), 2)


#Diplay original image with contour
print('Original Format Image')
Image.fromarray(original)

#Diplay tampered image with contour
print('Tampered Image')
Image.fromarray(tampered)

#Diplay difference image with black
print('Different Image')
Image.fromarray(diff)



#Display threshold image with white
print('Threshold Image')
Image.fromarray(thresh)


