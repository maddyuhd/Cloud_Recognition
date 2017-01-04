import cv2
import numpy as np

def imread(img):
	img = cv2.imread(img,1)
	return img;

def imd(img):
	(h,w) = img.shape[:2]
	return (h,w);

def imresize(image, w, h):
	ar=w/float(h)
	if h>w:
		ar=w/float(h)
		newH=320
		newW=int(newH*ar)
	elif h<w:
		ar=h/float(w)
		newW=320
		newH=int(newW*ar)
	else:
		newH=320
		newW=320
				
	img = cv2.resize(image, (newW, newH)) 
	return img;
	
def imwrite(imagename,img2):
	cv2.imwrite(imagename,img2)
	return ;
	
def rgb2gray(img):
	imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	return imgGray;

def gray2rgb(img):
	imgRgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
	return imgRgb;

def drawLines(image, x, y):
	# cv2.line(image, (int(x)-2,int(y)), (int(x)+2,int(y)),(0,255,255),1)
	return ;

def crop(image, w, h):
	imgTL = image[0:h/2, 0:w/2] 
	imgTR = image[0:h/2, w/2:w] 
	imgBL = image[h/2:h, 0:w/2]
	imgBR = image[h/2:h, w/2:w] 
	return imgTL,imgTR,imgBL,imgBR;

def cropEllipse(image, w, h):
	(axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
	(axesX1, axesY1) = (int(w) / 2, int(h) / 2)
	
	ellipCenterMask = np.zeros(image.shape[:2], dtype = "uint8")
	ellipMidMask = np.zeros(image.shape[:2], dtype = "uint8")
	
	cv2.ellipse(ellipCenterMask, (w/2, h/2), (axesX, axesY), 0, 0, 360, 255, -1)
	cv2.ellipse(ellipMidMask, (w/2, h/2), (axesX1, axesY1), 0, 0, 360, 255, -1)
	
	ellipEndMask=255-ellipMidMask
	ellipMidMask=(255-ellipCenterMask)-ellipEndMask

	maskedCdata = cv2.bitwise_and(image, image, mask=ellipCenterMask)
	maskedMdata = cv2.bitwise_and(image, image, mask=ellipMidMask)
	maskedEdata = cv2.bitwise_and(image, image, mask=ellipEndMask)
	return maskedCdata,maskedMdata,maskedEdata
