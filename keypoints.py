import cv2
import numpy as np

from imgutil import imresize,imwrite

def findnkp(image, thr):
	fast = cv2.FastFeatureDetector_create(thr)
	kp = fast.detect(image,None)
	size=len(kp)
	# print "Number of kp :",size
	return size;

def drawkeyPointsOnImage(image ,thr):
	fast = cv2.FastFeatureDetector_create(thr)
	kp = fast.detect(image,None)
	img2 = cv2.drawKeypoints(image, kp,None, color=(255,0,0))
	cv2.imshow('dst_rt', img2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return;
	
def savekeyPointsOnImage(image,imname ,kp,w,h):
	# fast = cv2.FastFeatureDetector_create(thr)
	# kp = fast.detect(image,None)
	img = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
	for k in kp:
		x,y=k.pt
		cv2.circle(img, (int(x),int(y)), 2, (50,50,50), thickness=1, lineType=8, shift=0)
		cv2.line(img, (int(x)-2,int(y)), (int(x)+2,int(y)),(0,252,248),1)
		cv2.line(img, (int(x),int(y)+2), (int(x),int(y)-2),(0,252,248),1)
	imwrite(imname,img)
	return;


def kpForCenter(image,thr,w,h):
	(axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
	(axesX1, axesY1) = (int(w) / 2, int(h) / 2)
	
	ellipMidMask =  np.zeros(image.shape[:2], dtype = "uint8")
	ellipCenterMask = np.zeros(image.shape[:2], dtype = "uint8")
	
	cv2.ellipse(ellipCenterMask, (w/2, h/2), (axesX, axesY), 0, 0, 360, 255, -1)
	cv2.ellipse(ellipMidMask, (w/2, h/2), (axesX1, axesY1), 0, 0, 360, 255, -1)
	
	ellipEndMask=255-ellipMidMask
	ellipMidMask=(255-ellipCenterMask)-ellipEndMask

	fast = cv2.FastFeatureDetector_create(thr)
	kp = fast.detect(image,None)
	
	ccount=mcount=ecount=0
	
	for k in kp:
		x,y=k.pt
		if ellipCenterMask[y][x]==255:
			ccount+=1
		elif ellipMidMask[y][x]==255:
			mcount+=1
		elif ellipEndMask[y][x]==255:
			ecount+=1

	return ccount,mcount,ecount


def kpForEqDist(image,thr,w,h):

	imgBR= np.zeros(image.shape[:2], dtype = "uint8")
	imgBL= np.zeros(image.shape[:2], dtype = "uint8")
	imgTR= np.zeros(image.shape[:2], dtype = "uint8")
	imgTL = np.zeros(image.shape[:2], dtype = "uint8")
	
	imgTL[0:h/2, 0:w/2] =255
	imgTR[0:h/2, w/2:w] =255
	imgBL[h/2:h, 0:w/2] =255
	imgBR[h/2:h, w/2:w] =255 
	
	fast = cv2.FastFeatureDetector_create(thr)
	kp = fast.detect(image,None)
	tkp=len(kp)
	
	tlcount=trcount=blcount=brcount=0
	
	for k in kp:
		x,y=k.pt
		if imgTL[y][x]==255:
			tlcount+=1
		elif imgTR[y][x]==255:
			trcount+=1
		elif imgBL[y][x]==255:
			blcount+=1
		elif imgBR[y][x]==255:
			brcount+=1

	return tkp,tlcount,trcount,blcount,brcount
