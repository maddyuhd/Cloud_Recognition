import cv2
import numpy as np

def maskdata(scn_img,h,w):

	mask =  np.zeros(scn_img.shape[:2], dtype = "uint8")

	h=h-10
	w=w-15

	mask = np.zeros(scn_img.shape,np.uint8)
	mask[10:h,15:w] = scn_img[10:h,15:w]
	masked_data = cv2.bitwise_and(scn_img, scn_img, mask=mask)
	return masked_data