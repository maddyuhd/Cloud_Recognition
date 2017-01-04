import cv2
import numpy as np
from matplotlib import pyplot as plt
from resize import resize
from mask import maskdata
from output import result
from keypoints import savekeyPointsOnImage
import time
import sys
import csv

a=[]
hello = [["IMG_Name","Number of Pos","Pos","Result"]]

def input_value(scn_img):
	h1,w1 = scn_img.shape[:2]
	scn_img=resize(h1,w1,640,scn_img)
	hr1,wr1 = scn_img.shape[:2]

	#==================================================>KPT_SCN
	# fast:

	# 	fast = cv2.FastFeatureDetector_create(49)
	# 	kp1 = fast.detect(scn_img,masked_data)

	# surf:

	detector=cv2.xfeatures2d.SURF_create(400, 5, 5)
	kp1, desc = detector.detectAndCompute(scn_img, None)

	# for k in kp1:
	# 	x,y=k.pt
	# 	print x,y

	#==================================================>DESCRIPTOR_SCN
	# freak:
		
	freakExtractor = cv2.xfeatures2d.FREAK_create()
	kp1,des1= freakExtractor.compute(scn_img,kp1)

	# savekeyPointsOnImage(scn_img,"input1.jpg" ,kp1,wr1,hr1)

	# img4 = cv2.drawKeypoints(scn_img, kp1,None, color=(0,255,255))
	# cv2.imwrite('input1.jpg', img4)

	return kp1,des1 


def sceen_value(src_img):
	#==============================================>Resize
	h2,w2 = src_img.shape[:2]
	# src_img=resize(h2,w2,640,src_img)
	# hr2,wr2 = src_img.shape[:2]

	#==================================================>KPT_SRC
	
	# fast:
		# kp2 = fast.detect(src_img,None)

	# surf:

	masked_data = maskdata(src_img,h2,w2)

	detector=cv2.xfeatures2d.SURF_create(400, 5, 5)
	freakExtractor = cv2.xfeatures2d.FREAK_create()
	
	kp2, descs = detector.detectAndCompute(src_img, masked_data)
	
	#==================================================>DESCRIPTOR_SRC
	
	# freak:
	kp2,des2= freakExtractor.compute(src_img,kp2)

	return kp2,des2


def match(scn_img,src_img,filename,kp1,des1,kp2,des2,pos):

	#==============================================>FLANN MATCHING
	start_time = time.time()

	FLANN_INDEX_LSH=0    
	flann_params= dict(algorithm = FLANN_INDEX_LSH,table_number = 6,# 12
			       key_size = 12,# 20
			       multi_probe_level = 1) #2
	matcher = cv2.FlannBasedMatcher(flann_params, {})

	if(len(kp2)>10):
		matches=matcher.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32), 2)
	
	#==============================================>Ratio Test

		good = []
		for m,n in matches:
			if m.distance < 0.67*n.distance:
				good.append(m)
		# print ("good matches",len(good))

		MIN_MATCH_COUNT=10

		if len(good)>MIN_MATCH_COUNT:
				
			src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
			try:
				matchesMask = mask.ravel().tolist()
				draw_params = dict(matchColor = (0,255,0), # draw matches in green color
			                        singlePointColor = None,
			                        matchesMask = matchesMask, # draw only inliers
			                        flags = 2)

				posadd=str(filename)+" = "+str(len(good))+" matches"
				# pos.append("      "+str(filename))
				pos.append(posadd)


				# img3=cv2.drawMatches(scn_img,kp1,src_img,kp2,good,None,**draw_params)
				# name=str(time.time())+".jpg"
				# cv2.imwrite("input1.jpg", img3)

				# h,w= scn_img.shape
				# pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
				# dst = cv2.perspectiveTransform(pts,M)
				# src_img = cv2.polylines(src_img,[np.int32(dst)],True,255,3, cv2.LINE_AA)
				#==============================================>DRAW KPT

				# savekeyPointsOnImage(src_img,"src.jpg" ,kp2,w2,h2)

				# img5 = cv2.drawKeypoints(src_img, kp2 ,None, color=(0,255,255))
				# cv2.imwrite('src.jpg', img5)
			except:
				print "       false +ve - "+str(filename)

		count=time.time() - start_time
		a.append(count)
		# print("Server Side--- %s seconds ---" % (count))

		
		# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
		                        # singlePointColor = None,
		                        # matchesMask = matchesMask, # draw only inliers
		                        # flags = 2)

		# img3=cv2.drawMatches(scn_img,kp1,src_img,kp2,good,None,**draw_params)
		# plt.imshow(img3, 'gray'),plt.show()
	else:
		print "error - "+str(filename)

	return a

	# hello.append([filename,len(kp1),len(pos),pos])