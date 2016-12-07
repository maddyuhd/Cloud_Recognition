import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
from keypoints import drawkeyPointsOnImage 
import sys
import os

a=[]
pos=[]
neg=[]
loop=0	

def resize(h1,w1,res,scn_img):

	if h1>w1:
			ar=w1/float(h1)
			newH=res
			newW=int(newH*ar)
	elif h1<w1:
		ar=h1/float(w1)
		newW=res
		newH=int(newW*ar)
	else:
		newH=res
		newW=res
					
	scn_img = cv2.resize(scn_img, (newW, newH))
	return scn_img

def match(scn_img,imagePath1,filename,kp1):
		src_img = cv2.imread(imagePath1,0)
	#==============================================>Resize
		h2,w2 = src_img.shape[:2]
		src_img=resize(h2,w2,320,src_img)
		#2448

	#==============================================>KPT

		# kp2 = fast.detect(src_img,None)
		kp2, descs = detector.detectAndCompute(src_img, None)
		# print("input:",len(kp2))

	#==============================================>DRAW KPT

		img5 = cv2.drawKeypoints(src_img, kp2,None, color=(0,255,255))
		cv2.imwrite('k.jpg', img5)
	#==============================================>FREAK
		freakExtractor = cv2.xfeatures2d.FREAK_create()

		kp1,des1= freakExtractor.compute(scn_img,kp1)
		
		# dst = cv2.cornerHarris(scn_img,2,3,0.04)
		# scn_img[dst>0.01*dst.max()]=[0,0,255]
		kp2,des2= freakExtractor.compute(src_img,kp2)
	#==============================================>FLANN MATCHING
		start_time = time.time()

		FLANN_INDEX_LSH=0    
		flann_params= dict(algorithm = FLANN_INDEX_LSH,table_number = 6,# 12
			       key_size = 12,# 20
			       multi_probe_level = 1) #2
		matcher = cv2.FlannBasedMatcher(flann_params, {})
		if(len(kp2)>10):
			matches=matcher.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32), 2)

		#==============================================>GOOD MATCHING

			good = []
			for m,n in matches:
				if m.distance < 0.625*n.distance:
					good.append(m)
			# print ("good matches",len(good))

			MIN_MATCH_COUNT=10

			if len(good)>MIN_MATCH_COUNT:
				filename=str(filename)+" = "+str(len(good))+" matches"
				pos.append("      "+str(filename))

				# src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
				# dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
				# M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
				# matchesMask = mask.ravel().tolist()

				# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
			 #                        singlePointColor = None,
			 #                        matchesMask = matchesMask, # draw only inliers
			 #                        flags = 2)

				# img3=cv2.drawMatches(scn_img,kp1,src_img,kp2,good,None,**draw_params)
				# name=str(time.time())+".jpg"
				# cv2.imwrite(filename, img3)


				# h,w= scn_img.shape
				# pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
				# dst = cv2.perspectiveTransform(pts,M)
				# src_img = cv2.polylines(src_img,[np.int32(dst)],True,255,3, cv2.LINE_AA)
			else:
				# print "Nope - %d/%d" % (len(good),MIN_MATCH_COUNT)
				filename=str(filename)+" = "+str(len(good))+" matches"
				neg.append("      "+str(filename))
				# matchesMask = None

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

def result(imagePath,loop,pos,neg,a):

	print(str(imagePath) +" ====> "+"Compared with "+str(loop) +" images")
	print("No. of positive matches = "+str(len(pos)))
	print '\n'.join(str(p) for p in pos)
	print("No. of negative matches = "+str(len(neg)))
	print '\n'.join(str(p) for p in neg)
	print "Total time (server)- ",str(sum(a))

try:
	#==============================================>Input
	imagePath = sys.argv[1]
	scn_img = cv2.imread(imagePath,0)

except :
	imagePath="pics/boat1.jpg"
	# imagePath="/home/smacar/Desktop/dev/online/db/movingobjects/barrywhitemov/barrywhitemovVGA_00001.jpg"
	scn_img= cv2.imread(imagePath,0)

path = '/home/smacar/Desktop/dev/online/pics/src'
# path="/home/smacar/Desktop/dev/online/db/database"

#==============================================>Constants
# fast = cv2.FastFeatureDetector_create(49)
detector=cv2.xfeatures2d.SURF_create(400, 5, 5)

h1,w1 = scn_img.shape[:2]
scn_img=resize(h1,w1,1280,scn_img)

#==============================================>MASK

mask =  np.zeros(scn_img.shape[:2], dtype = "uint8")

h,w = scn_img.shape[:2]
h=h-10
w=w-15

mask = np.zeros(scn_img.shape,np.uint8)
mask[10:h,15:w] = scn_img[10:h,15:w]
masked_data = cv2.bitwise_and(scn_img, scn_img, mask=mask)



kp1, desc = detector.detectAndCompute(scn_img, masked_data)
# kp1 = fast.detect(scn_img,masked_data)

img4 = cv2.drawKeypoints(scn_img, kp1,None, color=(0,255,255))
cv2.imwrite('k_input.jpg', img4)

#==============================================>Multiple Samples
for filename in os.listdir(path):
	loop+=1
	imagePath1=path+"/"+filename
	match(scn_img,imagePath1,filename,kp1)

result(imagePath,loop,pos,neg,a)