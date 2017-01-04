import cv2
import numpy as np
from matplotlib import pyplot as plt
from resize import resize
from match_surf import match,input_value,sceen_value
from output import result,test,per
from keypoints import savekeyPointsOnImage
import time
import os

# import sys
# scn_path = sys.argv[1]
# scn_img = cv2.imread(scn_path,0)

# scn_path="box1.jpg"
# scn_img= cv2.imread("pics/"+str(scn_path),0)

scn_path="/home/smacar/Online_Recognition/db/cd_covers/Droid/0"
	

# src_path = '/home/smacar/Online_Recognition/pics/src'
src_path="/home/smacar/Online_Recognition/db/cd_covers/Reference"

loop=0
found=0
correct=0
notcorrect=0
notfound=0

for im in range(1,99):
	Path = str(scn_path) + str(im) + ".jpg"
	scn_img= cv2.imread(Path,0)

	loop+=1
	pos=[]

	print("==> "+str(im)+".jpg with DB")

	kp1,des1 = input_value(scn_img)

	#==============================================>Match
	for src_name in os.listdir(src_path):
	
		src_path_name=src_path+"/"+src_name
		src_img = cv2.imread(src_path_name,0)
		kp2,des2 = sceen_value(src_img)
		a = match(scn_img,src_img,src_name,kp1,des1,kp2,des2,pos)


	found,correct,notcorrect,notfound = test(pos,found,correct,notcorrect,notfound)
	print"-------------------------------------"

per(a,loop,found,correct,notcorrect,notfound)
