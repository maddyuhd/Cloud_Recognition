import cv2


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