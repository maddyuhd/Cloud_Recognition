

def result(imagePath,loop,pos,neg,a):

	print(str(imagePath) +" ====> "+"Compared with "+str(loop) +" images")
	print("No. of positive matches = "+str(len(pos)))
	print '\n'.join(str(p) for p in pos)
	print("No. of negative matches = "+str(len(neg)))
	# print '\n'.join(str(p) for p in neg)
	print "Total time (Server)- ",str(sum(a))

def test(pos,found,correct,notcorrect,notfound):
	if len(pos)>0:
		found+=1
		print " found"
		if len(pos)==1:
			correct+=1
			print "     "+str(pos[0])
		else:
			print " !correct"
			notcorrect+=1
			print "       "+str(pos)
	else:
		notfound+=1
		print "\033 !found"
	
	return found,correct,notcorrect,notfound

	# length = len(hello[0])

	# with open('test1.csv', 'wb') as testfile:
	#     csv_writer = csv.writer(testfile)
	#     for y in range(length):
	#         csv_writer.writerow([x[y] for x in hello])

	# result(imagePath,loop,pos,neg,a)

def per(a,loop,found,correct,notcorrect,notfound):
	loop=float(loop)
	found=float(found)
	print "Total Input image :"+str(int(loop))
	if (found!=0):
		print "       "+str(found/loop*100)+" % is found"
		print "               "+str(correct/found*100)+" % is found true "
		print "               "+str(notcorrect/found*100)+" % is found false"
		print "       "+str(notfound/loop*100)+" % is not found"
	else:
		print "No Matches"

	print "Time Taken : "+str(sum(a))+" seconds"