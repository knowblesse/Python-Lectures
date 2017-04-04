from sys import stdout
import time
import re	
import numpy as np
import matplotlib.pyplot as plt

startTime = time.time()
u = []
b = []
t = []
# this is good, since we don't have to close it
with open('shakespeare.txt') as f2:
	# count the lines 
	c = 0
	# this will loop through the lines in the file
	for chunk in f2:
		# this converts the line to lower case and keeps only the letters
		l = re.sub(r'\W+', ' ',chunk.lower())
		# this splits the chunk into "words"
		l = l.split()
		# we read a line
		c = c + 1
		# append all words to list
		for el in l:
			u.append(el)

# go through the text and append all words to bigram
for i in range(0,len(u)-2):
	b.append(u[i] + u[i+1])

# go through the text and append all words to trigram		
for i in range(0,len(u)-3):
	t.append(u[i] + u[i+1] + u[i+2])
	
print("Read %d lines: resulting dictionary is %d elements long. Took %f seconds" % (c,len(u),time.time()-startTime))			


# unigram processing

# we can use all of the return values here
startTime = time.time()
un, ind, inv, cou = np.unique(u, return_index=True, return_inverse=True, return_counts=True)
print("Unigram has %d unique words, Inverse indices are %d long. Took %f seconds" % (len(un),len(inv),time.time()-startTime))
# this sorts the counts from high to low
sorted_cou_ind=np.argsort(cou)[::-1]
# print out the most counts and words for the thirty most common unigrams
print(sorted_cou_ind[0:30],un[sorted_cou_ind[0:30]])
# make bar plot
plt.figure()
plt.bar(np.arange(0,30),cou[sorted_cou_ind[0:30]])
plt.xticks(np.arange(0,30),un[sorted_cou_ind[0:30]],rotation='vertical')
plt.show()

startTime = time.time()
# now let's do the bigram - for this, we first go through the original list and make another 
# list to which we append the INDICES of the UNIGRAMs - for this the INV array is very helpful!
bitmp=[]
for i in range(0,len(u)-2,2):
	bitmp.append(inv[i])
	bitmp.append(inv[i+1])

# matrix of unigrams - note, we could declare this as np.sparse!
bi=np.zeros(shape=(len(un),len(un)))

# now go through our bitmp list and count the occurrances
for i in range(0,len(bitmp)-2):
	bi[bitmp[i],bitmp[i+1]]=bi[bitmp[i],bitmp[i+1]]+1
print("Bigram processing took %f seconds" %(time.time()-startTime))

startTime = time.time()
# create a sentence
c=1;
word = "random"
# what is the index of the word?
foundI=np.where(un==word)
# no error checking :(
print('found %s at %d' % (word,foundI[0]))
# print out 30 words
while c<=30:
	# what is the most common next word associated with our word?
	Imax = np.argmax(bi[foundI,:])
	# print out the index, the next word and the unigram
	print(Imax,bi[foundI,Imax],un[Imax])
	# current word becomes next word
	foundI=Imax
	c=c+1
print("Sentence processing took %f seconds" %(time.time()-startTime))

# bigram processing
startTime = time.time()
# the fastest way is to use the "b" array we created above
bc, ind, inv, cou = np.unique(b, return_index=True, return_inverse=True, return_counts=True)
print("Bigram has %d unique pairs, Inverse indices are %d long. Took %f seconds." % (len(bc),len(inv),time.time()-startTime))		
sorted_cou_ind=np.argsort(cou)[::-1]
print(sorted_cou_ind[0:30],bc[sorted_cou_ind[0:30]])
plt.figure()
plt.bar(np.arange(0,30),cou[sorted_cou_ind[0:30]])
plt.xticks(np.arange(0,30),bc[sorted_cou_ind[0:30]],rotation='vertical')
plt.subplots_adjust(bottom=0.15)
plt.show()

# we can also do this from our bigram matrix
startTime = time.time()
indices = bi.argpartition(bi.size - 30, axis=None)[-30:]
x, y = np.unravel_index(indices, bi.shape)
print(bi[x,y])
print(un[x],un[y])
print("Took %f seconds." % (time.time()-startTime))	

