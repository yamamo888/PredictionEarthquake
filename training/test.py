import pickle
import pdb

with open('process_5000.pickle','rb') as fp:
	pred = pickle.load(fp)
	x = pickle.load(fp)
	y = pickle.load(fp)
	loss = pickle.load(fp)
	minY = pickle.load(fp)
	maxY = pickle.load(fp)

print('pred:',pred.T)
#print('x:',x)
print('y:',y)
print('loss:',loss)
print('minY:',minY)
print('maxY:',maxY)
pdb.set_trace()
