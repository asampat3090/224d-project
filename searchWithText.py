import numpy as np
import cPickle as pickle
import argparse
import json

def main(params):
	dim = 300 # Dimension of the GloVe vectors chosen
	glove_dict_path = '../../cs224d/project/vecDict.pickle'
	print 'Going to load vecDict'
  	with open(glove_dict_path, 'rb') as handle:
  		vec_dict = pickle.load(handle)

  	print 'Going to load fileNameToVector'
  	with open('fileNameToVector.pickle', 'rb') as handle:
  		fileNameToVector = pickle.load(handle)
    
	print 'Loaded both dicts'
	sentence = params['searchQuery']
	sentenceVector = np.zeros(dim)
   	numWords = 0
   	for word in sentence.split():
   		if word in vec_dict:
   			sentenceVector += vec_dict[word].astype(np.float)
   			numWords += 1

   	sentenceVector /= numWords

    #distances = np.zeros(len(fileNameToVector))
   	currMin = -1
   	for fileName in fileNameToVector.keys():
   		currDist = np.linalg.norm(fileNameToVector[fileName]-sentenceVector)
   		if currMin==-1:
   			currMin = currDist
   			currName = fileName
   		if currDist < currMin:
   			currMin = currDist
   			currName = fileName

   	print "Best file found: ",currName

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('searchQuery', type=str, help='the sentence search query')
  
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  main(params)
