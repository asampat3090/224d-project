import numpy as np
import cPickle as pickle
import argparse
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import pdb
from nltk.stem.snowball import EnglishStemmer

def main():
  dim = 300 # Dimension of the GloVe vectors chosen
  glove_dict_path = '../vecDict.pickle'
  print 'Going to load vecDict'
  with open(glove_dict_path, 'rb') as handle:
    vec_dict = pickle.load(handle)

  print 'Going to load fileNameToVector'
  with open('fileNameToVector.pickle', 'rb') as handle:
    fileNameToVector = pickle.load(handle)

  output_path = 'output/'

  # Find images from Flicker dataset
  flikr8k_path = '../Flicker8k_Dataset/'

  # initialize stemmer for search in GLoVe vector space
  st = EnglishStemmer()

  while True: 
    sentence = raw_input('Please enter search query:')
    sentenceVector = np.zeros(dim)
    numWords = 0
    for word in sentence.split():
      if st.stem(word) in vec_dict:
        sentenceVector += vec_dict[st.stem(word)].astype(np.float)
        numWords += 1
      elif st.stem(word)+'e' in vec_dict:
        sentenceVector += vec_dict[st.stem(word)+'e'].astype(np.float)
        numWords += 1

   	sentenceVector /= numWords

    # write to results for this query to file
    output_filename = '_'.join(sentence.split(' '))+'.txt'

    with open(os.path.join(output_path,output_filename), 'w') as f:

      distArr = []
      for fileName in fileNameToVector.keys():
        distArr.append(np.linalg.norm(fileNameToVector[fileName]-sentenceVector))

      distArr = np.array(distArr)
      sortedInd = np.argsort(distArr)

      top10filenames = [fileNameToVector.keys()[i] for i in sortedInd[:10]]
      top10distances = [distArr[i] for i in sortedInd[:10]]

      # list the top 10 image names + distances + get relevance from user
      rel_score_arr = []

      f.write('filename, distance, relevance score(-1 to 3)\n')
      for filename,dist in zip(top10filenames,top10distances):
        # show the image to the user 
        curr_img = Image.open(os.path.join(flikr8k_path,filename))
        plt.imshow(curr_img)
        plt.show()
        # ask user to enter the relevance score
        while True:
          try: 
            rel_score = int(raw_input('Please enter the relevance score between -1 and 3.'))
            if rel_score >= -1 and rel_score <= 3: 
              break
            else: 
              print 'Please enter the relevance score within the range -1 and 3.'
          except: 
            print 'Please make sure the relevance score is an integer.'
        rel_score_arr.append(rel_score)
        f.write('%s,%s,%d\n' % (filename,dist,rel_score))

      # Calculate the nDCG score for this query

      # OLD FORMULATION
      # DCG = rel_score_arr[0]
      # for i in xrange(1,10): 
      #   DCG += float(rel_score_arr[i])/(np.log(i+1)/np.log(2))

      # sorted_rel_score_arr = np.sort(np.array(rel_score_arr))[::-1]

      # iDCG = sorted_rel_score_arr[0]
      # for i in xrange(1,10):
      #   iDCG += float(sorted_rel_score_arr[i])/(np.log(i+1)/np.log(2))

      # NEW FORMULATION
      DCG = float(0)
      for i in xrange(10):
        DCG += ((2**float(rel_score_arr[i]))-1)/(np.log(i+2)/np.log(2))

      pdb.set_trace()
      sorted_rel_score_arr = np.sort(np.array(rel_score_arr))[::-1]

      iDCG = float(0)
      for i in xrange(10):
        iDCG += ((2**float(sorted_rel_score_arr[i]))-1)/(np.log(i+2)/np.log(2))

      nDCG = DCG/iDCG

      f.write('query: %s\n' % sentence)
      f.write('nDCG score: %f\n' % nDCG)

    print "wrote to file %s" % output_filename
if __name__ == "__main__":

  # parser = argparse.ArgumentParser()
  # parser.add_argument('searchQuery', type=str, help='the sentence search query')
  
  # args = parser.parse_args()
  # params = vars(args) # convert to ordinary dict
  # print 'parsed parameters:'
  # print json.dumps(params, indent = 2)
  main()
