import argparse
import json
import time
import datetime
import numpy as np
import code
import os
import cPickle as pickle
import math
import scipy.io
import pdb
from nltk.stem.snowball import EnglishStemmer

from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeGenerator, eval_split

"""
This script is used to predict sentences for arbitrary images
that are located in a folder we call root_folder. It is assumed that
the root_folder contains:
- the raw images
- a file tasks.txt that lists the images you'd like to use
- a file vgg_feats.mat that contains the CNN features. 
  You'll need to use the Matlab script I provided and point it at the
  root folder and its tasks.txt file to save the features.

Then point this script at the folder and at a checkpoint model you'd
like to evaluate.
"""

def main(params):
  dim = 300
  # load the checkpoint
  checkpoint_path = params['checkpoint_path']
  # load glove vect dict
  #glove_dict_path = '../vecDict.pickle'
  #with open(glove_dict_path, 'rb') as handle:
  #  vec_dict = pickle.load(handle)

  print 'loading checkpoint %s' % (checkpoint_path, )
  checkpoint = pickle.load(open(checkpoint_path, 'rb'))
  checkpoint_params = checkpoint['params']
  dataset = checkpoint_params['dataset']
  model = checkpoint['model']
  misc = {}
  misc['wordtoix'] = checkpoint['wordtoix']
  ixtoword = checkpoint['ixtoword']

  # output blob which we will dump to JSON for visualizing the results
  blob = {} 
  blob['params'] = params
  blob['checkpoint_params'] = checkpoint_params
  blob['imgblobs'] = []

  # load the tasks.txt file
  root_path = params['root_path']
  img_names = open(os.path.join(root_path, 'tasks.txt'), 'r').read().splitlines()

  # load the features for all images
  features_path = os.path.join(root_path, 'vgg_feats.mat')
  features_struct = scipy.io.loadmat(features_path)
  features = features_struct['feats'] # this is a 4096 x N numpy array of features
  D,N = features.shape

  fileNameToVector = {}
  # iterate over all images and predict sentences
  BatchGenerator = decodeGenerator(checkpoint_params)
  for n in xrange(N):
    print 'image %d/%d:' % (n, N)

    # encode the image
    img = {}
    img['feat'] = features[:, n]
    img['local_file_path'] =img_names[n]

    # perform the work. heavy lifting happens inside
    kwparams = { 'beam_size' : params['beam_size'] }
    Ys = BatchGenerator.predict([{'image':img}], model, checkpoint_params, **kwparams)

    # build up the output
    img_blob = {}
    img_blob['img_path'] = img['local_file_path']

    # encode the top prediction
    top_predictions = Ys[0] # take predictions for the first (and only) image we passed in
    top_prediction = top_predictions[0] # these are sorted with highest on top
    candidate = ' '.join([ixtoword[ix] for ix in top_prediction[1] if ix > 0]) # ix 0 is the END token, skip that
    print 'PRED: (%f) %s' % (top_prediction[0], candidate)

    
    # Calculate the sentence vector using avg of GLoVe word vectors
    #st = EnglishStemmer()
    #currSentenceVector = np.zeros(dim)
    #numWords = 0
    #for word in candidate.split():
    #  if st.stem(word) in vec_dict:
    #    currSentenceVector += vec_dict[st.stem(word)].astype(np.float)
    #    numWords += 1
    #  elif st.stem(word)+'e' in vec_dict:
    #    currSentenceVector += vec_dict[st.stem(word)+'e'].astype(np.float)
    #    numWords += 1

    #currSentenceVector /= numWords
    #fileNameToVector[img['local_file_path']] = currSentenceVector
 
    img_blob['candidate'] = {'text': candidate, 'logprob': top_prediction[0]}    
    blob['imgblobs'].append(img_blob)

  # dump result struct to file
  save_file = os.path.join(root_path, 'result_struct.json')
  print 'writing predictions to %s...' % (save_file, )
  json.dump(blob, open(save_file, 'w'))

  # dump the fileNameToVector mapping to a pickle file
  #with open('fileNameToVector.pickle', 'wb') as handle:
  #  pickle.dump(fileNameToVector, handle)

  # dump output html
  html = ''
  for img in blob['imgblobs']:
    html += '<img src="%s" height="400"><br>' % (img['img_path'], )
    html += '(%f) %s <br><br>' % (img['candidate']['logprob'], img['candidate']['text'])
  html_file = os.path.join(root_path, 'result.html')
  print 'writing html result file to %s...' % (html_file, )
  open(html_file, 'w').write(html)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_path', type=str, help='the input checkpoint')
  parser.add_argument('-r', '--root_path', default='data/example_images', type=str, help='folder with the images, tasks.txt file, and corresponding vgg_feats.mat file')
  parser.add_argument('-b', '--beam_size', type=int, default=1, help='beam size in inference. 1 indicates greedy per-word max procedure. Good value is approx 20 or so, and more = better.')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  main(params)
