import numpy as np
import cPickle as pickle
import argparse
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import pdb
from nltk.stem.snowball import EnglishStemmer
from math import sqrt, ceil

def load_glove_vectors():
  glove_dict_path = '../vecDict.pickle'
  print 'Going to load vecDict'
  with open(glove_dict_path, 'rb') as handle:
    vec_dict = pickle.load(handle)
  return vec_dict

def load_image_caption_vectors():
  print 'Going to load filename_to_vector'
  with open('filename_to_vector.pickle', 'rb') as handle:
    filename_to_vector = pickle.load(handle)
  return filename_to_vector

def get_query(vec_dict):
  dim = 300 # Dimension of the GloVe vectors chosen

  # initialize stemmer for search in GLoVe vector space
  st = EnglishStemmer()

  query = raw_input('Please enter search query:')
  query_vector = np.zeros(dim)
  numWords = 0
  for word in query.split():
    if st.stem(word) in vec_dict:
      query_vector += vec_dict[st.stem(word)].astype(np.float)
      numWords += 1
    elif st.stem(word)+'e' in vec_dict:
      query_vector += vec_dict[st.stem(word)+'e'].astype(np.float)
      numWords += 1

  query_vector /= numWords

  return query, query_vector

def top_k_search(query,query_vector,filename_to_vector,k):
  distArr = []
  for fileName in filename_to_vector.keys():
    distArr.append(np.linalg.norm(filename_to_vector[fileName]-query_vector))

  distArr = np.array(distArr)
  sortedInd = np.argsort(distArr)

  top_k_filenames = [filename_to_vector.keys()[i] for i in sortedInd[:k]]
  top_k_distances = [distArr[i] for i in sortedInd[:k]]

  return top_k_filenames,top_k_distances

def nDCG_score(top_k_filenames,top_k_distances,images_path):
  
  # list the top k image names + distances + get relevance from user
  rel_score_arr = []

  file_to_write = {'filename': [], 'distance': [], 'rel_score':[]} 
  for filename,dist in zip(top_k_filenames,top_k_distances):
    # show the image to the user 
    curr_img = Image.open(os.path.join(images_path,filename))
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
    file_to_write['filename'].append(filename)
    file_to_write['distance'].append(dist)
    file_to_write['rel_score'].append(rel_score)

  # Calculate the nDCG score for this query

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

  return nDCG,file_to_write

def mean_rank_score(top_k_filenames, top_k_distances,images_path):
  # convert all k images to proper format
  num_images = len(top_k_filenames)
  h = 128
  w = 128
  c = 3

  ubound=255.0
  padding=1

  Xs = np.zeros((num_images,h,w,c))
  for i, (filename,dist) in enumerate(zip(top_k_filenames,top_k_distances)):
    # convert image to numpy array and add to input data Xs
    curr_img = Image.open(os.path.join(images_path,filename))
    curr_img = curr_img.resize((h,w),Image.ANTIALIAS)
    curr_img = np.asarray(curr_img)
    Xs[i] = curr_img

  # visualize all k images
  (N, H, W, C) = Xs.shape
  grid_size = int(ceil(sqrt(N)))
  grid_height = H * grid_size + padding * (grid_size - 1)
  grid_width = W * grid_size + padding * (grid_size - 1)
  grid = np.zeros((grid_height, grid_width, C))
  next_idx = 0
  y0, y1 = 0, H
  for y in xrange(grid_size):
    x0, x1 = 0, W
    for x in xrange(grid_size):
      if next_idx < N:
        img = Xs[next_idx]
        low, high = np.min(img), np.max(img)
        grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
        # grid[y0:y1, x0:x1] = Xs[next_idx]
        next_idx += 1
      x0 += W + padding
      x1 += W + padding
    y0 += H + padding
    y1 += H + padding

  plt.imshow(grid.astype('uint8'))
  plt.show()

  # Have the user enter the relevant images
  csv_indices = raw_input('Please input indices of relevant images for query')
  if csv_indices=='':
    mean_score = 0
  else:
    ind_list = [int(c) for c in csv_indices.split(',')]
    mean_score = float(sum(ind_list))/len(ind_list)
  return mean_score

def write_to_file(query,file_to_write):
  output_path = 'output/'
  # write to results for this query to file
  output_filename = '_'.join(query.split(' '))+'.txt'

  with open(os.path.join(output_path,output_filename), 'w') as f:
    f.write(','.join(file_to_write.keys()))
    for f,d,r in zip(file_to_write['filename'],file_to_write['distance'],file_to_write['rel_score']):
      f.write('%s,%s,%d\n' % (f,d,r))
    f.write('\nquery: %s\n' % query)
    f.write('\nnDCG score: %f\n' % nDCG)

  print "wrote to file %s" % output_filename

if __name__ == "__main__":

  # Command line parse

  # parser = argparse.ArgumentParser()
  # parser.add_argument('searchQuery', type=str, help='the query search query')
  
  # args = parser.parse_args()
  # params = vars(args) # convert to ordinary dict
  # print 'parsed parameters:'
  # print json.dumps(params, indent = 2)

  vec_dict = load_glove_vectors()
  filename_to_vect = load_image_caption_vectors()
  # Find images from Flicker dataset
  flikr8k_path = '../Flicker8k_Dataset/'

  # # Use nDCG Score for results
  # while True:
  #   query, query_vector = get_query(vec_dict)
  #   top_k_filenames,top_k_distances = top_k_search(query,query_vector,filename_to_vect,10)
  #   nDCG, file_data = nDCG_score(top_k_filenames,top_k_distances,flikr8k_path)
  #   write_to_file(query,file_data)

  # Try out mean_rank_score
  while True:
    k = int(raw_input('please enter number of top results you would like.'))
    query, query_vector = get_query(vec_dict)
    top_k_filenames,top_k_distances = top_k_search(query,query_vector,filename_to_vect,k)
    mean_rank = mean_rank_score(top_k_filenames,top_k_distances,flikr8k_path)
    print mean_rank
