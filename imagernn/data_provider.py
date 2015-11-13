import json
import os
import random
import scipy.io
import codecs
from collections import defaultdict
import pdb
from sets import Set

class BasicDataProvider:
  def __init__(self, dataset):
    print 'Initializing data provider for dataset %s...' % (dataset, )

    # !assumptions on folder structure
    self.dataset_root = os.path.join('data', dataset)
    self.image_root = os.path.join('data', dataset, 'imgs')

    # load the dataset into memory
    dataset_path = os.path.join(self.dataset_root, 'dataset.json')
    print 'BasicDataProvider: reading %s' % (dataset_path, )
    # pdb.set_trace()
    self.dataset = json.load(open(dataset_path, 'r'))

    # pdb.set_trace()

    # load the image features into memory
    features_path = os.path.join(self.dataset_root, 'vgg_feats.mat')
    print 'BasicDataProvider: reading %s' % (features_path, )
    features_struct = scipy.io.loadmat(features_path)
    self.features = features_struct['feats']
    print self.features.shape

    # Load tasks.txt file
    img_order = os.path.join(self.dataset_root, 'tasks.txt')
    self.image_filenames = []
    for l in open(img_order,'rb'):
      self.image_filenames.append(l.strip())

    # group images by their train/val/test split into a dictionary -> list structure
    self.split = defaultdict(list)
    # checkId = 318556
    # for 
    # pdb.set_trace()
    imgIdSet = Set([])
    annotationList = []
    for annotation in self.dataset['annotations']:      
      if annotation['image_id'] not in imgIdSet:
        imgIdSet.add(annotation['image_id'])
        annotationList.append(annotation)

    for img in annotationList:      
      self.split['train'].append(img)

  # "PRIVATE" FUNCTIONS
  # in future we may want to create copies here so that we don't touch the 
  # data provider class data, but for now lets do the simple thing and 
  # just return raw internal img sent structs. This also has the advantage
  # that the driver could store various useful caching stuff in these structs
  # and they will be returned in the future with the cache present
  def _getImage(self, img):
    """ create an image structure for the driver """

    # lazily fill in some attributes
    #if not 'local_file_path' in img: img['local_file_path'] = os.path.join(self.image_root, img['filename'])
    if not 'feat' in img: # also fill in the features

      real_file_name = 'COCO_train2014_' + str(img['image_id']).zfill(12) + '.jpg'
      # pdb.set_trace()
      i = 0

      for fileName in self.image_filenames:
        i+=1
        if real_file_name == fileName:
          feature_index = i
          # pdb.set_trace()
          img['feat'] = self.features[:,feature_index]
    return img

  def _getSentence(self, sent):
    """ create a sentence structure for the driver """
    # NOOP for now
    return sent

  # PUBLIC FUNCTIONS

  def getSplitSize(self, split, ofwhat = 'sentences'):
    """ return size of a split, either number of sentences or number of images """
    # if ofwhat == 'sentences': 
    #   return sum([1 for img in self.split[split]]) 
    # else: # assume images
    return len(self.split[split])

  def sampleImageSentencePair(self, split = 'train'):
    """ sample image sentence pair from a split """
    images = self.split[split]

    img = random.choice(images)
    sent = img['caption']
    out = {}
    out['image'] = self._getImage(img)
    out['sentence'] = self._getSentence(sent)
    return out

  def iterImageSentencePair(self, split = 'train', max_images = -1):
    for i,img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      out = {}
      out['image'] = self._getImage(img)
      sent = img['caption']
      out['sentence'] = self._getSentence(sent)
      yield out

  def iterImageSentencePairBatch(self, split = 'train', max_images = -1, max_batch_size = 100):
    batch = []
    for i,img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      out = {}
      out['image'] = self._getImage(img)
      sent = img['caption']
      out['sentence'] = self._getSentence(sent)
      batch.append(out)
      if len(batch) >= max_batch_size:
        yield batch
        batch = []
    if batch:
      yield batch

  def iterSentences(self, split = 'train'):
    for img in self.split[split]: 
      sent = img['caption']
      yield self._getSentence(sent)

  def iterImages(self, split = 'train', shuffle = False, max_images = -1):
    imglist = self.split[split]
    ix = range(len(imglist))
    if shuffle:
      random.shuffle(ix)
    if max_images > 0:
      ix = ix[:min(len(ix),max_images)] # crop the list
    for i in ix:
      yield self._getImage(imglist[i])

def getDataProvider(dataset):
  """ we could intercept a special dataset and return different data providers """
  assert dataset in ['flickr8k', 'flickr30k', 'coco','example_images'], 'dataset %s unknown' % (dataset, )
  return BasicDataProvider(dataset)
