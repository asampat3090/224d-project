import os
import time
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil

import sys
import argparse

from scipy.misc import imread, imresize
import scipy.io as io
import string
import cPickle as pickle

import json
import time
import datetime
import code
import os
import cPickle as pickle
import math
import pdb


sys.path.append('../')
from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeGenerator, eval_split

caffepath = '~/caffe/python'
sys.path.append(caffepath)
import matplotlib
matplotlib.use('Agg')
import caffe

ROOT_DIRNAME = os.path.abspath(os.path.dirname(__file__) + '/..')
UPLOAD_FOLDER = '/tmp/captionly_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

# Obtain the flask app object
app = flask.Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)

@app.route('/caption_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    result = app.clf.classify_image(image)
    return flask.render_template(
        'index.html', has_result=True, result=result, imagesrc=imageurl)

@app.route('/captionly_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    result = app.clf.classify_image(image)
    return flask.render_template(
        'index.html', has_result=True, result=result,
        imagesrc=embed_image_html(image)
    )

def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data

def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


########## DEFINE THE IMAGE CAPTION CLASS ######################

class ImageCaptioner(object):

    default_args = {
        'cnn_model_def' : '../python_features/deploy_features.prototext',
        'cnn_model_params' : '~/VGG_ILSVRC_16_layers.caffemodel',
        'rnn_model' : '../cv/model_checkpoint_coco_visionlab43.stanford.edu_lstm_11.14.p'
    }
    # default_args = {
    #     'model_def_file': (
    #         '{}/models/bvlc_reference_caffenet/deploy.prototxt'.format(REPO_DIRNAME)),
    #     'pretrained_model_file': (
    #         '{}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'.format(REPO_DIRNAME)),
    #     'mean_file': (
    #         '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
    #     'class_labels_file': (
    #         '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
    #     'bet_file': (
    #         '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
    # }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 256
    default_args['raw_scale'] = 255.

    # def __init__(self, model_def_file, pretrained_model_file, mean_file,
    #              raw_scale, class_labels_file, bet_file, image_dim, gpu_mode):
        
    def __init__(self,caffe_path, cnn_model_def, cnn_model_params,rnn_model,gpu_mode=True):
        
        logging.info('Loading CNN and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.net = caffe.Classifier(
            model_def_file, pretrained_model_file,
            image_dims=(image_dim, image_dim), raw_scale=raw_scale,
            mean=np.load(mean_file).mean(1).mean(1), channel_swap=(2, 1, 0)
            )

        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
        self.labels = labels_df.sort('synset_id')['name'].values

        self.bet = cPickle.load(open(bet_file))
        # A bias to prefer children nodes in single-chain paths
        # I am setting the value to 0.1 as a quick, simple model.
        # We could use better psychological models here...
        self.bet['infogain'] -= np.array(self.bet['preferences']) * 0.1

    def predict_sentence(self, image):
        try:

            ################ FEATURE EXTRACTION ##############

            cnn_model_def = default_args['cnn_model_def']
            cnn_model_params = default_args['cnn_model_params']
            rnn_model = default_args['rnn_model']


            def predict(in_data, net):
                """
                Get the features for a batch of data using network

                Inputs:
                in_data: data batch
                """

                out = net.forward(**{net.inputs[0]: in_data})
                features = out[net.outputs[0]].squeeze(axis=(2,3))
                return features


                def batch_predict(filenames, net):
                    """
                    Get the features for all images from filenames using a network

                    Inputs:
                    filenames: a list of names of image files

                    Returns:
                    an array of feature vectors for the images in that file
                    """

                    N, C, H, W = net.blobs[net.inputs[0]].data.shape
                    F = net.blobs[net.outputs[0]].data.shape[1]
                    Nf = len(filenames)
                #pdb.set_trace()
                Hi, Wi, _ = imread(IMAGE_PATH + '/' + filenames[0]).shape
                allftrs = np.zeros((Nf, F))
                for i in range(0, Nf, N):
                    in_data = np.zeros((N, C, H, W), dtype=np.float32)

                    batch_range = range(i, min(i+N, Nf))
                    batch_filenames = [filenames[j] for j in batch_range]
                    Nb = len(batch_range)

                    batch_images = np.zeros((Nb, 3, H, W))
                    for j,fname in enumerate(batch_filenames):
                        im = imread(IMAGE_PATH + '/' + fname)
                        if len(im.shape) == 2:
                            im = np.tile(im[:,:,np.newaxis], (1,1,3))
                        # RGB -> BGR
                        im = im[:,:,(2,1,0)]
                        # mean subtraction
                        im = im - np.array([103.939, 116.779, 123.68])
                        # resize
                        im = imresize(im, (H, W))
                        # get channel in correct dimension
                        im = np.transpose(im, (2, 0, 1))
                        batch_images[j,:,:,:] = im

                    # insert into correct place
                    in_data[0:len(batch_range), :, :, :] = batch_images

                    # predict features
                    ftrs = predict(in_data, net)

                    for j in range(len(batch_range)):
                        allftrs[i+j,:] = ftrs[j,:]

                        print 'Done %d/%d files' % (i+len(batch_range), len(filenames))

                        return allftrs


                        if args.gpu:
                            caffe.set_mode_gpu()
                        else:   
                            caffe.set_mode_cpu()

                            net = caffe.Net(args.model_def, args.model)
                            caffe.set_phase_test()

                            filenames = []
                            with open(args.files) as fp:
                                for line in fp:
                                    filename = line.strip().split()[0]
                                    filenames.append(filename)

                                    allftrs = batch_predict(filenames, net)



            #################### PREDICTION ##################

            dim = 300
            # load the checkpoint
            checkpoint_path = rnn_model
            # load glove vect dict

            glove_dict_path = '../../vecDict.pickle'
            with open(glove_dict_path, 'rb') as handle:
                vec_dict = pickle.load(handle)

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

            # create and load the tasks.txt file
            # root_path = params['root_path']
            allImages = os.listdir(UPLOAD_FOLDER)
            with open(os.path.join(UPLOAD_FOLDER, 'tasks.txt'), 'w') as f:
                for k, v in enumerate(allImages):
                    if k==len(allImages)-1: 
                        f.write(v)
                    else: 
                        f.write(v + '\n')

                        img_names = open(os.path.join(root_path, 'tasks.txt'), 'r').read().splitlines()

            # starttime = time.time()
            # scores = self.net.predict([image], oversample=True).flatten()
            # endtime = time.time()

            # indices = (-scores).argsort()[:5]
            # predictions = self.labels[indices]

            # # In addition to the prediction text, we will also produce
            # # the length for the progress bar visualization.
            # meta = [
            #     (p, '%.5f' % scores[i])
            #     for i, p in zip(indices, predictions)
            # ]
            # logging.info('result: %s', str(meta))

            # # Compute expected information gain
            # expected_infogain = np.dot(
            #     self.bet['probmat'], scores[self.bet['idmapping']])
            # expected_infogain *= self.bet['infogain']

            # # sort the scores
            # infogain_sort = expected_infogain.argsort()[::-1]
            # bet_result = [(self.bet['words'][v], '%.5f' % expected_infogain[v])
            #               for v in infogain_sort[:5]]
            # logging.info('bet result: %s', str(bet_result))

            # return (True, meta, bet_result, '%.3f' % (endtime - starttime))

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')


################## END THE IMAGE CAPTION CLASS ######################






def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()

def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier + warm start by forward for allocation
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    app.clf.net.forward()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)

