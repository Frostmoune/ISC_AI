# suppress warnings from earlier versions of h5py (imported by tensorflow)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# don't write bytecode for imported modules to disk - different ranks can
#  collide on the file and/or make the file system angry
import sys
sys.dont_write_bytecode = True

import tensorflow as tf
import numpy as np
import argparse

# instead of scipy, use PIL directly to save images
try:
    import PIL
    def imsave(filename, data):
        PIL.Image.fromarray(data.astype(np.uint8)).save(filename)
    have_imsave = True
except ImportError:
    have_imsave = False

import h5py as h5
import os
import time
import json

# limit tensorflow spewage to just warnings and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from models import createModel, createLoss
from utils import *

#GLOBAL CONSTANTS
image_height_orig = 768
image_width_orig = 1152

def getValue(now_dict, key, default=None):
    return now_dict[key] if key in now_dict else default

def str2dict(line):
    my_dict = {}
    for kv in line.split(","):
        k, v = kv.split("=")
        my_dict[k] = v
    return my_dict

#main function
def main(opt):
    frequencies = [0.991, 0.0266, 0.13]
    weights = [1./x for x in frequencies]
    weights /= np.sum(weights)
    device = getValue(opt, 'device', '/device:gpu:0')
    input_path_test = getValue(opt['dataset'], 'test_dir')
    tst_sz = getValue(opt['dataset'], 'test_size')
    downsampling_fact = getValue(opt['dataset'], 'downsampling', 4)
    downsampling_mode = getValue(opt['dataset'], 'downsampling_mode', 'center-crop')
    channels = getValue(opt['dataset'], 'channels', list(range(0, 16)))
    data_format = getValue(opt['dataset'], 'data_format', 'channels_last')
    label_id = getValue(opt['dataset'], 'label_id', 0)
    fs_type = getValue(opt['dataset'], 'fs', 'local')
    dtype = getattr(tf, getValue(opt['dataset'], 'dtype', 'float32')) 
    image_dir = getValue(opt['output'], 'image_dir', 'output')
    checkpoint_dir = getValue(opt['output'], 'checkpoint_dir', 'checkpoint')
    output_sampling = getValue(opt['output'], 'sample', None)
    batch = getValue(opt['test'], 'batch')
    scale_factor = getValue(opt['test'], 'scale_factor', 1.0)
    load_checkpoint = getValue(opt['test'], 'checkpoint', None)
    model_type = getValue(opt['model'], 'type', 'Unet')
    #init horovod
    comm_rank = 0
    comm_local_rank = 0
    comm_size = 1
    comm_local_size = 1

    #downsampling? recompute image dims
    image_height =  image_height_orig // downsampling_fact
    image_width = image_width_orig // downsampling_fact

    #parameters
    per_rank_output = False
    loss_print_interval = 10

    #session config
    sess_config=tf.ConfigProto(inter_op_parallelism_threads=1, #1
                               intra_op_parallelism_threads=6, #6
                               log_device_placement=False,
                               allow_soft_placement=True)
    sess_config.gpu_options.visible_device_list = str(comm_local_rank)
    sess_config.gpu_options.force_gpu_compatible = True

    #get data
    training_graph = tf.Graph()
    if comm_rank == 0:
        print("Loading data...")
    tst_data = load_data(input_path_test, shuffle=False, max_files=tst_sz, use_horovod=False)
    if comm_rank == 0:
        print("Shape of tst_data is {}".format(tst_data.shape[0]))
        print("done.")

    #print some stats
    if comm_rank==0:
        print("Num workers: {}".format(comm_size))
        print("Local batch size: {}".format(batch))
        if dtype == tf.float32:
            print("Precision: {}".format("FP32"))
        else:
            print("Precision: {}".format("FP16"))
        print("Model: {}".format(model_type))
        print("Channels: {}".format(channels))
        print("Loss weights: {}".format(weights))
        print("Loss scale factor: {}".format(scale_factor))
        print("Output sampling target: {}".format(output_sampling))
        print("Num test samples: {}".format(tst_data.shape[0]))

    #compute epochs and stuff:
    if fs_type == "local":
        num_samples = tst_data.shape[0] // comm_local_size
    else:
        num_samples = tst_data.shape[0] // comm_size

    with training_graph.as_default():
        #create readers
        tst_reader = h5_input_reader(input_path_test, channels, weights, dtype, normalization_file="stats.h5", update_on_read=False, data_format=data_format, label_id=label_id)
        #create datasets
        if fs_type == "local":
            tst_dataset = create_dataset(tst_reader, tst_data, batch, 1, comm_local_size, comm_local_rank, dtype, shuffle=False)
        else:
            tst_dataset = create_dataset(tst_reader, tst_data, batch, 1, comm_size, comm_rank, dtype, shuffle=False)

        #create iterators
        handle = tf.placeholder(tf.string, shape=[], name="iterator-placeholder")
        iterator = tf.data.Iterator.from_string_handle(handle, (dtype, tf.int32, dtype, tf.string),
                                                       ((batch, len(channels), image_height_orig, image_width_orig) if data_format=="channels_first" else (batch, image_height_orig, image_width_orig, len(channels)),
                                                        (batch, image_height_orig, image_width_orig),
                                                        (batch, image_height_orig, image_width_orig),
                                                        (batch))
                                                       )
        next_elem = iterator.get_next()

        #if downsampling, do some preprocessing
        if downsampling_fact != 1:
            if downsampling_mode == "scale":
                rand_select = tf.cast(tf.one_hot(tf.random_uniform((batch, image_height, image_width), minval=0, maxval=downsampling_fact*downsampling_fact, dtype=tf.int32), depth=downsampling_fact*downsampling_fact, axis=-1), dtype=tf.int32)
                next_elem = (tf.layers.average_pooling2d(next_elem[0], downsampling_fact, downsampling_fact, 'valid', data_format), \
                            tf.reduce_max(tf.multiply(tf.image.extract_image_patches(tf.expand_dims(next_elem[1], axis=-1), \
                                                                              [1, downsampling_fact, downsampling_fact, 1], \
                                                                              [1, downsampling_fact, downsampling_fact, 1], \
                                                                              [1,1,1,1], 'VALID'), rand_select), axis=-1), \
                            tf.squeeze(tf.layers.average_pooling2d(tf.expand_dims(next_elem[2], axis=-1), downsampling_fact, downsampling_fact, 'valid', "channels_last"), axis=-1), \
                            next_elem[3])
        
            elif downsampling_mode == "center-crop":
                #some parameters
                length = 1./float(downsampling_fact)
                offset = length/2.
                boxes = [[ offset, offset, offset+length, offset+length ]]*batch
                box_ind = list(range(0,batch))
                crop_size = [image_height, image_width]
                
                #be careful with data order
                if data_format=="channels_first":
                    next_elem[0] = tf.transpose(next_elem[0], perm=[0,2,3,1])
                    
                #crop
                next_elem = (tf.image.crop_and_resize(next_elem[0], boxes, box_ind, crop_size, method='bilinear', extrapolation_value=0, name="data_cropping"), \
                             ensure_type(tf.squeeze(tf.image.crop_and_resize(tf.expand_dims(next_elem[1],axis=-1), boxes, box_ind, crop_size, method='nearest', extrapolation_value=0, name="label_cropping"), axis=-1), tf.int32), \
                             tf.squeeze(tf.image.crop_and_resize(tf.expand_dims(next_elem[2],axis=-1), boxes, box_ind, crop_size, method='bilinear', extrapolation_value=0, name="weight_cropping"), axis=-1), \
                             next_elem[3])
                
                #be careful with data order
                if data_format=="channels_first":
                    next_elem[0] = tf.transpose(next_elem[0], perm=[0,3,1,2])
                    
            else:
                raise ValueError("Error, downsampling mode {} not supported. Supported are [center-crop, scale]".format(downsampling_mode))

        #create init handles
        #tst
        tst_iterator = tst_dataset.make_initializable_iterator()
        tst_handle_string = tst_iterator.string_handle()
        tst_init_op = iterator.make_initializer(tst_dataset)

        #set up model
        logit, prediction, fake, real = createModel(next_elem, opt['model'])
        loss, dloss = createLoss(opt['model'], logit, next_elem, fake, real)

        num_channels = len(channels)

        #set up streaming metrics
        iou_op, iou_update_op = tf.metrics.mean_iou(labels=next_elem[1],
                                                    predictions=tf.argmax(prediction, axis=3),
                                                    num_classes=3,
                                                    weights=None,
                                                    metrics_collections=None,
                                                    updates_collections=None,
                                                    name="iou_score")
        iou_reset_op = tf.variables_initializer([ i for i in tf.local_variables() if i.name.startswith('iou_score/') ])

        init_op =  tf.global_variables_initializer()
        init_local_op = tf.local_variables_initializer()

        #create image dir if not exists
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)

        #start session
        with tf.Session(config=sess_config) as sess:
            #initialize
            sess.run([init_op, init_local_op])
            load_model(sess, tf.train.Saver(), checkpoint_dir, load_checkpoint=load_checkpoint)
            #create iterator handles
            tst_handle = sess.run(tst_handle_string)
            #init iterators
            sess.run(tst_init_op, feed_dict={handle: tst_handle})

            #start training
            eval_loss = 0.0
            eval_steps = 0
            print("Starting evaluation on test set")
            while True:
                try:
                    #construct feed dict
                    _, tmp_loss, tst_model_predictions, tst_model_labels, tst_model_filenames = sess.run([iou_update_op,
                                                                                                          loss,
                                                                                                          prediction,
                                                                                                          next_elem[1],
                                                                                                          next_elem[3]],
                                                                                                          feed_dict={handle: tst_handle})
                    #print some images
                    if have_imsave:
                        imsave(image_dir+'/test_pred_estep'
                               +str(eval_steps)+'_rank'+str(comm_rank)+'.png', np.argmax(tst_model_predictions[0,...],axis=-1)*100)
                        imsave(image_dir+'/test_label_estep'
                               +str(eval_steps)+'_rank'+str(comm_rank)+'.png', tst_model_labels[0,...]*100)
                        imsave(image_dir+'/test_combined_estep'
                               +str(eval_steps)+'_rank'+str(comm_rank)+'.png', plot_colormap[tst_model_labels[0,...],np.argmax(tst_model_predictions[0,...],axis=-1)])
                    else:
                        np.savez(image_dir+'/test_estep'
                                 +str(eval_steps)+'_rank'+str(comm_rank)+'.npz', prediction=np.argmax(tst_model_predictions[...],axis=-1)*100,
                                                                                                 label=tst_model_labels[...]*100, filename=tst_model_filenames)

                    #update loss
                    eval_loss += tmp_loss
                    eval_steps += 1

                except tf.errors.OutOfRangeError:
                    eval_steps = np.max([eval_steps,1])
                    eval_loss /= eval_steps
                    print("COMPLETED: evaluation loss is {}".format(eval_loss))
                    iou_score = sess.run(iou_op)
                    print("COMPLETED: evaluation IoU is {}".format(iou_score))
                    break

if __name__ == '__main__':
    AP = argparse.ArgumentParser()
    AP.add_argument("--opt_dir",type=str,default='options/UnetPlus6Test.json',help="Defines the location of options")
    parsed = AP.parse_args()

    opt = None
    with open(parsed.opt_dir, 'r') as f:
        opt = json.load(f)
        
    #invoke main function
    main(opt)