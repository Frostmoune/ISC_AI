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

#horovod, yes or no?
horovod = True
try:
    import horovod.tensorflow as hvd
except:
    horovod = False

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
    input_path_train = getValue(opt['dataset'], 'train_dir')
    input_path_validation = getValue(opt['dataset'], 'val_dir')
    trn_sz = getValue(opt['dataset'], 'train_size')
    val_sz = getValue(opt['dataset'], 'val_size')
    downsampling_fact = getValue(opt['dataset'], 'downsampling', 4)
    downsampling_mode = getValue(opt['dataset'], 'downsampling_mode', 'center-crop')
    channels = getValue(opt['dataset'], 'channels', list(range(0, 16)))
    data_format = getValue(opt['dataset'], 'data_format', 'channels_last')
    label_id = getValue(opt['dataset'], 'label_id', 0)
    fs_type = getValue(opt['dataset'], 'fs', 'local')
    dtype = getattr(tf, getValue(opt['dataset'], 'dtype', 'float32')) 
    image_dir = getValue(opt['output'], 'image_dir', 'output')
    disable_imsave = getValue(opt['output'], 'disable_imsave', True)
    checkpoint_dir = getValue(opt['output'], 'checkpoint_dir', 'checkpoint')
    disable_checkpoints = getValue(opt['output'], 'disable_checkpoint', False)
    output_sampling = getValue(opt['output'], 'sample', None)
    optimizer = str2dict(getValue(opt['train'], 'optimizer'))
    batch = getValue(opt['train'], 'batch')
    num_epochs = getValue(opt['train'], 'epoch')
    scale_factor = getValue(opt['train'], 'scale_factor', 1.0)
    tracing = getValue(opt, 'tracing')
    model_type = getValue(opt['model'], 'type', 'Unet')
    if tracing:
        trace_dir = getValue(opt['tracing'], 'dir')
    load_checkpoint = getValue(opt['train'], 'checkpoint', None)
    #init horovod
    comm_rank = 0
    comm_local_rank = 0
    comm_size = 1
    comm_local_size = 1
    if horovod:
        hvd.init()
        comm_rank = hvd.rank()
        comm_local_rank = hvd.local_rank()
        comm_size = hvd.size()
        #not all horovod versions have that implemented
        try:
            comm_local_size = hvd.local_size()
        except:
            comm_local_size = 1
        if comm_rank == 0:
            print("Using distributed computation with Horovod: {} total ranks".format(comm_size,comm_rank))

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
    trn_data = load_data(input_path_train, True, trn_sz, horovod)
    val_data = load_data(input_path_validation, False, val_sz, horovod)
    if comm_rank == 0:
        print("Shape of trn_data is {}".format(trn_data.shape[0]))
        print("Shape of val_data is {}".format(val_data.shape[0]))
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
        #print optimizer parameters
        for k,v in optimizer.items():
            print("Solver Parameters: {k}: {v}".format(k=k,v=v))
        print("Num training samples: {}".format(trn_data.shape[0]))
        print("Num validation samples: {}".format(val_data.shape[0]))
        print("Disable checkpoints: {}".format(disable_checkpoints))
        print("Disable image save: {}".format(disable_imsave))

    #compute epochs and stuff:
    if fs_type == "local":
        num_samples = trn_data.shape[0] // comm_local_size
    else:
        num_samples = trn_data.shape[0] // comm_size
    num_steps_per_epoch = num_samples // batch
    num_steps = num_epochs*num_steps_per_epoch
    if per_rank_output:
        print("Rank {} does {} steps per epoch".format(comm_rank,
                                                       num_steps_per_epoch))

    with training_graph.as_default():
        #create readers
        trn_reader = h5_input_reader(input_path_train, channels, weights, dtype, normalization_file="stats.h5", update_on_read=False, data_format=data_format, label_id=label_id, sample_target=output_sampling)
        val_reader = h5_input_reader(input_path_validation, channels, weights, dtype, normalization_file="stats.h5", update_on_read=False, data_format=data_format, label_id=label_id)
        #create datasets
        if fs_type == "local":
            trn_dataset = create_dataset(trn_reader, trn_data, batch, num_epochs, comm_local_size, comm_local_rank, dtype, shuffle=True)
            val_dataset = create_dataset(val_reader, val_data, batch, 1, comm_local_size, comm_local_rank, dtype, shuffle=False)
        else:
            trn_dataset = create_dataset(trn_reader, trn_data, batch, num_epochs, comm_size, comm_rank, dtype, shuffle=True)
            val_dataset = create_dataset(val_reader, val_data, batch, 1, comm_size, comm_rank, dtype, shuffle=False)

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
                #do downsampling
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
        #trn
        trn_iterator = trn_dataset.make_initializable_iterator()
        trn_handle_string = trn_iterator.string_handle()
        trn_init_op = iterator.make_initializer(trn_dataset)
        #val
        val_iterator = val_dataset.make_initializable_iterator()
        val_handle_string = val_iterator.string_handle()
        val_init_op = iterator.make_initializer(val_dataset)

        #set up model
        logit, prediction, fake, real = createModel(next_elem, opt['model'])
        loss, dloss = createLoss(opt['model'], logit, next_elem, fake, real)

        #determine flops
        flops = graph_flops(format="NHWC" if data_format=="channels_last" else "NCHW", batch=batch, sess_config=sess_config)
        flops *= comm_size
        if comm_rank == 0:
            print('training flops: {:.3f} TF/step'.format(flops * 1e-12))

        #number of trainable parameters
        if comm_rank ==0:
            num_params = get_number_of_trainable_parameters()
            print('number of trainable parameters: {} ({} MB)'.format(num_params, num_params*(4 if dtype==tf.float32 else 2)*(2**-20)))
            
        if horovod:
            loss_avg = hvd.allreduce(ensure_type(loss, tf.float32))
            if not (dloss is None):
                dloss_avg = hvd.allreduce(ensure_type(dloss, tf.float32))
            else:
                dloss_avg = None
        else:
            loss_avg = tf.identity(loss)
            if not (dloss is None):
                dloss_avg = tf.identity(dloss, tf.float32)
            else:
                dloss_avg = None

        #set up global step - keep on CPU
        with tf.device('/device:CPU:0'):
            global_step = tf.train.get_or_create_global_step()

        #set up optimizer
        if optimizer['opt_type'].startswith("LARC"):
            if comm_rank==0:
                print("Enabling LARC")
            train_op, lr = get_larc_optimizer(optimizer, loss, global_step,
                                              num_steps_per_epoch, horovod)
            if not (dloss is None):
                dtrain_op, dlr = get_larc_optimizer(optimizer, dloss, global_step,
                                              num_steps_per_epoch, horovod)
        else:
            train_op, lr = get_optimizer(optimizer, loss, global_step,
                                         num_steps_per_epoch, horovod)
            if not (dloss is None):
                dtrain_op, dlr = get_larc_optimizer(optimizer, dloss, global_step,
                                              num_steps_per_epoch, horovod)

        #set up streaming metrics
        iou_op, iou_update_op = tf.metrics.mean_iou(labels=next_elem[1],
                                                    predictions=tf.argmax(prediction, axis=3),
                                                    num_classes=3,
                                                    weights=None,
                                                    metrics_collections=None,
                                                    updates_collections=None,
                                                    name="iou_score")
        iou_reset_op = tf.variables_initializer([ i for i in tf.local_variables() if i.name.startswith('iou_score/') ])

        if horovod:
            iou_avg = hvd.allreduce(iou_op)
        else:
            iou_avg = tf.identity(iou_op)

        if "gpu" in device.lower():
            with tf.device(device):
                mem_usage_ops = [ tf.contrib.memory_stats.MaxBytesInUse(),
                                  tf.contrib.memory_stats.BytesLimit() ]
        #hooks
        #these hooks are essential. regularize the step hook by adding one additional step at the end
        hooks = [tf.train.StopAtStepHook(last_step=num_steps+1)]
        #bcast init for bcasting the model after start
        if horovod:
            init_bcast = hvd.broadcast_global_variables(0)
        #initializers:
        init_op =  tf.global_variables_initializer()
        init_local_op = tf.local_variables_initializer()

        #checkpointing
        if comm_rank == 0:
            checkpoint_save_freq = 2*num_steps_per_epoch
            checkpoint_saver = tf.train.Saver(max_to_keep = 1000)
            if (not disable_checkpoints):
                hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=checkpoint_dir, save_steps=checkpoint_save_freq, saver=checkpoint_saver))
            #create image dir if not exists
            if not os.path.isdir(image_dir):
                os.makedirs(image_dir)

        #tracing
        if tracing is not None:
            import tracehook
            tracing_hook = tracehook.TraceHook(steps_to_trace=tracing,
                                               cache_traces=True,
                                               trace_dir=trace_dir)
            hooks.append(tracing_hook)

        # instead of averaging losses over an entire epoch, use a moving
        #  window average
        recent_losses, recent_dlosses = [], []
        loss_window_size = 10
        #start session
        with tf.train.MonitoredTrainingSession(config=sess_config, hooks=hooks) as sess:
            #initialize
            sess.run([init_op, init_local_op])
            #restore from checkpoint:
            if comm_rank == 0 and not disable_checkpoints:
                load_model(sess, checkpoint_saver, checkpoint_dir, load_checkpoint=load_checkpoint)
            #broadcast loaded model variables
            if horovod:
                sess.run(init_bcast)
            #create iterator handles
            trn_handle, val_handle = sess.run([trn_handle_string, val_handle_string])
            #init iterators
            sess.run(trn_init_op, feed_dict={handle: trn_handle})
            sess.run(val_init_op, feed_dict={handle: val_handle})

            # figure out what step we're on (it won't be 0 if we are
            #  restoring from a checkpoint) so we can count from there
            train_steps = sess.run([global_step])[0]

            #do the training
            epoch = 1
            step = 1

            prev_mem_usage = 0
            t_sustained_start = time.time()
            r_peak = 0

            #start training
            start_time = time.time()
            while not sess.should_stop():

                #training loop
                try:
                    #construct feed dict
                    t_inst_start = time.time()
                    if not (dloss is None):
                        _, tmp_dloss, cur_dlr = sess.run([dtrain_op,
                                                        (dloss if per_rank_output else dloss_avg),
                                                        dlr],
                                                        feed_dict={handle: trn_handle})
                    _, tmp_loss, cur_lr = sess.run([train_op,
                                                    (loss if per_rank_output else loss_avg),
                                                    lr],
                                                   feed_dict={handle: trn_handle})
                    t_inst_end = time.time()
                    if "gpu" in device.lower():
                        mem_used = sess.run(mem_usage_ops)
                    else:
                        mem_used = [0, 0]
                    train_steps += 1
                    train_steps_in_epoch = train_steps%num_steps_per_epoch
                    recent_losses = [ tmp_loss ] + recent_losses[0:loss_window_size-1]
                    train_loss = sum(recent_losses) / len(recent_losses)
                    if not (dloss is None):
                        recent_dlosses = [ tmp_dloss ] + recent_losses[0:loss_window_size-1]
                        train_dloss = sum(recent_dlosses) / len(recent_dlosses)
                    step += 1

                    r_inst = 1e-12 * flops / (t_inst_end-t_inst_start)
                    r_peak = max(r_peak, r_inst)

                    if (train_steps % loss_print_interval) == 0:
                        if "gpu" in device.lower():
                            mem_used = sess.run(mem_usage_ops)
                        else:
                            mem_used = [0, 0]
                        if per_rank_output:
                            print("REPORT: rank {}, training loss for step {} (of {}) is {}, time {:.3f}".format(comm_rank, train_steps, num_steps, train_loss, time.time()-start_time))
                            if not (dloss is None):
                                print("REPORT: rank {}, training dloss for step {} (of {}) is {}, time {:.3f}".format(comm_rank, train_steps, num_steps, train_dloss, time.time()-start_time))
                        else:
                            if comm_rank == 0:
                                if mem_used[0] > prev_mem_usage:
                                    print("memory usage: {:.2f} GB / {:.2f} GB".format(mem_used[0] / 2.0**30, mem_used[1] / 2.0**30))
                                    prev_mem_usage = mem_used[0]
                                print("REPORT: training loss for step {} (of {}) is {}, time {:.3f}, r_inst {:.3f}, r_peak {:.3f}, lr {:.2g}".format(train_steps, num_steps, train_loss, time.time()-start_time, r_inst, r_peak, cur_lr))
                                if not (dloss is None):
                                    print("REPORT: training dloss for step {} (of {}) is {}, time {:.3f}, r_inst {:.3f}, r_peak {:.3f}, lr {:.2g}".format(train_steps, num_steps, train_dloss, time.time()-start_time, r_inst, r_peak, cur_dlr))

                    #do the validation phase
                    if train_steps_in_epoch == 0:
                        end_time = time.time()
                        #print epoch report
                        if per_rank_output:
                            print("COMPLETED: rank {}, training loss for epoch {} (of {}) is {}, time {:.3f}, r_sust {:.3f}".format(comm_rank, epoch, num_epochs, train_loss, time.time() - start_time, 1e-12 * flops * num_steps_per_epoch / (end_time-t_sustained_start)))
                            if not (dloss is None):
                                print("COMPLETED: rank {}, training dloss for epoch {} (of {}) is {}, time {:.3f}, r_sust {:.3f}".format(comm_rank, epoch, num_epochs, train_dloss, time.time() - start_time, 1e-12 * flops * num_steps_per_epoch / (end_time-t_sustained_start)))
                        else:
                            if comm_rank == 0:
                                print("COMPLETED: training loss for epoch {} (of {}) is {}, time {:.3f}, r_sust {:.3f}".format(epoch, num_epochs, train_loss, time.time() - start_time, 1e-12 * flops * num_steps_per_epoch / (end_time-t_sustained_start)))
                                if not (dloss is None):
                                    print("COMPLETED: training dloss for epoch {} (of {}) is {}, time {:.3f}, r_sust {:.3f}".format(epoch, num_epochs, train_dloss, time.time() - start_time, 1e-12 * flops * num_steps_per_epoch / (end_time-t_sustained_start)))

                        #evaluation loop
                        eval_loss = 0.
                        eval_steps = 0
                        while True:
                            try:
                                #construct feed dict
                                _, tmp_loss, val_model_predictions, val_model_labels, val_model_filenames = sess.run([iou_update_op,
                                                                                                                      (loss if per_rank_output else loss_avg),
                                                                                                                      prediction,
                                                                                                                      next_elem[1],
                                                                                                                      next_elem[3]],
                                                                                                                      feed_dict={handle: val_handle})
                                #print some images
                                if comm_rank == 0 and not disable_imsave:
                                    if have_imsave:
                                        imsave(image_dir+'/test_pred_epoch'+str(epoch)+'_estep'
                                               +str(eval_steps)+'_rank'+str(comm_rank)+'.png',np.argmax(val_model_predictions[0,...],axis=2)*100)
                                        imsave(image_dir+'/test_label_epoch'+str(epoch)+'_estep'
                                               +str(eval_steps)+'_rank'+str(comm_rank)+'.png',val_model_labels[0,...]*100)
                                        imsave(image_dir+'/test_combined_epoch'+str(epoch)+'_estep'
                                               +str(eval_steps)+'_rank'+str(comm_rank)+'.png',plot_colormap[val_model_labels[0,...],np.argmax(val_model_predictions[0,...],axis=2)])
                                    else:
                                        np.savez(image_dir+'/test_epoch'+str(epoch)+'_estep'
                                                 +str(eval_steps)+'_rank'+str(comm_rank)+'.npz', prediction=np.argmax(val_model_predictions[0,...],axis=2)*100,
                                                                                                 label=val_model_labels[0,...]*100,
                                                                                                 filename=val_model_filenames[0])
                                eval_loss += tmp_loss
                                eval_steps += 1
                            except tf.errors.OutOfRangeError:
                                eval_steps = np.max([eval_steps,1])
                                eval_loss /= eval_steps
                                if per_rank_output:
                                    print("COMPLETED: rank {}, evaluation loss for epoch {} (of {}) is {}".format(comm_rank, epoch, num_epochs, eval_loss))
                                else:
                                    if comm_rank == 0:
                                        print("COMPLETED: evaluation loss for epoch {} (of {}) is {}".format(epoch, num_epochs, eval_loss))
                                if per_rank_output:
                                    iou_score = sess.run(iou_op)
                                    print("COMPLETED: rank {}, evaluation IoU for epoch {} (of {}) is {}".format(comm_rank, epoch, num_epochs, iou_score))
                                else:
                                    iou_score = sess.run(iou_avg)
                                    if comm_rank == 0:
                                        print("COMPLETED: evaluation IoU for epoch {} (of {}) is {}".format(epoch, num_epochs, iou_score))
                                sess.run(iou_reset_op)
                                sess.run(val_init_op, feed_dict={handle: val_handle})
                                break

                        #reset counters
                        epoch += 1
                        step = 0
                        t_sustained_start = time.time()

                except tf.errors.OutOfRangeError:
                    break

        # write any cached traces to disk
        if tracing is not None:
            tracing_hook.write_traces()

if __name__ == '__main__':
    AP = argparse.ArgumentParser()
    AP.add_argument("--opt_dir",type=str,default='options/RRDB_Conv.json',help="Defines the location of options")
    parsed = AP.parse_args()

    opt = None
    with open(parsed.opt_dir, 'r') as f:
        opt = json.load(f)
        
    #invoke main function
    main(opt)