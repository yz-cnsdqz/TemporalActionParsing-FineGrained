from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys, glob
import matplotlib.pyplot as plt
import random
import scipy.io as sio
import metrics


time_conv_size = int(sys.argv[1])
seq_length = int(sys.argv[2])







## prepare the pretrained VGG-16 model on ImageNet
## train the recurrent model end to end jointly

## prepare VGG network and loss
# import tensorflow.contrib.slim.nets as nets
from temporal_ed_bilinear import end_to_end_tensor_flow
from sklearn.metrics import accuracy_score


bg_class = 0
trial_metrics = metrics.ComputeMetrics(overlap=.1, bg_class=bg_class)





for ss in range(5):
    splits = ss+1


    ## ----------------------- Overall config -----------------------
    eval_opt = { 'split':splits,
                 'train_log_dir':'/home/yzhang/workspace/tensorflow_proj/exercise3-LearableStructureTensor/logs-50Salads/TimeConvSize{:d}-Length{:d}-Split{:d}'.format(time_conv_size,seq_length,splits),
                 'dataset_path':'/mnt/hdd/Dataset_50Salads/'
                }



    # print('[EVAL INFO] - evaluting performance on split1/test')
    dataset_path = eval_opt['dataset_path']
    #model_path = '/home/yzhang/workspace/tensorflow-models/models-pretrained/VGG16-ImageNet/vgg_16.ckpt'
    ## TODO sepcify the checkpoint file
    model_path_list = sorted(glob.glob(os.path.join(eval_opt['train_log_dir'],'model-*.meta')),
                             key=os.path.getmtime)
    model_path = model_path_list[-1][:-5]
    print('[EVAL INFO] - model:' + model_path)



    # model_path = eval_opt['train_log_dir'] + '/model-split{:d}.ckpt-9065'.format(splits)


    with open(os.path.join(dataset_path, 'annotations/mid_level_actions.txt')) as f:
        lines = f.read().splitlines()
        action_list = [x.split()[0] for x in lines]
    action_list = ['background']+action_list

    n_classes = len(action_list) # actions + background actions

    if trial_metrics.n_classes is None:
        trial_metrics.set_classes(n_classes)


    feature_path = os.path.join(dataset_path, 'features/SpatialCNN_mid')
    split_path = os.path.join(dataset_path, 'annotations/split/Split_{:d}'.format(eval_opt['split']))

    with open(os.path.join(split_path, 'train.txt')) as f:
        lines = f.read().splitlines()
        train_trial_list = [x.split()[0] for x in lines]

    with open(os.path.join(split_path, 'test.txt')) as f:
        lines = f.read().splitlines()
        test_trial_list = [x.split()[0] for x in lines]



    feature_list = []
    label_list = []
    video_idx_list = []



    train_list = [ os.path.join(feature_path,feature_path,'Split_{:d}'.format(eval_opt['split']),
                   'rgb-{:s}.avi.mat'.format(x)) for x in train_trial_list ]


    test_list = [ os.path.join(feature_path,feature_path,'Split_{:d}'.format(eval_opt['split']),
                   'rgb-{:s}.avi.mat'.format(x)) for x in test_trial_list ]





    model_opt = {'regularizer':tf.contrib.layers.l2_regularizer(scale=1e-6),
             'time_conv_size':time_conv_size,
             'output_dims':n_classes,
             'dropout_keep_prob':0.5,
             'is_training':False,
             'scope':'conv_temp_ed',
             'reuse':tf.AUTO_REUSE
             }




    print('[EVAL INFO] read the testing data')


    feature_path = os.path.join(dataset_path, 'features/SpatialCNN_mid')
    feature_list = []
    label_list = []
    video_idx_list = []

    yt_list = []
    prd_list = []





    for trial in test_list:
        print('--processing:' + trial )
        spatial_feature = sio.loadmat(trial)
        feature = spatial_feature['A']
        label = np.squeeze(spatial_feature['Y'])
        
              
        frames_batch = np.expand_dims(feature, axis=0)


        ## #labels = #frames-2, since our network discard the first two frames
        print('frames_batch_shape = ' + str(frames_batch.shape) )
        print('label_batch_shape = ' + str(label.shape) )


        print('[EVAL INFO] --setup the inference graph, based on the video length')
        tf.reset_default_graph()


        net_input = tf.placeholder(tf.float32, shape=(1, 
                                             frames_batch.shape[1],
                                             128)) # time, width, heigth are flexible

        net= end_to_end_tensor_flow(net_input, model_opt ) # output should be [batch,time,n_classes]
        print(net.shape)
        n_frames_net = np.minimum(net.shape[1], label.shape[0])



        variables_to_restore =  tf.contrib.framework.get_variables_to_restore()
        init_fn_model = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            
            print('[EVAL INFO] --perform inference')
            sess.run(init_op)
            init_fn_model(sess)
            feed_dict = { net_input:frames_batch}
            net_output = sess.run(net, feed_dict=feed_dict)
            net_output = np.squeeze(net_output)
            prd_list.append(net_output[:n_frames_net])
            yt_list.append(label[:n_frames_net])
            # print(label)
            # print(net_output)
            # print('[EVAL INFO][ONE VIDEO],acc={:f}'.format(accuracy_score(label[:n_frames_net], 
            #                                     net_output[:n_frames_net])))

            sess.close()


    # trial_metrics.add_predictions(splits, prd_list, yt_list)       
    # trial_metrics.print_trials()
    # print()





