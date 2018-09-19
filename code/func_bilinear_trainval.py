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





from temporal_ed_bilinear import end_to_end_tensor_flow
import tf_models, datasets, utils, metrics



def train(dataset_name, x_train, y_train, n_classes, time_conv_size, n_nodes, activation, seq_length, splits):






    ## ----------------------- Overall config -----------------------

    train_log_dir = '/home/yzhang/workspace/tensorflow_proj/TemporalConvolutionalNetworks/logs-{:s}/TimeConvSize{:d}-Length{:d}-Split{:d}'.format(dataset_name,time_conv_size,seq_length,splits)
    if not tf.gfile.Exists(train_log_dir):
        tf.gfile.MakeDirs(train_log_dir)




    train_opt = {'start_lr':0.001,
             'n_epochs':150,
             'lr_decay_per_X_epochs':5, 
             'batch_size':8,
             'time_length':seq_length, 
             'lr_decay':0.9,
             'split':splits,
             'save_model_after_n_epochs':50,
             'gradient_clip_norm':None
            }


    model_opt = {'regularizer':tf.contrib.layers.l2_regularizer(scale=1e-6),
             'time_conv_size':time_conv_size,
             'n_nodes':n_nodes,
             'activation':activation,
             'output_dims':n_classes,
             'dropout_keep_prob':0.5,
             'is_training':True,
             'scope':'conv_temp_ed',
             'reuse':tf.AUTO_REUSE
             }






    ## ----------------------- initialize the inference graph -----------------------
    print('[TRAIN INFO] setup the inference graph')
    tf.reset_default_graph()
    net_input = tf.placeholder(tf.float32, shape=(train_opt['batch_size'], 
                                             train_opt['time_length'], 
                                             128))
    net_labels = tf.placeholder(tf.int32, shape=(train_opt['batch_size'],train_opt['time_length'],
                                                model_opt['output_dims']))

    # this is actually the labels. We detect boundaries in the network function 

    net= end_to_end_tensor_flow(net_input, model_opt ) # output should be [batch,time,n_classes]

    print('[TRAIN INFO][INFERENCE GRAPH] --net input image =' + str(net_input))
    print('[TRAIN INFO][INFERENCE GRAPH] --net input labels =' + str(net_labels))
    print('[TRAIN INFO][INFERENCE GRAPH] --net output logits =' + str(net))





    ## specify the loss
    print('[TRAIN INFO] setup the loss')

    net_labels1 = tf.reshape(net_labels, [net_labels.shape[0]*net_labels.shape[1], -1])
    net = tf.reshape(net, [net.shape[0]*net.shape[1], -1 ])


    loss = tf.losses.softmax_cross_entropy(net_labels1, net)

    print('[TRAIN INFO] setup the training ops')

    ## specify the optimizer and the training OP
    tf.summary.scalar('loss',loss)
    summary = tf.summary.merge_all()


    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
    #     start_lr = train_opt['start_lr']
    #     lr = tf.train.exponential_decay(start_lr, global_step, n_frames_per_epoch,
    #                                     train_opt['exponential_decay_rate'], staircase=True)
    train_eps = tf.placeholder(tf.float32, shape=[],name='ep')
    lr = train_opt['start_lr'] * pow(train_opt['lr_decay'],train_eps)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    if train_opt['gradient_clip_norm'] != None  :
        print('[TRAIN INFO] apply gradient clip, norm='+str(train_opt['gradient_clip_norm']))
        gradients = optimizer.compute_gradients(loss)

        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_norm(grad, train_opt['gradient_clip_norm'])

        clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gradients]
        print(clipped_gradients)
        train_op = optimizer.apply_gradients(clipped_gradients, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)





    ## ----------------------- specify the input pipeline -----------------------
    print('[TRAIN INFO] read the training data')


    feature_list = []
    label_list = []
    video_idx_list = []




    
    x_train_index = [x for x in range(len(x_train))]
    com = list(zip(x_train, x_train_index, y_train))

    for _ in range(train_opt['n_epochs']): 
        
        random.shuffle(com)

        for xs, idx, ys in com:
            
            feature_list.append(xs)
            label_list.append(ys)
            
            video_idx = idx * np.ones(ys.shape[0])
            video_idx_list.append(video_idx)


    feature = np.concatenate(feature_list)
    label = np.concatenate(label_list)
    video_idx = np.concatenate(video_idx_list)    

    n_frames_per_epoch = 1.0*feature.shape[0]/train_opt['n_epochs']
    n_iters_per_epoch = 1.0*n_frames_per_epoch/(train_opt['time_length']*train_opt['batch_size'])



    print('[TRAIN INFO][DATA PIPELINE] --input feature =' + str(feature.shape))
    print('[TRAIN INFO][DATA PIPELINE] --input labels =' + str(label.shape))





    ## ----------------------- initialize training process and train-----------------------
    print('[TRAIN INFO] initialize training process')

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


    saver = tf.train.Saver()
    checkpoint_file = os.path.join(train_log_dir, 'model-split{:d}.ckpt'.format(train_opt['split']))


    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:

        ## initialization training loop
        summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)    
        sess.run(init_op)
        # init_fn_model(sess)
        feature_iterator = 0
        model_saved = 0
        gl_steps_e = 0
        print ('[TRAIN INFO] --start training')
        while True:

            try:
                ## prepare feed-in data
                features_batch_list = []
                labels_batch_list = []
                batch_count = 0
                while batch_count < train_opt['batch_size']:
                    ts = feature_iterator
                    te = feature_iterator+train_opt['time_length']

                    if te > feature.shape[0]:
                        print ('[TRAIN INFO] --all training samples are used up. Training terminates.')
                        saver.save(sess,checkpoint_file,global_step=gl_steps_e)
                        
                        return

                    video_idx_seq = video_idx[ts:te]
                    if np.unique(video_idx_seq).shape[0] == 1:
                        features_batch_list.append(feature[ts:te,:])
                        labels_batch_list.append(label[ts:te,:])
                        batch_count+=1
                    feature_iterator=te


                features_batch = np.stack(features_batch_list, axis=0)
                labels_batch = np.stack(labels_batch_list, axis=0)


                ## prepare learning rate 
                gl_steps_e = tf.train.global_step(sess, global_step)
                ep = gl_steps_e // n_iters_per_epoch

                
                ## one step training
                feed_dict ={net_input:features_batch,
                           net_labels:labels_batch,
                           train_eps:ep/train_opt['lr_decay_per_X_epochs']}
                _, loss_val = sess.run([train_op,loss], 
                                       feed_dict=feed_dict
                        )


                ## loging data 
                if gl_steps_e % 10 ==0:
                
                    print('[TRAIN INFO][STATISTICS] --step=%d, epoch=%d, lr=%f, loss=%f' % (gl_steps_e, ep, train_opt['start_lr'] * pow(train_opt['lr_decay'],np.floor(ep/train_opt['lr_decay_per_X_epochs'])),loss_val))
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, gl_steps_e)
                    summary_writer.flush()

                if ep % train_opt['save_model_after_n_epochs']==0 and ep > 0 and model_saved == 0:
                    print ('[TRAINING INFO] --save the model at epoch '+str(ep))
                    saver.save(sess,checkpoint_file,global_step=gl_steps_e)
                    model_saved = 1

                if ep % train_opt['save_model_after_n_epochs']!=0:
                    model_saved = 0
            
            except KeyboardInterrupt:
                print('[TRAIN INFO] -- process is terminated manually.')
                print('[TRAIN INFO][STATISTICS] --step=%d, epoch=%d, loss=%f' % (gl_steps_e, ep,loss_val))
                print('[TRAIN INFO] -- save model and close input data threads')
                saver.save(sess,checkpoint_file,global_step=gl_steps_e)
                sys.exit()
                break





    
    
    







def test(dataset_name, x_test, y_test, n_classes, time_conv_size, n_nodes, activation,seq_length, splits):

    from sklearn.metrics import accuracy_score


    bg_class = 0 if dataset_name is not "JIGSAWS" else None
    trial_metrics = metrics.ComputeMetrics(overlap=.1, bg_class=bg_class)



    ## ----------------------- Overall config -----------------------
    eval_opt = { 'split':splits,
                 'train_log_dir':'/home/yzhang/workspace/tensorflow_proj/TemporalConvolutionalNetworks/logs-{:s}/TimeConvSize{:d}-Length{:d}-Split{:d}'.format(dataset_name,time_conv_size,seq_length,splits),
                 'dataset_path':'/mnt/hdd/Dataset_50Salads/'
                }



    model_path_list = sorted(glob.glob(os.path.join(eval_opt['train_log_dir'],'model-*.meta')),
                             key=os.path.getmtime)
    model_path = model_path_list[-1][:-5]
    print('[EVAL INFO] - model:' + model_path)



    # model_path = eval_opt['train_log_dir'] + '/model-split{:d}.ckpt-9065'.format(splits)


   
    if trial_metrics.n_classes is None:
        trial_metrics.set_classes(n_classes)



   

    model_opt = {'regularizer':tf.contrib.layers.l2_regularizer(scale=1e-6),
             'time_conv_size':time_conv_size,
             'n_nodes':n_nodes,
             'activation':activation,
             'output_dims':n_classes,
             'dropout_keep_prob':0.5,
             'is_training':False,
             'scope':'conv_temp_ed',
             'reuse':tf.AUTO_REUSE
             }




    print('[EVAL INFO] read the testing data')


   
    feature_list = []
    label_list = []
    video_idx_list = []

    yt_list = []
    prd_list = []


    print(len(x_test))


    for xt,yt in zip(x_test, y_test):
        
        
        feature = xt
        label = yt
        
              
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
            
            sess.close()

    return prd_list, yt_list
