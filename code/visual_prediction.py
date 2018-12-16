import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import tf_models, datasets, utils, metrics
from utils import imshow_



## load path config
dataset = ["50Salads", "JIGSAWS", "MERL", "GTEA"][1]
split = 'Split_2'
features = "SpatialCNN"
base_dir = os.path.expanduser("/is/ps2/yzhang/workspaces/TemporalActionParsing-FineGrained/")
data = datasets.Dataset(dataset, base_dir)
X_train, y_train, X_test, y_test = data.load_split(features, split=split, 
                                                    sample_rate=3, 
                                                    feature_type='A')


## load results
# result_path = os.path.join(base_dir, 'predictions/{}'.format(dataset),'mid_video_ED-Bilinear_max_NeighborSize_5_lowdim_True/{}.mat'.format(split))
# result_path = os.path.join(base_dir, 'predictions/{}'.format(dataset),'mid_video_ED-Bilinear_dbilinear_lowdim_False/{}.mat'.format(split))
# result_path = os.path.join(base_dir, 'predictions/{}'.format(dataset),'mid_video_ED-Bilinear_dbilinear_NeighborSize_5_lowdim_False/{}.mat'.format(split))
# result_path = os.path.join(base_dir, 'predictions/{}'.format(dataset),'mid_video_ED-Bilinear_compact_NeighborSize_5_lowdim_True/{}.mat'.format(split))
# result_path = os.path.join(base_dir, 'predictions/{}'.format(dataset),'mid_video_ED-Bilinear_dbilinear_linear_proj_NeighborSize_5_lowdim_True/{}.mat'.format(split))



result_path = os.path.join(base_dir, 'predictions/{}'.format(dataset),'eval_video_ED-Bilinear_compact_NeighborSize_11_lowdim_True/{}.mat'.format(split))



res = sio.loadmat(result_path)



P_test = res['P'][0]
y_test = res['Y'][0]





max_classes = data.n_classes - 1
n_classes = data.n_classes
n_test = len(X_test)


print('- dataset: {}'.format(dataset))
print('- max_classes: {}'.format(max_classes))


# # Output all truth/prediction pairs
plt.figure(split, figsize=(20,10))
P_test_ = np.array(P_test)/float(n_classes-1)
y_test_ = np.array(y_test)/float(n_classes-1)





for i in range(len(y_test)):
    P_tmp = np.vstack([y_test_[i], P_test_[i]])
    
    print(P_tmp.shape)

    plt.subplot(n_test,1,i+1)
    imshow_(P_tmp, vmin=0.0, vmax=1.0)
    plt.xticks([])
    plt.yticks([])
    acc = np.mean(y_test[i]==P_test[i])*100
    plt.ylabel("{:.01f}".format(acc))
    # plt.title("Acc: {:.03}%".format(100*np.mean(P_test[i]==y_test[i]))



plt.show()