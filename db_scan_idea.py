import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import tqdm
import time
import os
import wandb
import torch.backends.cudnn as cudnn

from external_utils import format_time
from utils.data_preprocessing import preprocess_dataset_get_dataset, generate_dataset_from_loader
from structure.dlgn_conv_config_structure import DatasetConfig
import numpy as np

from sklearn import metrics
from sklearn.cluster import DBSCAN

if __name__ == '__main__':
    datasetname = 'mnist'
    save_folder = "data/custom_datasets/db_scan_based/"
    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
    data_config = DatasetConfig(
                datasetname, is_normalize_data=True, valid_split_size=0.1, batch_size=128, list_of_classes=None,custom_dataset_path=None)

    filtered_X_train, filtered_y_train, X_valid, y_valid, filtered_X_test, filtered_y_test = preprocess_dataset_get_dataset(
            data_config, "dlgn", verbose=1, dataset_folder="./Datasets/", is_split_validation=False)
    filtered_X_train = np.array(filtered_X_train).reshape(60000,784)
    extended_x_train = []
    extended_y_train = []

    eps = 0.4
    metric = 'chebyshev'
    for min_samples in range(2,100):
        print("Running for min_samples:{} ".format(min_samples))

        dbscanresults_folder = "root/dbs_results/dataset_{}/eps_{}/metric_{}/min_samples_{}/".format(datasetname,eps,metric,min_samples)
        if not os.path.exists(dbscanresults_folder):
            os.makedirs(dbscanresults_folder)
        db = DBSCAN(eps=eps, min_samples=min_samples,metric=metric,n_jobs=-1,algorithm='brute').fit(filtered_X_train)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print("For min_samples:{} -> n_clusters_:{} and number of core points:{}".format(min_samples,n_clusters_,db.core_sample_indices_.shape))
        # Once we get zero clusters more min_samples also get zero clusters
        if(n_clusters_ == 0):
            break
        with open(dbscanresults_folder+"core_sample_indices_{}.npy".format(db.core_sample_indices_.shape), 'wb') as file:
            np.savez(file, core_sample_indices=db.core_sample_indices_)

    # with open(save_folder+str(datasetname)+"__"+str(modestr)+".npy", 'wb') as file:
    #     np.savez(file, X_train=mod_X_train,y_train=mod_y_train,X_test=X_test,y_test=y_test)
    print("Finished!!")
