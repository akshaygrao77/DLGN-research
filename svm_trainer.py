import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from utils.data_preprocessing import preprocess_dataset_get_dataset, generate_dataset_from_loader,get_data_loader
from structure.dlgn_conv_config_structure import DatasetConfig
from sklearn import datasets, metrics, svm
from tqdm import trange
import pickle
from conv4_models import get_model_instance_from_dataset, get_img_size
from structure.generic_structure import SaveFeatures
import tqdm
import numpy as np

def get_margin_folder(model_path):
    return model_path.replace(".pt","/MARGIN_ANALYSIS/")

def generate_preacts_all_examples(dataset,model_arch_type,mpath,trainloader,fc_width=128,fc_depth = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes_in_each_layer_list = [fc_width] * fc_depth
    model = get_model_instance_from_dataset(dataset,model_arch_type, seed=2022, num_classes=10, nodes_in_each_layer_list=nodes_in_each_layer_list)
    model.load_state_dict(torch.load(mpath).state_dict())
    model = model.to(device)
    model.eval()
    
    outcapturer = OrderedDict()
    dummy_input = torch.rand(get_img_size(dataset)).unsqueeze(0)
    dummy_input = dummy_input.to(device)
    
    conv_matrix_operations_in_each_layer, conv_bias_operations_in_each_layer, channel_outs_size_in_each_layer = model.exact_forward_vis(dummy_input)

    for key,cur_m in model.get_gate_layers_ordered_dict().items():
        if isinstance(cur_m, nn.Linear):
            print(key,cur_m.weight.size(),cur_m.bias.size())
            outcapturer[key] = SaveFeatures(cur_m)

    overall_dist_hp = None
    loader = tqdm.tqdm(trainloader, desc='Generating Distance from HP')
    for batch_idx, data in enumerate(loader, 0):
        (X, y) = data
        X, y = X.cuda(), y.cuda()
        model(X)

        cur_batch_margins = None
        for key,cur_m in model.get_gate_layers_ordered_dict().items():
            cur_eff_w_norm = torch.norm(conv_matrix_operations_in_each_layer[key],p=2,dim=1).unsqueeze(0)
            cur_margin = outcapturer[key].features/cur_eff_w_norm
            if(cur_batch_margins is None):
                cur_batch_margins = cur_margin.T
            else:
                cur_batch_margins = torch.vstack([cur_batch_margins,cur_margin.T])
        
        if(overall_dist_hp is None):
            overall_dist_hp = cur_batch_margins.detach().cpu()
        else:
            overall_dist_hp = torch.hstack([overall_dist_hp.detach().cpu(),cur_batch_margins.detach().cpu()])
    print("overall_distance_from_HP ",overall_dist_hp.size())
    
    return overall_dist_hp.numpy()

def train_and_save_svms(y_true_sf_gates,save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(save_folder+"/all_svm_models.pckl", "wb") as f:
        with trange(y_true_sf_gates.shape[0], unit="Indx", desc="Running SVM for gate:") as pbar:
            for i in pbar:
                c1_count = np.sum(y_true_sf_gates[i])
                if(c1_count == len(y_true_sf_gates[i])):
                    y_true_sf_gates[i][0] = 0
                elif(c1_count == 0):
                    y_true_sf_gates[i][0] = 1
                clf = svm.SVC(kernel='linear')
                clf.fit(filtered_X_train, y_true_sf_gates[i])
                pickle.dump(clf, f)


if __name__ == '__main__':
    dataset = 'mnist'
    model_arch_type = 'fc_sf_dlgn'
    data_config = DatasetConfig(
                dataset, is_normalize_data=True, valid_split_size=0.1, batch_size=128, list_of_classes=None,custom_dataset_path=None)
    filtered_X_train, filtered_y_train, X_valid, _, filtered_X_test, _ = preprocess_dataset_get_dataset(
                data_config, model_arch_type, verbose=0, dataset_folder="./Datasets/", is_split_validation=False)
    trainloader = get_data_loader(
            filtered_X_train, filtered_y_train, data_config.batch_size, transforms=data_config.train_transforms)
    filtered_X_train = np.reshape(filtered_X_train,(60000,784))

    model_path = "root/model/save/mnist/adversarial_training/MT_fc_sf_dlgn_W_128_D_4_ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_PGD/adv_type_PGD/EPS_0.3/batch_size_64/eps_stp_size_0.01/adv_steps_40/update_on_all/R_init_True/norm_inf/use_ytrue_True/adv_model_dir.pt"

    save_folder = get_margin_folder(model_path)

    model_preact = generate_preacts_all_examples(dataset,model_arch_type,model_path,trainloader)
    y_true_sf_gates = np.where(model_preact>0,1,0)

    train_and_save_svms(y_true_sf_gates,save_folder)

    print("Finished exec!!")

