import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import tqdm
import matplotlib.pyplot as plt
import os
import torch.backends.cudnn as cudnn
from utils.data_preprocessing import preprocess_dataset_get_data_loader, generate_dataset_from_loader
from structure.dlgn_conv_config_structure import DatasetConfig
from collections import OrderedDict

from visualization import recreate_image, save_image,  PerClassDataset
from utils.data_preprocessing import true_segregation
from structure.generic_structure import CustomSimpleDataset
from adversarial_attacks_tester import generate_adv_examples,get_adv_save_str

from keras.datasets import mnist, fashion_mnist

from conv4_models import get_model_instance, get_model_instance_from_dataset

def get_model_from_path(dataset, model_arch_type, model_path, mask_percentage=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp_model = torch.load(model_path, map_location=device)
    custom_model = get_model_instance_from_dataset(
        dataset, model_arch_type)
    if("masked" in model_arch_type):
        custom_model = get_model_instance_from_dataset(
            dataset, model_arch_type, mask_percentage=mask_percentage)
    if(isinstance(temp_model, dict)):
        if("module." in [*temp_model['state_dict'].keys()][0]):
            new_state_dict = OrderedDict()
            for k, v in temp_model['state_dict'].items():
                name = k[7:]  # remove 'module.' of dataparallel
                new_state_dict[name] = v
            custom_model.load_state_dict(new_state_dict)
        else:
            custom_model.load_state_dict(temp_model['state_dict'])
    else:
        custom_model.load_state_dict(temp_model.state_dict())

    return custom_model

def get_convouts(model,out_capturer):
    if(out_capturer is None):
        if(isinstance(model, torch.nn.DataParallel)):
            conv_outs = model.module.linear_conv_outputs
        else:
            conv_outs = model.linear_conv_outputs
    else:
        conv_outs = []
        for ev in out_capturer:
            conv_outs.append(ev.features)

    
    return conv_outs

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()

def generate_difference_in_filter_activity(dataloader1,dataloader2,model):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    is_use_capturer = False
    out_capturer = None
    model(torch.randn((1,1,28,28)).to(device))
    
    if(isinstance(model, torch.nn.DataParallel)):
            if(not hasattr(model.module,"linear_conv_outputs")):
                is_use_capturer = True
    else:
        if(not hasattr(model,"linear_conv_outputs")):
            is_use_capturer = True
    if(is_use_capturer):
        out_capturer = []
        for val in model.get_gate_layers_ordered_dict():
            out_capturer = SaveFeatures(val)
    
    sig = nn.Sigmoid()
    
    all_filter_acts = []
    iter_dataloader2 = enumerate(dataloader2)
    loader = tqdm.tqdm(dataloader1, desc='Generating difference activity per channel')
    for ind, data in enumerate(loader, 0):
        inputs, _ = data
        inputs = inputs.to(
            device)
        _,(inputs2,_) = next(iter_dataloader2)
        inputs2 = inputs2.to(
            device)
        assert inputs2.size()==inputs.size(), "Not same sizes:{} vs {} at ind:{}".format(inputs.size(),inputs2.size(),ind)

        outputs = model(inputs)
        
        conv_outs = get_convouts(model,out_capturer)
        with torch.no_grad():
            conv_outs = [sig(i) for i in conv_outs]
        
        outputs = model(inputs2)
        
        conv_outs2 = get_convouts(model,out_capturer)
        with torch.no_grad():
            conv_outs = [abs(sig(conv_outs2[i])-conv_outs[i]) for i in range(len(conv_outs2))]
            conv_outs = torch.cat([torch.mean(i,(0,2,3)) for i in conv_outs])
            all_filter_acts.append(conv_outs)
    
    all_filter_acts = torch.stack(all_filter_acts)
    all_filter_acts = torch.mean(all_filter_acts,dim=(0))

    return all_filter_acts
        

def generate_filter_activity(dataloader,model):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    is_use_capturer = False
    out_capturer = None
    model(torch.randn((1,1,28,28)).to(device))
    
    if(isinstance(model, torch.nn.DataParallel)):
            if(not hasattr(model.module,"linear_conv_outputs")):
                is_use_capturer = True
    else:
        if(not hasattr(model,"linear_conv_outputs")):
            is_use_capturer = True
    if(is_use_capturer):
        out_capturer = []
        for val in model.get_gate_layers_ordered_dict():
            out_capturer = SaveFeatures(val)
    
    sig = nn.Sigmoid()
    all_filter_acts = []
    all_convouts_grad = []
    all_corr_filter_acts = []
    all_corr_convouts_grad = []
    all_incorr_filter_acts = []
    all_incorr_convouts_grad = []
    jac_filter = []
    jac_corr_filter = []
    jac_incorr_filter = []
    loader = tqdm.tqdm(dataloader, desc='Generating per channel output information')
    for _, data in enumerate(loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(
            device), labels.to(device)
        inputs.requires_grad_()

        # forward + backward + optimize
        outputs = model(inputs)
        
        conv_outs = get_convouts(model,out_capturer)
        for x in conv_outs: x.retain_grad()
        
        loss = criterion(outputs, labels)
        loss.backward()
        _, predicted = torch.max(outputs.data, 1)
        
        cout_grad = [x.grad for x in conv_outs]

        with torch.no_grad():
            # fft_cout_grad = torch.fft.fft2(cout_grad)
            # fft_cout_grad = torch.abs(torch.fft.fftshift(fft_cout_grad))
            # jac_filter.append(fft_cout_grad)
            conv_outs = [sig(i) for i in conv_outs]
            conv_outs = torch.cat([torch.mean(i,(0,2,3)) for i in conv_outs])
            cout_grad = torch.cat([torch.mean(i,(0,2,3)) for i in cout_grad])
            all_filter_acts.append(conv_outs)
            all_convouts_grad.append(cout_grad)
            correct_indices = torch.squeeze((predicted==labels).nonzero(),1)
            corr_inputs = torch.index_select(inputs, dim=0, index=correct_indices)
        if(corr_inputs.size()[0] != 0):
            corr_labels = torch.index_select(labels, dim=0, index=correct_indices)
            corr_inputs.requires_grad_()
            
            outputs = model(corr_inputs)
            
            conv_outs = get_convouts(model,out_capturer)
            for x in conv_outs: x.retain_grad()
            
            loss = criterion(outputs, corr_labels)
            loss.backward()
            
            cout_grad = [x.grad for x in conv_outs]
            with torch.no_grad():
                # fft_cout_grad = torch.fft.fft2(cout_grad)
                # fft_cout_grad = torch.abs(torch.fft.fftshift(fft_cout_grad))
                # jac_corr_filter.append(fft_cout_grad)
                conv_outs = [sig(i) for i in conv_outs]
                conv_outs = torch.cat([torch.mean(i,(0,2,3)) for i in conv_outs])
                cout_grad = torch.cat([torch.mean(i,(0,2,3)) for i in cout_grad])
                all_corr_filter_acts.append(conv_outs)
                all_corr_convouts_grad.append(cout_grad)
            

        with torch.no_grad():
            incorrect_indices = torch.squeeze((predicted != labels).nonzero(),1)
            incorr_inputs = torch.index_select(inputs, dim=0, index=incorrect_indices)
        if(incorr_inputs.size()[0] != 0):
            incorr_labels = torch.index_select(labels, dim=0, index=incorrect_indices)
            incorr_inputs.requires_grad_()
            
            outputs = model(incorr_inputs)
            
            conv_outs = get_convouts(model,out_capturer)
            for x in conv_outs: x.retain_grad()

            loss = criterion(outputs, incorr_labels)
            loss.backward()
            
            cout_grad = [x.grad for x in conv_outs]
            with torch.no_grad():
                # fft_cout_grad = torch.fft.fft2(cout_grad)
                # fft_cout_grad = torch.abs(torch.fft.fftshift(fft_cout_grad))
                # jac_incorr_filter.append(fft_cout_grad)
                conv_outs = [sig(i) for i in conv_outs]
                conv_outs = torch.cat([torch.mean(i,(0,2,3)) for i in conv_outs])
                cout_grad = torch.cat([torch.mean(i,(0,2,3)) for i in cout_grad])
                all_incorr_filter_acts.append(conv_outs)
                all_incorr_convouts_grad.append(cout_grad)
    
    all_filter_acts = torch.stack(all_filter_acts)
    all_filter_acts = torch.mean(all_filter_acts,dim=(0))
    all_convouts_grad = torch.stack(all_convouts_grad)
    all_convouts_grad = torch.mean(all_convouts_grad,dim=(0))

    if(len(all_corr_filter_acts)>0):
        all_corr_filter_acts = torch.stack(all_corr_filter_acts)
        all_corr_filter_acts = torch.mean(all_corr_filter_acts,dim=(0))
    else:
        all_corr_filter_acts = torch.Tensor([0])
    if(len(all_corr_convouts_grad)>0):
        all_corr_convouts_grad = torch.stack(all_corr_convouts_grad)
        all_corr_convouts_grad = torch.mean(all_corr_convouts_grad,dim=(0))
    else:
        all_corr_convouts_grad = torch.Tensor([0])

    all_incorr_filter_acts = torch.stack(all_incorr_filter_acts)
    all_incorr_filter_acts = torch.mean(all_incorr_filter_acts,dim=(0))
    all_incorr_convouts_grad = torch.stack(all_incorr_convouts_grad)
    all_incorr_convouts_grad = torch.mean(all_incorr_convouts_grad,dim=(0))

    # jac_filter = torch.cat(jac_filter,dim=0)
    # jac_filter = torch.mean(jac_filter,dim=(0,1))

    # jac_corr_filter = torch.cat(jac_corr_filter,dim=0)
    # jac_corr_filter = torch.mean(jac_corr_filter,dim=(0,1))

    # jac_incorr_filter = torch.cat(jac_incorr_filter,dim=0)
    # jac_incorr_filter = torch.mean(jac_incorr_filter,dim=(0,1))

    return all_filter_acts,all_convouts_grad,all_corr_filter_acts,all_corr_convouts_grad,all_incorr_filter_acts,all_incorr_convouts_grad,jac_filter,jac_corr_filter,jac_incorr_filter

def output_plt_image(images,path,title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticks([])
    ax.set_yticks([])
    cax = ax.imshow(images.detach().cpu().numpy(), interpolation='nearest')
    ax.set_title(title)
    cbar = fig.colorbar(cax)
    fig.tight_layout()
    fig.savefig(path,bbox_inches='tight')

def output_bar_graph(xlist,ylist,path,title=""):
    col = ['blue'if i%2==0 else 'red' for i in range(len(xlist))]
    fig = plt.figure(figsize =(len(xlist)//10, 7))
    ax = fig.add_subplot(111)
    xticks = [0]
    while(xticks[-1]<len(xlist)):
        xticks.append(xticks[-1]+10)
    ax.set_xticks(xticks)
    ax.bar(xlist,ylist,color=col,align='edge')
    ax.set_title(title)
    # fig.tight_layout()
    fig.savefig(path,bbox_inches='tight')

if __name__ == '__main__':
    # fashion_mnist , mnist, cifar10
    dataset = 'mnist'
    # conv4_dlgn , plain_pure_conv4_dnn , conv4_dlgn_n16_small , plain_pure_conv4_dnn_n16_small , conv4_deep_gated_net , conv4_deep_gated_net_n16_small
    # fc_dnn , fc_dlgn , fc_dgn , dlgn__conv4_dlgn_pad_k_1_st1_bn_wo_bias__
    model_arch_type = 'dlgn__conv4_dlgn_pad_k_1_st1_bn_wo_bias__'
    batch_size = 64

    is_analysis_on_train = False

    torch_seed = 2022

    # None means that train on all classes
    list_of_classes_to_train_on = None
    # list_of_classes_to_train_on = [4, 9]

    # Percentage of information retention during PCA (values between 0-1)
    pca_exp_percent = None
    # pca_exp_percent = 0.85

    is_analyse_adv = True

    wandb_config_additional_dict = None
    # wandb_config_additional_dict = {"type_of_APR": "APRS"}

    direct_model_path = "root/model/save/mnist/CLEAN_TRAINING/ST_2022/dlgn__conv4_dlgn_pad_k_1_st1_bn_wo_bias___dir.pt"

    custom_dataset_path = None
    # custom_dataset_path = "data/custom_datasets/freq_band_dataset/mnist__MB_HB.npy"

    if(dataset == "cifar10"):
        inp_channel = 3
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = len(classes)

        cifar10_config = DatasetConfig(
            'cifar10', is_normalize_data=False, valid_split_size=0.1, batch_size=batch_size, 
            list_of_classes=list_of_classes_to_train_on,custom_dataset_path=custom_dataset_path)

        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            cifar10_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

    elif(dataset == "mnist"):
        inp_channel = 1
        classes = [str(i) for i in range(0, 10)]
        num_classes = len(classes)

        ds_config = DatasetConfig(
            'mnist', is_normalize_data=True, valid_split_size=0.1, batch_size=batch_size, 
            list_of_classes=list_of_classes_to_train_on,custom_dataset_path=custom_dataset_path)

        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            ds_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)

    elif(dataset == "fashion_mnist"):
        inp_channel = 1
        classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle-boot')
        num_classes = len(classes)

        ds_config = DatasetConfig(
            'fashion_mnist', is_normalize_data=True, valid_split_size=0.1, batch_size=batch_size, 
            list_of_classes=list_of_classes_to_train_on,custom_dataset_path=custom_dataset_path)

        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            ds_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=False)
    
    if(custom_dataset_path is not None):
        dataset = custom_dataset_path[custom_dataset_path.rfind("/")+1:custom_dataset_path.rfind(".npy")]
    
    print("Testing over "+dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes_trained_on = num_classes
    dataset_str = dataset

    list_of_classes_to_train_on_str = ""
    if(list_of_classes_to_train_on is not None):
        for each_class_to_train_on in list_of_classes_to_train_on:
            list_of_classes_to_train_on_str += \
                str(each_class_to_train_on)+"_"
        dataset_str += "_"+str(list_of_classes_to_train_on_str)
        list_of_classes_to_train_on_str = "TR_ON_" + \
            list_of_classes_to_train_on_str[0:-1]
        num_classes_trained_on = len(list_of_classes_to_train_on)
        temp_classes = []
        for ea_c in list_of_classes_to_train_on:
            temp_classes.append(classes[ea_c])
        classes = temp_classes

    model_arch_type_str = model_arch_type
    if("masked" in model_arch_type):
        mask_percentage = 90
        model_arch_type_str = model_arch_type_str + \
            "_PRC_"+str(mask_percentage)
        net = get_model_instance(
            model_arch_type, inp_channel, mask_percentage=mask_percentage, seed=torch_seed, num_classes=num_classes_trained_on)
    elif("fc" in model_arch_type):
        fc_width = 128
        fc_depth = 4
        nodes_in_each_layer_list = [fc_width] * fc_depth
        model_arch_type_str = model_arch_type_str + \
            "_W_"+str(fc_width)+"_D_"+str(fc_depth)
        net = get_model_instance_from_dataset(dataset,
                                              model_arch_type, seed=torch_seed, num_classes=num_classes_trained_on, nodes_in_each_layer_list=nodes_in_each_layer_list)
    else:
        net = get_model_instance(model_arch_type, inp_channel,
                                 seed=torch_seed, num_classes=num_classes_trained_on)

    if(pca_exp_percent is not None):
        dataset_for_pca = generate_dataset_from_loader(trainloader)
        if(isinstance(dataset_for_pca.list_of_x[0], torch.Tensor)):
            dataset_for_pca = torch.stack(
                dataset_for_pca.list_of_x), torch.stack(dataset_for_pca.list_of_y)
        else:
            dataset_for_pca = np.array(dataset_for_pca.list_of_x), np.array(
                dataset_for_pca.list_of_y)
        number_of_components_for_pca = net.initialize_PCA_transformation(
            dataset_for_pca[0], pca_exp_percent)
        model_arch_type_str = model_arch_type_str + \
            "_PCA_K"+str(number_of_components_for_pca) + \
            "_P_"+str(pca_exp_percent)

    if('CLEAN' in direct_model_path or 'APR_TRAINING' in direct_model_path or 'adv_model_dir_epoch' in direct_model_path):
        data_save_prefix = direct_model_path[0:direct_model_path.rfind(
            ".pt")]
    else:
        data_save_prefix = direct_model_path[0:direct_model_path.rfind(
            "/")+1]
    
    isExist = os.path.exists(direct_model_path)
    assert isExist == True, 'Model path does not have saved model'

    net = get_model_from_path(
        dataset, model_arch_type, direct_model_path)

    net = net.to(device)
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_str == 'cuda':
        cudnn.benchmark = True
    
    if(is_analysis_on_train == True):
        eval_loader = trainloader
    else:
        eval_loader = testloader
    
    class_indx_to_visualize = [i for i in range(len(classes))]
    # class_indx_to_visualize= []

    if(is_analyse_adv):
        number_of_adversarial_optimization_steps = 40
        adv_attack_type = "PGD"
        adv_target = None
        eps_step_size = 0.01
        eps = 0.3
        is_adv_attack_on_train = is_analysis_on_train
        final_adv_postfix_for_save = get_adv_save_str(adv_attack_type,eps,eps_step_size,number_of_adversarial_optimization_steps,is_adv_attack_on_train)
        adv_save_path = data_save_prefix + \
            final_adv_postfix_for_save+"/adv_dataset.npy"
        is_current_adv_aug_available = os.path.exists(
            adv_save_path)
        if(is_current_adv_aug_available):
            with open(adv_save_path, 'rb') as file:
                npzfile = np.load(adv_save_path)
                list_of_adv_images = npzfile['x']
                list_of_labels = npzfile['y']
                adv_dataset = CustomSimpleDataset(
                    list_of_adv_images, list_of_labels)
                print("Loading adversarial examples from path:",
                        adv_save_path)
        else:
            print("adv_save_path:", adv_save_path)
            adv_dataset = generate_adv_examples(
                eval_loader, net, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, adv_target, is_save_adv=True, save_path=adv_save_path)
        
        to_be_analysed_adversarial_dataloader = torch.utils.data.DataLoader(
                        adv_dataset, shuffle=False, batch_size=batch_size)
        true_tobe_analysed_dataset_per_class = true_segregation(
                        to_be_analysed_adversarial_dataloader, num_classes_trained_on)
    
    true_eval_dataset_per_class = true_segregation(
        eval_loader, num_classes_trained_on)
    
    final_data_save_postfix = "/CHANNEL_ACTIVITY_ANALYSIS/ON_TRAIN_{}/dataset_{}/allclasses/".format(
                    is_analysis_on_train,dataset)
    save_folder = data_save_prefix + final_data_save_postfix
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    print("Will be saved in folder:"+save_folder)
    
    all_filter_acts,all_convouts_grad,all_corr_filter_acts,all_corr_convouts_grad,all_incorr_filter_acts,all_incorr_convouts_grad,jac_filter,jac_corr_filter,jac_incorr_filter = generate_filter_activity(eval_loader,net)
    output_bar_graph([i+1 for i in range(all_filter_acts.size()[0])],all_filter_acts.tolist(),save_folder+'/ch_act_ov_org.jpeg',"Mean activity per Chn ov Org")
    output_bar_graph([i+1 for i in range(all_convouts_grad.size()[0])],all_convouts_grad.tolist(),save_folder+'/jacch_ov_org.jpeg',"Mean Jacobian per Chn ov Org")
    output_bar_graph([i+1 for i in range(all_corr_filter_acts.size()[0])],all_corr_filter_acts.tolist(),save_folder+'/ch_act_ov_corr_org.jpeg',"Mean activity per Chn ov Corr Org")
    output_bar_graph([i+1 for i in range(all_corr_convouts_grad.size()[0])],all_corr_convouts_grad.tolist(),save_folder+'/jacch_ov_corr_org.jpeg',"Mean Jacobian per Chn ov Corr Org")
    output_bar_graph([i+1 for i in range(all_incorr_filter_acts.size()[0])],all_incorr_filter_acts.tolist(),save_folder+'/ch_act_ov_incorr_org.jpeg',"Mean activity per Chn ov Incorr Org")
    output_bar_graph([i+1 for i in range(all_incorr_convouts_grad.size()[0])],all_incorr_convouts_grad.tolist(),save_folder+'/jacch_ov_incorr_org.jpeg',"Mean Jacobian per Chn ov Incorr Org")

    if(is_analyse_adv):
        all_filter_acts=generate_difference_in_filter_activity(eval_loader,to_be_analysed_adversarial_dataloader,net)
        output_bar_graph([i+1 for i in range(all_filter_acts.size()[0])],all_filter_acts.tolist(),save_folder+'/diff_ch_act_ov_org.jpeg',"Mean difference activity per Chn ov Adv")
        
        all_filter_acts,all_convouts_grad,all_corr_filter_acts,all_corr_convouts_grad,all_incorr_filter_acts,all_incorr_convouts_grad,jac_filter,jac_corr_filter,jac_incorr_filter = generate_filter_activity(to_be_analysed_adversarial_dataloader,net)
        output_bar_graph([i+1 for i in range(all_filter_acts.size()[0])],all_filter_acts.tolist(),save_folder+'/adv_ch_act_ov_org.jpeg',"Mean activity per Chn ov Adv")
        output_bar_graph([i+1 for i in range(all_convouts_grad.size()[0])],all_convouts_grad.tolist(),save_folder+'/adv_jacch_ov_org.jpeg',"Mean Jacobian per Chn ov Adv")
        output_bar_graph([i+1 for i in range(all_corr_filter_acts.size()[0])],all_corr_filter_acts.tolist(),save_folder+'/adv_ch_act_ov_corr_org.jpeg',"Mean activity per Chn ov Corr Adv")
        output_bar_graph([i+1 for i in range(all_corr_convouts_grad.size()[0])],all_corr_convouts_grad.tolist(),save_folder+'/adv_jacch_ov_corr_org.jpeg',"Mean Jacobian per Chn ov Corr Adv")
        output_bar_graph([i+1 for i in range(all_incorr_filter_acts.size()[0])],all_incorr_filter_acts.tolist(),save_folder+'/adv_ch_act_ov_incorr_org.jpeg',"Mean activity per Chn ov Incorr Adv")
        output_bar_graph([i+1 for i in range(all_incorr_convouts_grad.size()[0])],all_incorr_convouts_grad.tolist(),save_folder+'/adv_jacch_ov_incorr_org.jpeg',"Mean Jacobian per Chn ov Incorr Adv")

    for c_indx in class_indx_to_visualize:
        class_label = classes[c_indx]
        print(
            "************************************************************ Class:", class_label)
        per_class_eval_dataset = PerClassDataset(
            true_eval_dataset_per_class[c_indx], c_indx)
        if(is_analyse_adv):
            per_class_tobe_analysed_dataset = PerClassDataset(
                                    true_tobe_analysed_dataset_per_class[c_indx], c_indx)
            per_class_tobe_analysed_data_loader = torch.utils.data.DataLoader(per_class_tobe_analysed_dataset, batch_size=256,
                                                                        shuffle=False)
        
        final_data_save_postfix = "/CHANNEL_ACTIVITY_ANALYSIS/ON_TRAIN_{}/c_indx_{}_class_label_{}/".format(
                            is_analysis_on_train, c_indx, class_label)

        per_class_eval_data_loader = torch.utils.data.DataLoader(per_class_eval_dataset, batch_size=256,
                                                                    shuffle=False)
        save_folder = data_save_prefix + final_data_save_postfix
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        all_filter_acts,all_convouts_grad,all_corr_filter_acts,all_corr_convouts_grad,all_incorr_filter_acts,all_incorr_convouts_grad,jac_filter,jac_corr_filter,jac_incorr_filter = generate_filter_activity(per_class_eval_data_loader,net)
        output_bar_graph([i+1 for i in range(all_filter_acts.size()[0])],all_filter_acts.tolist(),save_folder+'/ch_act_ov_org.jpeg',"Mean activity per Chn ov Org")
        output_bar_graph([i+1 for i in range(all_convouts_grad.size()[0])],all_convouts_grad.tolist(),save_folder+'/jacch_ov_org.jpeg',"Mean Jacobian per Chn ov Org")
        output_bar_graph([i+1 for i in range(all_corr_filter_acts.size()[0])],all_corr_filter_acts.tolist(),save_folder+'/ch_act_ov_corr_org.jpeg',"Mean activity per Chn ov Corr Org")
        output_bar_graph([i+1 for i in range(all_corr_convouts_grad.size()[0])],all_corr_convouts_grad.tolist(),save_folder+'/jacch_ov_corr_org.jpeg',"Mean Jacobian per Chn ov Corr Org")
        output_bar_graph([i+1 for i in range(all_incorr_filter_acts.size()[0])],all_incorr_filter_acts.tolist(),save_folder+'/ch_act_ov_incorr_org.jpeg',"Mean activity per Chn ov Incorr Org")
        output_bar_graph([i+1 for i in range(all_incorr_convouts_grad.size()[0])],all_incorr_convouts_grad.tolist(),save_folder+'/jacch_ov_incorr_org.jpeg',"Mean Jacobian per Chn ov Incorr Org")

        if(is_analyse_adv):
            all_filter_acts=generate_difference_in_filter_activity(per_class_eval_data_loader,per_class_tobe_analysed_data_loader,net)
            output_bar_graph([i+1 for i in range(all_filter_acts.size()[0])],all_filter_acts.tolist(),save_folder+'/diff_ch_act_ov_org.jpeg',"Mean difference activity per Chn ov Adv")
            
            all_filter_acts,all_convouts_grad,all_corr_filter_acts,all_corr_convouts_grad,all_incorr_filter_acts,all_incorr_convouts_grad,jac_filter,jac_corr_filter,jac_incorr_filter = generate_filter_activity(per_class_tobe_analysed_data_loader,net)
            output_bar_graph([i+1 for i in range(all_filter_acts.size()[0])],all_filter_acts.tolist(),save_folder+'/adv_ch_act_ov_org.jpeg',"Mean activity per Chn ov Adv")
            output_bar_graph([i+1 for i in range(all_convouts_grad.size()[0])],all_convouts_grad.tolist(),save_folder+'/adv_jacch_ov_org.jpeg',"Mean Jacobian per Chn ov Adv")
            output_bar_graph([i+1 for i in range(all_corr_filter_acts.size()[0])],all_corr_filter_acts.tolist(),save_folder+'/adv_ch_act_ov_corr_org.jpeg',"Mean activity per Chn ov Corr Adv")
            output_bar_graph([i+1 for i in range(all_corr_convouts_grad.size()[0])],all_corr_convouts_grad.tolist(),save_folder+'/adv_jacch_ov_corr_org.jpeg',"Mean Jacobian per Chn ov Corr Adv")
            output_bar_graph([i+1 for i in range(all_incorr_filter_acts.size()[0])],all_incorr_filter_acts.tolist(),save_folder+'/adv_ch_act_ov_incorr_org.jpeg',"Mean activity per Chn ov Incorr Adv")
            output_bar_graph([i+1 for i in range(all_incorr_convouts_grad.size()[0])],all_incorr_convouts_grad.tolist(),save_folder+'/adv_jacch_ov_incorr_org.jpeg',"Mean Jacobian per Chn ov Incorr Adv")
            
    
    print("Completed")