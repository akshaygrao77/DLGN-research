import torch
import numpy as np

from vgg_net_16 import DLGN_VGG_Network, DLGN_VGG_LinearNetwork, DLGN_VGG_WeightNetwork
from cross_verification import Net
from cross_verification_conv4_sim_vgg_with_dn import Net_sim_VGG_with_BN
from cross_verification_conv4_sim_vgg_without_bn import Net_sim_VGG_without_BN
from vgg_dlgn import vgg19, vgg19_with_inbuilt_norm
from cross_verification_inbuilt_norm import Net_with_inbuilt_norm, Net_with_inbuilt_norm_with_bn
from external_utils import DataNormalization_Layer

import torch.backends.cudnn as cudnn


def get_model_from_loader(model_arch_type, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "RANDOM_INIT_UNTRAINED_MODEL"
    print("Loading model")
    if(dataset == "cifar10"):
        if(model_arch_type == 'cifar10_vgg_dlgn_16'):
            model_path = "root/model/save/vggnet_ext_parallel_16_dir.pt"
            model = torch.load(model_path)
        elif(model_arch_type == 'cifar10_conv4_dlgn'):
            model_path = "root/model/save/model_norm_dir_None.pt"
            model = torch.load(model_path)
        elif(model_arch_type == 'cifar10_conv4_dlgn_sim_vgg_with_bn'):
            model_path = "root/model/save/cross_verification_conv4_sim_vgg_with_bn_norm_dir.pt"
            model = torch.load(model_path)
        elif(model_arch_type == 'cifar10_conv4_dlgn_sim_vgg_wo_bn'):
            model_path = "root/model/save/cross_verification_conv4_sim_vgg_wo_bn_norm_dir.pt"
            model = torch.load(model_path)
        elif(model_arch_type == 'random_conv4_dlgn_sim_vgg_wo_bn'):
            model = Net_sim_VGG_without_BN()
        elif(model_arch_type == 'random_conv4_dlgn_sim_vgg_with_bn'):
            model = Net_sim_VGG_with_BN()
        elif(model_arch_type == 'random_conv4_dlgn'):
            model = Net()
        elif(model_arch_type == 'random_vggnet_dlgn'):
            allones = np.ones((1, 3, 32, 32)).astype(np.float32)
            allones = torch.tensor(allones)
            model = vgg19(allones)
        elif(model_arch_type == "cifar10_conv4_dlgn_with_inbuilt_norm"):
            model_path = "root/model/save/cross_verification_pure_inbuilt_norm_conv4_dir.pt"
            model = torch.load(model_path)
        elif(model_arch_type == "random_cifar10_conv4_dlgn_with_inbuilt_norm"):
            model = Net_with_inbuilt_norm()
        elif(model_arch_type == "random_cifar10_conv4_dlgn_with_bn_with_inbuilt_norm"):
            model = Net_with_inbuilt_norm_with_bn()

        elif(model_arch_type == "cifar10_vgg_dlgn_16_with_inbuilt_norm"):
            model_path = "root/model/save/vggnet_with_inbuilt_norm_ext_parallel_16_dir.pt"
            model = torch.load(model_path)
        elif(model_arch_type == "cifar10_vgg_dlgn_16_with_inbuilt_norm_wo_bn"):
            model_path = "root/model/save/vggnet_with_inbuilt_norm_wo_bn_ext_parallel_16_dir.pt"
            model = torch.load(model_path)
        elif(model_arch_type == "random_cifar10_vgg_dlgn_16_with_inbuilt_norm"):
            allones = np.ones((1, 3, 32, 32)).astype(np.float32)
            allones = torch.tensor(allones)
            model = vgg19_with_inbuilt_norm(allones)

        elif(model_arch_type == "cifar10_conv4_dlgn_with_bn_with_inbuilt_norm"):
            model_path = "root/model/save/cross_verification_pure_inbuilt_norm_with_bn_conv4_dir.pt"
            model = torch.load(model_path)
        elif(model_arch_type == "cifar10_conv4_dlgn_with_inbuilt_norm_with_flip_crop"):
            model_path = "root/model/save/cross_verification_pure_inbuilt_norm_conv4_with_flip_crop_dir.pt"
            model = torch.load(model_path)
        elif(model_arch_type == "cifar10_conv4_dlgn_with_bn_with_inbuilt_norm_with_flip_crop"):
            model_path = "root/model/save/cross_verification_pure_inbuilt_norm_with_bn_conv4_with_flip_crop_dir.pt"
            model = torch.load(model_path)
        elif(model_arch_type == "conv4_dlgn"):
            model_path = "root/model/save/cifar10/conv4_dlgn_dir.pt"
            model = torch.load(model_path)

        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device_str == 'cuda':
            if(torch.cuda.device_count() > 1):
                print("Parallelizing model")
                model = torch.nn.DataParallel(model)
            cudnn.benchmark = True

    elif(dataset == "mnist"):
        if(model_arch_type == 'cifar10_conv4_dlgn'):
            model_path = "root/model/save/model_mnist_norm_dir_None.pt"
            model = torch.load(model_path)
        elif(model_arch_type == "dlgn_fc_w_128_d_4"):
            model_path = "root/model/save/mnist10_dlgn_fc_w_128_d_4_dir.pt"
            model = torch.load(model_path)
        elif(model_arch_type == "plain_pure_conv4_dnn"):
            model_path = "root/model/save/mnist/plain_pure_conv4_dnn_dir.pt"
            model = torch.load(model_path)
        elif(model_arch_type == "conv4_dlgn"):
            model_path = "root/model/save/mnist/conv4_dlgn_dir.pt"
            model = torch.load(model_path)

        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device_str == 'cuda':
            if(torch.cuda.device_count() > 1):
                print("Parallelizing model")
                model = torch.nn.DataParallel(model)
            cudnn.benchmark = True
    elif(dataset == "fashion_mnist"):
        if(model_arch_type == 'conv4_dlgn'):
            model_path = "root/model/save/fashion_mnist/CLEAN_TRAINING/ST_2022/conv4_dlgn_dir.pt"
            model = torch.load(model_path)
        elif(model_arch_type == "conv4_dlgn_n16_small"):
            model_path = "root/model/save/fashion_mnist/CLEAN_TRAINING/ST_2022/conv4_dlgn_n16_small_dir.pt"
            model = torch.load(model_path)
        elif(model_arch_type == "plain_pure_conv4_dnn_n16_small"):
            model_path = "root/model/save/fashion_mnist/CLEAN_TRAINING/ST_2022/plain_pure_conv4_dnn_n16_small_dir.pt"
            model = torch.load(model_path)
        elif(model_arch_type == "plain_pure_conv4_dnn"):
            model_path = "root/model/save/fashion_mnist/CLEAN_TRAINING/ST_2022/plain_pure_conv4_dnn_dir.pt"
            model = torch.load(model_path)
        elif(model_arch_type == "conv4_deep_gated_net"):
            model_path = "root/model/save/fashion_mnist/CLEAN_TRAINING/ST_2022/conv4_deep_gated_net_dir.pt"
            model = torch.load(model_path)
        elif(model_arch_type == "conv4_deep_gated_net_n16_small"):
            model_path = "root/model/save/fashion_mnist/CLEAN_TRAINING/ST_2022/conv4_deep_gated_net_n16_small_dir.pt"
            model = torch.load(model_path)

        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device_str == 'cuda':
            if(torch.cuda.device_count() > 1):
                print("Parallelizing model")
                model = torch.nn.DataParallel(model)
            cudnn.benchmark = True

    model.to(device)
    print("Model loaded of type:{} for dataset:{}".format(model_arch_type, dataset))

    return model, model_path
