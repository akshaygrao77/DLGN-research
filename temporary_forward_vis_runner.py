import torch
import numpy as np
from tqdm import tqdm, trange

from configs.generic_configs import get_preprocessing_and_other_configs
from conv4_models import get_model_instance, get_img_size
from collections import OrderedDict


if __name__ == '__main__':
    print("Start")
    # mnist , cifar10 , fashion_mnist , imagenet_1000
    dataset = 'cifar10'

    # cifar10_conv4_dlgn , cifar10_vgg_dlgn_16 , dlgn_fc_w_128_d_4 , random_conv4_dlgn , random_vggnet_dlgn
    # random_conv4_dlgn_sim_vgg_wo_bn , cifar10_conv4_dlgn_sim_vgg_wo_bn , cifar10_conv4_dlgn_sim_vgg_with_bn
    # random_conv4_dlgn_sim_vgg_with_bn , cifar10_conv4_dlgn_with_inbuilt_norm , random_cifar10_conv4_dlgn_with_inbuilt_norm
    # cifar10_vgg_dlgn_16_with_inbuilt_norm , random_cifar10_vgg_dlgn_16_with_inbuilt_norm
    # random_cifar10_conv4_dlgn_with_bn_with_inbuilt_norm , cifar10_conv4_dlgn_with_bn_with_inbuilt_norm
    # cifar10_conv4_dlgn_with_inbuilt_norm_with_flip_crop
    # cifar10_conv4_dlgn_with_bn_with_inbuilt_norm_with_flip_crop
    # cifar10_vgg_dlgn_16_with_inbuilt_norm_wo_bn
    # plain_pure_conv4_dnn , conv4_dlgn , conv4_dlgn_n16_small , plain_pure_conv4_dnn_n16_small
    # conv4_deep_gated_net , conv4_deep_gated_net_n16_small ,
    # dlgn__resnet18__ , dgn__resnet18__,dnn__resnet18__ , dlgn__vgg16_bn__ , dlgn__pad2_vgg16_bn__ , dlgn__st1_pad2_vgg16_bn_wo_bias__
    model_arch_type = 'dlgn__st1_pad2_vgg16_bn_wo_bias__'

    valid_split_size = 0.1
    torch_seed = 2022

    custom_model_path = "root/model/save/cifar10/CLEAN_TRAINING/ST_2022/dlgn__st1_pad2_vgg16_bn_wo_bias___PRET_False_dir.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inp_channel = get_img_size(dataset)[0]

    temp_model = torch.load(custom_model_path, map_location=device)
    classes, num_classes, ret_config = get_preprocessing_and_other_configs(
        dataset, valid_split_size)
    custom_model = get_model_instance(
        model_arch_type, inp_channel, seed=torch_seed, num_classes=num_classes)
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

    custom_model = custom_model.to(device)
    custom_model.eval()

    dummy_input = torch.rand(get_img_size(dataset)).unsqueeze(0)
    dummy_input = dummy_input.to(device)
    conv_matrix_operations_in_each_layer, conv_bias_operations_in_each_layer = custom_model.exact_forward_vis(
        dummy_input)
    print("Execution completed")
