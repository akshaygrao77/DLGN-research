import torch

from torch.optim import SGD
import os
from PIL import Image
import numpy as np
from tqdm import tqdm, trange
from torch.autograd import Variable
import subprocess
import math

import wandb
import random

import time

from configs.generic_configs import get_preprocessing_and_other_configs
from configs.dlgn_conv_config import HardRelu
from utils.data_preprocessing import preprocess_dataset_get_data_loader, segregate_classes
from structure.generic_structure import PerClassDataset
from model.model_loader import get_model_from_loader
from conv4_models import get_model_instance, get_img_size
import scipy.ndimage as nd
from collections import OrderedDict
from external_utils import format_time
# from adversarial_attacks_tester import generate_adv_examples,get_adv_save_str

# from vgg_net_16 import DLGN_VGG_Network, DLGN_VGG_LinearNetwork, DLGN_VGG_WeightNetwork
# from mnist_dlgn_fc import DLGN_FC_Network, DLGN_FC_Gating_Network, DLGN_FC_Value_Network
# from cross_verification import Net
# from cross_verification_conv4_sim_vgg_with_dn import Net_sim_VGG_with_BN
# from cross_verification_conv4_sim_vgg_without_bn import Net_sim_VGG_without_BN
# from vgg_dlgn import vgg19, vgg19_with_inbuilt_norm
# from cross_verification_inbuilt_norm import Net_with_inbuilt_norm, Net_with_inbuilt_norm_with_bn
# from external_utils import DataNormalization_Layer
from utils.visualise_utils import save_image, recreate_image, add_lower_dimension_vectors_within_itself, format_np_output


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_id + worker_seed)
    random.seed(worker_id - worker_seed)


def show_gpu(msg):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """
    def query(field):
        return(subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
                '--format=csv,nounits,noheader'],
            encoding='utf-8'))

    def to_int(result):
        return int(result.strip().split('\n')[0])

    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used/total
    print('\n' + msg, f'{100*pct:2.1f}% ({used} out of {total})')


def blur_img(img, sigma):
    # print("blur img shape", img.shape)
    if sigma > 0:
        img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
        # img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
        # img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
    return img


def preprocess_image(im_as_arr, normalize=True, resize_im=False):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # Mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Ensure or transform incoming image to PIL image
    # if type(pil_im) != Image.Image:
    #     try:
    #         pil_im = Image.fromarray(pil_im)
    #     except Exception as e:
    #         print("could not transform PIL_img to a PIL Image object. Please check input.")

    # # Resize image
    # if resize_im:
    #     pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    # im_as_arr = np.float32(pil_im)
    # im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    if(normalize):
        # Normalize the channels
        for channel, _ in enumerate(im_as_arr):
            im_as_arr[channel] /= 255
            im_as_arr[channel] -= mean[channel]
            im_as_arr[channel] /= std[channel]

    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten)
    return im_as_var


def unroll_print_torch_3D_array(torch_arr):
    (l, x, y, z) = torch_arr.size()
    for a in range(l):
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    print(torch_arr[a][i][j][k].item(), end=" ")
                print("")
            print("---------------------")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^")


def replace_percent_of_tensor_values_with_exact_percentages(inp_tensor, const_value, percentage):
    tpr = torch.arange(0, 100, step=100/inp_tensor.numel(),
                       dtype=torch.float32)
    indexes = torch.randperm(tpr.shape[0])
    tpr = tpr[indexes]
    tpr = torch.reshape(tpr, inp_tensor.size())
    ret = torch.where(tpr >= percentage, inp_tensor, const_value)
    return ret


def replace_percent_of_source_values_with_target_values_with_exact_percentages(inp_tensor, source_value, const_value, percentage_or_number_not_to_replace, seed_gen):
    ft_inp_tensor = torch.flatten(inp_tensor)
    temp = ft_inp_tensor == source_value
    source_indices = temp.nonzero()
    filtered_ft_inp_tensor = ft_inp_tensor[source_indices]
    if(percentage_or_number_not_to_replace <= 1):
        num_samples_to_be_replaced = int(
            filtered_ft_inp_tensor.numel() * (1 - percentage_or_number_not_to_replace))
    else:
        num_samples_to_be_replaced = filtered_ft_inp_tensor.numel() - \
            percentage_or_number_not_to_replace
    indexes = torch.randperm(
        filtered_ft_inp_tensor.numel(), generator=seed_gen)
    indexes = indexes[:num_samples_to_be_replaced]
    filtered_ft_inp_tensor[indexes] = const_value
    ft_inp_tensor[source_indices] = filtered_ft_inp_tensor
    ret = torch.reshape(ft_inp_tensor, inp_tensor.size())

    num_gates_sampled = filtered_ft_inp_tensor.numel() - num_samples_to_be_replaced
    return ret, num_gates_sampled


def get_fft_phase_amp_vis_for_tensor(input_image):
    raw_dft = torch.fft.fft2(input_image)
    raw_centered_dft_out = torch.fft.fftshift(raw_dft)
    for_vis_dft_amp_out = torch.log(1+torch.abs(raw_centered_dft_out))
    for_vis_dft_phase_out = torch.angle(raw_centered_dft_out)
    return for_vis_dft_amp_out, for_vis_dft_phase_out


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        for each_output in self.outputs:
            del each_output
        self.outputs = []

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()

class TemplateImageGenerator():

    def __init__(self, model, start_image_np):
        self.model = model
        self.model.eval()
        # Generate a random image
        # self.created_image = Image.open(im_path).convert('RGB')
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.image_save_prefix_folder = "root/cifar10-vggnet_16/CCE_and_Template_loss_mixed/zero_image_init/"

        # self.initial_image = start_image_np[None, :]
        self.original_image = start_image_np
        # im_path = 'root/generated/template_from_firstim_image_c_' + \
        #     str(class_label) + '.jpg'
        # # numpy_image = self.created_image.cpu().clone().detach().numpy()
        # numpy_image = recreate_image(
        #     self.original_image, False)
        # save_image(numpy_image, im_path)

        self.y_plus_list = None
        self.y_minus_list = None

        self.layer_nums_to_visualize = None
        # Hook the layers to get result of the convolution
        # self.hook_layer()
        # Create the folder to export images if not exists

        self.entropy_active_list = None
        self.overall_entropy_list = None

        self.layer_wise_average_feature_maps_dict = None

    def reset_layer_wise_states(self):
        self.layer_wise_average_feature_maps_dict = None
        self.layers_to_visualize = None

    def reset_collection_state(self):
        self.y_plus_list = None
        self.y_minus_list = None
        self.layer_nums_to_visualize = None
        if(hasattr(self,"out_capturer")):
            for e in self.out_capturer:
                e.close()
        tmp = self.original_image.type(torch.float32)
        self.model(torch.unsqueeze(tmp.to(self.device),0))
        self.is_use_capturer = False
        if(isinstance(self.model, torch.nn.DataParallel)):
            if(not hasattr(self.model.module,"linear_conv_outputs")):
                self.is_use_capturer = True
        else:
            if(not hasattr(self.model,"linear_conv_outputs")):
                self.is_use_capturer = True

        if(self.is_use_capturer):
            self.out_capturer = []
            for _,val in self.model.get_gate_layers_ordered_dict():
                self.out_capturer = SaveFeatures(val)

    def reset_entropy_state(self):
        self.entropy_active_list = None
        self.overall_entropy_list = None

    def get_convouts(self):
        if(not self.is_use_capturer):
            if(isinstance(self.model, torch.nn.DataParallel)):
                conv_outs = self.model.module.linear_conv_outputs
            else:
                conv_outs = self.model.linear_conv_outputs
        else:
            conv_outs = []
            for ev in self.out_capturer:
                conv_outs.append(ev.features)

        if(self.layer_nums_to_visualize is None):
            return conv_outs
        ret_outs = []
        for i in self.layer_nums_to_visualize:
            ret_outs.append(conv_outs[i])

        return ret_outs

    def initialise_layerwise_states(self, layers_to_visualize=None):
        self.total_tcollect_img_count = 0
        self.total_lay_coll_img_count = 0
        if(isinstance(self.model, torch.nn.DataParallel)):
            conv_outs = self.model.module.linear_conv_outputs
        else:
            conv_outs = self.model.linear_conv_outputs

        self.layer_wise_average_feature_maps_dict = dict()
        if(layers_to_visualize is None):
            self.layers_to_visualize = []
            for i in range(len(conv_outs)):
                self.layers_to_visualize.append(i)
        else:
            self.layers_to_visualize = layers_to_visualize

        for i in self.layers_to_visualize:
            each_conv_output = conv_outs[i]
            self.layer_wise_average_feature_maps_dict[i] = torch.zeros(size=each_conv_output.size()[
                1:], requires_grad=True, device=self.device)

    def initialise_y_plus_and_y_minus(self):
        self.y_plus_list = []
        self.y_minus_list = []
        self.total_tcollect_img_count = 0

        conv_outs = self.get_convouts()
        for each_conv_output in conv_outs:
            current_y_plus = torch.zeros(size=each_conv_output.size()[
                1:], requires_grad=True, device=self.device)
            current_y_minus = torch.zeros(size=each_conv_output.size()[
                1:], requires_grad=True, device=self.device)

            self.y_plus_list.append(current_y_plus)
            self.y_minus_list.append(current_y_minus)

    def initialise_entropy_information(self):
        self.entropy_active_list = []
        self.total_entr_img_count = 0

        conv_outs = self.get_convouts()
        for each_conv_output in conv_outs:
            current_entropy_active = torch.zeros(size=each_conv_output.size()[
                1:], requires_grad=True, device=self.device)

            self.entropy_active_list.append(current_entropy_active)

    def hook_layer(self):
        self.saved_output = SaveOutput()

        hook_handles = []

        for layer in self.model.linear_conv_net.modules():
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                handle = layer.register_forward_hook(self.saved_output)
                hook_handles.append(handle)

    def update_y_lists(self):
        conv_outs = self.get_convouts()
        with torch.no_grad():
            for indx in range(len(conv_outs)):
                # for indx in range(0, 4):
                each_conv_output = conv_outs[indx]
                positives = HardRelu()(each_conv_output)
                # [B,C,W,H]
                # red_pos = add_lower_dimension_vectors_within_itself(
                #   positives)
                red_pos = torch.sum(positives, dim=0)
                self.y_plus_list[indx] += red_pos

                negatives = HardRelu()(-each_conv_output)
                # red_neg = add_lower_dimension_vectors_within_itself(
                #     negatives)
                red_neg = torch.sum(negatives, dim=0)
                self.y_minus_list[indx] += red_neg

            self.total_tcollect_img_count += conv_outs[0].size()[0]

    def update_layerwise_states(self):
        if(isinstance(self.model, torch.nn.DataParallel)):
            conv_outs = self.model.module.linear_conv_outputs
        else:
            conv_outs = self.model.linear_conv_outputs
        bs = conv_outs[0].size()[0]
        with torch.no_grad():
            for l_indx in self.layers_to_visualize:
                each_conv_output = conv_outs[l_indx]
                # [B,C,W,H]
                cur_sum_fmap = torch.sum(each_conv_output, dim=0)
                if(self.total_lay_coll_img_count == 0):
                    self.layer_wise_average_feature_maps_dict[l_indx] = cur_sum_fmap/bs
                else:
                    self.layer_wise_average_feature_maps_dict[l_indx] = (
                        (self.layer_wise_average_feature_maps_dict[l_indx] * self.total_lay_coll_img_count) + cur_sum_fmap)/(self.total_lay_coll_img_count+bs)

            self.total_tcollect_img_count += bs
            self.total_lay_coll_img_count += bs

    def update_entropy_y_lists(self):
        conv_outs = self.get_convouts()
        with torch.no_grad():
            for indx in range(len(conv_outs)):
                each_conv_output = conv_outs[indx]

                actives = torch.where(each_conv_output > 0, 1, 0)
                red_act = torch.sum(actives, dim=0)
                self.entropy_active_list[indx] += red_act

            self.total_entr_img_count += conv_outs[0].size()[0]

    def calculate_entropy_from_maps(self):
        self.overall_entropy_list = []
        with torch.no_grad():
            for indx in range(len(self.entropy_active_list)):
                current_active_pixels = self.entropy_active_list[indx]
                current_active_prob = current_active_pixels / self.total_entr_img_count
                current_inactive_prob = 1 - current_active_prob
                entropy_bin_list = [current_active_prob, current_inactive_prob]
                current_layer_entropy = torch.zeros(
                    size=current_active_pixels.size(), device=self.device)

                for each_bin_value in entropy_bin_list:
                    zero_default = torch.zeros(
                        size=each_bin_value.size(), device=self.device)

                    pre_entropy = torch.where(
                        each_bin_value == 0., zero_default, (each_bin_value * torch.log2(each_bin_value)))
                    current_layer_entropy += pre_entropy

                current_layer_entropy = -current_layer_entropy

                self.overall_entropy_list.append(current_layer_entropy)

    def entropy_of_pixel_per_batch(self, per_class_per_batch_data):
        c_inputs, _ = per_class_per_batch_data
        c_inputs = c_inputs.to(self.device)

        # Forward pass to store layer outputs from hooks
        self.model(c_inputs)

        # Intiialise the structure to hold i's for which pixels are positive or negative
        if(self.entropy_active_list is None):
            self.initialise_entropy_information()

        self.update_entropy_y_lists()

    def collect_layer_wise_states_per_batch(self, per_class_per_batch_data, layers_to_visualize=None):
        c_inputs, _ = per_class_per_batch_data
        c_inputs = c_inputs.to(self.device)

        # Forward pass to store layer outputs
        self.model(c_inputs)

        if(self.layer_wise_average_feature_maps_dict is None):
            self.initialise_layerwise_states(layers_to_visualize)

        self.update_layerwise_states()

    def collect_all_layer_wise_states(self, per_class_data_loader, class_label, number_of_batch_to_collect, layers_to_visualize=None, is_save_original_image=True):
        self.reset_layer_wise_states()
        self.model.train(False)

        per_class_data_loader = tqdm(
            per_class_data_loader, desc='Collecting layer-wise states for class label:'+str(class_label))
        for i, per_class_per_batch_data in enumerate(per_class_data_loader):
            torch.cuda.empty_cache()
            c_inputs, _ = per_class_per_batch_data
            if(i == 0 and c_inputs.size()[0] == 1 and is_save_original_image):
                with torch.no_grad():
                    temp_image = recreate_image(
                        c_inputs, False)
                    save_folder = self.image_save_prefix_folder + \
                        "class_"+str(class_label)+"/"
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    im_path = save_folder+'/original.jpg'

                    numpy_image = temp_image
                    save_image(numpy_image, im_path)

            self.collect_layer_wise_states_per_batch(
                per_class_per_batch_data, layers_to_visualize)

            if(not(number_of_batch_to_collect is None) and i == number_of_batch_to_collect - 1):
                break

    def collect_active_pixel_per_batch(self, per_class_per_batch_data,is_set_rolled_part_zero=None):
        c_inputs, _ = per_class_per_batch_data
        c_inputs = c_inputs.to(self.device)

        # Forward pass to store layer outputs from hooks
        self.model(c_inputs)
        if(is_set_rolled_part_zero is None):
            # Intiialise the structure to hold i's for which pixels are positive or negative
            if(self.y_plus_list is None or self.y_minus_list is None):
                self.initialise_y_plus_and_y_minus()
            self.update_y_lists()
        else:
            if(self.y_plus_list is None or self.y_minus_list is None):
                self.initialise_y_plus_and_y_minus()
                self.update_y_lists()
            else:
                min_gate_diff = float("inf")
                min_x = -1
                min_y = -1
                min_fl = 0
                pxlmin = c_inputs.min()
                for xs in range(c_inputs.size()[-1]):
                    for ys in range(c_inputs.size()[-2]):
                        for fl in range(2):
                            cur_trans = torch.roll(c_inputs,shifts=(xs, ys), dims=(-1, -2))
                            if(is_set_rolled_part_zero):
                                cur_trans[:,:,0:ys,:] = pxlmin
                                cur_trans[:,:,:,0:xs] = pxlmin
                            if(fl % 2 == 1):
                                cur_trans = cur_trans.flip(-1)
                            
                            self.model(cur_trans)
                            
                            conv_outs = self.get_convouts()
                            y_plus_list = []
                            y_minus_list = []
                            with torch.no_grad():
                                for indx in range(len(conv_outs)):
                                    each_conv_output = conv_outs[indx]
                                    positives = HardRelu()(each_conv_output)
                                    # [B,C,W,H]
                                    red_pos = torch.sum(positives, dim=0)
                                    y_plus_list.append(red_pos)

                                    negatives = HardRelu()(-each_conv_output)
                                    red_neg = torch.sum(negatives, dim=0)
                                    y_minus_list.append(red_neg)
                                
                                gate_diff = 0
                                for indx in range(len(self.y_minus_list)):
                                    gate_diff += torch.sum(torch.abs(self.y_minus_list[indx] - y_minus_list[indx])) + torch.sum(torch.abs(self.y_plus_list[indx] - y_plus_list[indx]))

                                if(gate_diff < min_gate_diff):
                                    # print(xs,ys,min_gate_diff)
                                    min_x = xs
                                    min_y = ys
                                    min_fl = fl
                                    min_gate_diff = gate_diff

                c_inputs =  torch.roll(c_inputs,shifts=(min_x, min_y), dims=(-1, -2))
                if(is_set_rolled_part_zero):
                    c_inputs[:,:,0:min_y,:] = pxlmin
                    c_inputs[:,:,:,0:min_x] = pxlmin
                if(min_fl % 2 == 1):
                    c_inputs = c_inputs.flip(-1)
                
                self.model(c_inputs)
                self.update_y_lists()
                return c_inputs

    def update_overall_y_maps(self, collect_threshold, random_sample_gate_percent=None, coll_gate_sample_seed_gen=None):
        temp_one = torch.tensor([1.], device=self.y_minus_list[0].get_device())
        temp_zero = torch.tensor(
            [0.], device=self.y_minus_list[0].get_device())
        with torch.no_grad():
            self.overall_y = []
            for indx in range(len(self.y_plus_list)):
                each_y_plus = self.y_plus_list[indx]
                each_y_minus = self.y_minus_list[indx]

                y_plus_passed_threshold = HardRelu()(each_y_plus - collect_threshold *
                                                     self.total_tcollect_img_count)
                if(random_sample_gate_percent is not None):
                    assert coll_gate_sample_seed_gen is not None, "When random sample gate is enabled we need seed generator"
                    y_plus_passed_threshold, num_positive_gates_sampled = replace_percent_of_source_values_with_target_values_with_exact_percentages(
                        y_plus_passed_threshold, temp_one, temp_zero, random_sample_gate_percent, coll_gate_sample_seed_gen)
                y_minus_passed_threshold = HardRelu()(each_y_minus - collect_threshold *
                                                      self.total_tcollect_img_count)
                if(random_sample_gate_percent is not None):
                    assert coll_gate_sample_seed_gen is not None, "When random sample gate is enabled we need seed generator"
                    y_minus_passed_threshold, num_negative_gates_sampled = replace_percent_of_source_values_with_target_values_with_exact_percentages(
                        y_minus_passed_threshold, temp_one, temp_zero, random_sample_gate_percent, coll_gate_sample_seed_gen)
                    print("num_positive_gates_sampled:{},num_negative_gates_sampled:{}".format(
                        num_positive_gates_sampled, num_negative_gates_sampled))
                current_y_map = y_plus_passed_threshold - y_minus_passed_threshold

                self.overall_y.append(current_y_map)

    def collect_entropy_of_pixels_into_ymaps(self, per_class_data_loader, class_label, number_of_batch_to_collect):
        self.reset_entropy_state()
        self.model.train(False)

        per_class_data_loader = tqdm(
            per_class_data_loader, desc='Collecting entropy of pixels into maps for class label:'+str(class_label))
        for i, per_class_per_batch_data in enumerate(per_class_data_loader):

            self.entropy_of_pixel_per_batch(per_class_per_batch_data)

            if(not(number_of_batch_to_collect is None) and i == number_of_batch_to_collect - 1):
                break

        self.calculate_entropy_from_maps()

    def collect_all_active_pixels_into_ymaps(self, per_class_data_loader, class_label, number_of_batch_to_collect, collect_threshold, is_save_original_image=True, random_sample_gate_percent=None, coll_gate_sample_seed_gen=None):
        self.reset_collection_state()
        self.model.train(False)

        per_class_data_loader = tqdm(
            per_class_data_loader, desc='Collecting active maps class label:'+str(class_label))
        for i, per_class_per_batch_data in enumerate(per_class_data_loader):
            torch.cuda.empty_cache()
            c_inputs, _ = per_class_per_batch_data
            if(i == 0 and c_inputs.size()[0] == 1 and is_save_original_image):
                with torch.no_grad():
                    temp_image = recreate_image(
                        c_inputs, False)
                    save_folder = self.image_save_prefix_folder + \
                        "class_"+str(class_label)+"/"
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    im_path = save_folder+'/original.jpg'

                    numpy_image = temp_image
                    save_image(numpy_image, im_path)

            self.collect_active_pixel_per_batch(
                per_class_per_batch_data)

            if(not(number_of_batch_to_collect is None) and i == number_of_batch_to_collect - 1):
                break

        self.update_overall_y_maps(
            collect_threshold, random_sample_gate_percent, coll_gate_sample_seed_gen)

    def calculate_loss_for_output_class_max_image(self, outputs, labels):
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)

        return loss

    def calculate_loss_to_maximise_logits_wrt_another_image_logits(self, outputs, targets):
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(outputs, targets)

        return loss

    def calculate_mixed_loss_maximise_logit_and_template_image(self, outputs, targets,alpha = 0.1):
        mse_loss = self.calculate_loss_to_maximise_logits_wrt_another_image_logits(
            outputs, targets)
        template_loss, active_pixel_points, total_pixel_points = self.new_calculate_loss_for_template_image()
        overall_loss = alpha * mse_loss + (1-alpha)*template_loss
        return overall_loss, active_pixel_points, total_pixel_points

    def calculate_mixed_loss_output_class_and_template_image(self, outputs, labels, temp_loss_type, alpha=0.1):
        cce_loss = self.calculate_loss_for_output_class_max_image(
            outputs, labels)
        if(temp_loss_type == "TEMP_LOSS"):
            template_loss, active_pixel_points, total_pixel_points,non_zero_pixel_points = self.new_calculate_loss_for_template_image()
        elif(temp_loss_type == "TEMP_ACT_ONLY_LOSS"):
            template_loss, active_pixel_points, total_pixel_points,non_zero_pixel_points = self.calculate_only_active_loss_for_template_image()

        overall_loss = alpha * cce_loss + (1-alpha)*template_loss
        return overall_loss, active_pixel_points, total_pixel_points,non_zero_pixel_points

    def calculate_mixed_loss_adv_attack(self, outputs, labels, temp_loss_type, alpha):
        cce_loss = torch.log(torch.pow(10,torch.Tensor([-7])).to(
                            self.device)+self.calculate_loss_for_output_class_max_image(outputs, labels))
        if("TANH_TEMP_LOSS" in temp_loss_type):
            template_loss, active_pixel_points, total_pixel_points,non_zero_pixel_points = self.calculate_tanh_loss_for_template_image()
        elif("TEMP_LOSS" in temp_loss_type):
            template_loss, active_pixel_points, total_pixel_points,non_zero_pixel_points = self.new_calculate_loss_for_template_image()
        elif("TEMP_ACT_ONLY_LOSS" in temp_loss_type):
            template_loss, active_pixel_points, total_pixel_points,non_zero_pixel_points = self.calculate_only_active_loss_for_template_image()

        overall_loss = template_loss - alpha * cce_loss
        return overall_loss, active_pixel_points, total_pixel_points,non_zero_pixel_points

    def calculate_mixed_loss_pgd_attack(self, outputs, labels, temp_loss_type, alpha):
        cce_loss = torch.log(torch.pow(10,torch.Tensor([-7])).to(
                            self.device)+self.calculate_loss_for_output_class_max_image(outputs, labels))
        if("TANH_TEMP_LOSS" in temp_loss_type):
            template_loss, active_pixel_points, total_pixel_points,non_zero_pixel_points = self.calculate_tanh_loss_for_template_image()
        elif("TEMP_LOSS" in temp_loss_type):
            template_loss, active_pixel_points, total_pixel_points,non_zero_pixel_points = self.new_calculate_loss_for_template_image()
        elif("TEMP_ACT_ONLY_LOSS" in temp_loss_type):
            template_loss, active_pixel_points, total_pixel_points,non_zero_pixel_points = self.calculate_only_active_loss_for_template_image()

        # overall_loss = -torch.log(1 + torch.exp((template_loss * cce_loss)))
        overall_loss = -(template_loss * cce_loss)
        print("overall_loss:{} template_loss:{} cce_loss:{}".format(overall_loss,template_loss,cce_loss))
        return overall_loss, active_pixel_points, total_pixel_points,non_zero_pixel_points

    def calculate_mixed_loss_output_class_and_template_image_with_entropy(self, outputs, labels, alpha=0.1):
        cce_loss = self.calculate_loss_for_output_class_max_image(
            outputs, labels)
        template_loss, active_pixel_points, total_pixel_points = self.calculate_template_loss_with_entropy()

        overall_loss = alpha * cce_loss + (1-alpha)*template_loss
        return overall_loss, active_pixel_points, total_pixel_points

    def calculate_loss_for_template_image(self):
        loss = None
        conv_outs = self.get_convouts()
        total_pixel_points = 0
        active_pixel_points = 0
        for indx in range(len(conv_outs)):
            each_conv_output = conv_outs[indx]
            each_overall_y = self.overall_y[indx]
            # print("each_conv_output size", each_conv_output.size())
            # print("each_overall_y size", each_overall_y.size())
            flattened_conv_output = torch.flatten(each_conv_output)
            flattened_each_overall_y = torch.flatten(each_overall_y)

            assert len(flattened_conv_output) == len(
                flattened_each_overall_y), 'Flattened conv output and flattened overall y length unequal'

            total_pixel_points += len(flattened_conv_output)
            for each_point_indx in range(len(flattened_conv_output)):
                each_conv_out_point = flattened_conv_output[each_point_indx]
                each_overall_y_point = flattened_each_overall_y[each_point_indx]
                if(math. isnan(each_conv_out_point) or math.isinf(each_conv_out_point)):
                    print("each_conv_out_point is nan", each_conv_out_point)
                if(math. isnan(each_overall_y_point) or math.isinf(each_overall_y_point)):
                    print("each_overall_y_point is nan", each_overall_y_point)

                if(each_overall_y_point != 0):
                    active_pixel_points += 1
                    # print("each_conv_out_point :{} ==>{}".format(
                    #     indx, str(each_conv_out_point)))
                    exp_product = torch.exp(-each_overall_y_point *
                                            each_conv_out_point * 0.004)
                    if(math. isnan(exp_product) or math.isinf(exp_product)):
                        print("exp_product is nan/inf", exp_product)
                        print("each_conv_out_point", each_conv_out_point)
                        print("each_overall_y_point", each_overall_y_point)
                    # print("exp_product :{} ==>{}".format(
                    #     indx, str(exp_product)))
                    log_term = torch.log(1 + exp_product)
                    if(math. isnan(log_term) or math.isinf(log_term)):
                        print("log_term is nan", log_term)
                        print("exp_product", exp_product)

                    if(loss is None):
                        loss = log_term
                    else:
                        loss += log_term

        print("old raw loss", loss.item())
        return loss/active_pixel_points, active_pixel_points, total_pixel_points

    def calculate_template_loss_with_entropy(self):
        loss = 0
        conv_outs = self.get_convouts()

        total_pixel_points = 0
        active_pixel_points = 0
        non_zero_pixel_points = 0
        for indx in range(len(conv_outs)):
            each_conv_output = conv_outs[indx]
            each_overall_entropy = self.overall_entropy_list[indx]
            each_overall_y = self.overall_y[indx]

            current_active_pixels = self.entropy_active_list[indx]
            current_inactive_pixels = self.total_entr_img_count - current_active_pixels
            # positive entry here means majority times they were active. Negative entry here means majority times they were inactive
            diff_in_active = current_active_pixels - current_inactive_pixels
            current_y_map = torch.where(
                diff_in_active >= 0, 1., -1.)

            total_pixel_points += torch.numel(each_conv_output)
            current_active_pixel = torch.count_nonzero(
                HardRelu()(current_y_map))
            active_pixel_points += current_active_pixel.item()

            pre_exponent = torch.exp(-each_overall_y *
                                     each_conv_output * 0.004)
            exp_product_active_pixels = torch.where(
                each_overall_y == 0, each_overall_y, pre_exponent)

            log_term = torch.log(1 + exp_product_active_pixels)

            entrpy_prdct_term = each_overall_entropy * log_term

            each_conv_loss = torch.sum(entrpy_prdct_term)

            loss += each_conv_loss

        return loss, active_pixel_points, total_pixel_points

    def get_active_maps(self):
        conv_outs = self.get_convouts()

        list_of_active_maps = []
        for indx in range(len(conv_outs)):
            each_overall_y = self.overall_y[indx]
            current_active_map = HardRelu()(each_overall_y)
            list_of_active_maps.append(current_active_map.clone().detach())

        return list_of_active_maps

    def calculate_tanh_loss_for_template_image(self):
        tanh = torch.nn.Tanh()
        loss = 0
        conv_outs = self.get_convouts()

        total_pixel_points = 0
        active_pixel_points = 0
        non_zero_pixel_points = 0
        for indx in range(len(conv_outs)):
            each_conv_output = conv_outs[indx]
            each_overall_y = self.overall_y[indx]

            total_pixel_points += torch.numel(each_overall_y)
            current_active_pixel = torch.count_nonzero(
                HardRelu()(each_overall_y))
            non_zero_pixel_points += torch.count_nonzero(each_overall_y).item()
            active_pixel_points += current_active_pixel.item()
            exp_pow_term = -each_overall_y * tanh(each_conv_output)
            exp_pow_term = torch.clamp(exp_pow_term, -10, 10)
            pre_exponent = torch.exp(exp_pow_term)

            exp_product_active_pixels = torch.where(
                each_overall_y == 0, each_overall_y, pre_exponent)
            # print("exp_product_active_pixels", exp_product_active_pixels)

            log_term = torch.log(1 + exp_product_active_pixels)
            # print("log_term", log_term)

            each_conv_loss = torch.sum(log_term)
            # print("each_conv_loss", each_conv_loss)

            assert not(torch.isnan(each_conv_output).any().item() or torch.isinf(each_conv_output).any(
            ).item()), 'Conv out has nan or inf values while calculating temp loss'+str(each_conv_output)
            assert not(torch.isnan(each_conv_loss).any().item() or torch.isinf(
                each_conv_loss).any().item()), 'Loss value is inf or nan while calculating temp loss'+str(each_conv_loss)
            loss += each_conv_loss

        return loss/active_pixel_points, active_pixel_points, total_pixel_points, non_zero_pixel_points

    def new_calculate_loss_for_template_image(self, verbose=1):
        loss = 0
        conv_outs = self.get_convouts()

        total_pixel_points = 0
        active_pixel_points = 0
        non_zero_pixel_points = 0
        if(verbose > 2):
            print("self.initial image", self.initial_image)
            print("max initial image", torch.max(self.initial_image))
            print("min initial image", torch.min(self.initial_image))
        for indx in range(len(conv_outs)):
            each_conv_output = conv_outs[indx]
            each_overall_y = self.overall_y[indx]

            total_pixel_points += torch.numel(each_overall_y)
            current_active_pixel = torch.count_nonzero(
                HardRelu()(each_overall_y))
            non_zero_pixel_points += torch.count_nonzero(each_overall_y).item()
            active_pixel_points += current_active_pixel.item()
            exp_pow_term = -each_overall_y * each_conv_output * 0.004
            exp_pow_term = torch.clamp(exp_pow_term, -10, 10)
            pre_exponent = torch.exp(exp_pow_term)

            if(verbose > 2):
                print("Layer ----------- ", str(indx))
                print("pre_exponent dtype", pre_exponent.dtype)
                agmax = torch.argmax(each_conv_output)
                agmin = torch.argmin(each_conv_output)
                # unroll_print_torch_3D_array(each_conv_output)
                print("max conv out", torch.max(each_conv_output))
                print("min conv out", torch.min(each_conv_output))
                print("argmax conv out", agmax)
                print("argmin conv out", agmin)
                print("each_overall_y size", each_overall_y.size())
                print("each_conv_output", torch.norm(each_conv_output))
                print("current active", current_active_pixel.item() /
                      torch.numel(each_overall_y))
                print("current inactive", torch.count_nonzero(
                    HardRelu()(-each_overall_y)).item()/torch.numel(each_overall_y))
                # print("pre_exponent shape:", pre_exponent.size())
                # print("pre_exponent", pre_exponent)

                zero = torch.zeros(1, device=self.device)
                exp_active = torch.where(
                    each_overall_y == 1, pre_exponent, zero)
                # print("exp_active", exp_active)
                active_loss = torch.sum(torch.log(1 + exp_active))
                # print("active_loss", active_loss)
                exp_inactive = torch.where(
                    each_overall_y == -1, pre_exponent, zero)
                # print("exp_inactive", exp_inactive)
                # unroll_print_torch_3D_array(each_conv_output)
                log_inactive = torch.log(1 + exp_inactive)
                # print("log_inactive:", log_inactive)
                inactive_loss = torch.sum(log_inactive)
                print("inactive_loss", inactive_loss)

            exp_product_active_pixels = torch.where(
                each_overall_y == 0, each_overall_y, pre_exponent)
            # print("exp_product_active_pixels", exp_product_active_pixels)

            log_term = torch.log(1 + exp_product_active_pixels)
            # print("log_term", log_term)

            each_conv_loss = torch.sum(log_term)
            # print("each_conv_loss", each_conv_loss)

            assert not(torch.isnan(each_conv_loss).any().item() or torch.isinf(
                each_conv_loss).any().item()), 'Loss value is inf or nan while calculating temp loss'+str(each_conv_loss)
            assert not(torch.isnan(each_conv_output).any().item() or torch.isinf(each_conv_output).any(
            ).item()), 'Conv out has nan or inf values while calculating temp loss'
            loss += each_conv_loss

        return loss/active_pixel_points, active_pixel_points, total_pixel_points, non_zero_pixel_points

    def calculate_layer_MSE_loss(self, layer_indx):
        loss = 0
        total_pixel_points = 0
        active_pixel_points = 0
        non_zero_pixel_points = 0

        conv_outs = self.get_convouts()
        cur_conv_out = conv_outs[layer_indx][0]

        expected_out = self.layer_wise_average_feature_maps_dict[layer_indx]
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(cur_conv_out, expected_out)

        total_pixel_points += torch.numel(cur_conv_out)
        current_active_pixel = torch.count_nonzero(
            HardRelu()(cur_conv_out))
        non_zero_pixel_points += torch.count_nonzero(cur_conv_out).item()

        active_pixel_points += current_active_pixel.item()

        return loss, active_pixel_points, total_pixel_points, non_zero_pixel_points

    def calculate_only_active_loss_for_template_image(self):
        loss = 0
        conv_outs = self.get_convouts()

        total_pixel_points = 0
        active_pixel_points = 0
        non_zero_pixel_points = 0
        for indx in range(len(conv_outs)):
            each_conv_output = conv_outs[indx]
            each_overall_y = self.overall_y[indx]

            total_pixel_points += torch.numel(each_overall_y)
            current_active_pixel = torch.count_nonzero(
                HardRelu()(each_overall_y))
            non_zero_pixel_points += torch.count_nonzero(each_overall_y).item()

            # Taking into account only active pixels into loss
            each_overall_y = HardRelu()(each_overall_y)
            active_pixel_points += current_active_pixel.item()
            pre_exponent = torch.exp(-each_overall_y *
                                     each_conv_output * 0.004)
            exp_product_active_pixels = torch.where(
                each_overall_y == 0, each_overall_y, pre_exponent)

            log_term = torch.log(1 + exp_product_active_pixels)

            each_conv_loss = torch.sum(log_term)

            loss += each_conv_loss

        return loss/active_pixel_points, active_pixel_points, total_pixel_points, non_zero_pixel_points

    def get_wandb_config(self, exp_type, class_label, class_indx, classes, model_arch_type, dataset, is_template_image_on_train,
                         is_class_segregation_on_ground_truth, template_initial_image_type,
                         template_image_calculation_batch_size, template_loss_type, torch_seed, number_of_image_optimization_steps,
                         plot_iteration_interval=None, number_of_batch_to_collect=None, collect_threshold=None):

        wandb_config = dict()
        wandb_config["class_label"] = class_label
        wandb_config["class_indx"] = class_indx
        wandb_config["classes"] = classes
        wandb_config["model_arch_type"] = model_arch_type
        wandb_config["dataset"] = dataset
        wandb_config["is_template_image_on_train"] = is_template_image_on_train
        wandb_config["is_class_segregation_on_ground_truth"] = is_class_segregation_on_ground_truth
        wandb_config["template_initial_image_type"] = template_initial_image_type
        wandb_config["template_image_calculation_batch_size"] = template_image_calculation_batch_size
        wandb_config["template_loss_type"] = template_loss_type
        wandb_config["torch_seed"] = torch_seed
        wandb_config["number_of_image_optimization_steps"] = number_of_image_optimization_steps
        wandb_config["exp_type"] = exp_type
        if(not(plot_iteration_interval is None)):
            wandb_config["plot_iteration_interval"] = plot_iteration_interval
        if(not(number_of_batch_to_collect is None)):
            wandb_config["number_of_batch_to_collect"] = number_of_batch_to_collect
        if(not(collect_threshold is None)):
            wandb_config["collect_threshold"] = collect_threshold

        return wandb_config

    def get_prediction(self, input_tensor, original_label):

        outputs_raw = self.model(input_tensor)
        outputs_logits = outputs_raw.softmax(dim=1)
        outputs_final = outputs_logits.max(1).indices

        correct = outputs_final.eq(original_label).sum().item()

        return outputs_raw, outputs_logits, outputs_final, correct

    def get_layer_loss_value(self, template_loss_type, class_indx, layer_indx, outputs=None, class_image=None, alpha=None, verbose=1):
        active_pixel_points = None
        total_pixel_points = None
        non_zero_pixel_points = None
        if(template_loss_type == "MSE_LAYER_LOSS"):
            loss, active_pixel_points, total_pixel_points, non_zero_pixel_points = self.calculate_layer_MSE_loss(
                layer_indx)

        return loss, active_pixel_points, total_pixel_points, non_zero_pixel_points

    def get_loss_value(self, template_loss_type, class_indx, outputs=None, class_image=None, alpha=0.1, verbose=1):
        active_pixel_points = None
        total_pixel_points = None
        non_zero_pixel_points = None
        
        if("RPLATTCK" in template_loss_type):
            actual = torch.tensor(
                [class_indx] * len(outputs), device=self.device)
            loss, active_pixel_points, total_pixel_points, non_zero_pixel_points = self.calculate_mixed_loss_pgd_attack(outputs, actual, template_loss_type, alpha)
        elif("LATTCK" in template_loss_type):
            actual = torch.tensor(
                [class_indx] * len(outputs), device=self.device)
            loss, active_pixel_points, total_pixel_points, non_zero_pixel_points = self.calculate_mixed_loss_adv_attack(outputs, actual, template_loss_type, alpha)
        elif(template_loss_type == "TEMP_LOSS"):
            loss, active_pixel_points, total_pixel_points, non_zero_pixel_points = self.new_calculate_loss_for_template_image(
                verbose=verbose)
        elif(template_loss_type == "TANH_TEMP_LOSS"):
            loss, active_pixel_points, total_pixel_points, non_zero_pixel_points = self.calculate_tanh_loss_for_template_image()
        elif(template_loss_type == "TEMP_ACT_ONLY_LOSS"):
            loss, active_pixel_points, total_pixel_points, non_zero_pixel_points = self.calculate_only_active_loss_for_template_image()
        elif(template_loss_type == "ENTR_TEMP_LOSS"):
            loss, active_pixel_points, total_pixel_points = self.calculate_template_loss_with_entropy()
        elif(template_loss_type == "CCE_TEMP_LOSS_MIXED"):
            actual = torch.tensor(
                [class_indx] * len(outputs), device=self.device)
            loss, active_pixel_points, total_pixel_points,non_zero_pixel_points = self.calculate_mixed_loss_output_class_and_template_image(
                outputs, actual, "TEMP_LOSS", alpha)
        elif(template_loss_type == "CCE_TEMP_ACT_ONLY_LOSS_MIXED"):
            actual = torch.tensor(
                [class_indx] * len(outputs), device=self.device)
            loss, active_pixel_points, total_pixel_points = self.calculate_mixed_loss_output_class_and_template_image(
                outputs, actual, "TEMP_ACT_ONLY_LOSS", alpha)

        elif(template_loss_type == "CCE_ENTR_TEMP_LOSS_MIXED"):
            actual = torch.tensor(
                [class_indx] * len(outputs), device=self.device)
            loss, active_pixel_points, total_pixel_points = self.calculate_mixed_loss_output_class_and_template_image_with_entropy(
                outputs, actual, alpha)
        elif(template_loss_type == "CCE_LOSS"):
            actual = torch.tensor(
                [class_indx] * len(outputs), device=self.device)
            loss = self.calculate_loss_for_output_class_max_image(
                outputs, actual)
        elif(template_loss_type == "MSE_LOSS"):
            targets = self.model(class_image)
            loss = self.calculate_loss_to_maximise_logits_wrt_another_image_logits(
                outputs, targets)
        elif(template_loss_type == "MSE_TEMP_LOSS_MIXED"):
            targets = self.model(class_image)
            loss, active_pixel_points, total_pixel_points = self.calculate_mixed_loss_maximise_logit_and_template_image(
                outputs, targets)
        assert not(torch.isnan(loss).any().item() or torch.isinf(
                loss).any().item()), 'Loss value is inf or nan while calculating temp loss'+str(loss.item())
        return loss, active_pixel_points, total_pixel_points, non_zero_pixel_points

    def generate_accuracies_of_template_image_per_class(self, per_class_dataset, class_label, class_indx, classes, model_arch_type, dataset, is_template_image_on_train,
                                                        is_class_segregation_on_ground_truth, template_initial_image_type,
                                                        template_image_calculation_batch_size, template_loss_type, wand_project_name, wandb_group_name, torch_seed, number_of_image_optimization_steps,
                                                        exp_type, collect_threshold, entropy_calculation_batch_size, number_of_batches_to_calculate_entropy_on,
                                                        plot_iteration_interval=None, root_save_prefix="root", final_postfix_for_save="", wandb_config_additional_dict=None, vis_version='V2', random_sample_gate_percent=None,alpha = 0,v4lr=0.1):
        coll_gate_sample_seed_gen = None
        random_sample_gate_percent_str = ""
        list_of_reconst_images = None
        list_of_reconst_preds = None
        is_log_wandb = not(wand_project_name is None)

        # torch.manual_seed(torch_seed)
        coll_seed_gen = torch.Generator()
        coll_seed_gen.manual_seed(torch_seed)

        entr_seed_gen = torch.Generator()
        entr_seed_gen.manual_seed(torch_seed)

        if(random_sample_gate_percent is not None):
            random_sample_gate_percent_str = "RD_GT_PER_LAY_SAMPLE_" + \
                str(random_sample_gate_percent)
            coll_gate_sample_seed_gen = torch.Generator()
            coll_gate_sample_seed_gen.manual_seed(torch_seed)

        per_class_per_batch_data_loader = torch.utils.data.DataLoader(per_class_dataset, batch_size=template_image_calculation_batch_size,
                                                                      shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)
        tmp_image_over_what_str = 'test'
        if(is_template_image_on_train):
            tmp_image_over_what_str = 'train'

        seg_over_what_str = 'MP'
        if(is_class_segregation_on_ground_truth):
            seg_over_what_str = 'GT'

        layer_nums_str = ""
        if(self.layer_nums_to_visualize is not None):
            layer_nums_str = "/LV_"
            for i in self.layer_nums_to_visualize:
                layer_nums_str += str(i)+" "
            layer_nums_str = layer_nums_str.strip()
            layer_nums_str = layer_nums_str.replace(" ", "_")

        self.model.train(False)
        self.image_save_prefix_folder = str(root_save_prefix)+"/"+str(dataset)+"/MT_"+str(model_arch_type)+"_ET_"+str(exp_type)+"/Ver_"+str(vis_version)+"/_COLL_OV_"+str(tmp_image_over_what_str)+"/SEG_"+str(
            seg_over_what_str)+"/TMP_COLL_BS_"+str(template_image_calculation_batch_size)+"_NO_TO_COLL_"+str(number_of_batch_to_collect)+"/TMP_LOSS_TP_"+str(template_loss_type)+"/TMP_INIT_"+str(template_initial_image_type)+"/_torch_seed_"+str(torch_seed)+"_c_thres_"+str(collect_threshold)+"/"+random_sample_gate_percent_str+str(layer_nums_str)+"/" + str(final_postfix_for_save) + "/"

        per_class_per_batch_data_loader = tqdm(
            per_class_per_batch_data_loader, desc='Image being processed:'+str(class_label))
        total = 0
        reconst_correct = 0
        original_correct = 0
        if(alpha != 0):
            self.image_save_prefix_folder += "_alp_" + str(alpha)+"/"
        if(vis_version == "V4"):
            self.image_save_prefix_folder += "/V4_lr_"+str(v4lr)+"/"
        normalize_image = False
        overall_step = 0
        average_percent_active_pixels = 0.
        max_percent_active_pixels = 0.
        min_percent_active_pixels = 100.

        np_save_filename = self.image_save_prefix_folder + \
            '/class_'+str(class_label) + '.npy'
        is_current_aug_available = os.path.exists(np_save_filename)
        if(is_current_aug_available):
            with open(np_save_filename, 'rb') as file:
                npzfile = np.load(np_save_filename)
                each_class_output_template_list = npzfile['x']
                return each_class_output_template_list

        if(vis_version == 'V2' or vis_version == 'V3' or vis_version == 'V4'):
            if('conv4_deep_gated_net' in model_arch_type):
                number_of_image_optimization_steps = 301
            else:
                # number_of_image_optimization_steps = 161
                pass

            if(vis_version == 'V2' or vis_version == 'V4'):
                start_sigma = 0.75
                end_sigma = 0.1
            elif(vis_version == 'V3'):
                start_sigma = 0.1
                end_sigma = 0.75

            if('conv4_deep_gated_net' in model_arch_type):
                start_step_size = 1
                end_step_size = 0.5
            else:
                start_step_size = 0.1
                end_step_size = 0.05
        if(not(plot_iteration_interval is None)):
            number_of_intervals = (
                number_of_image_optimization_steps//plot_iteration_interval) + 1
            list_correct_prediction_of_reconst_img = [0] * number_of_intervals
            list_sum_norm_prediction_of_reconst_img = [
                0.] * number_of_intervals
            list_sum_norm_unnorm_gradients = [
                0.] * number_of_intervals
            list_sum_loss = [
                0.] * number_of_intervals

            list_of_optimized_image = [None] * number_of_intervals

            list_of_unnorm_gradients = [None] * number_of_intervals

            list_of_norm_gradients = [None] * number_of_intervals

            num_classes = len(classes)
            list_sum_softmax_of_reconst_img = [
                None] * number_of_intervals
            for ind in range(len(list_sum_softmax_of_reconst_img)):
                list_sum_softmax_of_reconst_img[ind] = [0.] * num_classes

        if(is_log_wandb):
            wandb_run_name = self.image_save_prefix_folder.replace(
                "/", "").replace(root_save_prefix, class_label)
            wandb_config = self.get_wandb_config(exp_type, class_label, class_indx, classes, model_arch_type, dataset, is_template_image_on_train,
                                                 is_class_segregation_on_ground_truth, template_initial_image_type,
                                                 template_image_calculation_batch_size, template_loss_type, torch_seed, number_of_image_optimization_steps,
                                                 plot_iteration_interval, collect_threshold=collect_threshold)
            wandb_config["vis_version"] = vis_version
            if(vis_version == 'V2' or vis_version == 'V3'):
                wandb_config["start_sigma"] = start_sigma
                wandb_config["end_sigma"] = end_sigma
                wandb_config["start_step_size"] = start_step_size
                wandb_config["end_step_size"] = end_step_size
                wandb_config["per_layer_random_sample_gate_percent"] = random_sample_gate_percent
            if(vis_version == "V4"):
                wandb_config["v4lr"] = v4lr

            if(wandb_config_additional_dict is not None):
                wandb_config.update(wandb_config_additional_dict)
            wandb_config["alpha"] = alpha

            wandb.init(
                project=f"{wand_project_name}",
                name=f"{wandb_run_name}",
                group=f"{wandb_group_name}",
                config=wandb_config,
            )

        if("ENTR" in template_loss_type):
            self.image_save_prefix_folder += "_ENT_BS_" + \
                str(entropy_calculation_batch_size)+"_ENTR_COLL_" + \
                str(number_of_batches_to_calculate_entropy_on)+"/"
            entropy_per_class_data_loader = torch.utils.data.DataLoader(per_class_dataset, batch_size=entropy_calculation_batch_size,
                                                                        shuffle=True, num_workers=2, generator=entr_seed_gen, worker_init_fn=seed_worker)
            self.collect_entropy_of_pixels_into_ymaps(
                entropy_per_class_data_loader, class_label, number_of_batch_to_collect=number_of_batches_to_calculate_entropy_on)

        eps = None
        if("PLATTCK" in template_loss_type):
            eps = float(template_loss_type.split("__")[1])

        batch_count = 0
        num_correct = 0
        num_incorrect = 0
        for batch_indx, per_class_data in enumerate(per_class_per_batch_data_loader):
            batch_count += 1
            torch.cuda.empty_cache()

            class_image, original_label = per_class_data
            class_image = class_image.to(self.device, non_blocking=True)
            original_label = original_label.to(self.device, non_blocking=True)

            self.reset_collection_state()
            self.collect_active_pixel_per_batch(
                per_class_data)
            self.update_overall_y_maps(
                collect_threshold, random_sample_gate_percent, coll_gate_sample_seed_gen)

            self.initial_image = preprocess_image(
                self.original_image.cpu().clone().detach().numpy(), normalize_image)

            self.initial_image = self.initial_image.to(self.device)

            self.initial_image.requires_grad_()

            step_size = 0.01
            if(vis_version == 'V2' or vis_version == 'V3'):
                # with trange(number_of_image_optimization_steps, unit="iter", desc="Generating template image for current batch V2") as pbar:
                for step_iter in range(number_of_image_optimization_steps):
                    # pbar.set_description(f"Iteration {step_iter+1}")

                    step_size = start_step_size + \
                        ((end_step_size - start_step_size) * step_iter) / \
                        number_of_image_optimization_steps
                    # optimizer = torch.optim.SGD([img_var], lr=step_size)
                    sigma = start_sigma + \
                        ((end_sigma - start_sigma) * step_iter) / \
                        number_of_image_optimization_steps

                    outputs = self.model(self.initial_image)
                    for x in self.get_convouts():
                        assert not(torch.isnan(x).any().item() or torch.isinf(x).any().item()), 'self.get_convouts value is inf or nan while calculating convouts: self.initial_image->'+str(self.initial_image)

                    loss, active_pixel_points, total_pixel_points, non_zero_pixel_points = self.get_loss_value(
                        template_loss_type, class_indx, outputs, class_image,alpha=alpha)

                    if(step_iter == 0 and active_pixel_points is not None):
                        percent_active_pixels = float((
                            active_pixel_points/total_pixel_points)*100)
                        print("active_pixel_points", active_pixel_points)
                        print("total_pixel_points", total_pixel_points)
                        print("Percentage of active pixels:",
                              percent_active_pixels)
                        if(percent_active_pixels > max_percent_active_pixels):
                            max_percent_active_pixels = percent_active_pixels
                        if(percent_active_pixels < min_percent_active_pixels):
                            min_percent_active_pixels = percent_active_pixels
                        average_percent_active_pixels += percent_active_pixels
                        if(is_log_wandb):
                            wandb.log({
                                "active_pixel_points": active_pixel_points, "total_pixel_points": total_pixel_points, "non_zero_pixel_points": non_zero_pixel_points,
                                "Percent_active_pixels": percent_active_pixels, "final_postfix_for_save": final_postfix_for_save
                            }, step=(batch_indx+1))

                    # print("{} Loss: {}".format(template_loss_type, loss))

                    # Backward
                    loss.backward()

                    unnorm_gradients = self.initial_image.grad
                    assert not(torch.isnan(unnorm_gradients).any().item() or torch.isinf(
                            unnorm_gradients).any().item()), 'unnorm_gradients value is inf or nan while calculating self.initial_image'+str(self.initial_image)+"loss "+str(loss)
                    # std_unnorm_grad = torch.std(unnorm_gradients)
                    norm_grad = torch.norm(unnorm_gradients)
                    # print("torch.norm(unnorm_gradients):",
                    #       norm_grad)
                    # print("std_unnorm_grad:", std_unnorm_grad)
                    # print("Original self.initial_image gradients", gradients)

                    # gradients = unnorm_gradients / first_norm
                    gradients = unnorm_gradients / \
                        norm_grad + 1e-8

                    # print("torch.norm(gradients):",
                    #       torch.norm(gradients))

                    with torch.no_grad():
                        # self.initial_image = self.initial_image - gradients*step_size
                        blurred_grad = gradients.cpu().detach().numpy()[0]
                        blurred_grad = blur_img(
                            blurred_grad, sigma)
                        self.initial_image = self.initial_image.cpu().detach().numpy()[
                            0]
                        self.initial_image -= step_size / \
                            np.abs(blurred_grad).mean() * blurred_grad

                        # optimizer.step()
                        # print("sigma:", sigma)

                        self.initial_image = blur_img(
                            self.initial_image, sigma)
                        self.initial_image = torch.from_numpy(
                            self.initial_image[None])
                        if(step_iter % 100 == 0):
                            print("Loss:{},norm_image:{},norm_grad:{},norm_raw_grad:{}".format(loss.item(), torch.norm(
                                self.initial_image).item(), torch.norm(gradients).item(), norm_grad.item()))

                        # pbar.set_postfix(loss=loss.item(), norm_image=torch.norm(
                            # self.initial_image).item(), norm_grad=torch.norm(gradients).item(), norm_raw_grad=norm_grad.item())

                        self.initial_image = self.initial_image.to(
                            self.device)
                        if(eps is not None):
                            self.initial_image = torch.clamp(self.initial_image, class_image - eps, class_image + eps)
                            self.initial_image = torch.clamp(self.initial_image, 0, 1)
                        if(not(plot_iteration_interval is None) and step_iter % plot_iteration_interval == 0):
                            update_indx = step_iter // plot_iteration_interval
                            _, outputs_logits, outputs_final, correct = self.get_prediction(
                                self.initial_image, original_label)
                            list_correct_prediction_of_reconst_img[update_indx] += correct
                            list_sum_norm_prediction_of_reconst_img[update_indx] += torch.norm(
                                self.initial_image).item()
                            list_sum_norm_unnorm_gradients[update_indx] += torch.norm(
                                unnorm_gradients).item()
                            list_sum_loss[update_indx] += loss

                            init_image_cpu = self.initial_image.cpu().detach().numpy()
                            if(list_of_optimized_image[update_indx] is None):
                                list_of_optimized_image[update_indx] = init_image_cpu
                            else:
                                list_of_optimized_image[update_indx] += init_image_cpu

                            unnorm_grads_np = unnorm_gradients.cpu().detach().numpy()
                            if(list_of_unnorm_gradients[update_indx] is None):
                                list_of_unnorm_gradients[update_indx] = unnorm_grads_np
                            else:
                                list_of_unnorm_gradients[update_indx] += unnorm_grads_np

                            grads_np = gradients.cpu().detach().numpy()
                            if(list_of_norm_gradients[update_indx] is None):
                                list_of_norm_gradients[update_indx] = grads_np
                            else:
                                list_of_norm_gradients[update_indx] += grads_np

                            list_sum_softmax_of_reconst_img[update_indx] = [
                                x + y for x, y in zip((torch.sum(outputs_logits, 0)/outputs_logits.size()[0]), list_sum_softmax_of_reconst_img[update_indx])]

                    self.initial_image.requires_grad_()
                    overall_step += 1

            elif(vis_version == 'V4'):
                # with trange(number_of_image_optimization_steps, unit="iter", desc="Generating template image for current batch V2") as pbar:
                for step_iter in range(number_of_image_optimization_steps):
                    # pbar.set_description(f"Iteration {step_iter+1}")

                    outputs = self.model(self.initial_image)
                    
                    for x in self.get_convouts():
                        assert not(torch.isnan(x).any().item() or torch.isinf(x).any().item()), 'self.get_convouts value is inf or nan while calculating convouts: self.initial_image->'+str(self.initial_image)

                    loss, active_pixel_points, total_pixel_points, non_zero_pixel_points = self.get_loss_value(
                        template_loss_type, class_indx, outputs, class_image,alpha=alpha)

                    if(step_iter == 0 and active_pixel_points is not None):
                        percent_active_pixels = float((
                            active_pixel_points/total_pixel_points)*100)
                        print("active_pixel_points", active_pixel_points)
                        print("total_pixel_points", total_pixel_points)
                        print("Percentage of active pixels:",
                              percent_active_pixels)
                        if(percent_active_pixels > max_percent_active_pixels):
                            max_percent_active_pixels = percent_active_pixels
                        if(percent_active_pixels < min_percent_active_pixels):
                            min_percent_active_pixels = percent_active_pixels
                        average_percent_active_pixels += percent_active_pixels
                        if(is_log_wandb):
                            wandb.log({
                                "active_pixel_points": active_pixel_points, "total_pixel_points": total_pixel_points, "non_zero_pixel_points": non_zero_pixel_points,
                                "Percent_active_pixels": percent_active_pixels, "final_postfix_for_save": final_postfix_for_save
                            }, step=(batch_indx+1))

                    # print("{} Loss: {}".format(template_loss_type, loss))

                    # Backward
                    loss.backward()

                    unnorm_gradients = self.initial_image.grad
                    # std_unnorm_grad = torch.std(unnorm_gradients)
                    norm_grad = torch.norm(unnorm_gradients)
                    # print("torch.norm(unnorm_gradients):",
                    #       norm_grad)
                    # print("std_unnorm_grad:", std_unnorm_grad)
                    # print("Original self.initial_image gradients", gradients)

                    # gradients = unnorm_gradients / first_norm
                    gradients = unnorm_gradients / \
                        norm_grad + 1e-8

                    # print("torch.norm(gradients):",
                    #       torch.norm(gradients))

                    with torch.no_grad():
                        self.initial_image -= v4lr*torch.sign(unnorm_gradients)
                        # x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) 
                        self.initial_image = torch.clamp(
                                self.initial_image, 0, 1)
                        if(step_iter % 100 == 0):
                            print("Loss:{},norm_image:{},norm_grad:{},norm_raw_grad:{}".format(loss.item(), torch.norm(
                                self.initial_image).item(), torch.norm(gradients).item(), norm_grad.item()))

                        # pbar.set_postfix(loss=loss.item(), norm_image=torch.norm(
                            # self.initial_image).item(), norm_grad=torch.norm(gradients).item(), norm_raw_grad=norm_grad.item())

                        self.initial_image = self.initial_image.to(
                            self.device)
                        if(eps is not None):
                            self.initial_image = torch.clamp(self.initial_image, class_image - eps, class_image + eps)
                            self.initial_image = torch.clamp(self.initial_image, 0, 1)
                        assert not(torch.isnan(self.initial_image).any().item() or torch.isinf(
                            self.initial_image).any().item()), 'self.initial_image value is inf or nan while calculating unnorm_gradients'+str(unnorm_gradients)
                        if(not(plot_iteration_interval is None) and step_iter % plot_iteration_interval == 0):
                            update_indx = step_iter // plot_iteration_interval
                            _, outputs_logits, outputs_final, correct = self.get_prediction(
                                self.initial_image, original_label)
                            list_correct_prediction_of_reconst_img[update_indx] += correct
                            list_sum_norm_prediction_of_reconst_img[update_indx] += torch.norm(
                                self.initial_image).item()
                            list_sum_norm_unnorm_gradients[update_indx] += torch.norm(
                                unnorm_gradients).item()
                            list_sum_loss[update_indx] += loss

                            init_image_cpu = self.initial_image.cpu().detach().numpy()
                            if(list_of_optimized_image[update_indx] is None):
                                list_of_optimized_image[update_indx] = init_image_cpu
                            else:
                                list_of_optimized_image[update_indx] += init_image_cpu

                            unnorm_grads_np = unnorm_gradients.cpu().detach().numpy()
                            if(list_of_unnorm_gradients[update_indx] is None):
                                list_of_unnorm_gradients[update_indx] = unnorm_grads_np
                            else:
                                list_of_unnorm_gradients[update_indx] += unnorm_grads_np

                            grads_np = gradients.cpu().detach().numpy()
                            if(list_of_norm_gradients[update_indx] is None):
                                list_of_norm_gradients[update_indx] = grads_np
                            else:
                                list_of_norm_gradients[update_indx] += grads_np

                            list_sum_softmax_of_reconst_img[update_indx] = [
                                x + y for x, y in zip((torch.sum(outputs_logits, 0)/outputs_logits.size()[0]), list_sum_softmax_of_reconst_img[update_indx])]

                    self.initial_image.requires_grad_()
                    overall_step += 1

            else:
                with trange(number_of_image_optimization_steps, unit="iter", desc="Generating template image for current batch") as pbar:
                    for step_iter in pbar:
                        pbar.set_description(f"Iteration {step_iter+1}")
                        outputs = self.model(self.initial_image)
                        loss, active_pixel_points, total_pixel_points, non_zero_pixel_points = self.get_loss_value(
                            template_loss_type, class_indx, outputs, class_image,alpha=alpha)

                        if(step_iter == 0 and active_pixel_points is not None):
                            percent_active_pixels = float((
                                active_pixel_points/total_pixel_points)*100)
                            print("active_pixel_points", active_pixel_points)
                            print("total_pixel_points", total_pixel_points)
                            print("Percentage of active pixels:",
                                  percent_active_pixels)
                            if(percent_active_pixels > max_percent_active_pixels):
                                max_percent_active_pixels = percent_active_pixels
                            if(percent_active_pixels < min_percent_active_pixels):
                                min_percent_active_pixels = percent_active_pixels
                            average_percent_active_pixels += percent_active_pixels
                            if(is_log_wandb):
                                wandb.log(
                                    {"active_pixel_points": active_pixel_points, "total_pixel_points": total_pixel_points, "non_zero_pixel_points": non_zero_pixel_points,
                                     "Percent_active_pixels": percent_active_pixels, "final_postfix_for_save": final_postfix_for_save}, step=(batch_indx+1))

                        print("{} Loss: {}".format(template_loss_type, loss))
                        # Backward
                        loss.backward()

                        # print("self.initial_image", self.initial_image)
                        unnorm_gradients = self.initial_image.grad
                        # print("Original self.initial_image gradients", gradients)

                        gradients = unnorm_gradients / \
                            torch.std(unnorm_gradients) + 1e-8

                        # print("After normalize self.initial_image gradients", gradients)

                        with torch.no_grad():
                            self.initial_image = self.initial_image - gradients*step_size
                            # self.initial_image = 0.9 * self.initial_image
                            if(eps is not None):
                                self.initial_image = torch.clamp(self.initial_image, class_image - eps, class_image + eps)
                                self.initial_image = torch.clamp(self.initial_image, 0, 1)

                            if(not(plot_iteration_interval is None) and step_iter % plot_iteration_interval == 0):
                                update_indx = step_iter // plot_iteration_interval
                                _, outputs_logits, outputs_final, correct = self.get_prediction(
                                    self.initial_image, original_label)
                                list_correct_prediction_of_reconst_img[update_indx] += correct
                                list_sum_norm_prediction_of_reconst_img[update_indx] += torch.norm(
                                    self.initial_image).item()
                                list_sum_norm_unnorm_gradients[update_indx] += torch.norm(
                                    unnorm_gradients).item()
                                list_sum_loss[update_indx] += loss

                                init_image_cpu = self.initial_image.cpu().detach().numpy()
                                if(list_of_optimized_image[update_indx] is None):
                                    list_of_optimized_image[update_indx] = init_image_cpu
                                else:
                                    list_of_optimized_image[update_indx] += init_image_cpu

                                unnorm_grads_np = unnorm_gradients.cpu().detach().numpy()
                                if(list_of_unnorm_gradients[update_indx] is None):
                                    list_of_unnorm_gradients[update_indx] = unnorm_grads_np
                                else:
                                    list_of_unnorm_gradients[update_indx] += unnorm_grads_np

                                grads_np = gradients.cpu().detach().numpy()
                                if(list_of_norm_gradients[update_indx] is None):
                                    list_of_norm_gradients[update_indx] = grads_np
                                else:
                                    list_of_norm_gradients[update_indx] += grads_np

                                list_sum_softmax_of_reconst_img[update_indx] = [
                                    x + y for x, y in zip((torch.sum(outputs_logits, 0)/outputs_logits.size()[0]), list_sum_softmax_of_reconst_img[update_indx])]

                        self.initial_image.requires_grad_()
                        overall_step += 1

            reconst_outputs = self.model(self.initial_image)
            reconst_outputs_softmax = reconst_outputs.softmax(dim=1)
            print("Confidence over Reconstructed image with alpha:", alpha)
            reconst_img_norm = torch.norm(self.initial_image)
            print("Norm of reconstructed image is:", reconst_img_norm)
            for i in range(len(reconst_outputs[0])):
                print("Class {} => {}".format(
                    classes[i], reconst_outputs[0][i]))
            reconst_pred = reconst_outputs_softmax.max(1).indices
            print("Reconstructed image Class predicted:",
                  classes[reconst_pred])

            total += original_label.size()[0]
            temp_var = reconst_pred.eq(original_label).sum().item()
            reconst_correct += temp_var

            image = preprocess_image(
                class_image[0].cpu().clone().detach().numpy(), normalize_image)
            image = image.to(self.device)

            image.requires_grad_()
            original_image_outputs = self.model(image)
            original_image_outputs_softmax = original_image_outputs.softmax(
                dim=1)
            print("Confidence over original image with alpha:", alpha)
            for i in range(len(original_image_outputs_softmax[0])):
                print("Class {} => {}".format(
                    classes[i], original_image_outputs_softmax[0][i]))
            original_image_pred = original_image_outputs_softmax.max(1).indices
            original_correct += original_image_pred.eq(
                original_label).sum().item()
            print("Class predicted on original image was :",
                  classes[original_image_pred])
            print("Original label was:", class_label)

            per_class_per_batch_data_loader.set_postfix(loss=loss.item(),rec_acc=100. * reconst_correct/total,
                                                        orig_acc=100.*original_correct/total, rec_ratio="{}/{}".format(reconst_correct, total), orig_ratio="{}/{}".format(original_correct, total))

            with torch.no_grad():
                init_img_np = self.initial_image.cpu().clone().detach().numpy()
                temp_reconst_pred = reconst_pred.cpu().clone().detach().numpy()
                if(list_of_reconst_images is None):
                    list_of_reconst_images = init_img_np
                else:
                    list_of_reconst_images = np.vstack(
                        (list_of_reconst_images, init_img_np))
                if(list_of_reconst_preds is None):
                    list_of_reconst_preds = temp_reconst_pred
                else:
                    list_of_reconst_preds = np.vstack(
                        (list_of_reconst_preds, temp_reconst_pred))
                is_out_image = False
                # This means it was correctly classified
                if(temp_var > 0):
                    if(num_correct % 150 == 0):
                        is_out_image = True
                    num_correct += 1
                    save_folder = self.image_save_prefix_folder + \
                        "/corr_clssfied/true_class_"+str(class_label)+"/"
                else:
                    if(num_incorrect % 150 == 0):
                        is_out_image = True
                    num_incorrect += 1
                    save_folder = self.image_save_prefix_folder + \
                        "/in_corr_clssfied/true_class_"+str(class_label)+"/"
                if(is_out_image):
                    self.created_image = recreate_image(
                        self.initial_image, normalize_image)
                    model_input_orig_image = recreate_image(
                        image, normalize_image)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    reconst_im_path = save_folder+'/batch_indx_' + \
                        str(batch_indx)+'_no_optimizer_reconst_img_pred_c_' + \
                        str(classes[reconst_pred]) + \
                        'act_pred_'+str(class_label) + '.jpg'
                    input_orig_im_path = save_folder + '/batch_indx_' + \
                        str(batch_indx)+'_minput_orig_img.jpg'

                    numpy_image = self.created_image
                    save_image(numpy_image, reconst_im_path)
                    # save_image(model_input_orig_image, input_orig_im_path)

                    for_vis_dft_amp_out, for_vis_dft_phase_out = get_fft_phase_amp_vis_for_tensor(
                        torch.from_numpy(self.created_image).to(device=self.initial_image.get_device()))
                    reconst_std01_vis_dft_amp_out = recreate_image(
                        for_vis_dft_amp_out, unnormalize=False)
                    reconst_std01_vis_dft_phase_out = recreate_image(
                        for_vis_dft_phase_out, unnormalize=False)

                    reconst_dft_amp_im_path = save_folder + '/batch_indx_' + \
                        str(batch_indx)+'_reconst_dft_amp.jpg'
                    reconst_dft_phase_im_path = save_folder + '/batch_indx_' + \
                        str(batch_indx)+'_reconst_dft_phase.jpg'
                    # save_image(reconst_std01_vis_dft_amp_out,
                    #            reconst_dft_amp_im_path)
                    # save_image(reconst_std01_vis_dft_phase_out,
                    #            reconst_dft_phase_im_path)

                    for_vis_dft_amp_out, for_vis_dft_phase_out = get_fft_phase_amp_vis_for_tensor(
                        image)
                    orig_std01_vis_dft_amp_out = recreate_image(
                        for_vis_dft_amp_out, unnormalize=False)
                    orig_std01_vis_dft_phase_out = recreate_image(
                        for_vis_dft_phase_out, unnormalize=False)
                    orig_dft_amp_im_path = save_folder + '/batch_indx_' + \
                        str(batch_indx)+'_orig_dft_amp.jpg'
                    orig_dft_phase_im_path = save_folder + '/batch_indx_' + \
                        str(batch_indx)+'_orig_dft_phase.jpg'
                    # save_image(orig_std01_vis_dft_amp_out,
                    #            orig_dft_amp_im_path)
                    # save_image(orig_std01_vis_dft_phase_out,
                    #            orig_dft_phase_im_path)

            if(is_log_wandb):
                wandb.log(
                    {"batch_indx": batch_indx,
                     "original_image_outputs_softmax": original_image_outputs_softmax, "original_img_label_pred": classes[original_image_pred], "original_img_pred_indx": original_image_pred,
                     }, step=(batch_indx+1))

        average_percent_active_pixels = average_percent_active_pixels / batch_count
        final_original_accuracy = (100. * original_correct/total)
        final_accuracy = (100. * reconst_correct/total)
        print("Overall class accuracy over original images:",
              final_original_accuracy)
        print("Overall class accuracy over template images:",
              final_accuracy)

        if(not(plot_iteration_interval is None)):
            number_of_intervals = (
                number_of_image_optimization_steps//plot_iteration_interval)+1

            list_avg_softmax_of_reconst_img = [None] * number_of_intervals

            for indx in range(number_of_intervals):
                list_avg_softmax_of_reconst_img[indx] = [
                    x/total for x in list_sum_softmax_of_reconst_img[indx]]

                if(is_log_wandb):
                    current_avg_reconst_accuracy = 100. * \
                        list_correct_prediction_of_reconst_img[indx] / total
                    current_avg_reconst_img = list_sum_norm_prediction_of_reconst_img[indx] / total
                    current_avg_norm_unnorm_grads = list_sum_norm_unnorm_gradients[indx] / total
                    current_avg_optimized_image = list_of_optimized_image[indx] / total
                    current_avg_unnorm_grads = list_of_unnorm_gradients[indx] / total
                    current_avg_norm_grads = list_of_norm_gradients[indx] / total
                    current_avg_loss = list_sum_loss[indx] / total

                    wandb.log({
                        "avg_optimized_Image": wandb.Histogram(current_avg_optimized_image), "avg_norm_optimized_image": current_avg_reconst_img,
                        "avg_normalized_gradients": wandb.Histogram(current_avg_norm_grads), "avg_unnormalized_gradients": wandb.Histogram(current_avg_unnorm_grads),
                        "avg_loss": current_avg_loss, "avg_norm_unnormalized_grad": current_avg_norm_unnorm_grads, "avg_acc_reconst_img": current_avg_reconst_accuracy}, step=(total+indx))

            if(is_log_wandb):
                # accuracies_of_reconst_img_data = [[(plot_iteration_interval * ind), list_of_accuracies_of_reconst_img[ind]]
                #                                   for ind in range(number_of_intervals)]

                # accuracies_of_reconst_img_table = wandb.Table(
                #     data=accuracies_of_reconst_img_data, columns=["Optimization_iteration", "Average_Accuracies_Reconstructed_Images"])

                # avg_norm_of_reconst_img_data = [[(plot_iteration_interval * ind), list_of_avg_norm_of_reconst_img[ind]]
                #                                 for ind in range(number_of_intervals)]
                # avg_norm_of_reconst_img_table = wandb.Table(
                #     data=avg_norm_of_reconst_img_data, columns=["Optimization_iteration", "Average_Norm_Reconstructed_Images"])

                step_lists = [(plot_iteration_interval * indx)
                              for indx in range(number_of_intervals)]
                each_class_softmax_ordered_by_steps = [
                    None] * num_classes

                for c_ind in range(num_classes):
                    current_class_softmax_list = []
                    for indx in range(number_of_intervals):
                        current_class_softmax_list.append(
                            list_avg_softmax_of_reconst_img[indx][c_ind])
                    each_class_softmax_ordered_by_steps[c_ind] = current_class_softmax_list

                wandb.log({"final_acc_ov_reconst": final_accuracy, "final_acc_ov_orig": final_original_accuracy,
                           "average_percent_active_pixels": average_percent_active_pixels,
                          "min_percent_active_pixels": min_percent_active_pixels,
                           "max_percent_active_pixels": max_percent_active_pixels,
                           "softmax_reconst_img_opt_steps_plt": wandb.plot.line_series(xs=step_lists,
                                                                                       ys=each_class_softmax_ordered_by_steps,
                                                                                       keys=classes,
                                                                                       title="Variation of softmax values across classes vs Optimization steps",
                                                                                       xname="Optimization steps")
                           })

        else:
            if(is_log_wandb):
                wandb.log({"final_acc_ov_reconst": final_accuracy,
                          "final_acc_ov_orig": final_original_accuracy,
                           "average_percent_active_pixels": average_percent_active_pixels,
                           "min_percent_active_pixels": min_percent_active_pixels,
                           "max_percent_active_pixels": max_percent_active_pixels})

        if(is_log_wandb):
            wandb.finish()
        print("Reconstructed images written at:",
              self.image_save_prefix_folder)
        current_y_s = np.full(
            list_of_reconst_images.shape[0], original_label[0].cpu().clone().detach().numpy())

        with open(np_save_filename, 'wb') as file:
            np.savez(
                file, x=list_of_reconst_images, y=current_y_s, y_pred=list_of_reconst_preds)
        return list_of_reconst_images

    def generate_projection_data_and_accuracies_per_class(self,epsilon, per_class_dataset, class_label, class_indx, classes, model_arch_type, dataset, is_template_image_on_train,
                                                        is_class_segregation_on_ground_truth, template_initial_image_type,
                                                        template_image_calculation_batch_size, template_loss_type, wand_project_name, wandb_group_name, torch_seed, number_of_image_optimization_steps,
                                                        exp_type, collect_threshold, entropy_calculation_batch_size, number_of_batches_to_calculate_entropy_on,
                                                        plot_iteration_interval=None, root_save_prefix="root", final_postfix_for_save="", wandb_config_additional_dict=None, vis_version='V2', random_sample_gate_percent=None,alpha = 0):
        random_sample_gate_percent_str = ""
        list_of_reconst_images = None
        list_of_reconst_preds = None
        is_log_wandb = not(wand_project_name is None)

        # torch.manual_seed(torch_seed)
        coll_seed_gen = torch.Generator()
        coll_seed_gen.manual_seed(torch_seed)

        if(random_sample_gate_percent is not None):
            random_sample_gate_percent_str = "RD_GT_PER_LAY_SAMPLE_" + \
                str(random_sample_gate_percent)

        per_class_per_batch_data_loader = torch.utils.data.DataLoader(per_class_dataset, batch_size=template_image_calculation_batch_size,
                                                                      shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)
        tmp_image_over_what_str = 'test'
        if(is_template_image_on_train):
            tmp_image_over_what_str = 'train'

        seg_over_what_str = 'MP'
        if(is_class_segregation_on_ground_truth):
            seg_over_what_str = 'GT'

        layer_nums_str = ""
        if(self.layer_nums_to_visualize is not None):
            layer_nums_str = "/LV_"
            for i in self.layer_nums_to_visualize:
                layer_nums_str += str(i)+" "
            layer_nums_str = layer_nums_str.strip()
            layer_nums_str = layer_nums_str.replace(" ", "_")

        self.model.train(False)
        self.image_save_prefix_folder = str(root_save_prefix)+"/"+str(dataset)+"/MT_"+str(model_arch_type)+"_ET_"+str(exp_type)+"/Ver_"+str(vis_version)+"/_COLL_OV_"+str(tmp_image_over_what_str)+"/SEG_"+str(
            seg_over_what_str)+"/TMP_COLL_BS_"+str(template_image_calculation_batch_size)+"_NO_TO_COLL_"+str(number_of_batch_to_collect)+"/TMP_LOSS_TP_"+str(template_loss_type)+"/TMP_INIT_"+str(template_initial_image_type)+"/_torch_seed_"+str(torch_seed)+"_c_thres_"+str(collect_threshold)+"/"+random_sample_gate_percent_str+str(layer_nums_str)+"/" + str(final_postfix_for_save) + "/"
        self.image_file_save_pref = str(root_save_prefix)+"/"+str(dataset)+"/MT_"+str(model_arch_type)+"_ET_GENERATE_PROJECTED/EPS_"+str(epsilon)+"/Ver_"+str(vis_version)+"/_COLL_OV_"+str(tmp_image_over_what_str)+"/SEG_"+str(
            seg_over_what_str)+"/TMP_COLL_BS_"+str(template_image_calculation_batch_size)+"_NO_TO_COLL_"+str(number_of_batch_to_collect)+"/TMP_LOSS_TP_"+str(template_loss_type)+"/TMP_INIT_"+str(template_initial_image_type)+"/_torch_seed_"+str(torch_seed)+"_c_thres_"+str(collect_threshold)+"/"+random_sample_gate_percent_str+str(layer_nums_str)+"/" + str(final_postfix_for_save) + "/"

        per_class_per_batch_data_loader = tqdm(
            per_class_per_batch_data_loader, desc='Image being processed:'+str(class_label))
        total = 0
        reconst_correct = 0
        original_correct = 0
        if(alpha != 0):
            self.image_save_prefix_folder += "_alp_" + str(alpha)+"/"
            self.image_file_save_pref += "_alp_" + str(alpha)+"/"
        normalize_image = False

        np_save_filename = self.image_save_prefix_folder + \
            '/class_'+str(class_label) + '.npy'
        is_current_aug_available = os.path.exists(np_save_filename)
        assert is_current_aug_available==True, 'Reconstructed images not found at:'+str(np_save_filename)
        if(is_current_aug_available):
            print("Reading reconstruction images from:{}".format(np_save_filename))
            with open(np_save_filename, 'rb') as file:
                npzfile = np.load(np_save_filename)
                each_class_output_template_list = npzfile['x']
        
        print("each_class_output_template_list:",each_class_output_template_list.shape)
        per_class_reconstructed_batch_data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(each_class_output_template_list)), batch_size=template_image_calculation_batch_size,shuffle=False)

        if(vis_version == 'V2' or vis_version == 'V3' or vis_version == 'V4'):
            if('conv4_deep_gated_net' in model_arch_type):
                number_of_image_optimization_steps = 301
            else:
                # number_of_image_optimization_steps = 161
                pass

            if(vis_version == 'V2' or vis_version == 'V4'):
                start_sigma = 0.75
                end_sigma = 0.1
            elif(vis_version == 'V3'):
                start_sigma = 0.1
                end_sigma = 0.75

            if('conv4_deep_gated_net' in model_arch_type):
                start_step_size = 1
                end_step_size = 0.5
            else:
                start_step_size = 0.1
                end_step_size = 0.05

        if(is_log_wandb):
            wandb_run_name = self.image_file_save_pref.replace(
                "/", "").replace(root_save_prefix, class_label)
            wandb_config = self.get_wandb_config("GENERATE_PROJECTED", class_label, class_indx, classes, model_arch_type, dataset, is_template_image_on_train,
                                                 is_class_segregation_on_ground_truth, template_initial_image_type,
                                                 template_image_calculation_batch_size, template_loss_type, torch_seed, number_of_image_optimization_steps,
                                                 plot_iteration_interval, collect_threshold=collect_threshold)
            wandb_config["vis_version"] = vis_version
            wandb_config["p_eps"]=epsilon
            if(vis_version == 'V2' or vis_version == 'V3'):
                wandb_config["start_sigma"] = start_sigma
                wandb_config["end_sigma"] = end_sigma
                wandb_config["start_step_size"] = start_step_size
                wandb_config["end_step_size"] = end_step_size
                wandb_config["per_layer_random_sample_gate_percent"] = random_sample_gate_percent

            if(wandb_config_additional_dict is not None):
                wandb_config.update(wandb_config_additional_dict)
            wandb_config["alpha"] = alpha

            wandb.init(
                project=f"{wand_project_name}",
                name=f"{wandb_run_name}",
                group=f"{wandb_group_name}",
                config=wandb_config,
            )

        if("ENTR" in template_loss_type):
            self.image_file_save_pref += "_ENT_BS_" + \
                str(entropy_calculation_batch_size)+"_ENTR_COLL_" + \
                str(number_of_batches_to_calculate_entropy_on)+"/"

        batch_count = 0
        num_correct = 0
        num_incorrect = 0
        for batch_indx, (per_class_data,reconstructed_data) in enumerate(zip(per_class_per_batch_data_loader,per_class_reconstructed_batch_data_loader)):
            batch_count += 1
            torch.cuda.empty_cache()

            class_image, original_label = per_class_data
            class_image = class_image.to(self.device, non_blocking=True)
            original_label = original_label.to(self.device, non_blocking=True)
            reconstructed_data = reconstructed_data[0].to(self.device, non_blocking=True)

            assert reconstructed_data.size()==class_image.size(),'class_image size:'+str(class_image.size())+" reconstructed_data size:"+str(reconstructed_data.size())

            self.initial_image = torch.clamp(reconstructed_data, class_image - epsilon, class_image + epsilon)
            self.initial_image = torch.clamp(self.initial_image, 0, 1)
            
            # During reconstruction projection, the reconstructed image acts like original example
            class_image = reconstructed_data

            reconst_outputs = self.model(self.initial_image)
            reconst_outputs_softmax = reconst_outputs.softmax(dim=1)
            print("Confidence over Reconstructed image with alpha:", alpha)
            reconst_img_norm = torch.norm(self.initial_image)
            print("Norm of reconstructed image is:", reconst_img_norm)
            for i in range(len(reconst_outputs[0])):
                print("Class {} => {}".format(
                    classes[i], reconst_outputs[0][i]))
            reconst_pred = reconst_outputs_softmax.max(1).indices
            print("Reconstructed image Class predicted:",
                  classes[reconst_pred])

            total += original_label.size()[0]
            temp_var = reconst_pred.eq(original_label).sum().item()
            reconst_correct += temp_var

            image = preprocess_image(
                class_image[0].cpu().clone().detach().numpy(), normalize_image)
            image = image.to(self.device)

            image.requires_grad_()
            original_image_outputs = self.model(image)
            original_image_outputs_softmax = original_image_outputs.softmax(
                dim=1)
            print("Confidence over original image with alpha:", alpha)
            for i in range(len(original_image_outputs_softmax[0])):
                print("Class {} => {}".format(
                    classes[i], original_image_outputs_softmax[0][i]))
            original_image_pred = original_image_outputs_softmax.max(1).indices
            original_correct += original_image_pred.eq(
                original_label).sum().item()
            print("Class predicted on original image was :",
                  classes[original_image_pred])
            print("Original label was:", class_label)

            per_class_per_batch_data_loader.set_postfix(rec_acc=100. * reconst_correct/total,
                                                        orig_acc=100.*original_correct/total, rec_ratio="{}/{}".format(reconst_correct, total), orig_ratio="{}/{}".format(original_correct, total))

            with torch.no_grad():
                init_img_np = self.initial_image.cpu().clone().detach().numpy()
                temp_reconst_pred = reconst_pred.cpu().clone().detach().numpy()
                if(list_of_reconst_images is None):
                    list_of_reconst_images = init_img_np
                else:
                    list_of_reconst_images = np.vstack(
                        (list_of_reconst_images, init_img_np))
                if(list_of_reconst_preds is None):
                    list_of_reconst_preds = temp_reconst_pred
                else:
                    list_of_reconst_preds = np.vstack(
                        (list_of_reconst_preds, temp_reconst_pred))
                is_out_image = False
                # This means it was correctly classified
                if(temp_var > 0):
                    if(num_correct % 250 == 0):
                        is_out_image = True
                    num_correct += 1
                    save_folder = self.image_file_save_pref + \
                        "/corr_clssfied/true_class_"+str(class_label)+"/"
                else:
                    if(num_incorrect % 250 == 0):
                        is_out_image = True
                    num_incorrect += 1
                    save_folder = self.image_file_save_pref + \
                        "/in_corr_clssfied/true_class_"+str(class_label)+"/"
                if(is_out_image):
                    self.created_image = recreate_image(
                        self.initial_image, normalize_image)
                    model_input_orig_image = recreate_image(
                        image, normalize_image)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    reconst_im_path = save_folder+'/batch_indx_' + \
                        str(batch_indx)+'_no_optimizer_reconst_img_pred_c_' + \
                        str(classes[reconst_pred]) + \
                        'act_pred_'+str(class_label) + '.jpg'
                    input_orig_im_path = save_folder + '/batch_indx_' + \
                        str(batch_indx)+'_minput_orig_img.jpg'

                    numpy_image = self.created_image
                    save_image(numpy_image, reconst_im_path)
                    save_image(model_input_orig_image, input_orig_im_path)

                    input_orig_im_path = save_folder + '/batch_indx_' + \
                        str(batch_indx)+'_root_orig_img.jpg'
                    save_image(recreate_image(
                        per_class_data[0].cpu().detach().numpy(), normalize_image), input_orig_im_path)

                    for_vis_dft_amp_out, for_vis_dft_phase_out = get_fft_phase_amp_vis_for_tensor(
                        torch.from_numpy(self.created_image).to(device=self.initial_image.get_device()))
                    reconst_std01_vis_dft_amp_out = recreate_image(
                        for_vis_dft_amp_out, unnormalize=False)
                    reconst_std01_vis_dft_phase_out = recreate_image(
                        for_vis_dft_phase_out, unnormalize=False)

                    reconst_dft_amp_im_path = save_folder + '/batch_indx_' + \
                        str(batch_indx)+'_reconst_dft_amp.jpg'
                    reconst_dft_phase_im_path = save_folder + '/batch_indx_' + \
                        str(batch_indx)+'_reconst_dft_phase.jpg'
                    # save_image(reconst_std01_vis_dft_amp_out,
                    #            reconst_dft_amp_im_path)
                    # save_image(reconst_std01_vis_dft_phase_out,
                    #            reconst_dft_phase_im_path)

                    for_vis_dft_amp_out, for_vis_dft_phase_out = get_fft_phase_amp_vis_for_tensor(
                        image)
                    orig_std01_vis_dft_amp_out = recreate_image(
                        for_vis_dft_amp_out, unnormalize=False)
                    orig_std01_vis_dft_phase_out = recreate_image(
                        for_vis_dft_phase_out, unnormalize=False)
                    orig_dft_amp_im_path = save_folder + '/batch_indx_' + \
                        str(batch_indx)+'_orig_dft_amp.jpg'
                    orig_dft_phase_im_path = save_folder + '/batch_indx_' + \
                        str(batch_indx)+'_orig_dft_phase.jpg'
                    # save_image(orig_std01_vis_dft_amp_out,
                    #            orig_dft_amp_im_path)
                    # save_image(orig_std01_vis_dft_phase_out,
                    #            orig_dft_phase_im_path)

            if(is_log_wandb):
                wandb.log(
                    {"batch_indx": batch_indx,
                     "original_image_outputs_softmax": original_image_outputs_softmax, "original_img_label_pred": classes[original_image_pred], "original_img_pred_indx": original_image_pred,
                     }, step=(batch_indx+1))

        final_original_accuracy = (100. * original_correct/total)
        final_accuracy = (100. * reconst_correct/total)
        print("Overall class accuracy over original images:",
              final_original_accuracy)
        print("Overall class accuracy over template images:",
              final_accuracy)

        if(is_log_wandb):
            wandb.log({"final_acc_ov_reconst": final_accuracy,
                        "final_acc_ov_orig": final_original_accuracy})
            wandb.finish()
        print("Reconstructed images written at:",
              self.image_file_save_pref)
        current_y_s = np.full(
            list_of_reconst_images.shape[0], original_label[0].cpu().clone().detach().numpy())

        # np_file_save_filename = self.image_file_save_pref + \
        #     '/class_'+str(class_label) + '.npy'
        # is_current_aug_available = os.path.exists(np_file_save_filename)
        # with open(np_file_save_filename, 'wb') as file:
        #     np.savez(
        #         file, x=list_of_reconst_images, y=current_y_s, y_pred=list_of_reconst_preds)
        return list_of_reconst_images


    def generate_template_image_over_given_image(self, model_arch_type, image_to_collect_upon, number_of_image_optimization_steps, template_loss_type, vis_version='V2',alpha=0):
        self.initial_image = preprocess_image(
            self.original_image.cpu().clone().detach().numpy(), normalize=False)
        self.initial_image = self.initial_image.to(self.device)
        self.initial_image.requires_grad_()

        image_to_collect_upon = torch.unsqueeze(image_to_collect_upon, 0)
        image_to_collect_upon = image_to_collect_upon.to(self.device)
        per_class_data = image_to_collect_upon, None
        self.reset_collection_state()
        self.collect_active_pixel_per_batch(
            per_class_data)
        # Collect threshold doesn't matter since collection it is over a single image
        self.update_overall_y_maps(collect_threshold=0.95)

        step_size = 0.01
        eps = None
        if("PLATTCK" in template_loss_type):
            eps = float(template_loss_type.split("__")[1])
        if(vis_version == "V2" or vis_version == 'V3' or vis_version == 'V4'):
            if('conv4_deep_gated_net' in model_arch_type):
                number_of_image_optimization_steps = 301
            else:
                # number_of_image_optimization_steps = 161
                pass

            if(vis_version == 'V2'):
                start_sigma = 0.75
                end_sigma = 0.1
            elif(vis_version == 'V3'):
                start_sigma = 0.1
                end_sigma = 0.75
            if('conv4_deep_gated_net' in model_arch_type):
                start_step_size = 1
                end_step_size = 0.5
            else:
                start_step_size = 0.1
                end_step_size = 0.05
            if(vis_version == "V2" or vis_version == 'V3'):
                with trange(number_of_image_optimization_steps, unit="iter", desc="Generating template image for given image "+vis_version) as pbar:
                    for step_iter in pbar:
                        pbar.set_description(f"Iteration {step_iter+1}")

                        step_size = start_step_size + \
                            ((end_step_size - start_step_size) * step_iter) / \
                            number_of_image_optimization_steps
                        # optimizer = torch.optim.SGD([img_var], lr=step_size)
                        sigma = start_sigma + \
                            ((end_sigma - start_sigma) * step_iter) / \
                            number_of_image_optimization_steps

                        outputs = self.model(self.initial_image)

                        verbose = 1
                        loss, active_pixel_points, total_pixel_points, non_zero_pixel_points = self.get_loss_value(
                            template_loss_type, class_indx=0, outputs=outputs, verbose=verbose,alpha=alpha)

                        # print("{} Loss: {}".format(template_loss_type, loss))
                        if(step_iter == 0 and active_pixel_points is not None):
                            percent_active_pixels = float((
                                active_pixel_points/total_pixel_points)*100)
                            print("active_pixel_points", active_pixel_points)
                            print("total_pixel_points", total_pixel_points)
                            print("Percentage of active pixels:",
                                percent_active_pixels)

                        # Backward
                        loss.backward()

                        unnorm_gradients = self.initial_image.grad
                        # std_unnorm_grad = torch.std(unnorm_gradients)
                        norm_grad = torch.norm(unnorm_gradients)

                        # gradients = unnorm_gradients / first_norm
                        gradients = unnorm_gradients / \
                            norm_grad + 1e-8

                        with torch.no_grad():
                            # self.initial_image = self.initial_image - gradients*step_size
                            blurred_grad = gradients.cpu().detach().numpy()[0]
                            blurred_grad = blur_img(
                                blurred_grad, sigma)
                            self.initial_image = self.initial_image.cpu().detach().numpy()[
                                0]
                            self.initial_image -= step_size / \
                                np.abs(blurred_grad).mean() * blurred_grad

                            self.initial_image = blur_img(
                                self.initial_image, sigma)
                            self.initial_image = torch.from_numpy(
                                self.initial_image[None])

                        self.initial_image = self.initial_image.to(self.device)
                        if(eps is not None):
                            self.initial_image = torch.clamp(self.initial_image, image_to_collect_upon - eps, image_to_collect_upon + eps)
                            self.initial_image = torch.clamp(self.initial_image, 0, 1)
                        self.initial_image.requires_grad_()

                        pbar.set_postfix(loss=loss.item(), norm_image=torch.norm(
                            self.initial_image).item(), norm_grad=torch.norm(gradients).item(), norm_raw_grad=norm_grad.item())
            else:
                with trange(number_of_image_optimization_steps, unit="iter", desc="Generating template image for given image "+vis_version) as pbar:
                    for step_iter in pbar:
                        pbar.set_description(f"Iteration {step_iter+1}")

                        verbose = 1
                        loss, active_pixel_points, total_pixel_points, non_zero_pixel_points = self.get_loss_value(
                            template_loss_type, class_indx=0, outputs=outputs, verbose=verbose,alpha=alpha)

                        # print("{} Loss: {}".format(template_loss_type, loss))
                        if(step_iter == 0 and active_pixel_points is not None):
                            percent_active_pixels = float((
                                active_pixel_points/total_pixel_points)*100)
                            print("active_pixel_points", active_pixel_points)
                            print("total_pixel_points", total_pixel_points)
                            print("Percentage of active pixels:",
                                percent_active_pixels)

                        # Backward
                        loss.backward()

                        unnorm_gradients = self.initial_image.grad
                        with torch.no_grad():
                            self.initial_image -= 0.1*torch.sign(unnorm_gradients)
                            if(eps is not None):
                                self.initial_image = torch.clamp(self.initial_image, image_to_collect_upon - eps, image_to_collect_upon + eps)
                            self.initial_image = torch.clamp(
                                self.initial_image, 0, 1)

                        self.initial_image.requires_grad_()

                        pbar.set_postfix(loss=loss.item(), norm_image=torch.norm(
                            self.initial_image).item())
        else:
            with trange(number_of_image_optimization_steps, unit="iter", desc="Generating template image for given image") as pbar:
                for step_iter in pbar:
                    pbar.set_description(f"Iteration {step_iter+1}")

                    outputs = self.model(self.initial_image)

                    loss, active_pixel_points, total_pixel_points, non_zero_pixel_points = self.get_loss_value(
                        template_loss_type, class_indx=0, outputs=outputs,alpha=alpha)

                    if(step_iter == 0 and active_pixel_points is not None):
                        percent_active_pixels = float((
                            active_pixel_points/total_pixel_points)*100)
                        print("active_pixel_points", active_pixel_points)
                        print("total_pixel_points", total_pixel_points)
                        print("Percentage of active pixels:",
                              percent_active_pixels)

                    # print("loss", loss)
                    # Backward
                    loss.backward()

                    unnorm_gradients = self.initial_image.grad
                    # print("Original self.initial_image gradients", gradients)

                    gradients = unnorm_gradients / \
                        torch.std(unnorm_gradients) + 1e-8

                    # print("After normalize self.initial_image gradients", gradients)

                    with torch.no_grad():
                        self.initial_image = self.initial_image - gradients*step_size
                        # self.initial_image = 0.9 * self.initial_image
                        if(eps is not None):
                            self.initial_image = torch.clamp(self.initial_image, image_to_collect_upon - eps, image_to_collect_upon + eps)
                        self.initial_image = torch.clamp(self.initial_image, 0, 1)

                    self.initial_image.requires_grad_()

        return self.initial_image

    def generate_layer_wise_template_image_per_class(self, exp_type, per_class_dataset, class_label, class_indx, number_of_batch_to_collect, classes, model_arch_type, dataset, is_template_image_on_train,
                                                     is_class_segregation_on_ground_truth, template_initial_image_type,
                                                     template_image_calculation_batch_size, template_loss_type, wand_project_name, wandb_group_name, torch_seed, number_of_image_optimization_steps,
                                                     layer_nums_to_visualize, root_save_prefix="root", final_postfix_for_save="",
                                                     wandb_config_additional_dict=None, vis_version='V2', random_sample_gate_percent=None):
        coll_gate_sample_seed_gen = None
        random_sample_gate_percent_str = ""
        is_log_wandb = not(wand_project_name is None)
        plot_iteration_interval = 25

        # torch.manual_seed(torch_seed)
        coll_seed_gen = torch.Generator()
        coll_seed_gen.manual_seed(torch_seed)

        entr_seed_gen = torch.Generator()
        entr_seed_gen.manual_seed(torch_seed)

        if(random_sample_gate_percent is not None):
            random_sample_gate_percent_str = "RD_GT_PER_LAY_SAMPLE_" + \
                str(random_sample_gate_percent)
            coll_gate_sample_seed_gen = torch.Generator()
            coll_gate_sample_seed_gen.manual_seed(torch_seed)

        self.model.train(False)
        per_class_data_loader = torch.utils.data.DataLoader(per_class_dataset, batch_size=template_image_calculation_batch_size,
                                                            shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)

        tmp_image_over_what_str = 'test'
        if(is_template_image_on_train):
            tmp_image_over_what_str = 'train'

        seg_over_what_str = 'MP'
        if(is_class_segregation_on_ground_truth):
            seg_over_what_str = 'GT'

        alpha = 0
        normalize_image = False

        self.collect_all_layer_wise_states(
            per_class_data_loader, class_label, number_of_batch_to_collect, layer_nums_to_visualize)
        print("self.total_lay_coll_img_count:", self.total_lay_coll_img_count)
        class_image, _ = next(iter(per_class_data_loader))
        class_image = class_image.to(self.device, non_blocking=True)
        if(vis_version == 'V2' or vis_version == 'V3'):
            if('conv4_deep_gated_net' in model_arch_type):
                number_of_image_optimization_steps = 301
            else:
                # number_of_image_optimization_steps = 161
                pass
            if(vis_version == 'V2'):
                start_sigma = 0.75
                end_sigma = 0.1
            elif(vis_version == 'V3'):
                start_sigma = 0.1
                end_sigma = 0.75
            if('conv4_deep_gated_net' in model_arch_type):
                start_step_size = 1
                end_step_size = 0.5
            else:
                start_step_size = 0.1
                end_step_size = 0.05

        if(layer_nums_to_visualize is None):
            layer_nums_to_visualize = []
            self.model(class_image)
            if(isinstance(self.model, torch.nn.DataParallel)):
                conv_outs = self.model.module.linear_conv_outputs
            else:
                conv_outs = self.model.linear_conv_outputs

            for i in range(len(conv_outs)):
                layer_nums_to_visualize.append(i)

        for current_layer_num in layer_nums_to_visualize:
            self.image_save_prefix_folder = str(root_save_prefix)+"/"+str(dataset)+"/MT_"+str(model_arch_type)+"_ET_"+str(exp_type)+"/Ver_"+str(vis_version)+"/_COLL_OV_"+str(tmp_image_over_what_str)+"/SEG_"+str(
                seg_over_what_str)+"/TMP_COLL_BS_"+str(template_image_calculation_batch_size)+"_NO_TO_COLL_"+str(number_of_batch_to_collect)+"/TMP_LOSS_TP_"+str(template_loss_type)+"/TMP_INIT_"+str(template_initial_image_type)+"/_torch_seed_"+str(torch_seed)+"/"+random_sample_gate_percent_str+"/" + str(final_postfix_for_save) + "/"

            self.image_save_prefix_folder += "/Layer_" + \
                str(current_layer_num)+"/"

            print(
                "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Layer", current_layer_num)

            num_classes = len(classes)
            number_of_intervals = (
                number_of_image_optimization_steps // plot_iteration_interval) + 1
            list_of_reconst_softmax_pred = [None] * number_of_intervals

            if(is_log_wandb):
                wandb_run_name = self.image_save_prefix_folder.replace(
                    "/", "").replace(root_save_prefix, class_label)
                wandb_config = self.get_wandb_config(exp_type, class_label, class_indx, classes, model_arch_type, dataset, is_template_image_on_train,
                                                     is_class_segregation_on_ground_truth, template_initial_image_type,
                                                     template_image_calculation_batch_size, template_loss_type, torch_seed, number_of_image_optimization_steps,
                                                     number_of_batch_to_collect=number_of_batch_to_collect)
                if(vis_version == 'V2' or vis_version == 'V3'):
                    wandb_config["vis_version"] = vis_version
                    wandb_config["start_sigma"] = start_sigma
                    wandb_config["end_sigma"] = end_sigma
                    wandb_config["start_step_size"] = start_step_size
                    wandb_config["end_step_size"] = end_step_size
                    wandb_config["per_layer_random_sample_gate_percent"] = random_sample_gate_percent
                    wandb_config["current_layer_num"] = current_layer_num
                if(wandb_config_additional_dict is not None):
                    wandb_config.update(wandb_config_additional_dict)
                wandb_config["alpha"] = alpha

                wandb.init(
                    project=f"{wand_project_name}",
                    name=f"{wandb_run_name}",
                    group=f"{wandb_group_name}",
                    config=wandb_config,
                )

            self.initial_image = preprocess_image(
                self.original_image.cpu().clone().detach().numpy(), normalize_image)

            self.initial_image = self.initial_image.to(self.device)

            self.initial_image.requires_grad_()

            step_size = 0.01
            if(vis_version == 'V2' or vis_version == 'V3'):
                with trange(number_of_image_optimization_steps, unit="iter", desc="Generating template image V2") as pbar:
                    for step_iter in pbar:
                        begin_time = time.time()
                        pbar.set_description(f"Iteration {step_iter+1}")

                        step_size = start_step_size + \
                            ((end_step_size - start_step_size) * step_iter) / \
                            number_of_image_optimization_steps
                        # optimizer = torch.optim.SGD([img_var], lr=step_size)
                        sigma = start_sigma + \
                            ((end_sigma - start_sigma) * step_iter) / \
                            number_of_image_optimization_steps

                        outputs = self.model(self.initial_image)

                        verbose = 1
                        loss, active_pixel_points, total_pixel_points, non_zero_pixel_points = self.get_layer_loss_value(
                            template_loss_type, class_indx, current_layer_num, outputs, class_image, alpha, verbose)

                        # print("{} Loss: {}".format(template_loss_type, loss))
                        if(step_iter == 0 and active_pixel_points is not None):
                            percent_active_pixels = float((
                                active_pixel_points/total_pixel_points)*100)
                            print("active_pixel_points", active_pixel_points)
                            print("total_pixel_points", total_pixel_points)
                            print("Percentage of active pixels:",
                                  percent_active_pixels)
                            if(is_log_wandb):
                                wandb.log(
                                    {"active_pixel_points": active_pixel_points, "total_pixel_points": total_pixel_points,
                                     "Percent_active_pixels": percent_active_pixels, "non_zero_pixel_points": non_zero_pixel_points,
                                     "final_postfix_for_save": final_postfix_for_save}, step=(step_iter+1))

                        # Backward
                        loss.backward()

                        unnorm_gradients = self.initial_image.grad
                        # std_unnorm_grad = torch.std(unnorm_gradients)
                        norm_grad = torch.norm(unnorm_gradients)
                        # print("torch.norm(unnorm_gradients):",
                        #       norm_grad)
                        # print("std_unnorm_grad:", std_unnorm_grad)
                        # print("Original self.initial_image gradients", gradients)

                        # gradients = unnorm_gradients / first_norm
                        gradients = unnorm_gradients / \
                            norm_grad + 1e-8

                        # print("torch.norm(gradients):",
                        #       torch.norm(gradients))

                        with torch.no_grad():
                            # self.initial_image = self.initial_image - gradients*step_size
                            blurred_grad = gradients.cpu().detach().numpy()[0]
                            blurred_grad = blur_img(
                                blurred_grad, sigma)
                            self.initial_image = self.initial_image.cpu().detach().numpy()[
                                0]
                            self.initial_image -= step_size / \
                                np.abs(blurred_grad).mean() * blurred_grad

                            # optimizer.step()
                            # print("sigma:", sigma)

                            self.initial_image = blur_img(
                                self.initial_image, sigma)
                            self.initial_image = torch.from_numpy(
                                self.initial_image[None])

                        self.initial_image = self.initial_image.to(self.device)
                        self.initial_image.requires_grad_()

                        if(plot_iteration_interval is not None and step_iter % plot_iteration_interval == 0):
                            with torch.no_grad():
                                reconst_outputs = self.model(
                                    self.initial_image)
                                reconst_outputs_softmax = reconst_outputs.softmax(
                                    dim=1)
                                reconst_img_norm = torch.norm(
                                    self.initial_image)
                                reconst_pred = reconst_outputs_softmax.max(
                                    1).indices

                                update_indx = step_iter // plot_iteration_interval
                                list_of_reconst_softmax_pred[update_indx] = reconst_outputs_softmax[0]

                                if(is_log_wandb):
                                    wandb.log(
                                        {"reconst_img_pred_indx":  reconst_pred, "reconst_img_norm": reconst_img_norm,
                                         "normalized_gradients": wandb.Histogram(gradients.cpu().detach().numpy()),
                                         "unnormalized_gradients": wandb.Histogram(unnorm_gradients.cpu().detach().numpy()),
                                         "reconst_img_label_pred": classes[reconst_pred], "loss": loss, "optimizing_img_gradient_norm": torch.norm(unnorm_gradients).item()}, step=(step_iter+1))

                                self.created_image = recreate_image(
                                    self.initial_image, normalize_image)
                                print("self.created_image.shape::",
                                      self.created_image.shape)
                                save_folder = self.image_save_prefix_folder + \
                                    "class_"+str(class_label)+"/"
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                im_path = save_folder+'/iter_' + \
                                    str(step_iter)+'_no_optimizer_actual_c_' + \
                                    str(class_label)+'.jpg'

                                dft_im_path = save_folder+'/iter_' + \
                                    str(step_iter)+'_centered_dft_amp_' + \
                                    str(class_label)+'.jpg'

                                dft_phase_im_path = save_folder+'/iter_' + \
                                    str(step_iter)+'_centered_dft_phase_' + \
                                    str(class_label)+'.jpg'

                                raw_dft = np.fft.fft2(self.created_image)
                                raw_centered_dft_out = np.fft.fftshift(raw_dft)
                                for_vis_dft_amp_out = np.log(
                                    1+np.abs(raw_centered_dft_out))
                                std01_vis_dft_amp_out = recreate_image(
                                    for_vis_dft_amp_out, unnormalize=False)

                                for_vis_dft_phase_out = np.angle(
                                    raw_centered_dft_out)
                                std01_vis_dft_phase_out = recreate_image(
                                    for_vis_dft_phase_out, unnormalize=False)

                                numpy_image = self.created_image
                                save_image(numpy_image, im_path)

                                # save_image(std01_vis_dft_amp_out, dft_im_path)
                                # save_image(std01_vis_dft_phase_out,
                                #            dft_phase_im_path)

                        cur_time = time.time()
                        tot_time = cur_time - begin_time

                        pbar.set_postfix(loss=loss.item(), norm_image=torch.norm(
                            self.initial_image).item(), norm_grad=torch.norm(gradients).item(), norm_raw_grad=norm_grad.item(), it_time=format_time(tot_time))
            else:
                with trange(number_of_image_optimization_steps, unit="iter") as pbar:
                    for step_iter in pbar:
                        begin_time = time.time()
                        pbar.set_description(f"Iteration {step_iter+1}")
                        # print("self.initial_image grad", self.initial_image.grad)
                        # self.initial_image.grad = None

                        # conv = torch.nn.Conv2d(
                        #     3, 3, 3, padding=1)
                        # conv = conv.to(self.device)
                        # self.initial_image_tilda = conv(self.initial_image)

                        outputs = self.model(self.initial_image)

                        loss, active_pixel_points, total_pixel_points, non_zero_pixel_points = self.get_layer_loss_value(
                            template_loss_type, class_indx, current_layer_num, outputs, class_image, alpha, verbose)

                        print("{} Loss: {}".format(template_loss_type, loss))

                        if(step_iter == 0 and active_pixel_points is not None):
                            percent_active_pixels = float((
                                active_pixel_points/total_pixel_points)*100)
                            print("active_pixel_points", active_pixel_points)
                            print("total_pixel_points", total_pixel_points)
                            print("Percentage of active pixels:",
                                  percent_active_pixels)
                            if(is_log_wandb):
                                wandb.log(
                                    {"active_pixel_points": active_pixel_points, "total_pixel_points": total_pixel_points,
                                     "Percent_active_pixels": percent_active_pixels, "non_zero_pixel_points": non_zero_pixel_points,
                                     "final_postfix_for_save": final_postfix_for_save}, step=(step_iter+1))
                        # Backward
                        loss.backward()

                        unnorm_gradients = self.initial_image.grad
                        # print("Original self.initial_image gradients",
                        #       unnorm_gradients)

                        gradients = unnorm_gradients / \
                            torch.std(unnorm_gradients) + 1e-8
                        # print("After normalize self.initial_image gradients", gradients)

                        with torch.no_grad():
                            self.initial_image = self.initial_image - gradients*step_size
                            # self.initial_image = 0.9 * self.initial_image
                            self.initial_image = torch.clamp(
                                self.initial_image, -1, 1)

                        # Save image every plot_interval iteration
                        if step_iter % plot_iteration_interval == 0:
                            with torch.no_grad():
                                reconst_outputs = self.model(
                                    self.initial_image)
                                reconst_outputs_softmax = reconst_outputs.softmax(
                                    dim=1)
                                reconst_img_norm = torch.norm(
                                    self.initial_image)
                                reconst_pred = reconst_outputs_softmax.max(
                                    1).indices

                                update_indx = step_iter // plot_iteration_interval
                                list_of_reconst_softmax_pred[update_indx] = reconst_outputs_softmax[0]

                                if(is_log_wandb):
                                    wandb.log(
                                        {"reconst_img_pred_indx":  reconst_pred, "reconst_img_norm": reconst_img_norm,
                                         "normalized_gradients": wandb.Histogram(gradients.cpu().detach().numpy()),
                                         "unnormalized_gradients": wandb.Histogram(unnorm_gradients.cpu().detach().numpy()),
                                         "reconst_img_label_pred": classes[reconst_pred], "loss": loss, "optimizing_img_gradient_norm": torch.norm(unnorm_gradients).item()}, step=(step_iter+1))

                                self.created_image = recreate_image(
                                    self.initial_image, normalize_image)
                                print("self.created_image.shape::",
                                      self.created_image.shape)
                                save_folder = self.image_save_prefix_folder + \
                                    "class_"+str(class_label)+"/"
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                im_path = save_folder+'/no_optimizer_actual_c_' + \
                                    str(class_label)+'_iter' + \
                                    str(step_iter) + '.jpg'

                                numpy_image = self.created_image
                                save_image(numpy_image, im_path)

                        self.initial_image.requires_grad_()
                        # Recreate image
                        print("self.initial_image", self.initial_image)

                        cur_time = time.time()
                        tot_time = cur_time - begin_time

                        pbar.set_postfix(
                            loss=loss, it_time=format_time(tot_time))

            reconst_outputs = self.model(self.initial_image)
            reconst_outputs_softmax = reconst_outputs.softmax(dim=1)
            reconst_img_norm = torch.norm(self.initial_image)
            print("Norm of reconstructed image is:", reconst_img_norm)
            print("Confidence over Reconstructed image with alpha:", alpha)
            for i in range(len(reconst_outputs_softmax[0])):
                print("Class {} => {}".format(
                    classes[i], reconst_outputs_softmax[0][i]))
            reconst_pred = reconst_outputs_softmax.max(1).indices
            print("Reconstructed image Class predicted:",
                  classes[reconst_pred])

            image = preprocess_image(
                class_image[0].cpu().clone().detach().numpy(), normalize_image)
            image = image.to(self.device)

            image.requires_grad_()
            original_image_outputs = self.model(image)
            original_image_outputs_softmax = original_image_outputs.softmax(
                dim=1)
            print("Confidence over original image with alpha:", alpha)
            for i in range(len(original_image_outputs_softmax[0])):
                print("Class {} => {}".format(
                    classes[i], original_image_outputs_softmax[0][i]))
            original_image_pred = original_image_outputs_softmax.max(1).indices
            print("Class predicted on original image was :",
                  classes[original_image_pred])
            print("Original label was:", class_label)

            if(is_log_wandb):
                step_lists = [(plot_iteration_interval * indx)
                              for indx in range(number_of_intervals)]
                each_class_softmax_ordered_by_steps = [None] * num_classes

                for c_ind in range(num_classes):
                    current_class_softmax_list = []
                    for indx in range(number_of_intervals):
                        current_class_softmax_list.append(
                            list_of_reconst_softmax_pred[indx][c_ind])
                    each_class_softmax_ordered_by_steps[c_ind] = current_class_softmax_list

                reconst_img = recreate_image(
                    self.initial_image, normalize_image)
                if isinstance(reconst_img, (np.ndarray, np.generic)):
                    reconst_img = format_np_output(reconst_img)
                    reconst_img = Image.fromarray(reconst_img)
                reconst_img = wandb.Image(
                    reconst_img, caption="Reconstructed final image")

                orig_img = recreate_image(
                    class_image, normalize_image)
                if isinstance(orig_img, (np.ndarray, np.generic)):
                    orig_img = format_np_output(orig_img)
                    orig_img = Image.fromarray(orig_img)
                orig_img = wandb.Image(
                    orig_img, caption="Original image")

                wandb.log(
                    {
                        "reconst_img": reconst_img, "orig_img": orig_img,
                        "reconst_img_norm": reconst_img_norm,
                        "reconst_img_label_pred": classes[reconst_pred], "reconst_img_pred_indx":  reconst_pred,
                        "original_image_outputs_softmax": original_image_outputs_softmax, "original_img_label_pred": classes[original_image_pred], "original_img_pred_indx": original_image_pred,
                        "softmax_reconst_img_opt_steps_plt": wandb.plot.line_series(xs=step_lists,
                                                                                    ys=each_class_softmax_ordered_by_steps,
                                                                                    keys=classes,
                                                                                    title="Variation of softmax values across classes vs Optimization steps",
                                                                                    xname="Optimization steps")
                    }, step=(step_iter+1))
                wandb.finish()
            print("Reconstructed images written at:",
                  self.image_save_prefix_folder)

        return self.initial_image.cpu().clone().detach().numpy()

    def generate_template_image_per_class(self, exp_type, per_class_dataset, class_label, class_indx, number_of_batch_to_collect, classes, model_arch_type, dataset, is_template_image_on_train,
                                          is_class_segregation_on_ground_truth, template_initial_image_type,
                                          template_image_calculation_batch_size, template_loss_type, wand_project_name, wandb_group_name, torch_seed, number_of_image_optimization_steps, collect_threshold,
                                          entropy_calculation_batch_size, number_of_batches_to_calculate_entropy_on, root_save_prefix="root", final_postfix_for_save="",
                                          wandb_config_additional_dict=None, vis_version='V2', random_sample_gate_percent=None,alpha = 0,v4lr=0.01,is_skip_collection=False):
        coll_gate_sample_seed_gen = None
        random_sample_gate_percent_str = ""
        is_log_wandb = not(wand_project_name is None)
        plot_iteration_interval = 5

        # torch.manual_seed(torch_seed)
        coll_seed_gen = torch.Generator()
        coll_seed_gen.manual_seed(torch_seed)

        entr_seed_gen = torch.Generator()
        entr_seed_gen.manual_seed(torch_seed)

        if(random_sample_gate_percent is not None):
            random_sample_gate_percent_str = "RD_GT_PER_LAY_SAMPLE_" + \
                str(random_sample_gate_percent)
            coll_gate_sample_seed_gen = torch.Generator()
            coll_gate_sample_seed_gen.manual_seed(torch_seed)

        self.model.train(False)
        per_class_data_loader = torch.utils.data.DataLoader(per_class_dataset, batch_size=template_image_calculation_batch_size,
                                                            shuffle=True, generator=coll_seed_gen, worker_init_fn=seed_worker)

        tmp_image_over_what_str = 'test'
        if(is_template_image_on_train):
            tmp_image_over_what_str = 'train'

        seg_over_what_str = 'MP'
        if(is_class_segregation_on_ground_truth):
            seg_over_what_str = 'GT'

        layer_nums_str = ""
        if(self.layer_nums_to_visualize is not None):
            layer_nums_str = "/LV_"
            for i in self.layer_nums_to_visualize:
                layer_nums_str += str(i)+" "
            layer_nums_str = layer_nums_str.strip()
            layer_nums_str = layer_nums_str.replace(" ", "_")

        self.image_save_prefix_folder = str(root_save_prefix)+"/"+str(dataset)+"/MT_"+str(model_arch_type)+"_ET_"+str(exp_type)+"/Ver_"+str(vis_version)+"/_COLL_OV_"+str(tmp_image_over_what_str)+"/SEG_"+str(
            seg_over_what_str)+"/TMP_COLL_BS_"+str(template_image_calculation_batch_size)+"_NO_TO_COLL_"+str(number_of_batch_to_collect)+"/TMP_LOSS_TP_"+str(template_loss_type)+"/TMP_INIT_"+str(template_initial_image_type)+"/_torch_seed_"+str(torch_seed)+"_c_thres_"+str(collect_threshold)+"/"+random_sample_gate_percent_str+str(layer_nums_str)+"/" + str(final_postfix_for_save) + "/"

        self.image_save_prefix_folder += "_alp_" + str(alpha)+"/"
        if(vis_version == "V4"):
            self.image_save_prefix_folder += "/V4_lr_"+str(v4lr)+"/"
        normalize_image = False
        
        if not is_skip_collection:
            if("ENTR" in template_loss_type):
                entropy_per_class_data_loader = torch.utils.data.DataLoader(per_class_dataset, batch_size=entropy_calculation_batch_size,
                                                                            shuffle=True, num_workers=2, generator=entr_seed_gen, worker_init_fn=seed_worker)
                self.collect_entropy_of_pixels_into_ymaps(
                    entropy_per_class_data_loader, class_label, number_of_batch_to_collect=number_of_batches_to_calculate_entropy_on)

            self.collect_all_active_pixels_into_ymaps(
                per_class_data_loader, class_label, number_of_batch_to_collect, collect_threshold, random_sample_gate_percent=random_sample_gate_percent, coll_gate_sample_seed_gen=coll_gate_sample_seed_gen)

        class_image, _ = next(iter(per_class_data_loader))
        class_image = class_image.to(self.device, non_blocking=True)
        if(vis_version == 'V2' or vis_version == 'V3' or vis_version == 'V4'):
            if('conv4_deep_gated_net' in model_arch_type):
                number_of_image_optimization_steps = 301
            else:
                # number_of_image_optimization_steps = 161
                pass
            if(vis_version == 'V2' or vis_version == 'V4'):
                start_sigma = 0.75
                end_sigma = 0.1
            elif(vis_version == 'V3'):
                start_sigma = 0.1
                end_sigma = 0.75
            if('conv4_deep_gated_net' in model_arch_type):
                start_step_size = 1
                end_step_size = 0.5
            else:
                start_step_size = 0.1
                end_step_size = 0.05

        for repeat in range(1):

            if(repeat == 1):
                alpha = 0.1
            self.image_save_prefix_folder = str(root_save_prefix)+"/"+str(dataset)+"/MT_"+str(model_arch_type)+"_ET_"+str(exp_type)+"/Ver_"+str(vis_version)+"/_COLL_OV_"+str(tmp_image_over_what_str)+"/SEG_"+str(
                seg_over_what_str)+"/TMP_COLL_BS_"+str(template_image_calculation_batch_size)+"_NO_TO_COLL_"+str(number_of_batch_to_collect)+"/TMP_LOSS_TP_"+str(template_loss_type)+"/TMP_INIT_"+str(template_initial_image_type)+"/_torch_seed_"+str(torch_seed)+"_c_thres_"+str(collect_threshold)+"/"+random_sample_gate_percent_str+str(layer_nums_str)+"/" + str(final_postfix_for_save) + "/"

            self.image_save_prefix_folder += "_alp_" + str(alpha)+"/"
            if("ENTR" in template_loss_type):
                self.image_save_prefix_folder += "_ENT_BS_" + \
                    str(entropy_calculation_batch_size)+"_ENTR_COLL_" + \
                    str(number_of_batches_to_calculate_entropy_on)+"/"
            if(vis_version == "V4"):
                self.image_save_prefix_folder += "/V4_lr_"+str(v4lr)+"/"

            print(
                "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ alpha", alpha)

            num_classes = len(classes)
            number_of_intervals = (
                number_of_image_optimization_steps // plot_iteration_interval) + 1
            list_of_reconst_softmax_pred = [None] * number_of_intervals

            if(is_log_wandb):
                wandb_run_name = self.image_save_prefix_folder.replace(
                    "/", "").replace(root_save_prefix, class_label)
                wandb_config = self.get_wandb_config(exp_type, class_label, class_indx, classes, model_arch_type, dataset, is_template_image_on_train,
                                                     is_class_segregation_on_ground_truth, template_initial_image_type,
                                                     template_image_calculation_batch_size, template_loss_type, torch_seed, number_of_image_optimization_steps,
                                                     number_of_batch_to_collect=number_of_batch_to_collect, collect_threshold=collect_threshold)
                wandb_config["vis_version"] = vis_version
                if(vis_version == 'V2' or vis_version == 'V3'):
                    wandb_config["start_sigma"] = start_sigma
                    wandb_config["end_sigma"] = end_sigma
                    wandb_config["start_step_size"] = start_step_size
                    wandb_config["end_step_size"] = end_step_size
                    wandb_config["per_layer_random_sample_gate_percent"] = random_sample_gate_percent
                if(vis_version == "V4"):
                    wandb_config["v4lr"] = v4lr
                if(wandb_config_additional_dict is not None):
                    wandb_config.update(wandb_config_additional_dict)
                wandb_config["alpha"] = alpha

                wandb.init(
                    project=f"{wand_project_name}",
                    name=f"{wandb_run_name}",
                    group=f"{wandb_group_name}",
                    config=wandb_config,
                )

            self.initial_image = preprocess_image(
                self.original_image.cpu().clone().detach().numpy(), normalize_image)

            self.initial_image = self.initial_image.to(self.device)

            self.initial_image.requires_grad_()

            step_size = 0.01
            eps = None
            if("PLATTCK" in template_loss_type):
                eps = float(template_loss_type.split("__")[1])
            
            if(vis_version == 'V2' or vis_version == 'V3'):
                with trange(number_of_image_optimization_steps, unit="iter", desc="Generating template image V2") as pbar:
                    for step_iter in pbar:
                        begin_time = time.time()
                        pbar.set_description(f"Iteration {step_iter+1}")

                        step_size = start_step_size + \
                            ((end_step_size - start_step_size) * step_iter) / \
                            number_of_image_optimization_steps
                        # optimizer = torch.optim.SGD([img_var], lr=step_size)
                        sigma = start_sigma + \
                            ((end_sigma - start_sigma) * step_iter) / \
                            number_of_image_optimization_steps

                        outputs = self.model(self.initial_image)

                        verbose = 1
                        loss, active_pixel_points, total_pixel_points, non_zero_pixel_points = self.get_loss_value(
                            template_loss_type, class_indx, outputs, class_image, verbose=verbose,alpha=alpha)

                        # print("{} Loss: {}".format(template_loss_type, loss))
                        if(step_iter == 0 and active_pixel_points is not None):
                            percent_active_pixels = float((
                                active_pixel_points/total_pixel_points)*100)
                            print("active_pixel_points", active_pixel_points)
                            print("total_pixel_points", total_pixel_points)
                            print("Percentage of active pixels:",
                                  percent_active_pixels)
                            if(is_log_wandb):
                                wandb.log(
                                    {"active_pixel_points": active_pixel_points, "total_pixel_points": total_pixel_points,
                                     "Percent_active_pixels": percent_active_pixels, "non_zero_pixel_points": non_zero_pixel_points,
                                     "final_postfix_for_save": final_postfix_for_save}, step=(step_iter+1))

                        # Backward
                        loss.backward()

                        unnorm_gradients = self.initial_image.grad
                        # std_unnorm_grad = torch.std(unnorm_gradients)
                        norm_grad = torch.norm(unnorm_gradients)
                        # print("torch.norm(unnorm_gradients):",
                        #       norm_grad)
                        # print("std_unnorm_grad:", std_unnorm_grad)
                        # print("Original self.initial_image gradients", gradients)

                        # gradients = unnorm_gradients / first_norm
                        gradients = unnorm_gradients / \
                            norm_grad + 1e-8

                        # print("torch.norm(gradients):",
                        #       torch.norm(gradients))

                        with torch.no_grad():
                            # self.initial_image = self.initial_image - gradients*step_size
                            blurred_grad = gradients.cpu().detach().numpy()[0]
                            blurred_grad = blur_img(
                                blurred_grad, sigma)
                            self.initial_image = self.initial_image.cpu().detach().numpy()[
                                0]
                            self.initial_image -= step_size / \
                                np.abs(blurred_grad).mean() * blurred_grad

                            # optimizer.step()
                            # print("sigma:", sigma)

                            self.initial_image = blur_img(
                                self.initial_image, sigma)
                            self.initial_image = torch.from_numpy(
                                self.initial_image[None])

                        self.initial_image = self.initial_image.to(self.device)
                        if(eps is not None):
                            self.initial_image = torch.clamp(self.initial_image, class_image - eps, class_image + eps)
                            self.initial_image = torch.clamp(self.initial_image, 0, 1)
                        self.initial_image.requires_grad_()

                        if(plot_iteration_interval is not None and step_iter % plot_iteration_interval == 0):
                            with torch.no_grad():
                                reconst_outputs = self.model(
                                    self.initial_image)
                                reconst_outputs_softmax = reconst_outputs.softmax(
                                    dim=1)
                                reconst_img_norm = torch.norm(
                                    self.initial_image)
                                reconst_pred = reconst_outputs_softmax.max(
                                    1).indices

                                update_indx = step_iter // plot_iteration_interval
                                list_of_reconst_softmax_pred[update_indx] = reconst_outputs_softmax[0]

                                if(is_log_wandb):
                                    wandb.log(
                                        {"reconst_img_pred_indx":  reconst_pred, "reconst_img_norm": reconst_img_norm,
                                         "normalized_gradients": wandb.Histogram(gradients.cpu().detach().numpy()),
                                         "unnormalized_gradients": wandb.Histogram(unnorm_gradients.cpu().detach().numpy()),
                                         "reconst_img_label_pred": classes[reconst_pred], "loss": loss, "optimizing_img_gradient_norm": torch.norm(unnorm_gradients).item()}, step=(step_iter+1))

                                self.created_image = recreate_image(
                                    self.initial_image, normalize_image)
                                print("self.created_image.shape::",
                                      self.created_image.shape)
                                save_folder = self.image_save_prefix_folder + \
                                    "class_"+str(class_label)+"/"
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                im_path = save_folder+'/iter_' + \
                                    str(step_iter)+'_no_optimizer_actual_c_' + \
                                    str(class_label)+'.jpg'

                                dft_im_path = save_folder+'/iter_' + \
                                    str(step_iter)+'_centered_dft_amp_' + \
                                    str(class_label)+'.jpg'

                                dft_phase_im_path = save_folder+'/iter_' + \
                                    str(step_iter)+'_centered_dft_phase_' + \
                                    str(class_label)+'.jpg'

                                raw_dft = np.fft.fft2(self.created_image)
                                raw_centered_dft_out = np.fft.fftshift(raw_dft)
                                for_vis_dft_amp_out = np.log(
                                    1+np.abs(raw_centered_dft_out))
                                std01_vis_dft_amp_out = recreate_image(
                                    for_vis_dft_amp_out, unnormalize=False)

                                for_vis_dft_phase_out = np.angle(
                                    raw_centered_dft_out)
                                std01_vis_dft_phase_out = recreate_image(
                                    for_vis_dft_phase_out, unnormalize=False)

                                numpy_image = self.created_image
                                save_image(numpy_image, im_path)

                                # save_image(std01_vis_dft_amp_out, dft_im_path)
                                # save_image(std01_vis_dft_phase_out,
                                #            dft_phase_im_path)

                        cur_time = time.time()
                        tot_time = cur_time - begin_time

                        pbar.set_postfix(loss=loss.item(), norm_image=torch.norm(
                            self.initial_image).item(), norm_grad=torch.norm(gradients).item(), norm_raw_grad=norm_grad.item(), it_time=format_time(tot_time))
            elif(vis_version == 'V4'):
                with trange(number_of_image_optimization_steps, unit="iter", desc="Generating template image V4") as pbar:
                    for step_iter in pbar:
                        begin_time = time.time()
                        pbar.set_description(f"Iteration {step_iter+1}")

                        outputs = self.model(self.initial_image)

                        verbose = 1
                        loss, active_pixel_points, total_pixel_points, non_zero_pixel_points = self.get_loss_value(
                            template_loss_type, class_indx, outputs, class_image, verbose=verbose,alpha=alpha)

                        # print("{} Loss: {}".format(template_loss_type, loss))
                        if(step_iter == 0 and active_pixel_points is not None):
                            percent_active_pixels = float((
                                active_pixel_points/total_pixel_points)*100)
                            print("active_pixel_points", active_pixel_points)
                            print("total_pixel_points", total_pixel_points)
                            print("Percentage of active pixels:",
                                  percent_active_pixels)
                            if(is_log_wandb):
                                wandb.log(
                                    {"active_pixel_points": active_pixel_points, "total_pixel_points": total_pixel_points,
                                     "Percent_active_pixels": percent_active_pixels, "non_zero_pixel_points": non_zero_pixel_points,
                                     "final_postfix_for_save": final_postfix_for_save}, step=(step_iter+1))

                        # Backward
                        loss.backward()

                        unnorm_gradients = self.initial_image.grad
                        norm_grad = torch.norm(unnorm_gradients)
                        gradients = unnorm_gradients / \
                            norm_grad + 1e-8
                        with torch.no_grad():
                            self.initial_image -= v4lr *torch.sign(unnorm_gradients)
                            if(eps is not None):
                                self.initial_image = torch.clamp(self.initial_image, class_image - eps, class_image + eps)
                            self.initial_image = torch.clamp(
                                self.initial_image, 0, 1)
                        
                        self.initial_image.requires_grad_()

                        if(plot_iteration_interval is not None and step_iter % plot_iteration_interval == 0):
                            with torch.no_grad():
                                reconst_outputs = self.model(
                                    self.initial_image)
                                reconst_outputs_softmax = reconst_outputs.softmax(
                                    dim=1)
                                reconst_img_norm = torch.norm(
                                    self.initial_image)
                                reconst_pred = reconst_outputs_softmax.max(
                                    1).indices

                                update_indx = step_iter // plot_iteration_interval
                                list_of_reconst_softmax_pred[update_indx] = reconst_outputs_softmax[0]

                                if(is_log_wandb):
                                    wandb.log(
                                        {"reconst_img_pred_indx":  reconst_pred, "reconst_img_norm": reconst_img_norm,
                                         "normalized_gradients": wandb.Histogram(gradients.cpu().detach().numpy()),
                                         "unnormalized_gradients": wandb.Histogram(unnorm_gradients.cpu().detach().numpy()),
                                         "reconst_img_label_pred": classes[reconst_pred], "loss": loss, "optimizing_img_gradient_norm": torch.norm(unnorm_gradients).item()}, step=(step_iter+1))

                                self.created_image = recreate_image(
                                    self.initial_image, normalize_image)
                                print("self.created_image.shape::",
                                      self.created_image.shape)
                                save_folder = self.image_save_prefix_folder + \
                                    "class_"+str(class_label)+"/"
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                im_path = save_folder+'/iter_' + \
                                    str(step_iter)+'_no_optimizer_actual_c_' + \
                                    str(class_label)+'.jpg'

                                dft_im_path = save_folder+'/iter_' + \
                                    str(step_iter)+'_centered_dft_amp_' + \
                                    str(class_label)+'.jpg'

                                dft_phase_im_path = save_folder+'/iter_' + \
                                    str(step_iter)+'_centered_dft_phase_' + \
                                    str(class_label)+'.jpg'

                                raw_dft = np.fft.fft2(self.created_image)
                                raw_centered_dft_out = np.fft.fftshift(raw_dft)
                                for_vis_dft_amp_out = np.log(
                                    1+np.abs(raw_centered_dft_out))
                                std01_vis_dft_amp_out = recreate_image(
                                    for_vis_dft_amp_out, unnormalize=False)

                                for_vis_dft_phase_out = np.angle(
                                    raw_centered_dft_out)
                                std01_vis_dft_phase_out = recreate_image(
                                    for_vis_dft_phase_out, unnormalize=False)

                                numpy_image = self.created_image
                                save_image(numpy_image, im_path)

                                # save_image(std01_vis_dft_amp_out, dft_im_path)
                                # save_image(std01_vis_dft_phase_out,
                                #            dft_phase_im_path)

                        cur_time = time.time()
                        tot_time = cur_time - begin_time

                        pbar.set_postfix(loss=loss.item(), norm_image=torch.norm(
                            self.initial_image).item(), norm_grad=torch.norm(gradients).item(), norm_raw_grad=norm_grad.item(), it_time=format_time(tot_time))
            
            else:
                with trange(number_of_image_optimization_steps, unit="iter") as pbar:
                    for step_iter in pbar:
                        begin_time = time.time()
                        pbar.set_description(f"Iteration {step_iter+1}")
                        # print("self.initial_image grad", self.initial_image.grad)
                        # self.initial_image.grad = None

                        # conv = torch.nn.Conv2d(
                        #     3, 3, 3, padding=1)
                        # conv = conv.to(self.device)
                        # self.initial_image_tilda = conv(self.initial_image)

                        outputs = self.model(self.initial_image)

                        loss, active_pixel_points, total_pixel_points, non_zero_pixel_points = self.get_loss_value(
                            template_loss_type, class_indx, outputs, class_image, alpha=alpha)

                        print("{} Loss: {}".format(template_loss_type, loss))

                        if(step_iter == 0 and active_pixel_points is not None):
                            percent_active_pixels = float((
                                active_pixel_points/total_pixel_points)*100)
                            print("active_pixel_points", active_pixel_points)
                            print("total_pixel_points", total_pixel_points)
                            print("Percentage of active pixels:",
                                  percent_active_pixels)
                            if(is_log_wandb):
                                wandb.log(
                                    {"active_pixel_points": active_pixel_points, "total_pixel_points": total_pixel_points,
                                     "Percent_active_pixels": percent_active_pixels, "non_zero_pixel_points": non_zero_pixel_points,
                                     "final_postfix_for_save": final_postfix_for_save}, step=(step_iter+1))
                        # Backward
                        loss.backward()

                        unnorm_gradients = self.initial_image.grad
                        # print("Original self.initial_image gradients",
                        #       unnorm_gradients)

                        gradients = unnorm_gradients / \
                            torch.std(unnorm_gradients) + 1e-8
                        # print("After normalize self.initial_image gradients", gradients)

                        with torch.no_grad():
                            self.initial_image = self.initial_image - gradients*step_size
                            # self.initial_image = 0.9 * self.initial_image
                            if(eps is not None):
                                self.initial_image = torch.clamp(self.initial_image, class_image - eps, class_image + eps)
                                self.initial_image = torch.clamp(self.initial_image, 0, 1)

                        # Save image every plot_interval iteration
                        if step_iter % plot_iteration_interval == 0:
                            with torch.no_grad():
                                reconst_outputs = self.model(
                                    self.initial_image)
                                reconst_outputs_softmax = reconst_outputs.softmax(
                                    dim=1)
                                reconst_img_norm = torch.norm(
                                    self.initial_image)
                                reconst_pred = reconst_outputs_softmax.max(
                                    1).indices

                                update_indx = step_iter // plot_iteration_interval
                                list_of_reconst_softmax_pred[update_indx] = reconst_outputs_softmax[0]

                                if(is_log_wandb):
                                    wandb.log(
                                        {"reconst_img_pred_indx":  reconst_pred, "reconst_img_norm": reconst_img_norm,
                                         "normalized_gradients": wandb.Histogram(gradients.cpu().detach().numpy()),
                                         "unnormalized_gradients": wandb.Histogram(unnorm_gradients.cpu().detach().numpy()),
                                         "reconst_img_label_pred": classes[reconst_pred], "loss": loss, "optimizing_img_gradient_norm": torch.norm(unnorm_gradients).item()}, step=(step_iter+1))

                                self.created_image = recreate_image(
                                    self.initial_image, normalize_image)
                                print("self.created_image.shape::",
                                      self.created_image.shape)
                                save_folder = self.image_save_prefix_folder + \
                                    "class_"+str(class_label)+"/"
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                im_path = save_folder+'/no_optimizer_actual_c_' + \
                                    str(class_label)+'_iter' + \
                                    str(step_iter) + '.jpg'

                                numpy_image = self.created_image
                                save_image(numpy_image, im_path)

                        self.initial_image.requires_grad_()
                        # Recreate image
                        print("self.initial_image", self.initial_image)

                        cur_time = time.time()
                        tot_time = cur_time - begin_time

                        pbar.set_postfix(
                            loss=loss, it_time=format_time(tot_time))

            reconst_outputs = self.model(self.initial_image)
            reconst_outputs_softmax = reconst_outputs.softmax(dim=1)
            reconst_img_norm = torch.norm(self.initial_image)
            print("Norm of reconstructed image is:", reconst_img_norm)
            print("Confidence over Reconstructed image with alpha:", alpha)
            for i in range(len(reconst_outputs_softmax[0])):
                print("Class {} => {}".format(
                    classes[i], reconst_outputs_softmax[0][i]))
            reconst_pred = reconst_outputs_softmax.max(1).indices
            print("Reconstructed image Class predicted:",
                  classes[reconst_pred])

            image = preprocess_image(
                class_image[0].cpu().clone().detach().numpy(), normalize_image)
            image = image.to(self.device)

            image.requires_grad_()
            original_image_outputs = self.model(image)
            original_image_outputs_softmax = original_image_outputs.softmax(
                dim=1)
            print("Confidence over original image with alpha:", alpha)
            for i in range(len(original_image_outputs_softmax[0])):
                print("Class {} => {}".format(
                    classes[i], original_image_outputs_softmax[0][i]))
            original_image_pred = original_image_outputs_softmax.max(1).indices
            print("Class predicted on original image was :",
                  classes[original_image_pred])
            print("Original label was:", class_label)

            if(is_log_wandb):
                step_lists = [(plot_iteration_interval * indx)
                              for indx in range(number_of_intervals)]
                each_class_softmax_ordered_by_steps = [None] * num_classes

                for c_ind in range(num_classes):
                    current_class_softmax_list = []
                    for indx in range(number_of_intervals):
                        current_class_softmax_list.append(
                            list_of_reconst_softmax_pred[indx][c_ind])
                    each_class_softmax_ordered_by_steps[c_ind] = current_class_softmax_list

                reconst_img = recreate_image(
                    self.initial_image, normalize_image)
                if isinstance(reconst_img, (np.ndarray, np.generic)):
                    reconst_img = format_np_output(reconst_img)
                    reconst_img = Image.fromarray(reconst_img)
                reconst_img = wandb.Image(
                    reconst_img, caption="Reconstructed final image")

                orig_img = recreate_image(
                    class_image, normalize_image)
                if isinstance(orig_img, (np.ndarray, np.generic)):
                    orig_img = format_np_output(orig_img)
                    orig_img = Image.fromarray(orig_img)
                orig_img = wandb.Image(
                    orig_img, caption="Original image")

                wandb.log(
                    {
                        "reconst_img": reconst_img, "orig_img": orig_img,
                        "reconst_img_norm": reconst_img_norm,
                        "reconst_img_label_pred": classes[reconst_pred], "reconst_img_pred_indx":  reconst_pred,
                        "original_image_outputs_softmax": original_image_outputs_softmax, "original_img_label_pred": classes[original_image_pred], "original_img_pred_indx": original_image_pred,
                        "softmax_reconst_img_opt_steps_plt": wandb.plot.line_series(xs=step_lists,
                                                                                    ys=each_class_softmax_ordered_by_steps,
                                                                                    keys=classes,
                                                                                    title="Variation of softmax values across classes vs Optimization steps",
                                                                                    xname="Optimization steps")
                    }, step=(step_iter+1))
                wandb.finish()
                print("Reconstructed images written at:",
                      self.image_save_prefix_folder)

        return self.initial_image.cpu().clone().detach().numpy()


def get_initial_image(dataset, template_initial_image_type, size=None):
    if(dataset == "cifar10"):
        if(size is None):
            size = 32
        if(template_initial_image_type == 'zero_init_image'):
            return torch.from_numpy(np.uint8(np.random.uniform(0, 1, (3, size, size))))
        elif(template_initial_image_type == 'uniform_init_image'):
            return torch.from_numpy(np.uint8(np.random.uniform(100, 180, (3, size, size)))/255)
        elif(template_initial_image_type == 'gaussian_init_image'):
            return torch.from_numpy(np.uint8(np.random.random((3, size, size)) * 20 + 128.)/255)
    elif(dataset == "mnist" or dataset == "fashion_mnist"):
        if(size is None):
            size = 28
        if(template_initial_image_type == 'zero_init_image'):
            return torch.from_numpy(np.uint8(np.random.uniform(0, 1, (1, size, size))))
        elif(template_initial_image_type == 'uniform_init_image'):
            return torch.from_numpy(np.uint8(np.random.uniform(100, 180, (1, size, size)))/255)
        elif(template_initial_image_type == 'gaussian_init_image'):
            return torch.from_numpy(np.uint8(np.random.random((1, size, size)) * 20 + 128.)/255)
        elif(template_initial_image_type == 'normal_init_image'):
            return torch.from_numpy(np.random.normal(128, 8, (1, 28, 28)).astype('float32')/255)


def quick_visualization_on_config(model_arch_type, model, dataset, exp_type, template_initial_image_type, images_to_collect_upon, number_of_image_optimization_steps, template_loss_type, vis_version='V2'):
    tmp_gen = TemplateImageGenerator(
        model, get_initial_image(dataset, template_initial_image_type))
    if(exp_type == "GENERATE_TEMPLATE_GIVEN_IMAGE"):
        return tmp_gen.generate_template_image_over_given_image(
            model_arch_type, images_to_collect_upon, number_of_image_optimization_steps, template_loss_type, vis_version)
    elif(exp_type == "GENERATE_TEMPLATE_GIVEN_BATCH_OF_IMAGES"):
        list_of_reconst_images = None
        for each_image_to_collect_upon in images_to_collect_upon:
            current_reconst_image = tmp_gen.generate_template_image_over_given_image(
                model_arch_type, each_image_to_collect_upon, number_of_image_optimization_steps, template_loss_type, vis_version)
            if(list_of_reconst_images is None):
                list_of_reconst_images = current_reconst_image
            else:
                list_of_reconst_images = torch.vstack(
                    (list_of_reconst_images, current_reconst_image))

        return list_of_reconst_images


def run_visualization_on_config(dataset, model_arch_type, is_template_image_on_train, is_class_segregation_on_ground_truth, template_initial_image_type,
                                template_image_calculation_batch_size, template_loss_type, number_of_batch_to_collect, wand_project_name, is_split_validation,
                                valid_split_size, torch_seed, number_of_image_optimization_steps, wandb_group_name, exp_type, collect_threshold,
                                entropy_calculation_batch_size, number_of_batches_to_calculate_entropy_on, root_save_prefix='root', final_postfix_for_save="aug_indx_1",
                                custom_model=None, custom_data_loader=None, class_indx_to_visualize=None, wandb_config_additional_dict=None,
                                vis_version='V2', random_sample_gate_percent=None, layer_nums_to_visualize=None,alpha=0,p_epsilon=None,v4lr=0.1,
                                analysis_mode=None,src_classes_set=None,list_of_dest_classes_set=None):
    output_template_list_per_class = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running for "+str(dataset))
    classes, num_classes, ret_config = get_preprocessing_and_other_configs(
        dataset, valid_split_size, batch_size=32)

    if(custom_data_loader is None):
        trainloader, _, testloader = preprocess_dataset_get_data_loader(
            ret_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=is_split_validation)
    else:
        trainloader, testloader = custom_data_loader

    print("Preprocessing and dataloader process completed of type:{} for dataset:{}".format(
        model_arch_type, dataset))
    if(custom_model is None):
        model, _ = get_model_from_loader(model_arch_type, dataset)
        print("Model loaded is:", model)
    else:
        model = custom_model
        print("Custom model provided in arguments will be used")

    if(class_indx_to_visualize is None):
        class_indx_to_visualize = [i for i in range(len(classes))]

    if(len(class_indx_to_visualize) != 0):
        input_data_list_per_class = segregate_classes(
            model, trainloader, testloader, num_classes, is_template_image_on_train, is_class_segregation_on_ground_truth)

    if(exp_type == "GENERATE_ALL_FINAL_TEMPLATE_IMAGES"):
        output_template_list_per_class = [None] * num_classes
        for i in range(num_classes):
            output_template_list_per_class[i] = []
    
    if(exp_type == "GENERATE_TEMPLATE_IMAGES_FOR_CLASS_ANALYSIS"):
        current_src_overall_y = None
        # In this stage all active gate overlaps are aggregated
        for c_indx in src_classes_set:
            c_indx = c_indx % len(classes)
            class_label = classes[c_indx]
            print("CLASS ANALYSIS MODE *********************************************************** Class:", class_label)
            per_class_dataset = PerClassDataset(
                input_data_list_per_class[c_indx], c_indx)
            tmp_gen = TemplateImageGenerator(
                model, get_initial_image(dataset, template_initial_image_type))
            per_class_data_loader = torch.utils.data.DataLoader(per_class_dataset, batch_size=template_image_calculation_batch_size)
            tmp_gen.collect_all_active_pixels_into_ymaps(per_class_data_loader, class_label, number_of_batch_to_collect, collect_threshold)
            if current_src_overall_y is None:
                # We are interested in only the gates which was switched on over the threshold
                current_src_overall_y = [HardRelu()(a) for a in tmp_gen.overall_y]
            else:
                for lind in range(len(current_src_overall_y)):
                    current_src_overall_y[lind] = current_src_overall_y[lind] * HardRelu()(tmp_gen.overall_y[lind])

        for dest_classes_set in list_of_dest_classes_set:
            current_overall_y = [torch.clone(a) for a in current_src_overall_y]
            # The gates which are active in src set but not in destination classes set are considered here
            for c_indx in dest_classes_set:
                c_indx = c_indx % len(classes)
                class_label = classes[c_indx]
                print("CLASS ANALYSIS MODE DEST ************************************************************ Class:", class_label)
                per_class_dataset = PerClassDataset(
                    input_data_list_per_class[c_indx], c_indx)
                tmp_gen = TemplateImageGenerator(
                    model, get_initial_image(dataset, template_initial_image_type))
                per_class_data_loader = torch.utils.data.DataLoader(per_class_dataset, batch_size=template_image_calculation_batch_size)
                tmp_gen.collect_all_active_pixels_into_ymaps(per_class_data_loader, class_label, number_of_batch_to_collect, collect_threshold)
                for lind in range(len(current_overall_y)):
                    if analysis_mode == "DIFF":
                        current_overall_y[lind] = HardRelu()(current_overall_y[lind] - HardRelu()(tmp_gen.overall_y[lind]))
                    elif analysis_mode == "OVRLP":
                        current_overall_y[lind] = current_overall_y[lind] * HardRelu()(tmp_gen.overall_y[lind])

            c_indx = dest_classes_set[0]
            # The gates which are overlapped between the all classes in the src parameter
            class_label = classes[c_indx]
            per_class_dataset = PerClassDataset(
                input_data_list_per_class[c_indx], c_indx)
            tmp_gen = TemplateImageGenerator(
                model, get_initial_image(dataset, template_initial_image_type))
            tmp_gen.reset_collection_state()
            # This is important to decide
            # for lind in range(len(current_overall_y)):
            #     current_overall_y[lind] = torch.where(current_overall_y[lind]>0,1.0,-1.0)
            tmp_gen.overall_y = current_overall_y
            postfix_save_append_str = str(analysis_mode)+'/SRC'+'_'.join(map(str,src_classes_set))+"/DST"+'_'.join(map(str,dest_classes_set))+'_'
            tmp_gen.generate_template_image_per_class(exp_type,
                                                    per_class_dataset, class_label, c_indx, number_of_batch_to_collect, classes, model_arch_type, dataset, is_template_image_on_train,
                                                    is_class_segregation_on_ground_truth, template_initial_image_type,
                                                    template_image_calculation_batch_size, template_loss_type, wand_project_name, wandb_group_name, torch_seed, number_of_image_optimization_steps, collect_threshold, entropy_calculation_batch_size,
                                                    number_of_batches_to_calculate_entropy_on, root_save_prefix=root_save_prefix, final_postfix_for_save=final_postfix_for_save+postfix_save_append_str, wandb_config_additional_dict=wandb_config_additional_dict,
                                                    vis_version=vis_version, random_sample_gate_percent=random_sample_gate_percent,alpha = alpha,v4lr=v4lr,is_skip_collection=True)
    
        return 

    for c_indx in class_indx_to_visualize:
        class_label = classes[c_indx]
        print("************************************************************ Class:", class_label)
        per_class_dataset = PerClassDataset(
            input_data_list_per_class[c_indx], c_indx)
        # per_class_loader = torch.utils.data.DataLoader(per_class_dataset, batch_size=32,
        #                                                shuffle=False)

        tmp_gen = TemplateImageGenerator(
            model, get_initial_image(dataset, template_initial_image_type))
        tmp_gen.layer_nums_to_visualize = layer_nums_to_visualize
        if(p_epsilon is not None):
            print("Running projection generator")
            list_of_reconst_images = tmp_gen.generate_projection_data_and_accuracies_per_class(p_epsilon,per_class_dataset, class_label, c_indx, classes, model_arch_type, dataset, is_template_image_on_train,
                                                                                             is_class_segregation_on_ground_truth, template_initial_image_type,
                                                                                             template_image_calculation_batch_size, template_loss_type, wand_project_name, wandb_group_name, torch_seed, number_of_image_optimization_steps,
                                                                                             exp_type, collect_threshold, entropy_calculation_batch_size, number_of_batches_to_calculate_entropy_on,
                                                                                             plot_iteration_interval=10, root_save_prefix=root_save_prefix, final_postfix_for_save=final_postfix_for_save,
                                                                                             wandb_config_additional_dict=wandb_config_additional_dict, vis_version=vis_version, random_sample_gate_percent=random_sample_gate_percent,alpha = alpha)
            output_template_list_per_class[c_indx] = list_of_reconst_images
        else:
            if(exp_type == "GENERATE_TEMPLATE_IMAGES_ADV_ORIG"):
                per_class_loader = torch.utils.data.DataLoader(per_class_dataset, batch_size=template_image_calculation_batch_size,shuffle=False)
                eps = 0.3
                adv_attack_type = 'PGD'
                number_of_adversarial_optimization_steps = 40
                eps_step_size = 0.01
                adv_target = None
                residue_vname = None
                update_on='all'
                rand_init=True
                norm=np.inf
                use_ytrue=True
                criterion= None
                adv_save_path = None
                if("bc_" in model_arch_type):
                    # criterion = torch.nn.BCEWithLogitsLoss()
                    criterion = Y_Logits_Binary_class_Loss()

                adv_dataset = generate_adv_examples(
                    per_class_loader, model, eps, adv_attack_type, number_of_adversarial_optimization_steps, eps_step_size, adv_target, number_of_batch_to_collect, is_save_adv=False, save_path=adv_save_path,residue_vname=residue_vname,update_on=update_on,rand_init=rand_init,norm=norm,use_ytrue=use_ytrue)

                tmp_gen = TemplateImageGenerator(
                    model, get_initial_image(dataset, template_initial_image_type))
                per_class_adv_data_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=template_image_calculation_batch_size)
                tmp_gen.collect_all_active_pixels_into_ymaps(per_class_adv_data_loader, class_label, number_of_batch_to_collect, collect_threshold)
                # current_adv_overall_y = [HardRelu()(a) for a in tmp_gen.overall_y]
                current_adv_overall_y = [a for a in tmp_gen.overall_y]
                
                if analysis_mode != "ONLY_ADV":
                    tmp_gen = TemplateImageGenerator(
                        model, get_initial_image(dataset, template_initial_image_type))
                    tmp_gen.collect_all_active_pixels_into_ymaps(per_class_loader, class_label, number_of_batch_to_collect, collect_threshold)
                    # current_orig_overall_y = [HardRelu()(a) for a in tmp_gen.overall_y]
                    current_orig_overall_y = [a for a in tmp_gen.overall_y]

                    current_overall_y = [None]*len(current_orig_overall_y)
                    for lind in range(len(current_overall_y)):
                        tmp1 = current_adv_overall_y[lind] * current_orig_overall_y[lind]
                        z=torch.zeros(1,device=tmp1.device)
                        if analysis_mode == "ADV_DIFF_ORIG":
                            #Both these versions gave almost same output
                            # current_overall_y[lind] = HardRelu()(current_adv_overall_y[lind] - HardRelu()(current_orig_overall_y[lind]))
                            current_overall_y[lind] = torch.where(tmp1<0,current_adv_overall_y[lind],z)
                            print(current_overall_y[lind])
                        elif analysis_mode == "ORIG_DIFF_ADV":
                            current_overall_y[lind] = torch.where(tmp1<0,current_orig_overall_y[lind],z)
                            # current_overall_y[lind] = HardRelu()(current_orig_overall_y[lind] - HardRelu()(current_adv_overall_y[lind]))
                        elif analysis_mode == "ADV_OVRLP_ORIG":
                            current_overall_y[lind] = torch.where(tmp1>0,current_orig_overall_y[lind],z)
                            # current_overall_y[lind] = current_adv_overall_y[lind] * HardRelu()(current_orig_overall_y[lind])
                else:
                    current_overall_y = current_adv_overall_y

                tmp_gen = TemplateImageGenerator(
                    model, get_initial_image(dataset, template_initial_image_type))
                tmp_gen.reset_collection_state()
                # This is important to decide
                for lind in range(len(current_overall_y)):
                    current_overall_y[lind] = torch.where(current_overall_y[lind]>0,1.0,-1.0)
                tmp_gen.overall_y = current_overall_y
                postfix_save_append_str = "_"+str(analysis_mode)+"_"

                tmp_gen.generate_template_image_per_class(exp_type,
                                                        per_class_dataset, class_label, c_indx, number_of_batch_to_collect, classes, model_arch_type, dataset, is_template_image_on_train,
                                                        is_class_segregation_on_ground_truth, template_initial_image_type,
                                                        template_image_calculation_batch_size, template_loss_type, wand_project_name, wandb_group_name, torch_seed, number_of_image_optimization_steps, collect_threshold, entropy_calculation_batch_size,
                                                        number_of_batches_to_calculate_entropy_on, root_save_prefix=root_save_prefix, final_postfix_for_save=final_postfix_for_save+postfix_save_append_str, wandb_config_additional_dict=wandb_config_additional_dict,
                                                        vis_version=vis_version, random_sample_gate_percent=random_sample_gate_percent,alpha = alpha,v4lr=v4lr,is_skip_collection=True)
            elif(exp_type == "GENERATE_TEMPLATE_IMAGES"):
                tmp_gen.generate_template_image_per_class(exp_type,
                                                        per_class_dataset, class_label, c_indx, number_of_batch_to_collect, classes, model_arch_type, dataset, is_template_image_on_train,
                                                        is_class_segregation_on_ground_truth, template_initial_image_type,
                                                        template_image_calculation_batch_size, template_loss_type, wand_project_name, wandb_group_name, torch_seed, number_of_image_optimization_steps, collect_threshold, entropy_calculation_batch_size,
                                                        number_of_batches_to_calculate_entropy_on, root_save_prefix=root_save_prefix, final_postfix_for_save=final_postfix_for_save, wandb_config_additional_dict=wandb_config_additional_dict,
                                                        vis_version=vis_version, random_sample_gate_percent=random_sample_gate_percent,alpha = alpha,v4lr=v4lr)

            elif(exp_type == "GENERATE_TEMPLATE_IMAGES_LAYER_WISE_AVG"):
                tmp_gen.generate_layer_wise_template_image_per_class(exp_type, per_class_dataset, class_label, c_indx, number_of_batch_to_collect, classes, model_arch_type, dataset, is_template_image_on_train,
                                                                    is_class_segregation_on_ground_truth, template_initial_image_type,
                                                                    template_image_calculation_batch_size, template_loss_type, wand_project_name, wandb_group_name, torch_seed, number_of_image_optimization_steps,
                                                                    layer_nums_to_visualize, root_save_prefix=root_save_prefix, final_postfix_for_save=final_postfix_for_save,
                                                                    wandb_config_additional_dict=wandb_config_additional_dict, vis_version=vis_version, random_sample_gate_percent=random_sample_gate_percent,alpha = alpha,v4lr=v4lr)
            elif(exp_type == "TEMPLATE_ACC_WITH_CUSTOM_PLOTS"):
                tmp_gen.generate_accuracies_of_template_image_per_class(
                    per_class_dataset, class_label, c_indx, classes, model_arch_type, dataset, is_template_image_on_train,
                    is_class_segregation_on_ground_truth, template_initial_image_type,
                    template_image_calculation_batch_size, template_loss_type, wand_project_name, wandb_group_name, torch_seed, number_of_image_optimization_steps,
                    exp_type, collect_threshold, entropy_calculation_batch_size, number_of_batches_to_calculate_entropy_on, plot_iteration_interval=10, wandb_config_additional_dict=wandb_config_additional_dict,
                    vis_version=vis_version, random_sample_gate_percent=random_sample_gate_percent,alpha = alpha,v4lr=v4lr)

            elif(exp_type == "TEMPLATE_ACC"):
                tmp_gen.generate_accuracies_of_template_image_per_class(
                    per_class_dataset, class_label, c_indx, classes, model_arch_type, dataset, is_template_image_on_train,
                    is_class_segregation_on_ground_truth, template_initial_image_type,
                    template_image_calculation_batch_size, template_loss_type, wand_project_name, wandb_group_name, torch_seed, number_of_image_optimization_steps, exp_type,
                    collect_threshold, entropy_calculation_batch_size, number_of_batches_to_calculate_entropy_on, wandb_config_additional_dict=wandb_config_additional_dict,
                    vis_version=vis_version, random_sample_gate_percent=random_sample_gate_percent,alpha = alpha,v4lr=v4lr)
            elif(exp_type == "GENERATE_ALL_FINAL_TEMPLATE_IMAGES"):
                list_of_reconst_images = tmp_gen.generate_accuracies_of_template_image_per_class(per_class_dataset, class_label, c_indx, classes, model_arch_type, dataset, is_template_image_on_train,
                                                                                                is_class_segregation_on_ground_truth, template_initial_image_type,
                                                                                                template_image_calculation_batch_size, template_loss_type, wand_project_name, wandb_group_name, torch_seed, number_of_image_optimization_steps,
                                                                                                exp_type, collect_threshold, entropy_calculation_batch_size, number_of_batches_to_calculate_entropy_on,
                                                                                                plot_iteration_interval=10, root_save_prefix=root_save_prefix, final_postfix_for_save=final_postfix_for_save,
                                                                                                wandb_config_additional_dict=wandb_config_additional_dict, vis_version=vis_version, random_sample_gate_percent=random_sample_gate_percent,alpha = alpha,v4lr=v4lr)
                output_template_list_per_class[c_indx] = list_of_reconst_images

    return output_template_list_per_class


if __name__ == '__main__':
    print("Start")
    # mnist , cifar10 , fashion_mnist , imagenet_1000
    dataset = 'fashion_mnist'
    # cifar10_conv4_dlgn , cifar10_vgg_dlgn_16 , dlgn_fc_w_128_d_4 , random_conv4_dlgn , random_vggnet_dlgn
    # random_conv4_dlgn_sim_vgg_wo_bn , cifar10_conv4_dlgn_sim_vgg_wo_bn , cifar10_conv4_dlgn_sim_vgg_with_bn
    # random_conv4_dlgn_sim_vgg_with_bn , cifar10_conv4_dlgn_with_inbuilt_norm , random_cifar10_conv4_dlgn_with_inbuilt_norm
    # cifar10_vgg_dlgn_16_with_inbuilt_norm , random_cifar10_vgg_dlgn_16_with_inbuilt_norm
    # random_cifar10_conv4_dlgn_with_bn_with_inbuilt_norm , cifar10_conv4_dlgn_with_bn_with_inbuilt_norm
    # cifar10_conv4_dlgn_with_inbuilt_norm_with_flip_crop
    # cifar10_conv4_dlgn_with_bn_with_inbuilt_norm_with_flip_crop
    # cifar10_vgg_dlgn_16_with_inbuilt_norm_wo_bn
    # plain_pure_conv4_dnn , conv4_dlgn , conv4_dlgn_n16_small , plain_pure_conv4_dnn_n16_small
    # conv4_deep_gated_net , conv4_deep_gated_net_n16_small ,dlgn__conv4_dlgn_pad_k_1_st1_bn_wo_bias__
    # dlgn__resnet18__ , dgn__resnet18__,dnn__resnet18__ , dlgn__vgg16_bn__ , dlgn__st1_pad1_vgg16_bn_wo_bias__ ,dlgn__st1_pad2_vgg16_bn_wo_bias__, dnn__st1_pad2_vgg16_bn_wo_bias__
    model_arch_type = 'conv4_dlgn'
    # If False, then on test
    is_template_image_on_train = True
    # If False, then segregation is over model prediction
    is_class_segregation_on_ground_truth = True
    # uniform_init_image , zero_init_image , gaussian_init_image
    template_initial_image_type = 'zero_init_image'
    template_image_calculation_batch_size = 32
    # MSE_LOSS , MSE_TEMP_LOSS_MIXED , ENTR_TEMP_LOSS , CCE_TEMP_LOSS_MIXED , TEMP_LOSS , CCE_ENTR_TEMP_LOSS_MIXED , TEMP_ACT_ONLY_LOSS
    # CCE_TEMP_ACT_ONLY_LOSS_MIXED , TANH_TEMP_LOSS
    # MSE_LAYER_LOSS
    # TEMP_LOSS_LATTCK , TANH_TEMP_LOSS_LATTCK
    # TANH_TEMP_LOSS_PLATTCK_EPS__0.3__ , TANH_TEMP_LOSS_RPLATTCK_EPS__0.3__
    template_loss_type = "TANH_TEMP_LOSS"
    number_of_batch_to_collect = None
    vis_version = "V4"
    wand_project_name = None
    # wand_project_name = "layerwise_avg_template_visualisation_augmentation"
    # wand_project_name = "latest_recheck_visualisation_augmentation"
    # wand_project_name = "fast_adv_tr_visualisation"
    # wand_project_name = "test_template_visualisation_augmentation"
    wandb_group_name = "TP_"+str(template_loss_type) + \
        "_DS_"+str(dataset)+"_MT_"+str(model_arch_type)
    is_split_validation = False
    valid_split_size = 0.1
    torch_seed = 2022
    alpha = 1
    p_epsilon = None
    # p_epsilon = 0.3
    v4lr = 0.1
    number_of_image_optimization_steps = 51
    # TEMPLATE_ACC,GENERATE_TEMPLATE_IMAGES , TEMPLATE_ACC_WITH_CUSTOM_PLOTS , GENERATE_ALL_FINAL_TEMPLATE_IMAGES , GENERATE_TEMPLATE_IMAGES_LAYER_WISE_AVG 
    # GENERATE_TEMPLATE_IMAGES_ADV_ORIG , GENERATE_TEMPLATE_IMAGES_FOR_CLASS_ANALYSIS
    exp_type = "GENERATE_TEMPLATE_IMAGES_ADV_ORIG"
    # ADV_DIFF_ORIG , ORIG_DIFF_ADV , ADV_OVRLP_ORIG , ONLY_ADV
    analysis_mode = "ONLY_ADV"
    temp_collect_threshold = 0.90
    entropy_calculation_batch_size = 64
    number_of_batches_to_calculate_entropy_on = None

    random_per_layer_sample_gate_percent = None
    # random_per_layer_sample_gate_percent = 1000

    layer_nums_to_visualize = None
    # layer_nums_to_visualize = [2]
    # layer_nums_to_visualize = [i for i in range(16)]

    class_indx_to_visualize = None
    if(dataset == 'imagenet_1000'):
        class_indx_to_visualize = [i for i in range(100, 111)]

    if(not(wand_project_name is None)):
        wandb.login()

    # for torch_seed in [2022]:
    #     for is_template_image_on_train in [True, False]:
    #         for template_loss_type in ["CCE_TEMP_LOSS_MIXED", "CCE_TEMP_ACT_ONLY_LOSS_MIXED"]:
    #             for collect_threshold in [0.51, 0.74, 0.81, 0.88, 0.93, 0.97]:
    #                 for template_image_calculation_batch_size in [2, 5, 10]:
    #                     wandb_group_name = "iter_500_seed_" + \
    #                         str(torch_seed)+"on_train_"+str(is_template_image_on_train) + \
    #                         "_LSS_"+str(template_loss_type) + \
    #                         "_thres_"+str(collect_threshold)+"_BS_COLL_" + \
    #                         str(template_image_calculation_batch_size)
    #                     for model_arch_type in ["cifar10_conv4_dlgn", "cifar10_vgg_dlgn_16"]:
    #                         if("ENTR" in template_loss_type):
    #                             for number_of_batches_to_calculate_entropy_on in [10, 20, None]:
    #                                 wandb_group_name += "_BS_ENTR_" + \
    #                                     str(number_of_batches_to_calculate_entropy_on)
    #                                 run_visualization_on_config(dataset, model_arch_type, is_template_image_on_train, is_class_segregation_on_ground_truth, template_initial_image_type,
    #                                                             template_image_calculation_batch_size, template_loss_type, number_of_batch_to_collect, wand_project_name, is_split_validation,
    #                                                             valid_split_size, torch_seed, number_of_image_optimization_steps, wandb_group_name, exp_type, collect_threshold, entropy_calculation_batch_size, number_of_batches_to_calculate_entropy_on)
    #                         else:
    #                             run_visualization_on_config(dataset, model_arch_type, is_template_image_on_train, is_class_segregation_on_ground_truth, template_initial_image_type,
    #                                                         template_image_calculation_batch_size, template_loss_type, number_of_batch_to_collect, wand_project_name, is_split_validation,
    #                                                         valid_split_size, torch_seed, number_of_image_optimization_steps, wandb_group_name, exp_type, collect_threshold, entropy_calculation_batch_size, number_of_batches_to_calculate_entropy_on)

    wandb_config = dict()
    custom_model_path = None

    # custom_model_path = "root/model/save/mnist/adversarial_training/MT_conv4_dlgn_ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_PGD/adv_type_PGD/EPS_0.3/batch_size_64/eps_stp_size_0.005/adv_steps_40/update_on_all/R_init_True/norm_inf/use_ytrue_True/adv_model_dir.pt"
    # custom_model_path = "root/model/save/mnist/CLEAN_TRAINING/ST_2022/conv4_dlgn_dir.pt"
    custom_model_path = "root/model/save/fashion_mnist/adversarial_training/MT_conv4_dlgn_ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_PGD/adv_type_PGD/EPS_0.3/OPT_Adam (Parameter Group 0    amsgrad: False    betas: (0.9, 0.999)    eps: 1e-08    lr: 0.0001    weight_decay: 0)/batch_size_64/eps_stp_size_0.01/adv_steps_40/update_on_all/R_init_True/norm_inf/use_ytrue_True/out_lossfn_CrossEntropyLoss()/inner_lossfn_CrossEntropyLoss()/adv_model_dir.pt"
    # custom_model_path = "root/model/save/fashion_mnist/CLEAN_TRAINING/ST_2022/conv4_dlgn_dir.pt"

    if(custom_model_path is not None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device", device)
        inp_channel = get_img_size(dataset)[0]

        wandb_config["custom_model_path"] = custom_model_path
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
        root_save_prefix = custom_model_path[0:custom_model_path.rfind(
            ".pt")]+"/"+"RECONST_IMAGES/"
    else:
        root_save_prefix = None
        custom_model = None
    
    if exp_type == "GENERATE_TEMPLATE_IMAGES_FOR_CLASS_ANALYSIS":
        # OVRLP , DIFF
        analysis_mode="DIFF"
        class_comb_length = 2

        for src_class_indx in range(len(classes)):
            src_classes_set = [src_class_indx]
            list_of_dest_classes_set = []
            if analysis_mode == "OVRLP":
                for dest_class_indx in range(src_class_indx+1,len(classes)) :
                    tmp = []
                    i=0
                    cur=dest_class_indx
                    while(i < class_comb_length-1):
                        if(cur % len(classes) not in src_classes_set):
                            tmp.append(cur % len(classes))
                            i+=1
                        cur+=1
                    list_of_dest_classes_set.append(tmp)
            elif analysis_mode == "DIFF":
                dest_class_indx = src_class_indx+1
                while((dest_class_indx+class_comb_length)%len(classes)-2 != src_class_indx ):
                    tmp = []
                    i=0
                    cur=dest_class_indx
                    print(dest_class_indx)
                    while(i < class_comb_length-1):
                        if(cur % len(classes) not in src_classes_set):
                            tmp.append(cur % len(classes))
                            i+=1
                        cur+=1
                    list_of_dest_classes_set.append(tmp)
                    dest_class_indx = (dest_class_indx+1) % len(classes)
            
            print("list_of_dest_classes_set ",list_of_dest_classes_set)
            run_visualization_on_config(dataset, model_arch_type, is_template_image_on_train, is_class_segregation_on_ground_truth, template_initial_image_type,
                                    template_image_calculation_batch_size, template_loss_type, number_of_batch_to_collect, wand_project_name, is_split_validation,
                                    valid_split_size, torch_seed, number_of_image_optimization_steps, wandb_group_name, exp_type, temp_collect_threshold, entropy_calculation_batch_size,
                                    number_of_batches_to_calculate_entropy_on, custom_model=custom_model, wandb_config_additional_dict=wandb_config, vis_version=vis_version,
                                    root_save_prefix=root_save_prefix, random_sample_gate_percent=random_per_layer_sample_gate_percent,
                                    class_indx_to_visualize=class_indx_to_visualize, layer_nums_to_visualize=layer_nums_to_visualize,alpha=alpha,p_epsilon=p_epsilon,
                                    v4lr=v4lr,analysis_mode=analysis_mode,src_classes_set=src_classes_set,list_of_dest_classes_set=list_of_dest_classes_set)

    else:
        run_visualization_on_config(dataset, model_arch_type, is_template_image_on_train, is_class_segregation_on_ground_truth, template_initial_image_type,
                                    template_image_calculation_batch_size, template_loss_type, number_of_batch_to_collect, wand_project_name, is_split_validation,
                                    valid_split_size, torch_seed, number_of_image_optimization_steps, wandb_group_name, exp_type, temp_collect_threshold, entropy_calculation_batch_size,
                                    number_of_batches_to_calculate_entropy_on, custom_model=custom_model, wandb_config_additional_dict=wandb_config, vis_version=vis_version,
                                    root_save_prefix=root_save_prefix, random_sample_gate_percent=random_per_layer_sample_gate_percent,analysis_mode=analysis_mode,
                                    class_indx_to_visualize=class_indx_to_visualize, layer_nums_to_visualize=layer_nums_to_visualize,alpha=alpha,p_epsilon=p_epsilon,v4lr=v4lr)

    print("Execution completed")
