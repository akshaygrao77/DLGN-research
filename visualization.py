import torch
import torchvision
from keras.datasets import mnist
from torch.optim import SGD
import os
from PIL import Image
import numpy as np
from tqdm import tqdm, trange
from torch.autograd import Variable
import subprocess
import copy
import math

from structure.dlgn_conv_config_structure import DatasetConfig
from algos.dlgn_conv_preprocess import add_channel_to_image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from configs.dlgn_conv_config import HardRelu

import torch.backends.cudnn as cudnn

import wandb

import time
from external_utils import format_time

from vgg_net_16 import DLGN_VGG_Network, DLGN_VGG_LinearNetwork, DLGN_VGG_WeightNetwork


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def get_data_loader(x_data, labels, bs, orig_labels=None):
    merged_data = []
    if(orig_labels is None):
        for i in range(len(x_data)):
            merged_data.append([x_data[i], labels[i]])
    else:
        for i in range(len(x_data)):
            merged_data.append([x_data[i], labels[i], orig_labels[i]])
    dataloader = torch.utils.data.DataLoader(
        merged_data, shuffle=True, batch_size=bs)
    return dataloader


def preprocess_dataset_get_data_loader(dataset_config, model_arch_type, verbose=1, dataset_folder='./Datasets/', is_split_validation=True):
    if(dataset_config.name == 'cifar10'):
        if(model_arch_type == 'cifar10_vgg_dlgn_16'):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
        elif(model_arch_type == 'cifar10_conv4_dlgn'):
            transform = transforms.Compose([
                transforms.ToTensor()])

        validloader = None
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        if(is_split_validation):
            trainset, val_set = torch.utils.data.random_split(trainset, [math.ceil(
                0.9 * len(trainset)), len(trainset) - (math.ceil(0.9 * len(trainset)))])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=dataset_config.batch_size,
                                                  shuffle=False, num_workers=2)
        if(is_split_validation):
            validloader = torch.utils.data.DataLoader(val_set, batch_size=dataset_config.batch_size,
                                                      shuffle=False, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=False, transform=transform)

        testloader = torch.utils.data.DataLoader(testset, batch_size=dataset_config.batch_size,
                                                 shuffle=False, num_workers=2)

        return trainloader, validloader, testloader
    elif(dataset_config.name == 'mnist'):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        # print("X_train[0].type", X_train.dtype)
        # print("X_train[0].shape", X_train.shape)
        # print("y_train[0].type", y_train.dtype)
        # print("y_train[0].shape", y_train.shape)

        if(dataset_config.is_normalize_data == True):
            max = np.max(X_train)
            X_train = X_train / max
            X_test = X_test / max
            if(verbose > 2):
                print("After normalizing dataset")
                print("Max value:{}".format(max))
                print("filtered_X_train size:{} filtered_y_train size:{}".format(
                    X_train.shape, y_train.shape))
                print("filtered_X_test size:{} y_test size:{}".format(
                    X_test.shape, y_test.shape))

        X_train = add_channel_to_image(X_train)
        X_test = add_channel_to_image(X_test)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=dataset_config.valid_split_size, random_state=42)

        train_data_loader = get_data_loader(
            X_train, y_train, dataset_config.batch_size)
        valid_data_loader = get_data_loader(
            X_valid, y_valid, dataset_config.batch_size)
        test_data_loader = get_data_loader(
            X_test, y_test, dataset_config.batch_size)

        return train_data_loader, valid_data_loader, test_data_loader


def true_segregation(data_loader, num_classes):
    input_data_list_per_class = [0] * num_classes
    for i in range(num_classes):
        input_data_list_per_class[i] = []

    data_loader = tqdm(data_loader, desc='Processing original loader')
    for i, inp_data in enumerate(data_loader):
        input_image, labels = inp_data
        for indx in range(len(labels)):
            each_label = labels[indx]
            input_data_list_per_class[each_label].append(input_image[indx])
    return input_data_list_per_class


def segregate_input_over_labels(model, data_loader, num_classes):
    print("Segregating predicted labels")
    # We don't need gradients on to do reporting
    model.train(False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_data_list_per_class = [None] * num_classes
    for i in range(num_classes):
        input_data_list_per_class[i] = []

    data_loader = tqdm(data_loader, desc='Processing loader')
    for i, inp_data in enumerate(data_loader):
        input_data, _ = inp_data

        input_data = input_data.to(device)

        outputs = model(input_data)

        outputs = outputs.softmax(dim=1).max(1).indices

        for indx in range(len(outputs)):
            each_out = outputs[indx]
            input_data_list_per_class[each_out].append(input_data[indx])

    return input_data_list_per_class


def multiply_lower_dimension_vectors_within_itself(input_tensor):
    init_dim = input_tensor[0]
    for i in range(1, input_tensor.size()[0]):
        t = input_tensor[i]
        # init_dim = init_dim * t
        init_dim = init_dim + t

    init_dim = HardRelu()(init_dim - 0.50 * input_tensor.size()[0])
    return init_dim


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


def recreate_image(im_as_var, unnormalize=True):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [0.4914, 0.4822, 0.4465]
    reverse_std = [1/0.2023, 1/0.1994, 1/0.2010]

    recreated_im = copy.copy(im_as_var.cpu().clone().detach().numpy()[0])
    arr_max = np.amax(recreated_im)
    arr_min = np.amin(recreated_im)
    recreated_im = (recreated_im-arr_min)/(arr_max-arr_min)

    if(unnormalize):
        for c in range(3):
            recreated_im[c] /= reverse_std[c]
            recreated_im[c] -= reverse_mean[c]
    # recreated_im[recreated_im > 1] = 1
    # recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im)
    # recreated_im = recreated_im..transpose(1, 2, 0)
    return recreated_im


class PerClassDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_images, label):
        self.list_of_images = list_of_images
        self.label = label

    def __len__(self):
        return len(self.list_of_images)

    def __getitem__(self, idx):
        image = self.list_of_images[idx]

        return image, self.label


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        for each_output in self.outputs:
            del each_output
        self.outputs = []


class TemplateImageGenerator():

    def __init__(self, model, start_image_np):
        self.model = model
        self.model.eval()
        # Generate a random image
        # self.created_image = Image.open(im_path).convert('RGB')
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

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
        # Hook the layers to get result of the convolution
        # self.hook_layer()
        # Create the folder to export images if not exists

    def reset_state(self):
        self.y_plus_list = None
        self.y_minus_list = None

    def initialise_y_plus_and_y_minus(self):
        self.y_plus_list = []
        self.y_minus_list = []
        if(isinstance(self.model, torch.nn.DataParallel)):
            conv_outs = self.model.module.linear_conv_outputs
        else:
            conv_outs = self.model.linear_conv_outputs
        for each_conv_output in conv_outs:
            current_y_plus = torch.ones(size=each_conv_output.size()[
                                        1:], requires_grad=True, device=self.device)
            current_y_minus = -torch.ones(size=each_conv_output.size()[
                1:], requires_grad=True, device=self.device)

            self.y_plus_list.append(current_y_plus)
            self.y_minus_list.append(current_y_minus)

    def hook_layer(self):
        self.saved_output = SaveOutput()

        hook_handles = []

        for layer in self.model.linear_conv_net.modules():
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                handle = layer.register_forward_hook(self.saved_output)
                hook_handles.append(handle)

    def update_y_lists(self):
        if(isinstance(self.model, torch.nn.DataParallel)):
            conv_outs = self.model.module.linear_conv_outputs
        else:
            conv_outs = self.model.linear_conv_outputs
        with torch.no_grad():
            for indx in range(len(conv_outs)):
                y_plus = self.y_plus_list[indx]
                each_conv_output = conv_outs[indx]
                positives = HardRelu()(each_conv_output)
                # [B,C,W,H]
                red_pos = multiply_lower_dimension_vectors_within_itself(
                    positives)
                self.y_plus_list[indx] = y_plus * red_pos

                y_minus = self.y_minus_list[indx]
                negatives = HardRelu()(-each_conv_output)
                red_neg = multiply_lower_dimension_vectors_within_itself(
                    negatives)
                self.y_minus_list[indx] = y_minus * red_neg

    def collect_active_pixel_per_batch(self, per_class_per_batch_data):
        c_inputs, _ = per_class_per_batch_data
        c_inputs = c_inputs.to(self.device)

        # Forward pass to store layer outputs from hooks
        self.model(c_inputs)

        # Intiialise the structure to hold i's for which pixels are positive or negative
        if(self.y_plus_list is None or self.y_minus_list is None):
            self.initialise_y_plus_and_y_minus()

        self.update_y_lists()

    def update_overall_y_maps(self):
        with torch.no_grad():
            self.overall_y = []
            for indx in range(len(self.y_plus_list)):
                each_y_plus = self.y_plus_list[indx]
                each_y_minus = self.y_minus_list[indx]
                # print("each_y_plus :{} ==>{}".format(indx, each_y_plus))
                # print("each_y_minus :{} ==>{}".format(indx, each_y_minus))

                self.overall_y.append(each_y_plus + each_y_minus)

    def collect_all_active_pixels_into_ymaps(self, per_class_data_loader, class_label, number_of_batch_to_collect):
        self.model.train(False)

        per_class_data_loader = tqdm(
            per_class_data_loader, desc='Collecting active maps class label:'+str(class_label))
        for i, per_class_per_batch_data in enumerate(per_class_data_loader):
            torch.cuda.empty_cache()
            c_inputs, _ = per_class_per_batch_data
            if(i == 0 and c_inputs.size()[0] == 1):
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

            self.collect_active_pixel_per_batch(per_class_per_batch_data)

            if(i == number_of_batch_to_collect - 1):
                break

        self.update_overall_y_maps()

    def calculate_loss_for_output_class_max_image(self, outputs, labels):
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)

        return loss

    def calculate_loss_to_maximise_logits_wrt_another_image_logits(self, outputs, targets):
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(outputs, targets)

        return loss

    def calculate_mixed_loss_maximise_logit_and_template_image(self, outputs, targets):
        mse_loss = self.calculate_loss_to_maximise_logits_wrt_another_image_logits(
            outputs, targets)
        template_loss, active_pixel_points, total_pixel_points = self.new_calculate_loss_for_template_image()
        alpha = 0.1
        overall_loss = alpha * mse_loss + (1-alpha)*template_loss
        return overall_loss, active_pixel_points, total_pixel_points

    def calculate_mixed_loss_output_class_and_template_image(self, outputs, labels, alpha=0.1):
        cce_loss = self.calculate_loss_for_output_class_max_image(
            outputs, labels)
        template_loss, active_pixel_points, total_pixel_points = self.new_calculate_loss_for_template_image()

        overall_loss = alpha * cce_loss + (1-alpha)*template_loss
        return overall_loss, active_pixel_points, total_pixel_points

    def calculate_loss_for_template_image(self):
        loss = None
        if(isinstance(self.model, torch.nn.DataParallel)):
            conv_outs = self.model.module.linear_conv_outputs
        else:
            conv_outs = self.model.linear_conv_outputs
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

    def new_calculate_loss_for_template_image(self):
        loss = 0
        if(isinstance(self.model, torch.nn.DataParallel)):
            conv_outs = self.model.module.linear_conv_outputs
        else:
            conv_outs = self.model.linear_conv_outputs
        total_pixel_points = 0
        active_pixel_points = 0
        for indx in range(len(conv_outs)):
            each_conv_output = conv_outs[indx]
            each_overall_y = self.overall_y[indx]

            total_pixel_points += torch.numel(each_conv_output)
            current_active_pixel = torch.count_nonzero(each_overall_y)
            active_pixel_points += current_active_pixel.item()
            pre_exponent = torch.exp(-each_overall_y *
                                     each_conv_output * 0.004)
            exp_product_active_pixels = torch.where(
                each_overall_y == 0, each_overall_y, pre_exponent)

            log_term = torch.log(1 + exp_product_active_pixels)

            each_conv_loss = torch.sum(log_term)

            loss += each_conv_loss

        return loss/active_pixel_points, active_pixel_points, total_pixel_points

    def get_wandb_config(self, exp_type, class_label, class_indx, classes, model_arch_type, dataset, is_template_image_on_train,
                         is_class_segregation_on_ground_truth, template_initial_image_type,
                         template_image_calculation_batch_size, template_loss_type, torch_seed, number_of_image_optimization_steps,
                         plot_iteration_interval=None, number_of_batch_to_collect=None):

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

        return wandb_config

    def get_prediction(self, input_tensor, original_label):

        outputs_raw = self.model(input_tensor)
        outputs_logits = outputs_raw.softmax(dim=1)
        outputs_final = outputs_logits.max(1).indices

        correct = outputs_final.eq(original_label).sum().item()

        return outputs_raw, outputs_logits, outputs_final, correct

    def generate_accuracies_of_template_image_per_class(self, per_class_dataset, class_label, class_indx, classes, model_arch_type, dataset, is_template_image_on_train,
                                                        is_class_segregation_on_ground_truth, template_initial_image_type,
                                                        template_image_calculation_batch_size, template_loss_type, wand_project_name, wandb_group_name, torch_seed, number_of_image_optimization_steps, exp_type, plot_iteration_interval=None):
        is_log_wandb = not(wand_project_name is None)

        torch.manual_seed(torch_seed)
        per_class_one_img_per_batch_data_loader = torch.utils.data.DataLoader(per_class_dataset, batch_size=template_image_calculation_batch_size,
                                                                              shuffle=True)
        tmp_image_over_what_str = 'test'
        if(is_template_image_on_train):
            tmp_image_over_what_str = 'train'

        seg_over_what_str = 'MP'
        if(is_class_segregation_on_ground_truth):
            seg_over_what_str = 'GT'

        self.model.train(False)
        self.image_save_prefix_folder = "root/"+str(dataset)+"/MT_"+str(model_arch_type)+"_ET_"+str(exp_type)+"/_COLL_OV_"+str(tmp_image_over_what_str)+"/SEG_"+str(
            seg_over_what_str)+"/TMP_COLL_BS_"+str(template_image_calculation_batch_size)+"/TMP_LOSS_TP_"+str(template_loss_type)+"/TMP_INIT_"+str(template_initial_image_type)+"/_torch_seed_"+str(torch_seed)+"/"

        per_class_one_img_per_batch_data_loader = tqdm(
            per_class_one_img_per_batch_data_loader, desc='Image being processed:'+str(class_label))
        total = 0
        reconst_correct = 0
        original_correct = 0
        alpha = 0
        normalize_image = False
        overall_step = 0

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
                "/", "").replace("root", class_label)
            wandb_config = self.get_wandb_config(exp_type, class_label, class_indx, classes, model_arch_type, dataset, is_template_image_on_train,
                                                 is_class_segregation_on_ground_truth, template_initial_image_type,
                                                 template_image_calculation_batch_size, template_loss_type, torch_seed, number_of_image_optimization_steps,
                                                 plot_iteration_interval)
            wandb_config["alpha"] = alpha

            wandb.init(
                project=f"{wand_project_name}",
                name=f"{wandb_run_name}",
                group=f"{wandb_group_name}",
                config=wandb_config,
            )

        for batch_indx, per_class_data in enumerate(per_class_one_img_per_batch_data_loader):
            self.reset_state()
            torch.cuda.empty_cache()

            class_image, original_label = per_class_data
            class_image = class_image.to(self.device, non_blocking=True)
            original_label = original_label.to(self.device, non_blocking=True)
            self.collect_active_pixel_per_batch(per_class_data)
            self.update_overall_y_maps()

            self.initial_image = preprocess_image(
                self.original_image.cpu().clone().detach().numpy(), normalize_image)

            self.initial_image = self.initial_image.to(self.device)

            self.initial_image.requires_grad_()

            step_size = 0.01
            with trange(number_of_image_optimization_steps, unit="iter", desc="Generating template image") as pbar:
                for step_iter in pbar:
                    pbar.set_description(f"Iteration {step_iter+1}")
                    # print("self.initial_image grad", self.initial_image.grad)
                    # self.initial_image.grad = None

                    # conv = torch.nn.Conv2d(
                    #     3, 3, 3, padding=1)
                    # conv = conv.to(self.device)
                    # self.initial_image_tilda = conv()

                    outputs = self.model(self.initial_image)

                    # loss, active_pixel_points, total_pixel_points = self.calculate_loss_for_template_image()
                    if(template_loss_type == "TEMP_LOSS"):
                        loss, active_pixel_points, total_pixel_points = self.new_calculate_loss_for_template_image()
                    elif(template_loss_type == "CCE_TEMP_LOSS_MIXED"):
                        actual = torch.tensor(
                            [class_indx] * len(outputs), device=self.device)
                        loss, active_pixel_points, total_pixel_points = self.calculate_mixed_loss_output_class_and_template_image(
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

                    if(step_iter == 0 and template_loss_type == "CCE_TEMP_LOSS_MIXED"):
                        percent_active_pixels = float((
                            active_pixel_points/total_pixel_points)*100)
                        print("active_pixel_points", active_pixel_points)
                        print("total_pixel_points", total_pixel_points)
                        print("Percentage of active pixels:",
                              percent_active_pixels)
                        if(is_log_wandb):
                            wandb.log(
                                {"active_pixel_points": active_pixel_points, "total_pixel_points": total_pixel_points,
                                 "Percent_active_pixels": percent_active_pixels}, step=(batch_indx+1))

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
                        self.initial_image = torch.clamp(
                            self.initial_image, -1, 1)

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

            with torch.no_grad():
                self.created_image = recreate_image(
                    self.initial_image, normalize_image)
                save_folder = self.image_save_prefix_folder + \
                    "class_"+str(class_label)+"/"
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                im_path = save_folder+'/no_optimizer_actual_c_' + \
                    str(class_label)+'_batch_indx' + str(batch_indx) + '.jpg'

                numpy_image = self.created_image
                save_image(numpy_image, im_path)

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

            total += 1
            reconst_correct += reconst_pred.eq(original_label).sum().item()

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

            if(is_log_wandb):
                wandb.log(
                    {"batch_indx": batch_indx,
                     "original_image_outputs_softmax": original_image_outputs_softmax, "original_img_label_pred": classes[original_image_pred], "original_img_pred_indx": original_image_pred,
                     }, step=(batch_indx+1))

        final_original_accuracy = (100. * original_correct/total)
        final_accuracy = (100. * reconst_correct/total)
        print("Overall class accuracy by template images:", final_accuracy)
        print("Overall class accuracy over template images:",
              final_original_accuracy)

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
                           "softmax_reconst_img_opt_steps_plt": wandb.plot.line_series(xs=step_lists,
                                                                                       ys=each_class_softmax_ordered_by_steps,
                                                                                       keys=classes,
                                                                                       title="Variation of softmax values across classes vs Optimization steps",
                                                                                       xname="Optimization steps")
                           })

        else:
            if(is_log_wandb):
                wandb.log({"final_acc_ov_reconst": final_accuracy,
                          "final_acc_ov_orig": final_original_accuracy})

        if(is_log_wandb):
            wandb.finish()

    def generate_template_image_per_class(self, exp_type, per_class_dataset, class_label, class_indx, number_of_batch_to_collect, classes, model_arch_type, dataset, is_template_image_on_train,
                                          is_class_segregation_on_ground_truth, template_initial_image_type,
                                          template_image_calculation_batch_size, template_loss_type, wand_project_name, wandb_group_name, torch_seed, number_of_image_optimization_steps):
        is_log_wandb = not(wand_project_name is None)
        plot_iteration_interval = 5

        torch.manual_seed(torch_seed)
        self.model.train(False)
        per_class_data_loader = torch.utils.data.DataLoader(per_class_dataset, batch_size=template_image_calculation_batch_size,
                                                            shuffle=True)

        tmp_image_over_what_str = 'test'
        if(is_template_image_on_train):
            tmp_image_over_what_str = 'train'

        seg_over_what_str = 'MP'
        if(is_class_segregation_on_ground_truth):
            seg_over_what_str = 'GT'

        alpha = 0
        self.image_save_prefix_folder = "root/"+str(dataset)+"/MT_"+str(model_arch_type)+"_ET_"+str(exp_type)+"/_COLL_OV_"+str(tmp_image_over_what_str)+"/SEG_"+str(
            seg_over_what_str)+"/TMP_COLL_BS_"+str(template_image_calculation_batch_size)+"/TMP_LOSS_TP_"+str(template_loss_type)+"/TMP_INIT_"+str(template_initial_image_type)+"/_torch_seed_"+str(torch_seed)+"/"

        self.image_save_prefix_folder += "_alp_" + str(alpha)+"/"
        normalize_image = False

        self.collect_all_active_pixels_into_ymaps(
            per_class_data_loader, class_label, number_of_batch_to_collect)

        class_image, _ = next(iter(per_class_data_loader))
        class_image = class_image.to(self.device, non_blocking=True)

        for repeat in range(2):

            if(repeat == 1):
                alpha = 0.1
                self.image_save_prefix_folder = "root/"+str(dataset)+"/MT_"+str(model_arch_type)+"_ET_"+str(exp_type)+"/_COLL_OV_"+str(tmp_image_over_what_str)+"/SEG_"+str(
                    seg_over_what_str)+"/TMP_COLL_BS_"+str(template_image_calculation_batch_size)+"/TMP_LOSS_TP_"+str(template_loss_type)+"/TMP_INIT_"+str(template_initial_image_type)+"/_torch_seed_"+str(torch_seed)+"/"
                self.image_save_prefix_folder += "_alp_" + str(alpha)+"/"

            print(
                "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ alpha", alpha)

            num_classes = len(classes)
            number_of_intervals = (
                number_of_image_optimization_steps // plot_iteration_interval) + 1
            list_of_reconst_softmax_pred = [None] * number_of_intervals

            if(is_log_wandb):
                wandb_run_name = self.image_save_prefix_folder.replace(
                    "/", "").replace("root", class_label)
                wandb_config = self.get_wandb_config(exp_type, class_label, class_indx, classes, model_arch_type, dataset, is_template_image_on_train,
                                                     is_class_segregation_on_ground_truth, template_initial_image_type,
                                                     template_image_calculation_batch_size, template_loss_type, torch_seed, number_of_image_optimization_steps,
                                                     number_of_batch_to_collect=number_of_batch_to_collect)
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

            with trange(number_of_image_optimization_steps, unit="iter") as pbar:
                for step_iter in pbar:
                    begin_time = time.time()
                    pbar.set_description(f"Iteration {step_iter+1}")
                    print("self.initial_image grad", self.initial_image.grad)
                    # self.initial_image.grad = None

                    # conv = torch.nn.Conv2d(
                    #     3, 3, 3, padding=1)
                    # conv = conv.to(self.device)
                    # self.initial_image_tilda = conv(self.initial_image)

                    outputs = self.model(self.initial_image)

                    if(template_loss_type == "TEMP_LOSS"):
                        loss, active_pixel_points, total_pixel_points = self.new_calculate_loss_for_template_image()
                    elif(template_loss_type == "CCE_TEMP_LOSS_MIXED"):
                        actual = torch.tensor(
                            [class_indx] * len(outputs), device=self.device)
                        loss, active_pixel_points, total_pixel_points = self.calculate_mixed_loss_output_class_and_template_image(
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

                    if(step_iter == 0 and template_loss_type == "CCE_TEMP_LOSS_MIXED"):
                        percent_active_pixels = float((
                            active_pixel_points/total_pixel_points)*100)
                        print("active_pixel_points", active_pixel_points)
                        print("total_pixel_points", total_pixel_points)
                        print("Percentage of active pixels:",
                              percent_active_pixels)
                        if(is_log_wandb):
                            wandb.log(
                                {"active_pixel_points": active_pixel_points, "total_pixel_points": total_pixel_points,
                                 "Percent_active_pixels": percent_active_pixels}, step=(step_iter+1))
                    # Backward
                    loss.backward()

                    unnorm_gradients = self.initial_image.grad
                    print("Original self.initial_image gradients",
                          unnorm_gradients)

                    gradients = unnorm_gradients / \
                        torch.std(unnorm_gradients) + 1e-8
                    print("After normalize self.initial_image gradients", gradients)

                    with torch.no_grad():
                        self.initial_image = self.initial_image - gradients*step_size
                        # self.initial_image = 0.9 * self.initial_image
                        self.initial_image = torch.clamp(
                            self.initial_image, -1, 1)

                    # Save image every plot_interval iteration
                    if step_iter % plot_iteration_interval == 0:
                        with torch.no_grad():
                            reconst_outputs = self.model(self.initial_image)
                            reconst_outputs_softmax = reconst_outputs.softmax(
                                dim=1)
                            reconst_img_norm = torch.norm(self.initial_image)
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

                    pbar.set_postfix(loss=loss, it_time=format_time(tot_time))

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
                each_class_softmax_ordered_by_steps = [
                    None] * num_classes

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


def print_segregation_info(input_data_list_per_class):
    sum = 0
    for indx in range(len(input_data_list_per_class)):
        each_inp = input_data_list_per_class[indx]
        length = len(each_inp)
        sum += length
        print("Indx {} len:{}".format(indx, length))
    print("Sum", sum)


def get_model_from_loader(model_arch_type, dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Loading model")
    if(dataset == "cifar10"):
        if(model_arch_type == 'cifar10_vgg_dlgn_16'):
            model = torch.load("root/model/save/vggnet_ext_parallel_16_dir.pt")
        elif(model_arch_type == 'cifar10_conv4_dlgn'):
            model = torch.load("root/model/save/model_norm_dir_None.pt")

        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device_str == 'cuda':
            if(torch.cuda.device_count() > 1):
                print("Parallelizing model")
                model = torch.nn.DataParallel(model)
            cudnn.benchmark = True

    elif(dataset == "mnist"):
        if(model_arch_type == 'cifar10_conv4_dlgn'):
            model = torch.load("root/model/save/model_mnist_norm_dir_None.pt")

        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device_str == 'cuda':
            if(torch.cuda.device_count() > 1):
                print("Parallelizing model")
                model = torch.nn.DataParallel(model)
            cudnn.benchmark = True

    model.to(device)
    print("Model loaded of type:{} for dataset:{}".format(model_arch_type, dataset))

    return model


def run_visualization_on_config(dataset, model_arch_type, is_template_image_on_train, is_class_segregation_on_ground_truth, template_initial_image_type,
                                template_image_calculation_batch_size, template_loss_type, number_of_batch_to_collect, wand_project_name, is_split_validation, valid_split_size, torch_seed, number_of_image_optimization_steps, wandb_group_name, exp_type):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if(dataset == "cifar10"):
        print("Running for CIFAR 10")
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        cifar10_config = DatasetConfig(
            'cifar10', is_normalize_data=False, valid_split_size=valid_split_size, batch_size=128)

        trainloader, validloader, testloader = preprocess_dataset_get_data_loader(
            cifar10_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=is_split_validation)

    elif(dataset == "mnist"):
        print("Running for MNIST")
        classes = [i for i in range(0, 10)]
        mnist_config = DatasetConfig(
            'mnist', is_normalize_data=True, valid_split_size=valid_split_size, batch_size=128)

        trainloader, validloader, testloader = preprocess_dataset_get_data_loader(
            mnist_config, model_arch_type, verbose=1, dataset_folder="./Datasets/", is_split_validation=is_split_validation)

    print("Preprocessing and dataloader process completed of type:{} for dataset:{}".format(
        model_arch_type, dataset))
    model = get_model_from_loader(model_arch_type, dataset)
    input_data_list_per_class = None

    if(is_template_image_on_train):
        train_repdicted_input_data_list_per_class = segregate_input_over_labels(
            model, trainloader, 10)

        print("train Model segregation of classes:")
        print_segregation_info(train_repdicted_input_data_list_per_class)

        train_true_input_data_list_per_class = true_segregation(
            trainloader, 10)

        print("trainset Ground truth segregation of classes:")
        print_segregation_info(train_true_input_data_list_per_class)
        if(is_class_segregation_on_ground_truth):
            input_data_list_per_class = train_true_input_data_list_per_class
        else:
            input_data_list_per_class = train_repdicted_input_data_list_per_class
    else:
        test_predicted_input_data_list_per_class = segregate_input_over_labels(
            model, testloader, 10)

        print("Model segregation of classes:")
        print_segregation_info(test_predicted_input_data_list_per_class)

        test_true_input_data_list_per_class = true_segregation(testloader, 10)

        print("Ground truth segregation of classes:")
        print_segregation_info(test_true_input_data_list_per_class)

        if(is_class_segregation_on_ground_truth):
            input_data_list_per_class = test_true_input_data_list_per_class
        else:
            input_data_list_per_class = test_predicted_input_data_list_per_class

    for c_indx in range(len(classes)):
        class_label = classes[c_indx]
        print("************************************************************ Class:", class_label)
        per_class_dataset = PerClassDataset(
            input_data_list_per_class[c_indx], c_indx)
        # per_class_loader = torch.utils.data.DataLoader(per_class_dataset, batch_size=32,
        #                                                shuffle=False)

        if(dataset == "cifar10"):
            if(template_initial_image_type == 'zero_init_image'):
                tmp_gen = TemplateImageGenerator(
                    model, torch.from_numpy(np.uint8(np.random.uniform(0, 1, (3, 32, 32)))))
        elif(dataset == "mnist"):
            if(template_initial_image_type == 'zero_init_image'):
                tmp_gen = TemplateImageGenerator(
                    model, torch.from_numpy(np.uint8(np.random.uniform(0, 1, (1, 28, 28)))))

        if(exp_type == "GENERATE_TEMPLATE_IMAGES"):
            tmp_gen.generate_template_image_per_class(exp_type,
                                                      per_class_dataset, class_label, c_indx, number_of_batch_to_collect, classes, model_arch_type, dataset, is_template_image_on_train,
                                                      is_class_segregation_on_ground_truth, template_initial_image_type,
                                                      template_image_calculation_batch_size, template_loss_type, wand_project_name, wandb_group_name, torch_seed, number_of_image_optimization_steps)

        elif(exp_type == "TEMPLATE_ACC_WITH_CUSTOM_PLOTS"):
            tmp_gen.generate_accuracies_of_template_image_per_class(
                per_class_dataset, class_label, c_indx, classes, model_arch_type, dataset, is_template_image_on_train,
                is_class_segregation_on_ground_truth, template_initial_image_type,
                template_image_calculation_batch_size, template_loss_type, wand_project_name, wandb_group_name, torch_seed, number_of_image_optimization_steps, exp_type, plot_iteration_interval=10)

        elif(exp_type == "TEMPLATE_ACC"):
            tmp_gen.generate_accuracies_of_template_image_per_class(
                per_class_dataset, class_label, c_indx, classes, model_arch_type, dataset, is_template_image_on_train,
                is_class_segregation_on_ground_truth, template_initial_image_type,
                template_image_calculation_batch_size, template_loss_type, wand_project_name, wandb_group_name, torch_seed, number_of_image_optimization_steps, exp_type)


if __name__ == '__main__':
    # THIS OPERATION IS MEMORY HUNGRY! #
    # Because of the selected image is very large
    # If it gives out of memory error or locks the computer
    # Try it with a smaller image
    print("Start")
    dataset = 'cifar10'
    # cifar10_conv4_dlgn , cifar10_vgg_dlgn_16
    model_arch_type = 'cifar10_conv4_dlgn'
    # If False, then on test
    is_template_image_on_train = False
    # If False, then segregation is over model prediction
    is_class_segregation_on_ground_truth = True
    template_initial_image_type = 'zero_init_image'
    template_image_calculation_batch_size = 1
    # MSE_LOSS , MSE_TEMP_LOSS_MIXED
    template_loss_type = "CCE_TEMP_LOSS_MIXED"
    number_of_batch_to_collect = 1
    # wand_project_name = "template_visualization"
    wand_project_name = "test_gen_visualization"
    # wand_project_name = None
    wandb_group_name = "test_1_51"
    is_split_validation = True
    valid_split_size = 0.1
    torch_seed = 2022
    number_of_image_optimization_steps = 51
    # TEMPLATE_ACC,GENERATE_TEMPLATE_IMAGES , TEMPLATE_ACC_WITH_CUSTOM_PLOTS
    exp_type = "GENERATE_TEMPLATE_IMAGES"

    if(not(wand_project_name is None)):
        wandb.login()

    run_visualization_on_config(dataset, model_arch_type, is_template_image_on_train, is_class_segregation_on_ground_truth, template_initial_image_type,
                                template_image_calculation_batch_size, template_loss_type, number_of_batch_to_collect, wand_project_name, is_split_validation,
                                valid_split_size, torch_seed, number_of_image_optimization_steps, wandb_group_name, exp_type)

    print("Execution completed")
