import torch
from torch.optim import SGD
import os
from PIL import Image
import numpy as np
import tqdm
from torch.autograd import Variable
import subprocess
import copy
import math

from structure.dlgn_conv_config_structure import DatasetConfig
from algos.dlgn_conv_preprocess import preprocess_dataset_get_data_loader
from configs.dlgn_conv_config import HardRelu

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


def segregate_input_over_labels(model, data_loader, num_classes):
    print("Segregating predicted labels")
    # We don't need gradients on to do reporting
    model.train(False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_data_list_per_class = [None] * num_classes
    for i in range(num_classes):
        input_data_list_per_class[i] = []

    data_loader = tqdm.tqdm(data_loader, desc='Processing loader')
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

    init_dim = HardRelu()(init_dim - 0.5 * input_tensor.size()[0])
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
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]

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


image_save_prefix_folder = "root/cifar10-vggnet_16/generated/zero_image_init_50_active/"


class TemplateImageGenerator():

    def __init__(self, model, start_image_np):
        self.model = model
        self.model.eval()
        # Generate a random image
        # self.created_image = Image.open(im_path).convert('RGB')
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

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

    def initialise_y_plus_and_y_minus(self):
        self.y_plus_list = []
        self.y_minus_list = []
        # conv_outs = self.saved_output.outputs
        conv_outs = self.model.linear_conv_outputs
        for each_conv_output in conv_outs:
            current_y_plus = torch.ones(size=each_conv_output.size()[
                                        1:], requires_grad=True, device=device)
            current_y_minus = -torch.ones(size=each_conv_output.size()[
                1:], requires_grad=True, device=self.device)

            print("current_y_plus size", current_y_plus.size())

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
        # conv_outs = self.saved_output.outputs
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

    def collect_all_active_pixels_into_ymaps(self, per_class_data_loader, class_label):
        self.model.train(False)

        per_class_data_loader = tqdm.tqdm(
            per_class_data_loader, desc='Collecting active maps class label:'+str(class_label))
        for i, per_class_data in enumerate(per_class_data_loader):
            torch.cuda.empty_cache()

            c_inputs, _ = per_class_data
            c_inputs = c_inputs.to(self.device)

            # Forward pass to store layer outputs from hooks
            self.model(c_inputs)

            # Intiialise the structure to hold i's for which pixels are positive or negative
            if(self.y_plus_list is None or self.y_minus_list is None):
                self.initialise_y_plus_and_y_minus()

            self.update_y_lists()

        with torch.no_grad():
            self.overall_y = []
            for indx in range(len(self.y_plus_list)):
                each_y_plus = self.y_plus_list[indx]
                each_y_minus = self.y_minus_list[indx]
                # print("each_y_plus :{} ==>{}".format(indx, each_y_plus))
                # print("each_y_minus :{} ==>{}".format(indx, each_y_minus))

                self.overall_y.append(each_y_plus + each_y_minus)
        # print("self.overall_y", self.overall_y)

    def calculate_loss_for_output_class_max_image(self, outputs, labels):
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)

        return loss

    def calculate_loss_for_template_image(self):
        loss = None
        # conv_outs = self.saved_output.outputs
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

        print("Percentage of active pixels:", float((
            active_pixel_points/total_pixel_points)*100))
        return loss/active_pixel_points

    def generate_template_image_per_class(self, per_class_data_loader, class_label, class_indx):
        normalize_image = False
        self.collect_all_active_pixels_into_ymaps(
            per_class_data_loader, class_label)

        self.initial_image = preprocess_image(
            self.original_image.cpu().clone().detach().numpy(), normalize_image)

        self.initial_image = self.initial_image.to(self.device)

        self.initial_image.requires_grad_()

        print("self.initial_image size", self.initial_image.size())

        step_size = 0.01
        for i in range(11):
            print("Iteration number:", i)
            print("self.initial_image grad", self.initial_image.grad)
            # self.initial_image.grad = None

            # conv = torch.nn.Conv2d(
            #     3, 3, 3, padding=1)
            # conv = conv.to(self.device)
            # self.initial_image_tilda = conv(self.initial_image)

            outputs = self.model(self.initial_image)

            loss = self.calculate_loss_for_template_image()
            # actual = torch.tensor(
            #     [class_indx] * len(outputs), device=self.device)
            # loss = self.calculate_loss_for_output_class_max_image(
            #     outputs, actual)

            print('Loss:', loss)
            # Backward
            loss.backward()

            gradients = self.initial_image.grad
            print("Original self.initial_image gradients", gradients)

            gradients /= torch.std(gradients) + 1e-8
            print("After normalize self.initial_image gradients", gradients)

            with torch.no_grad():
                self.initial_image = self.initial_image - gradients*step_size
                # self.initial_image = 0.9 * self.initial_image
                self.initial_image = torch.clamp(self.initial_image, -1, 1)

            self.initial_image.requires_grad_()
            # Recreate image
            print("self.initial_image", self.initial_image)

            # Save image every 20 iteration
            if i % 5 == 0:
                # self.created_image = recreate_image(
                #     self.initial_image_tilda, normalize_image)
                # print("self.created_image.shape::", self.created_image.shape)
                # save_folder = image_save_prefix_folder + \
                #     "class_"+str(class_label)+"/"
                # if not os.path.exists(save_folder):
                #     os.makedirs(save_folder)
                # im_path = save_folder+'/no_optimizer_tilda_c_' + \
                #     str(class_label)+'_iter' + str(i) + '.jpg'
                # # numpy_image = self.created_image.cpu().clone().detach().numpy()
                # numpy_image = self.created_image
                # save_image(numpy_image, im_path)

                self.created_image = recreate_image(
                    self.initial_image, normalize_image)
                print("self.created_image.shape::", self.created_image.shape)
                save_folder = image_save_prefix_folder + \
                    "class_"+str(class_label)+"/"
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                im_path = save_folder+'/no_optimizer_actual_c_' + \
                    str(class_label)+'_iter' + str(i) + '.jpg'
                # numpy_image = self.created_image.cpu().clone().detach().numpy()
                numpy_image = self.created_image
                save_image(numpy_image, im_path)

    # def generate_template_image_per_class(self, per_class_data_loader, class_label, class_indx):
    #     normalize_image = False
    #     self.collect_all_active_pixels_into_ymaps(
    #         per_class_data_loader, class_label)

    #     self.initial_image = preprocess_image(
    #         self.original_image.cpu().clone().detach().numpy(), normalize_image)

    #     self.initial_image = self.initial_image.to(self.device)

    #     self.initial_image.requires_grad_()

    #     print("self.initial_image size", self.initial_image.size())

    #     step_size = 0.01
    #     for i in range(100):
    #         print("Iteration number:", i)
    #         print("self.initial_image grad", self.initial_image.grad)
    #         # self.initial_image.grad = None

    #         outputs = self.model(self.initial_image)

    #         loss = self.calculate_loss_for_template_image()
    #         # actual = torch.tensor(
    #         #     [class_indx] * len(outputs), device=self.device)
    #         # loss = self.calculate_loss_for_output_class_max_image(
    #         #     outputs, actual)

    #         print('Loss:', loss)
    #         # Backward
    #         loss.backward()

    #         gradients = self.initial_image.grad
    #         print("Original self.initial_image gradients", gradients)

    #         gradients /= torch.std(gradients) + 1e-8
    #         print("After normalize self.initial_image gradients", gradients)

    #         with torch.no_grad():
    #             self.initial_image = self.initial_image - gradients*step_size
    #             # self.initial_image = 0.9 * self.initial_image
    #             self.initial_image = torch.clamp(self.initial_image, -1, 1)

    #         self.initial_image.requires_grad_()
    #         # Recreate image
    #         print("self.initial_image", self.initial_image)

    #         # Save image every 20 iteration
    #         if i % 5 == 0:
    #             self.created_image = recreate_image(
    #                 self.initial_image, normalize_image)
    #             print("self.created_image.shape::", self.created_image.shape)
    #             save_folder = image_save_prefix_folder + \
    #                 "class_"+str(class_label)+"/"
    #             if not os.path.exists(save_folder):
    #                 os.makedirs(save_folder)
    #             im_path = save_folder+'/no_optimizer_template_c_' + \
    #                 str(class_label)+'_iter' + str(i) + '.jpg'
    #             # numpy_image = self.created_image.cpu().clone().detach().numpy()
    #             numpy_image = self.created_image
    #             save_image(numpy_image, im_path)

    # def generate_template_image_per_class(self, per_class_data_loader, class_label, c_indx):
    #     normalize_image = False
    #     self.collect_all_active_pixels_into_ymaps(
    #         per_class_data_loader, class_label)

    #     self.initial_image = preprocess_image(
    #         self.original_image.cpu().clone().detach().numpy(), normalize_image)

    #     self.initial_image = self.initial_image.to(self.device)

    #     self.initial_image.requires_grad_()

    #     print("self.initial_image size", self.initial_image.size())

    #     # Define optimizer for the image
    #     optimizer = torch.optim.SGD(
    #         [self.initial_image], lr=100, momentum=0.9)
    #     for i in range(100):
    #         optimizer.zero_grad()

    #         self.model(self.initial_image)

    #         loss = self.calculate_loss_for_template_image()
    #         print('Loss:', loss)
    #         # Backward
    #         loss.backward()

    #         # Update image
    #         optimizer.step()
    #         # Recreate image
    #         print("Iteration number:", i)
    #         print("self.initial_image grad", self.initial_image.grad)
    #         print("self.initial_image", self.initial_image)

    #         # Save image every 20 iteration
    #         if i % 5 == 0:
    #             self.created_image = recreate_image(
    #                 self.initial_image, normalize_image)
    #             print("self.created_image.shape::", self.created_image.shape)
    #             im_path = 'root/generated/template_from_rand_image_c_' + \
    #                 str(class_label)+'_iter' + str(i) + '.jpg'
    #             # numpy_image = self.created_image.cpu().clone().detach().numpy()
    #             numpy_image = self.created_image
    #             save_image(numpy_image, im_path)


if __name__ == '__main__':
    # THIS OPERATION IS MEMORY HUNGRY! #
    # Because of the selected image is very large
    # If it gives out of memory error or locks the computer
    # Try it with a smaller image
    print("Start")
    dataset = 'cifar'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if(dataset == "cifar"):
        print("Running for CIFAR 10")
        cifar10_config = DatasetConfig(
            'cifar10', is_normalize_data=False, valid_split_size=0.1, batch_size=128)

        trainloader, validloader, testloader = preprocess_dataset_get_data_loader(
            cifar10_config, verbose=1, dataset_folder="./Datasets/")

        print("Loading model")
        # model = torch.load("root/model/save/model_dir_None.pt")
        model = torch.load("root/model/save/vggnet_16_dir.pt")
        print("Model loaded")

    elif(dataset == "mnist"):
        mnist_config = DatasetConfig(
            'mnist', is_normalize_data=True, valid_split_size=0.1, batch_size=128)

        trainloader, validloader, testloader = preprocess_dataset_get_data_loader(
            mnist_config, verbose=1, dataset_folder="./Datasets/")

        model = torch.load("root/model/save/model_mnist_norm_dir_None.pt")

    model.to(device)

    input_data_list_per_class = segregate_input_over_labels(
        model, trainloader, 10)

    sum = 0
    for indx in range(len(input_data_list_per_class)):
        each_inp = input_data_list_per_class[indx]
        length = len(each_inp)
        sum += length
        print("Indx {} len:{}".format(indx, length))
    print("Sum", sum)

    if(dataset == "cifar"):
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    elif(dataset == "mnist"):
        classes = [i for i in range(0, 10)]

    for c_indx in range(len(classes)):
        class_label = classes[c_indx]
        print("************************************************************ Class:", class_label)
        per_class_dataset = PerClassDataset(
            input_data_list_per_class[c_indx], c_indx)
        per_class_loader = torch.utils.data.DataLoader(per_class_dataset, batch_size=32,
                                                       shuffle=False)
        # tmp_gen = TemplateImageGenerator(
        #     model, input_data_list_per_class[c_indx][0])
        if(dataset == "cifar"):
            tmp_gen = TemplateImageGenerator(
                model, torch.tensor(np.uint8(np.random.uniform(0, 1, (3, 32, 32)))))
        elif(dataset == "mnist"):
            tmp_gen = TemplateImageGenerator(
                model, torch.tensor(np.uint8(np.random.uniform(0, 1, (1, 28, 28)))))

        tmp_gen.generate_template_image_per_class(
            per_class_loader, class_label, c_indx)
