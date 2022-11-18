from keras.datasets import mnist
import torch
import numpy as np
import random
from tqdm import tqdm, trange
from torch.autograd import Variable
from PIL import Image
import copy
import torch.optim as optim
from conv4_models import get_model_instance_from_dataset
import scipy.ndimage as nd

import torch.nn as nn


class HardRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()  # init the base class

    def forward(self, inputs):
        return HardReLU_F.apply(inputs)


class HardReLU_F(torch.autograd.Function):
    # both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)  # save input for backward pass
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        values = torch.tensor([0], dtype=inputs.dtype, device=device)
        retval = torch.heaviside(inputs, values)
        return retval

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_input = None  # set output to None

        inputs, = ctx.saved_tensors  # restore input from context

        # check that input requires grad
        # if not requires grad we will return None to speed up computation
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()

            grad_input[inputs <= 0] = 0

            grad_input[inputs > 0] = 0

        return grad_input


def blur_img(img, sigma):
    # print("blur img shape", img.shape)
    if sigma > 0:
        img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
        # img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
        # img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
    return img


class PerClassDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_images, label):
        self.list_of_images = list_of_images
        self.label = label

    def __len__(self):
        return len(self.list_of_images)

    def __getitem__(self, idx):
        image = self.list_of_images[idx]

        return image, self.label


def add_channel_to_image(X):
    out_X = []
    for each_X in X:
        out_X.append(each_X[None, :])
    return out_X


def get_data_loader(x_data, labels, bs, orig_labels=None):
    merged_data = []
    if(orig_labels is None):
        for i in range(len(x_data)):
            merged_data.append([x_data[i], labels[i]])
    else:
        for i in range(len(x_data)):
            merged_data.append([x_data[i], labels[i], orig_labels[i]])
    dataloader = torch.utils.data.DataLoader(
        merged_data, shuffle=False, batch_size=bs)
    return dataloader


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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_id + worker_seed)
    random.seed(worker_id - worker_seed)


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


def add_lower_dimension_vectors_within_itself(input_tensor):
    init_dim = input_tensor[0]
    for i in range(1, input_tensor.size()[0]):
        t = input_tensor[i]
        init_dim = init_dim + t

    return init_dim


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

        self.y_plus_list = None
        self.y_minus_list = None

    def reset_collection_state(self):
        self.y_plus_list = None
        self.y_minus_list = None

    def initialise_y_plus_and_y_minus(self):
        self.y_plus_list = []
        self.y_minus_list = []
        self.total_tcollect_img_count = 0
        if(isinstance(self.model, torch.nn.DataParallel)):
            conv_outs = self.model.module.linear_conv_outputs
        else:
            conv_outs = self.model.linear_conv_outputs
        for each_conv_output in conv_outs:
            current_y_plus = torch.zeros(size=each_conv_output.size()[
                1:], requires_grad=True, device=self.device)
            current_y_minus = torch.zeros(size=each_conv_output.size()[
                1:], requires_grad=True, device=self.device)

            self.y_plus_list.append(current_y_plus)
            self.y_minus_list.append(current_y_minus)

    def find_overall_between_two_images(self, inp1, inp2):
        inp1 = inp1.to(self.device)

        # Forward pass to store layer outputs from hooks
        self.model(inp1)
        if(isinstance(self.model, torch.nn.DataParallel)):
            conv_outs1 = self.model.module.linear_conv_outputs
        else:
            conv_outs1 = self.model.linear_conv_outputs

        inp2 = inp2.to(self.device)

        # Forward pass to store layer outputs from hooks
        self.model(inp2)
        if(isinstance(self.model, torch.nn.DataParallel)):
            conv_outs2 = self.model.module.linear_conv_outputs
        else:
            conv_outs2 = self.model.linear_conv_outputs

        overlap_count = 0
        total_pixel_points = 0

        with torch.no_grad():
            for indx in range(len(conv_outs1)):
                # for indx in range(0, 4):
                each_conv_output1 = conv_outs1[indx]
                each_conv_output2 = conv_outs2[indx]

                positives1 = HardRelu()(each_conv_output1)
                positives2 = HardRelu()(each_conv_output2)

                total_pixel_points += torch.numel(positives1)

                red_pos1 = add_lower_dimension_vectors_within_itself(
                    positives1)
                red_pos2 = add_lower_dimension_vectors_within_itself(
                    positives2)

                over_pos = red_pos1 * red_pos2
                # print("over_pos", over_pos)

                overlap_count += torch.count_nonzero(over_pos)

                negatives1 = HardRelu()(-each_conv_output1)
                negatives2 = HardRelu()(-each_conv_output2)
                red_neg1 = add_lower_dimension_vectors_within_itself(
                    negatives1)
                red_neg2 = add_lower_dimension_vectors_within_itself(
                    negatives2)

                over_neg = red_neg1 * red_neg2
                # print("over_neg", over_neg)
                overlap_count += torch.count_nonzero(over_neg)

        return overlap_count, total_pixel_points, 100. * overlap_count/total_pixel_points

    def update_y_lists(self):
        if(isinstance(self.model, torch.nn.DataParallel)):
            conv_outs = self.model.module.linear_conv_outputs
        else:
            conv_outs = self.model.linear_conv_outputs
        with torch.no_grad():
            for indx in range(len(conv_outs)):
                # for indx in range(0, 4):
                each_conv_output = conv_outs[indx]
                positives = HardRelu()(each_conv_output)
                # [B,C,W,H]
                red_pos = add_lower_dimension_vectors_within_itself(
                    positives)
                self.y_plus_list[indx] += red_pos

                negatives = HardRelu()(-each_conv_output)
                red_neg = add_lower_dimension_vectors_within_itself(
                    negatives)
                self.y_minus_list[indx] += red_neg

            self.total_tcollect_img_count += conv_outs[0].size()[0]

    def collect_active_pixel_per_batch(self, per_class_per_batch_data):
        c_inputs, _ = per_class_per_batch_data
        c_inputs = c_inputs.to(self.device)

        # Forward pass to store layer outputs from hooks
        self.model(c_inputs)

        # Intiialise the structure to hold i's for which pixels are positive or negative
        if(self.y_plus_list is None or self.y_minus_list is None):
            self.initialise_y_plus_and_y_minus()

        self.update_y_lists()

    def update_overall_y_maps(self, collect_threshold):

        with torch.no_grad():
            self.overall_y = []
            for indx in range(len(self.y_plus_list)):
                each_y_plus = self.y_plus_list[indx]
                each_y_minus = self.y_minus_list[indx]

                y_plus_passed_threshold = HardRelu()(each_y_plus - collect_threshold *
                                                     self.total_tcollect_img_count)
                y_minus_passed_threshold = HardRelu()(each_y_minus - collect_threshold *
                                                      self.total_tcollect_img_count)

                current_y_map = y_plus_passed_threshold - y_minus_passed_threshold

                self.overall_y.append(current_y_map)

    def collect_all_active_pixels_into_ymaps(self, per_class_data_loader, class_label, number_of_batch_to_collect, collect_threshold, is_save_original_image=True):
        self.reset_collection_state()
        self.model.train(False)

        per_class_data_loader = tqdm(
            per_class_data_loader, desc='Collecting active maps class label:'+str(class_label))
        for i, per_class_per_batch_data in enumerate(per_class_data_loader):
            torch.cuda.empty_cache()
            c_inputs, _ = per_class_per_batch_data

            self.collect_active_pixel_per_batch(
                per_class_per_batch_data)

            if(not(number_of_batch_to_collect is None) and i == number_of_batch_to_collect - 1):
                break

        self.update_overall_y_maps(collect_threshold)

    def new_calculate_loss_for_template_image(self):
        loss = 0
        if(isinstance(self.model, torch.nn.DataParallel)):
            conv_outs = self.model.module.linear_conv_outputs
        else:
            conv_outs = self.model.linear_conv_outputs

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
            pre_exponent = torch.exp(-each_overall_y *
                                     each_conv_output * 0.004)

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

    def get_loss_value(self, template_loss_type, class_indx, outputs=None, class_image=None, alpha=None):
        active_pixel_points = None
        total_pixel_points = None
        non_zero_pixel_points = None

        if(template_loss_type == "TEMP_LOSS"):
            loss, active_pixel_points, total_pixel_points, non_zero_pixel_points = self.new_calculate_loss_for_template_image()

        return loss, active_pixel_points, total_pixel_points, non_zero_pixel_points

    def generate_template_image_over_given_image(self, image_to_collect_upon, number_of_image_optimization_steps, template_loss_type):
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

        with trange(number_of_image_optimization_steps, unit="iter", desc="Generating template image for given image") as pbar:
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

                loss, active_pixel_points, total_pixel_points, non_zero_pixel_points = self.get_loss_value(
                    template_loss_type, class_indx=0, outputs=outputs)

                if(step_iter == 0 and "TEMP" in template_loss_type):
                    percent_active_pixels = float((
                        active_pixel_points/total_pixel_points)*100)
                    print("active_pixel_points", active_pixel_points)
                    print("total_pixel_points", total_pixel_points)
                    print("Percentage of active pixels:",
                          percent_active_pixels)

                # Backward
                loss.backward()

                unnorm_gradients = self.initial_image.grad

                if(step_iter == 0):
                    first_norm = torch.norm(unnorm_gradients) + 1e-8
                    # print("Original self.initial_image unnorm_gradients",
                    #       unnorm_gradients)
                # print("Original self.initial_image gradients", gradients)

                # gradients = unnorm_gradients / first_norm
                gradients = unnorm_gradients / \
                    torch.std(unnorm_gradients) + 1e-8

                # print("After normalize self.initial_image gradients", gradients)

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
                    pbar.set_postfix(loss=loss.item(), norm_image=torch.norm(
                        self.initial_image).item())
                    # self.initial_image = 0.9 * self.initial_image
                    # self.initial_image = torch.clamp(
                    #     self.initial_image, -1, 1)
                    # if(torch.norm(self.initial_image) > 32):
                    # self.initial_image = self.initial_image / \
                    # torch.std(self.initial_image) + 1e-8
                self.initial_image = self.initial_image.to(device)
                self.initial_image.requires_grad_()

        return self.initial_image


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


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)


X_train = add_channel_to_image(X_train)
X_test = add_channel_to_image(X_test)

train_data_loader = get_data_loader(
    X_train, y_train, 32)
test_data_loader = get_data_loader(
    X_test, y_test, 32)

dataset = 'mnist'
# conv4_dlgn , plain_pure_conv4_dnn , conv4_dlgn_n16_small , plain_pure_conv4_dnn_n16_small , conv4_deep_gated_net,conv4_deep_gated_net_with_actual_inp_in_wt_net
model_arch_type = 'conv4_deep_gated_net_with_actual_inp_in_wt_net'

models_base_path = None
# models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_deep_gated_net_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.93/aug_conv4_dlgn_iter_1_dir.pt"
# models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_plain_pure_conv4_dnn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.91/aug_conv4_dlgn_iter_1_dir.pt"
# models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.74/aug_conv4_dlgn_iter_1_dir.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vis_model = torch.load("/content/conv4_dgn_iter_1_dir.pt",map_location=device)
vis_model = get_model_instance_from_dataset(
    dataset, model_arch_type)
if(models_base_path is not None):
    custom_temp_model = torch.load(models_base_path, map_location=device)
    vis_model.load_state_dict(custom_temp_model.state_dict())
vis_model.train(False)
vis_model = vis_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vis_model.parameters(), lr=3e-4)
num_epochs_to_train = 32
for epoch in range(num_epochs_to_train):  # loop over the dataset multiple times
    correct = 0
    total = 0

    running_loss = 0.0
    loader = tqdm(train_data_loader, desc='Training')
    for batch_idx, data in enumerate(loader, 0):
        loader.set_description(f"Epoch {epoch+1}")
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(
            device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = vis_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()


image_ind = 220
template_loss_type = "TEMP_LOSS"
class_indx_to_visualize = [i for i in range(9)]
input_data_list_per_class = true_segregation(train_data_loader, 10)

intial_image = torch.from_numpy(np.random.normal(
    128, 8, (1, 28, 28)).astype('float32')/255)
# intial_image = torch.from_numpy(np.uint8(np.random.uniform(0, 1, (1, 28, 28))))
# intial_image = torch.from_numpy(np.uint8(np.random.uniform(0, 1, (1, 28, 28))))

print("intial_image dtype:", intial_image.dtype)

vis_model.train(False)
# print(per_class_dataset[image_ind])
overall_total = 0
overall_reconst_correct = 0
overall_overlap_percent_avg = 0
for c_indx in class_indx_to_visualize:
    total = 0
    reconst_correct = 0
    overlap_percent_avg = 0
    class_label = c_indx
    print("************************************************************ Class:", class_label)
    per_class_dataset = PerClassDataset(
        input_data_list_per_class[c_indx], c_indx)
    with trange(20, unit="Indx", desc="Generating template image for image of class:{}".format(class_label)) as pbar:
        for image_ind in pbar:
            images_to_collect_upon = per_class_dataset[image_ind][0]
            # orig_image = recreate_image(
            #     images_to_collect_upon, unnormalize=False)
            # save_image(orig_image, "root/temp/original_c_"+str(c_indx) +
            #            "_image_ind_"+str(image_ind)+"_"+str(model_arch_type)+".jpg")

            number_of_image_optimization_steps = 161
            start_sigma = 0.75
            end_sigma = 0.1
            start_step_size = 0.1
            end_step_size = 0.05

            tmp_gen = TemplateImageGenerator(
                vis_model, intial_image)
            vis_image = tmp_gen.generate_template_image_over_given_image(
                images_to_collect_upon, number_of_image_optimization_steps, template_loss_type)

            reconst_outputs = vis_model(vis_image)
            reconst_outputs_softmax = reconst_outputs.softmax(dim=1)
            print("Confidence over Reconstructed image")
            reconst_img_norm = torch.norm(vis_image)
            print("Norm of reconstructed image is:", reconst_img_norm)
            for i in range(len(reconst_outputs[0])):
                print("Class {} => {}".format(
                    class_label, reconst_outputs[0][i]))
            reconst_pred = reconst_outputs_softmax.max(1).indices
            print("Reconstructed image Class predicted:",
                  reconst_pred)

            overlap_bw_reconst_and_orig, total_pixel_points, overlap_bw_reconst_and_orig_percent = tmp_gen.find_overall_between_two_images(
                vis_image, tmp_gen.original_image[None])
            overlap_percent_avg += overlap_bw_reconst_and_orig_percent
            total += 1
            reconst_correct += reconst_pred.eq(class_label).sum().item()
            pbar.set_postfix(
                reconst_ratio="{}/{}".format(reconst_correct, total), recon_acc=100. * reconst_correct/total, ovlap="{}/{}".format(overlap_bw_reconst_and_orig, total_pixel_points), ovlap_per=overlap_bw_reconst_and_orig_percent.item(), ov_per_avg=overlap_percent_avg.item()/total)

            reconst_image = recreate_image(
                vis_image, unnormalize=False)
            path = "root/temp/reconst_c_"+str(c_indx) +\
                "_ep_"+str(num_epochs_to_train)+"_image_ind_"+str(image_ind) + \
                "_st_"+str(number_of_image_optimization_steps) + \
                "_"+str(model_arch_type)+"_recent.jpg"
            print("path saved:", path)
            save_image(reconst_image, path)
    overall_total += total
    overall_overlap_percent_avg += overlap_percent_avg
    overall_reconst_correct += reconst_correct

reconst_acc = 100. * overall_reconst_correct/overall_total
print("reconst_acc:", reconst_acc)
overall_overlap_percent_avg = overall_overlap_percent_avg / overall_total
print("overall_overlap_percent_avg:", overall_overlap_percent_avg)
