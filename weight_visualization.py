
import torch
import os
import numpy as np
import wandb
from utils.weight_utils import get_gating_layer_weights
from structure.conv4_models import get_model_instance_from_dataset
from visualization import preprocess_image, get_initial_image
from tqdm import tqdm, trange
from utils.visualise_utils import recreate_image, generate_plain_3DImage, save_image
import cv2
import scipy.ndimage as nd


class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()


def blur_img(img, sigma):
    # print("blur img shape", img.shape)
    if sigma > 0:
        img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
        # img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
        # img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
    return img


class FilterVisualizer():
    def __init__(self, model, weight_vis_initial_image_type, weight_vis_loss_type, size=10, upscaling_steps=10, upscaling_factor=1.2):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.size, self.upscaling_steps, self.upscaling_factor, self.weight_vis_loss_type = size, upscaling_steps, upscaling_factor, weight_vis_loss_type
        model = model.to(device)
        self.model = model
        self.original_image = get_initial_image(
            dataset, weight_vis_initial_image_type, self.size)
        print("self.original_image", self.original_image)

    def visualize(self, network_type, layer_num, filter_indx, number_of_image_optimization_steps=20, lr=0.1):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        sz = self.size

        img = preprocess_image(
            self.original_image.cpu().clone().detach().numpy(), False)
        img = img.to(device)
        print("img shape:", img.size())

        img.requires_grad_()

        layer_obj = self.model.get_layer_object(network_type, layer_num)
        save_features = SaveFeatures(layer_obj)
        start_sigma = 0.75
        end_sigma = 0.1
        start_step_size = 0.1
        end_step_size = 0.05

        for _ in range(self.upscaling_steps):  # scale the image up upscaling_steps times
            img_var = img
            # optimizer = torch.optim.Adam([img_var], lr=lr)
            # optimize pixel values for opt_steps times
            with trange(number_of_image_optimization_steps, unit="iter", desc="Generating template image for given image") as pbar:
                for optim_num in pbar:
                    step_size = start_step_size + \
                        ((end_step_size - start_step_size) * optim_num) / \
                        number_of_image_optimization_steps
                    # optimizer = torch.optim.SGD([img_var], lr=step_size)
                    sigma = start_sigma + \
                        ((end_sigma - start_sigma) * optim_num) / \
                        number_of_image_optimization_steps

                    # optimizer.zero_grad()
                    model_outputs = self.model(img_var)
                    loss = get_vis_loss_value(
                        self.weight_vis_loss_type, network_type, save_features, model_outputs, filter_indx, self.model)
                    loss.backward()

                    unnorm_gradients = img_var.grad

                    gradients = unnorm_gradients / \
                        torch.std(unnorm_gradients) + 1e-8

                    with torch.no_grad():
                        # self.initial_image = self.initial_image - gradients*step_size
                        blurred_grad = gradients.cpu().detach().numpy()[0]
                        blurred_grad = blur_img(
                            blurred_grad, sigma)
                        img_var = img_var.cpu().detach().numpy()[
                            0]
                        img_var -= step_size / \
                            np.abs(blurred_grad).mean() * blurred_grad

                        img_var = blur_img(
                            img_var, sigma)
                        img_var = torch.from_numpy(
                            img_var[None])
                        pbar.set_postfix(loss=loss.item(), norm_image=torch.norm(
                            img_var).item())
                        img_var = img_var.to(device)
                        img_var.requires_grad_()

            # print("img_var shape:", img_var.size())
            img = img_var
            if(self.upscaling_factor != 1):
                img = img_var.cpu().detach().numpy()[0][0]
                # calculate new image size
                sz = int(self.upscaling_factor * sz)
                # scale image up
                # print("img size:", img.size())
                img = cv2.resize(
                    img, (sz, sz), interpolation=cv2.INTER_CUBIC)
                # print("img size:", img.size())
                # if blur is not None:
                # blur image to reduce high frequency patterns
                # img = cv2.blur(img, (blur, blur))
                img = torch.from_numpy(img[None][None])
                img = img.to(device)
                img.requires_grad_()

            self.output = img

        # self.save(layer_num, filter)
        save_features.close()

        # print("Output shape:", self.output.size())
        return self.output.cpu().detach().numpy()[0]

    def save(self, layer_num, filter_indx, save_folder):
        # plt.imsave("layer_"+str(layer)+"_filter_"+str(filter) +
        #            ".jpg", np.clip(self.output, 0, 1))
        # created_image = np.clip(self.output.cpu().detach().numpy()[0], 0, 1)
        created_image = recreate_image(
            self.output, False)
        print("created_image", created_image)
        # print("created_image shape", created_image.shape)
        save_folder = save_folder+"/LAYER_NUM_"+str(layer_num)+"/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        im_path = save_folder+'/weight_vis_ind_'+str(filter_indx) + '.jpg'
        # print("im_path:", im_path)
        save_image(created_image, im_path)


def get_vis_loss_value(
        weight_vis_loss_type, network_type, save_features, outputs, filter_indx, model):
    activations = torch.squeeze(save_features.features)
    if(weight_vis_loss_type == "MAXIMIZE_FILTER_OUTPUT"):
        return -torch.mean(activations[filter_indx])


def generate_weight_initialisation_per_filter(model, weight_vis_initial_image_type, weight_vis_loss_type, number_of_image_optimization_steps,
                                              network_type, layer_num, filter_indx, save_folder):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    original_image = get_initial_image(dataset, weight_vis_initial_image_type)
    initial_image = preprocess_image(
        original_image.cpu().clone().detach().numpy(), False)
    initial_image = initial_image.to(device)
    model = model.to(device)

    initial_image.requires_grad_()

    layer_obj = model.get_layer_object(network_type, layer_num)
    save_features = SaveFeatures(layer_obj)

    step_size = 0.1

    # optimizer = torch.optim.Adam(
    #     [initial_image], lr=step_size, weight_decay=1e-6)

    for optim_num in range(number_of_image_optimization_steps):
        # optimizer.zero_grad()

        outputs = model(initial_image)

        loss = get_vis_loss_value(
            weight_vis_loss_type, network_type, save_features, outputs, filter_indx, model)

        # Backward
        loss.backward()

        # optimizer.step()

        unnorm_gradients = initial_image.grad

        gradients = unnorm_gradients / \
            torch.std(unnorm_gradients) + 1e-8

        if(optim_num % 5 == 0):
            print("Before initial_image", initial_image)

        with torch.no_grad():
            initial_image = initial_image - gradients*step_size
            # self.initial_image = 0.9 * self.initial_image
            initial_image = torch.clamp(
                initial_image, -1, 1)
            if(optim_num % 5 == 0):
                print("Original initial_image unnorm_gradients", unnorm_gradients)
                print("After normalize initial_image gradients", gradients)
                print("After initial_image", initial_image)

        initial_image.requires_grad_()

    with torch.no_grad():
        created_image = recreate_image(
            initial_image, False)
        print("created_image", created_image)
        save_folder = save_folder+"/LAYER_NUM_"+str(layer_num)+"/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        im_path = save_folder+'/weight_vis_ind_'+str(filter_indx) + '.jpg'
        print("im_path:", im_path)
        save_image(created_image, im_path)

        return created_image


def run_weight_visualization_on_config(model, weight_vis_initial_image_type, weight_vis_loss_type, number_of_image_optimization_steps,
                                       network_type, size, upscaling_steps, upscaling_factor,
                                       root_save_prefix='root/WEIGHT_VISUALIZATION', final_postfix_for_save="",
                                       is_save_graph_visualizations=True):
    if(root_save_prefix is None):
        root_save_prefix = 'root/WEIGHT_VISUALIZATION'
    if(final_postfix_for_save is None):
        final_postfix_for_save = ""
    if(is_save_graph_visualizations):

        list_of_weights, list_of_bias = get_gating_layer_weights(model)

        filter_vis_obj = FilterVisualizer(
            model, weight_vis_initial_image_type, weight_vis_loss_type, size, upscaling_steps, upscaling_factor,)

        # Iterate each layer
        for layer_num in range(1, len(list_of_weights)):
            print(
                " ******************************************** Visualizing layer:", layer_num)
            save_folder = root_save_prefix + "/" + \
                str(final_postfix_for_save)+"/PlainImages/NT_TP_" + \
                str(network_type)+"/"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            number_of_filters = list_of_weights[layer_num].shape[0]

            list_of_img_vis_per_filter = []
            with trange(number_of_filters, unit="filter_ind", desc="Generating image for weight_vis_loss_type:{} network_type:{} , layer_num {}".format(weight_vis_loss_type, network_type, layer_num)) as pbar:
                for filter_indx in pbar:
                    # current_image_vis = generate_weight_initialisation_per_filter(model, weight_vis_initial_image_type, weight_vis_loss_type, number_of_image_optimization_steps,
                    #                                                               network_type, layer_num, filter_indx, save_folder)
                    current_image_vis = filter_vis_obj.visualize(
                        network_type, layer_num, filter_indx, number_of_image_optimization_steps)
                    print("current_image_vis size:", current_image_vis.shape)
                    print("current_image_vis", current_image_vis)
                    filter_vis_obj.save(layer_num, filter_indx, save_folder)
                    list_of_img_vis_per_filter.append(current_image_vis)

                current_full_img_save_path = save_folder + \
                    "LAY_NUM_"+str(layer_num)+"_weight_vis_plot.jpg"

                list_of_img_vis_per_filter = np.array(
                    list_of_img_vis_per_filter)

                print("current_full_img_save_path:",
                      current_full_img_save_path)
                print("list_of_img_vis_per_filter",
                      list_of_img_vis_per_filter.shape)
                generate_plain_3DImage(
                    list_of_img_vis_per_filter, current_full_img_save_path, True)


def run_weight_visualization(models_base_path, weight_vis_initial_image_type, weight_vis_loss_type, network_type, wand_project_name,
                             wandb_group_name, number_of_image_optimization_steps=161, size=10, upscaling_steps=10, upscaling_factor=1.2, wandb_config_additional_dict=None,
                             is_save_graph_visualizations=True, it_start=1, num_iter=None):
    is_log_wandb = not(wand_project_name is None)
    if(num_iter is None):
        num_iter = it_start + 1

    list_of_model_paths = []
    if(models_base_path != None):
        list_of_save_prefixes = []
        list_of_save_postfixes = []

        for i in range(it_start, num_iter):
            each_model_prefix = "aug_conv4_dlgn_iter_{}_dir.pt".format(i)

            list_of_model_paths.append(models_base_path+each_model_prefix)
            list_of_save_prefixes.append(
                str(models_base_path)+"/WEIGHT_VISUALIZATION/")
            list_of_save_postfixes.append("/aug_indx_{}".format(i))

    else:
        list_of_model_paths = [None]
        list_of_save_prefixes = [
            "root/WEIGHT_VISUALIZATION/"]
        list_of_save_postfixes = [None]

    for ind in range(len(list_of_model_paths)):
        each_model_path = list_of_model_paths[ind]
        each_save_prefix = list_of_save_prefixes[ind]
        each_save_postfix = list_of_save_postfixes[ind]
        analysed_model_path = each_model_path

        custom_temp_model = torch.load(each_model_path)
        custom_model = get_model_instance_from_dataset(
            dataset, model_arch_type)
        custom_model.load_state_dict(custom_temp_model.state_dict())
        custom_model.train(False)
        custom_model.eval()
        print(" #*#*#*#*#*#*#*# Generating weights analysis for model path:{} with save prefix :{} and postfix:{}".format(
            each_model_path, each_save_prefix, each_save_postfix))

        image_save_prefix_folder = str(each_save_prefix)+"/VIS_TYP"+str(weight_vis_loss_type) + \
            "/VIS_INIT_"+str(weight_vis_initial_image_type) + \
            "/"

        if(is_log_wandb):
            wandb_run_name = image_save_prefix_folder.replace(
                "/", "")
            wandb_config = dict()
            wandb_config["weight_vis_init_img_tp"] = weight_vis_initial_image_type
            wandb_config["weight_vis_loss_tp"] = weight_vis_loss_type
            wandb_config["num_of_img_optim_stp"] = number_of_image_optimization_steps
            wandb_config["models_base_path"] = models_base_path
            wandb_config["final_postfix_for_save"] = each_save_postfix

            if(wandb_config_additional_dict is not None):
                wandb_config.update(wandb_config_additional_dict)

            wandb.init(
                project=f"{wand_project_name}",
                name=f"{wandb_run_name}",
                group=f"{wandb_group_name}",
                config=wandb_config,
            )

        run_weight_visualization_on_config(custom_model, weight_vis_initial_image_type, weight_vis_loss_type, number_of_image_optimization_steps,
                                           network_type, size, upscaling_steps, upscaling_factor,
                                           root_save_prefix=image_save_prefix_folder, final_postfix_for_save=each_save_postfix,
                                           is_save_graph_visualizations=is_save_graph_visualizations)
        if(is_log_wandb):
            wandb.finish()


if __name__ == '__main__':
    dataset = 'mnist'
    # conv4_dlgn , plain_pure_conv4_dnn , conv4_dlgn_n16_small , plain_pure_conv4_dnn_n16_small , conv4_deep_gated_net
    model_arch_type = 'conv4_dlgn'
    # uniform_init_image , zero_init_image , gaussian_init_image
    weight_vis_initial_image_type = 'normal_init_image'
    weight_vis_loss_type = "MAXIMIZE_FILTER_OUTPUT"
    wand_project_name = None
    wandb_group_name = "test"
    number_of_image_optimization_steps = 161
    is_save_graph_visualizations = True
    network_type = "GATE_NET"
    size = 28
    upscaling_steps = 1
    upscaling_factor = 1

    # models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_plain_pure_conv4_dnn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.91/"
    # models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_dlgn_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.95/"
    models_base_path = "root/model/save/mnist/iterative_augmenting/DS_mnist/MT_conv4_deep_gated_net_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.93/"

    wandb_additional_dict = None
    if(not(wand_project_name is None)):
        wandb.login()
        wandb_additional_dict = dict()
        wandb_additional_dict["dataset"] = dataset
        wandb_additional_dict["model_arch_type"] = model_arch_type

    num_iterations = 1
    start_index = 1
    for current_it_start in range(start_index, num_iterations + 1):
        run_weight_visualization(models_base_path, weight_vis_initial_image_type, weight_vis_loss_type, network_type, wand_project_name,
                                 wandb_group_name, number_of_image_optimization_steps, size, upscaling_steps, upscaling_factor, wandb_additional_dict,
                                 is_save_graph_visualizations, current_it_start)

    print("Finished execution!!!")
