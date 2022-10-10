import torch

# Activation map is a 1,0 tensor


def calculate_common_among_two_activation_patterns(active_map_one, active_map_two):
    common_active_pixels = 0
    total_pixels = 0
    active_pixels_one = 0
    active_pixels_two = 0
    for act_indx in range(len(active_map_one)):
        sub_activation_map_one = active_map_one[act_indx]
        sub_activation_map_two = active_map_two[act_indx]

        active_pixels_one += torch.count_nonzero(
            sub_activation_map_one).item()
        active_pixels_two += torch.count_nonzero(
            sub_activation_map_two).item()

        total_pixels += torch.numel(sub_activation_map_one)

        common_activations_map = sub_activation_map_one * sub_activation_map_two

        common_active_pixels += torch.count_nonzero(
            common_activations_map).item()

    common_active_percentage = (100. * (common_active_pixels/total_pixels))

    return common_active_percentage, common_active_pixels, total_pixels, active_pixels_one, active_pixels_two
