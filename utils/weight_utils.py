from conv4_models import Plain_CONV4_Net, Conv4_DLGN_Net, Conv4_DLGN_Net_N16_Small


def get_gating_layer_weights(input_model):
    list_of_weights = []
    list_of_bias = []
    if(isinstance(input_model, Conv4_DLGN_Net)):
        list_of_weights.append(input_model.conv1_g.weight)
        list_of_weights.append(input_model.conv2_g.weight)
        list_of_weights.append(input_model.conv3_g.weight)
        list_of_weights.append(input_model.conv4_g.weight)

        list_of_bias.append(input_model.conv1_g.bias)
        list_of_bias.append(input_model.conv2_g.bias)
        list_of_bias.append(input_model.conv3_g.bias)
        list_of_bias.append(input_model.conv4_g.bias)

    elif(isinstance(input_model, Conv4_DLGN_Net_N16_Small)):
        list_of_weights.append(input_model.conv1_g.weight)
        list_of_weights.append(input_model.conv2_g.weight)
        list_of_weights.append(input_model.conv3_g.weight)
        list_of_weights.append(input_model.conv4_g.weight)

        list_of_bias.append(input_model.conv1_g.bias)
        list_of_bias.append(input_model.conv2_g.bias)
        list_of_bias.append(input_model.conv3_g.bias)
        list_of_bias.append(input_model.conv4_g.bias)

    elif(isinstance(input_model, Plain_CONV4_Net)):
        list_of_weights.append(input_model.conv1_g.weight)
        list_of_weights.append(input_model.conv2_g.weight)
        list_of_weights.append(input_model.conv3_g.weight)
        list_of_weights.append(input_model.conv4_g.weight)

        list_of_bias.append(input_model.conv1_g.bias)
        list_of_bias.append(input_model.conv2_g.bias)
        list_of_bias.append(input_model.conv3_g.bias)
        list_of_bias.append(input_model.conv4_g.bias)

    return list_of_weights, list_of_bias
