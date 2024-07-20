from conv4_models import Plain_CONV4_Net, Plain_CONV4_Net_N16_Small, Conv4_DLGN_Net, Conv4_DeepGated_Net_N16_Small, Conv4_DLGN_Net_N16_Small, Conv4_DeepGated_Net, Conv4_DeepGated_Net_With_Actual_Inp_Over_WeightNet, Mask_Conv4_DLGN_Net_N16_Small, Mask_Conv4_DLGN_Net,DLGN_FC_Network,SF_DLGN_FC_Network,Conv4_SF_DLGN_Net


def get_gating_layer_weights(input_model):
    list_of_weights = []
    list_of_bias = []
    if(isinstance(input_model, Conv4_DLGN_Net) or isinstance(input_model, Conv4_SF_DLGN_Net)):
        list_of_weights.append(input_model.conv1_g.weight)
        list_of_weights.append(input_model.conv2_g.weight)
        list_of_weights.append(input_model.conv3_g.weight)
        list_of_weights.append(input_model.conv4_g.weight)

        list_of_bias.append(input_model.conv1_g.bias)
        list_of_bias.append(input_model.conv2_g.bias)
        list_of_bias.append(input_model.conv3_g.bias)
        list_of_bias.append(input_model.conv4_g.bias)
    elif(isinstance(input_model, DLGN_FC_Network) or isinstance(input_model,SF_DLGN_FC_Network)):
        for em in input_model.gating_network.list_of_modules:
            list_of_weights.append(em.weight)
            list_of_bias.append(em.bias)

    elif(isinstance(input_model, Conv4_DLGN_Net_N16_Small)):
        list_of_weights.append(input_model.conv1_g.weight)
        list_of_weights.append(input_model.conv2_g.weight)
        list_of_weights.append(input_model.conv3_g.weight)
        list_of_weights.append(input_model.conv4_g.weight)

        list_of_bias.append(input_model.conv1_g.bias)
        list_of_bias.append(input_model.conv2_g.bias)
        list_of_bias.append(input_model.conv3_g.bias)
        list_of_bias.append(input_model.conv4_g.bias)
    elif(isinstance(input_model, Mask_Conv4_DLGN_Net)):
        list_of_weights.append(input_model.conv1_g.weight)
        list_of_weights.append(input_model.conv2_g.weight)
        list_of_weights.append(input_model.conv3_g.weight)
        list_of_weights.append(input_model.conv4_g.weight)

        list_of_bias.append(input_model.conv1_g.bias)
        list_of_bias.append(input_model.conv2_g.bias)
        list_of_bias.append(input_model.conv3_g.bias)
        list_of_bias.append(input_model.conv4_g.bias)

    elif(isinstance(input_model, Mask_Conv4_DLGN_Net_N16_Small)):
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

    elif(isinstance(input_model, Plain_CONV4_Net_N16_Small)):
        list_of_weights.append(input_model.conv1_g.weight)
        list_of_weights.append(input_model.conv2_g.weight)
        list_of_weights.append(input_model.conv3_g.weight)
        list_of_weights.append(input_model.conv4_g.weight)

        list_of_bias.append(input_model.conv1_g.bias)
        list_of_bias.append(input_model.conv2_g.bias)
        list_of_bias.append(input_model.conv3_g.bias)
        list_of_bias.append(input_model.conv4_g.bias)

    elif(isinstance(input_model, Conv4_DeepGated_Net)):
        list_of_weights.append(input_model.conv1_g.weight)
        list_of_weights.append(input_model.conv2_g.weight)
        list_of_weights.append(input_model.conv3_g.weight)
        list_of_weights.append(input_model.conv4_g.weight)

        list_of_bias.append(input_model.conv1_g.bias)
        list_of_bias.append(input_model.conv2_g.bias)
        list_of_bias.append(input_model.conv3_g.bias)
        list_of_bias.append(input_model.conv4_g.bias)

    elif(isinstance(input_model, Conv4_DeepGated_Net_N16_Small)):
        list_of_weights.append(input_model.conv1_g.weight)
        list_of_weights.append(input_model.conv2_g.weight)
        list_of_weights.append(input_model.conv3_g.weight)
        list_of_weights.append(input_model.conv4_g.weight)

        list_of_bias.append(input_model.conv1_g.bias)
        list_of_bias.append(input_model.conv2_g.bias)
        list_of_bias.append(input_model.conv3_g.bias)
        list_of_bias.append(input_model.conv4_g.bias)

    elif(isinstance(input_model, Conv4_DeepGated_Net_With_Actual_Inp_Over_WeightNet)):
        list_of_weights.append(input_model.conv1_g.weight)
        list_of_weights.append(input_model.conv2_g.weight)
        list_of_weights.append(input_model.conv3_g.weight)
        list_of_weights.append(input_model.conv4_g.weight)

        list_of_bias.append(input_model.conv1_g.bias)
        list_of_bias.append(input_model.conv2_g.bias)
        list_of_bias.append(input_model.conv3_g.bias)
        list_of_bias.append(input_model.conv4_g.bias)

    return list_of_weights, list_of_bias
