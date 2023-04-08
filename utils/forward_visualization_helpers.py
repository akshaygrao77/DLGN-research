import torch
import numpy as np
import torch.nn.functional as F


def conv2d_to_conv_matrix(conv, image_shape):
    identity = torch.eye(np.prod(image_shape).item(
    ), dtype=conv.weight.dtype).reshape([-1] + list(image_shape))
    output = F.conv2d(identity, conv.weight, None, conv.stride, conv.padding)
    output_shape = output.size()[1:]
    W = output.reshape(-1, np.prod(output_shape).item()).T
    tbias = conv.bias
    if(conv.bias is None):
        tbias = torch.zeros(conv.weight.size()[0], dtype=conv.weight.dtype)
    b = torch.stack(
        [torch.ones(output_shape[1:], dtype=conv.weight.dtype) * bi for bi in tbias])
    b = b.reshape(-1, np.prod(output_shape).item())
    return W, b, output_shape


def merge_batchnorm_into_convmatrix(bn, raw_conv_matrix, raw_conv_bias):
    original_conv_matrix_shape = raw_conv_matrix.size()
    original_conv_bias_shape = raw_conv_bias.size()

    oc = bn.bias.size()[0]
    tl = list(raw_conv_matrix.size())
    pr = 1
    for e in tl:
        pr *= e
    sl = [oc, pr//oc]
    conv_matrix = raw_conv_matrix.reshape(sl)

    tl = list(raw_conv_bias.size())
    pr = 1
    for e in tl:
        pr *= e
    sl = [oc, pr//oc]
    conv_bias = raw_conv_bias.reshape(sl)

    mweight = conv_matrix
    div_term = torch.sqrt(bn.running_var + bn.eps)
    m_bias = torch.zeros_like(conv_bias, dtype=conv_bias.dtype)
    for i in range(len(conv_bias)):
        m_bias[i] = ((conv_bias[i] - bn.running_mean[i]) /
                     (div_term[i]))*bn.weight.data[i] + bn.bias.data[i]

    scale_factor = (bn.weight / div_term)
    for i in range(len(conv_matrix)):
        mweight[i] *= scale_factor[i]

    mweight = mweight.reshape(original_conv_matrix_shape)
    m_bias = m_bias.reshape(original_conv_bias_shape)
    return mweight, m_bias


def merge_conv_matrix(W1, b1, W2, b2):
    W = torch.matmul(W2, W1)
    b = torch.transpose(torch.matmul(W2, torch.transpose(b1, 1, 0)), 1, 0) + b2

    return W, b


def convert_avgpool_to_conv(avgp, ch, dtype=torch.float32):
    ks = avgp.kernel_size
    tp_weights = torch.zeros(size=(ch, ch, ks, ks))
    for i in range(len(tp_weights)):
        tp_weights[i][i] = torch.ones(size=(ks, ks))/(ks*ks)
    merged_layer = torch.nn.Conv2d(
        ch, ch, ks, stride=avgp.stride, padding=avgp.padding, bias=False, dtype=dtype)
    merged_layer.weight = torch.nn.Parameter(tp_weights.type(dtype))
    return merged_layer


def apply_input_on_conv_matrix(input, W, b):
    f_input = torch.flatten(input)
    ret = torch.matmul(W, f_input)+b

    return ret


def add_skip_conn_to_convmatrix(raw_conv_matrix):
    ch = raw_conv_matrix.size()[1]
    identity_matrix = torch.eye(ch, dtype=raw_conv_matrix.dtype)
    ret_matrix = raw_conv_matrix + identity_matrix
    return ret_matrix


def merge_operations_in_modules(modObj, current_tensor_size, merged_conv_matrix=None, merged_conv_bias=None):
    list_to_loop = list(enumerate(modObj.children()))
    if(len(list_to_loop) == 0):
        list_to_loop = [(0, modObj)]

    for (i, current_layer) in list_to_loop:
        print("current_layer", current_layer)
        if(isinstance(current_layer, torch.nn.Conv2d)):
            if(merged_conv_matrix is None):
                merged_conv_matrix, merged_conv_bias, current_tensor_size = conv2d_to_conv_matrix(
                    current_layer, current_tensor_size)
            else:
                cur_conv_matrix, cur_conv_bias, current_tensor_size = conv2d_to_conv_matrix(
                    current_layer, current_tensor_size)
                merged_conv_matrix, merged_conv_bias = merge_conv_matrix(
                    merged_conv_matrix, merged_conv_bias, cur_conv_matrix, cur_conv_bias)
        elif(isinstance(current_layer, torch.nn.BatchNorm2d)):
            merged_conv_matrix, merged_conv_bias = merge_batchnorm_into_convmatrix(
                current_layer, merged_conv_matrix, merged_conv_bias)
        elif(isinstance(current_layer, torch.nn.AvgPool2d)):
            tp_conv_avg = convert_avgpool_to_conv(
                current_layer, current_tensor_size[0])
            cur_conv_matrix, cur_conv_bias, current_tensor_size = conv2d_to_conv_matrix(
                tp_conv_avg, current_tensor_size)
            merged_conv_matrix, merged_conv_bias = merge_conv_matrix(
                merged_conv_matrix, merged_conv_bias, cur_conv_matrix, cur_conv_bias)
        elif(isinstance(current_layer, torch.nn.Linear)):
            merged_conv_matrix, merged_conv_bias = merge_conv_matrix(
                merged_conv_matrix, merged_conv_bias, current_layer.weight, current_layer.bias)
            current_tensor_size = (current_layer.out_features,)

        print("merged_conv_matrix:{} merged_conv_bias:{} current_tensor_size:{}".format(
            merged_conv_matrix.size(), merged_conv_bias.size(), current_tensor_size))

    return merged_conv_matrix, merged_conv_bias, current_tensor_size
