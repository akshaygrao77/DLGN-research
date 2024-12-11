import torch
import numpy as np
import torch.nn.functional as F


def merge_conv_layers(conv1, conv2):
    """
    
    ================ ATTENTION =====================
      Both the conv needs to be flipped in (-1,-2) dimensions or alternatively the input to the convolution can be flipped
    ================================================

    :input k1: A tensor of shape ``(out1, in1, s1, s1)``
    :input k1: A tensor of shape ``(out2, in2, s2, s2)``
    :returns: A tensor of shape ``(out2, in1, s1+s2-1, s1+s2-1)``
      so that convolving with it equals convolving with k1 and
      then with k2.
    """
    conv1_weight = conv1.weight.data.flip(-1,-2)
    if(conv2.weight.get_device() < 0):
        idevice = torch.device("cpu")
    else:
        idevice = conv2.weight.get_device()

    m_bias = True
    padding = (conv2.weight.data.size()[-2] - 1,conv2.weight.data.size()[-1] - 1)
    # Flip because this is actually correlation, and permute to adapt to BHCW
    new_conv_weights = torch.conv2d(conv1_weight.data.permute(1, 0, 2, 3), conv2.weight.data,
                                    padding=padding).permute(1, 0, 2, 3)

    if(conv1.bias is None and conv2.bias is not None):
        new_conv_bias = conv2.bias.data
    elif((conv1.bias is not None and conv2.bias is not None) or (conv2.bias is None and conv1.bias is not None)):
        add_x = torch.ones(1, conv1.out_channels, *conv2.kernel_size,
                           dtype=conv1_weight.dtype, device=idevice) * (conv1.bias.data[None, :, None, None]).to(idevice)
        # This operation simultaneously transfers the bias from the first convolution and adds the bias from the second.
        tc = conv2.padding
        conv2.padding = 0
        new_conv_bias = conv2(add_x).flatten()
        conv2.padding = tc
    else:
        m_bias = False

    # merged_layer = torch.nn.Conv2d(new_conv_weights.size()[1],new_conv_weights.size()[0],new_conv_weights.size()[2],stride=1,padding=new_conv_weights.size()[2]-1,bias=m_bias)
    merged_layer = torch.nn.Conv2d(new_conv_weights.size()[1], new_conv_weights.size()[0], (new_conv_weights.size()[
                                   -2],new_conv_weights.size()[-1]), stride=1, padding=(new_conv_weights.size()[-2] - 1,new_conv_weights.size()[-1] - 1), bias=m_bias, dtype=conv1.weight.dtype, device=idevice)
    merged_layer.weight = torch.nn.Parameter(new_conv_weights.flip(-1,-2))
    if(m_bias):
        merged_layer.bias = torch.nn.Parameter(new_conv_bias)
    return merged_layer

def convert_avgpool_to_conv_layer(avgp, ch, dtype=torch.float32, idevice=torch.device("cuda")):
    ks = avgp.kernel_size
    tp_weights = torch.zeros(size=(ch, ch, ks, ks),
                             dtype=dtype, device=idevice)
    for i in range(len(tp_weights)):
        tp_weights[i][i] = torch.ones(
            size=(ks, ks), dtype=dtype, device=idevice)/(ks*ks)
    merged_layer = torch.nn.Conv2d(
        ch, ch, ks, stride=1, padding=0, bias=False, dtype=dtype, device=idevice)
    merged_layer.weight = torch.nn.Parameter(tp_weights)
    return merged_layer


def add_skip_connection_to_conv(conv):
    tp_weights = conv.weight.data
    ks = conv.weight.size()[-1]
    for i in range(len(tp_weights)):
        tp_weights[i][i][(ks//2)][(ks//2)] += 1
    conv.weight = torch.nn.Parameter(tp_weights)
    return conv


def merge_batchnorm_into_conv(bn, conv):
    if(conv.weight.get_device() < 0):
        idevice = torch.device("cpu")
    else:
        idevice = conv.weight.get_device()

    mweight = conv.weight.data
    div_term = torch.sqrt(bn.running_var + bn.eps)
    if(conv.bias is not None):
        m_bias = ((conv.bias.data - bn.running_mean)/(div_term)) * \
            bn.weight.data + bn.bias.data
    else:
        m_bias = ((- bn.running_mean)/(div_term))*bn.weight.data + bn.bias.data

    scale_factor = (bn.weight / div_term)
    for i in range(len(conv.weight)):
        mweight[i] *= scale_factor[i]

    merged_layer = torch.nn.Conv2d(conv.weight.size()[1], conv.weight.size()[
                                   0], conv.weight.size()[2], stride=1, padding=conv.padding, bias=True, dtype=conv.weight.dtype, device=idevice)
    merged_layer.weight = torch.nn.Parameter(mweight.flip(-1,-2))
    merged_layer.bias = torch.nn.Parameter(m_bias)
    return merged_layer


def conv2d_to_conv_matrix(conv, image_shape):
    return conv2dparams_to_conv_matrix(conv.weight, conv.bias, conv.stride, conv.padding, image_shape)


def conv2dparams_to_conv_matrix(cweight, cbias, cstride, cpad, image_shape):
    print("cweight.get_device()", cweight.get_device())
    if(cweight.get_device() < 0):
        idevice = torch.device("cpu")
    else:
        # cweight = cweight.type(torch.float16)
        idevice = cweight.get_device()

    with torch.no_grad():
        identity = torch.eye(np.prod(image_shape).item(
        ), dtype=cweight.dtype, device=idevice).reshape([-1] + list(image_shape))
        # output = conv_layer(identity)
        output = []
        for i in range(cweight.size()[0]):
            conv_layer = torch.nn.Conv2d(1, cweight.size()[0], kernel_size=(cweight.size()[
                2], cweight.size()[3]), stride=cstride, padding=cpad, dtype=cweight.dtype, bias=False)
            conv_layer.weight = torch.nn.Parameter(
                torch.unsqueeze(cweight[i], 0))
            if(not cweight.get_device() < 0):
                conv_layer = torch.nn.DataParallel(conv_layer).cuda()
            # output.append(
            #     F.conv2d(identity, torch.unsqueeze(cweight[i], 0), None, cstride, cpad))
            output.append(conv_layer(identity))
        output = torch.squeeze(torch.stack(output, 1), 2)
        # output = F.conv2d(identity, cweight, None, cstride, cpad)
        output_shape = output.size()[1:]
        W = output.reshape(-1, np.prod(output_shape).item()).T

        if(cbias is None):
            b = torch.stack(
                [torch.ones(output_shape[1:], dtype=cweight.dtype, device=idevice) * bi for bi in torch.zeros(cweight.size()[
                    0], dtype=cweight.dtype, device=idevice)])
        else:
            b = torch.stack(
                [torch.ones(output_shape[1:], dtype=cweight.dtype, device=idevice) * bi for bi in cbias])
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
    conv_bias = ((conv_bias - bn.running_mean[:, None].type(raw_conv_matrix.dtype)) /
                 (div_term[:, None]).type(raw_conv_matrix.dtype))*bn.weight.data[:, None].type(raw_conv_matrix.dtype) + bn.bias.data[:, None].type(raw_conv_matrix.dtype)

    scale_factor = (bn.weight / div_term).type(raw_conv_matrix.dtype)
    mweight *= scale_factor[:, None]

    mweight = mweight.reshape(original_conv_matrix_shape)
    conv_bias = conv_bias.reshape(original_conv_bias_shape)
    return mweight, conv_bias


def merge_conv_matrix(W1, b1, W2, b2):
    W = torch.matmul(W2, W1)
    tmp = b1
    if(len(b1.size())==2):
      tmp = torch.transpose(b1, 1, 0)
    tmp2 = torch.matmul(W2, tmp)
    if(len(tmp2.size())==2):
      b = torch.transpose(tmp2, 1, 0) + b2
    else:
      b = tmp2  + b2

    return W, b


def convert_avgpool_to_convmatrix(avgp, image_shape, dtype=torch.float16):
    device = "cuda"
    ch = image_shape[0]
    ks = avgp.kernel_size
    tp_weights = torch.zeros(size=(ch, ch, ks, ks), device=device)
    for i in range(len(tp_weights)):
        tp_weights[i][i] = torch.ones(size=(ks, ks), device=device)/(ks*ks)
    tp_weights = tp_weights.type(dtype)
    return conv2dparams_to_conv_matrix(tp_weights, None, avgp.stride, avgp.padding, image_shape)


def apply_input_on_conv_matrix(input, W, b):
    f_input = torch.flatten(input).type(W.dtype).to(device=W.device)
    ret = torch.matmul(W, f_input)+b

    return ret


def add_skip_conn_to_convmatrix(raw_conv_matrix):
    ch = raw_conv_matrix.size()[1]
    identity_matrix = torch.eye(ch, dtype=raw_conv_matrix.dtype)
    ret_matrix = raw_conv_matrix + identity_matrix
    return ret_matrix


def merge_layers_operations_in_modules(modObj, current_tensor_size, lay_type, idevice, merged_conv_layer=None):
    list_to_loop = list(enumerate(modObj.children()))
    if(len(list_to_loop) == 0):
        list_to_loop = [(0, modObj)]

    for (i, current_layer) in list_to_loop:
        # print("current_layer", current_layer)
        if(isinstance(current_layer, torch.nn.Conv2d)):
            if(merged_conv_layer is None):
                merged_conv_layer = current_layer
            else:
                merged_conv_layer = merge_conv_layers(
                    merged_conv_layer, current_layer)
        elif(isinstance(current_layer, torch.nn.BatchNorm2d)):
            merged_conv_layer = merge_batchnorm_into_conv(
                current_layer, merged_conv_layer)
        elif(isinstance(current_layer, torch.nn.AvgPool2d)):
            tp_conv_avg = convert_avgpool_to_conv_layer(current_layer, merged_conv_layer.weight.size()[
                                                        0], lay_type, idevice)
            merged_conv_layer = merge_conv_layers(
                merged_conv_layer, tp_conv_avg)

        # print("merged_conv_layer:{} current_tensor_size:{}".format(
        #     merged_conv_layer, current_tensor_size))

    return merged_conv_layer, current_tensor_size


def merge_operations_in_modules(modObj, current_tensor_size, merged_conv_matrix=None, merged_conv_bias=None,is_no_grad=True):
    list_to_loop = list(enumerate(modObj.children()))
    if(len(list_to_loop) == 0):
        list_to_loop = [(0, modObj)]
    def subfunc():
        nonlocal merged_conv_matrix
        nonlocal merged_conv_bias
        nonlocal current_tensor_size
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
                    cur_conv_matrix, cur_conv_bias, current_tensor_size = convert_avgpool_to_convmatrix(
                        current_layer, current_tensor_size)
                    merged_conv_matrix, merged_conv_bias = merge_conv_matrix(
                        merged_conv_matrix, merged_conv_bias, cur_conv_matrix, cur_conv_bias)
                elif(isinstance(current_layer, torch.nn.Linear)):
                    if(merged_conv_matrix is None):
                        merged_conv_matrix, merged_conv_bias = current_layer.weight,current_layer.bias
                    else:
                        merged_conv_matrix, merged_conv_bias = merge_conv_matrix(
                            merged_conv_matrix, merged_conv_bias, current_layer.weight, current_layer.bias)
                    current_tensor_size = (current_layer.out_features,)
    if(is_no_grad):
        with torch.no_grad():
            subfunc()
    else:
        print("Running with grad enabled")
        subfunc()

        print("merged_conv_matrix:{} merged_conv_bias:{} current_tensor_size:{}".format(
            merged_conv_matrix.size(), merged_conv_bias.size(), current_tensor_size))

    return merged_conv_matrix, merged_conv_bias, current_tensor_size
