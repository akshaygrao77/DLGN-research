

class All_Conv_info:
    def __init__(self, list_of_each_conv_info, input_image_size):
        self.list_of_each_conv_info = list_of_each_conv_info
        self.input_image_size = input_image_size

    def __str__(self):
        ret = "input_image_size: "+str(self.input_image_size)+" \n"
        for each_conv_info in self.list_of_each_conv_info:
            ret += str(each_conv_info)+"\n"
        return ret


class Conv_info:
    def __init__(self, layer_type, layer_sub_type, in_ch=3, number_of_filters=128, padding=0, stride=1, kernel_size=(3, 3), num_nodes_in_fc=None, weight_init_type="XAVIER_UNIFORM"):
        self.layer_type = layer_type
        self.layer_sub_type = layer_sub_type
        self.in_ch = in_ch
        self.number_of_filters = number_of_filters
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.num_nodes_in_fc = num_nodes_in_fc
        self.weight_init_type = weight_init_type

    def __str__(self):
        return "layer_type: "+str(self.layer_type)+" layer_sub_type: "+str(self.layer_sub_type)+" in_ch: "+str(self.in_ch)+" number_of_filters: "+str(self.number_of_filters)+" padding: "+str(self.padding)+" stride: "+str(self.stride)+" kernel_size: "+str(self.kernel_size)+" weight_init_type: "+str(self.weight_init_type)+" num_nodes_in_fc: "+str(self.num_nodes_in_fc)
