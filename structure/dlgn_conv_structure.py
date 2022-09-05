

class All_Conv_info:
    def __init__(self, list_of_each_conv_info,input_image_size):
        self.list_of_each_conv_info = list_of_each_conv_info
        self.input_image_size = input_image_size

class Conv_info:
    def __init__(self, layer_type,layer_sub_type,in_ch=3,number_of_filters=128,padding=0,stride=1,kernel_size=(3,3),num_nodes_in_fc=None,weight_init_type = "XAVIER_UNIFORM"):
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
        return "layer_type: "+str(self.layer_type)+" layer_sub_type: "+str(self.layer_sub_type)+" in_ch: "+str(self.in_ch)+" out_ch: "+str(self.out_ch)+" padding: "+str(self.padding)+" stride: "+str(self.stride)+" kernel_size: "+str(self.kernel_size)

def convert_generic_object_list_to_All_Conv_info(list_of_gen_all_conv_obj):
    list_of_conv_obj = []
    for each_gen_hp_obj in list_of_gen_all_conv_obj:
        list_of_conv_obj.append(
            convert_generic_object_to_Conv_info_object(each_gen_hp_obj))
    
    return All_Conv_info(list_of_conv_obj,None)


def convert_generic_object_to_Conv_info_object(gen_conv_obj):
    return Conv_info(gen_conv_obj.layer_type,gen_conv_obj.layer_sub_type,gen_conv_obj.in_ch,gen_conv_obj.number_of_filters,gen_conv_obj.padding,gen_conv_obj.stride,gen_conv_obj.kernel_size,gen_conv_obj.num_nodes_in_fc,gen_conv_obj.weight_init_type)
