import torch

def get_activation_function_from_key(activ_func_key):
    if(activ_func_key == 'RELU'):
        return torch.nn.ReLU()
    elif(activ_func_key == 'SIGMOID'):
        return torch.nn.Sigmoid()
    return activ_func_key


def get_loss_function_from_key(loss_fn_key):
    if(loss_fn_key == 'BCE'):
        return torch.nn.BCELoss()
    elif(loss_fn_key == 'CCE'):
        return torch.nn.CrossEntropyLoss()
    return loss_fn_key