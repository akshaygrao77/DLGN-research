import torch


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
        values = torch.tensor([0.], device=device)
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
