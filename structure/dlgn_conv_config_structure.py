from configs.dlgn_conv_config import get_activation_function_from_key, get_loss_function_from_key
try:
    from types import SimpleNamespace as Namespace
except ImportError:
    from argparse import Namespace

class Configs(Namespace):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattribute__(self, value):
        try:
            return super().__getattribute__(value)
        except AttributeError:
            return None

class DatasetConfig:
    def __init__(self, name, is_normalize_data, valid_split_size, batch_size=64):
        self.name = name
        self.is_normalize_data = is_normalize_data
        self.valid_split_size = valid_split_size
        self.batch_size = batch_size

class AllParams:
    def __init__(self, *args):
        self.args = args
    
    def __str__(self):
        return "Args: "+str(self.args)

class HPParams:
    def __init__(self, optimizer, momentum, lr, epochs, activ_func, output_activ_func, weight_init, loss_fn, batch_size=64):
        self.optimizer = optimizer
        self.momentum = momentum
        self.lr = lr
        self.epochs = epochs
        self.activ_func = get_activation_function_from_key(activ_func)
        self.output_activ_func = get_activation_function_from_key(
            output_activ_func)
        #  XAVIER_UNIFORM , XAVIER_NORMAL
        self.weight_init = weight_init
        self.loss_fn = get_loss_function_from_key(loss_fn)
        self.batch_size = batch_size

    def __str__(self):
        return "optimizer: "+str(self.optimizer) + " momentum: "+str(self.momentum) + " learning rate: "+str(self.lr)+"\n epochs: "+str(self.epochs)+"\n Inner Activation fn:"+str(self.activ_func) + "\n Output Activation Fn:"+str(self.output_activ_func)+"\n Weight initialization:"+str(self.weight_init) + "\n Loss Fn:"+str(self.loss_fn)+"\n Batch size:"+str(self.batch_size)

    def __hash__(self):
        return hash((self.optimizer, self.momentum, self.lr, self.epochs, str(self.activ_func), str(self.output_activ_func), self.weight_init, str(self.loss_fn), self.batch_size))
