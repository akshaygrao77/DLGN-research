import torch
import torch.nn as nn
import numpy as np
import math
import wandb

def clip_eta(eta, norm, eps):
    """
    PyTorch implementation of the clip_eta in utils_tf.

    :param eta: Tensor
    :param norm: np.inf, 1, or 2
    :param eps: float
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")

    avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
    reduc_ind = list(range(1, len(eta.size())))
    if norm == np.inf:
        eta = torch.clamp(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("L1 clip is not implemented.")
            norm = torch.max(
                avoid_zero_div, torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True)
            )
        elif norm == 2:
            norm = torch.sqrt(
                torch.max(
                    avoid_zero_div, torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
                )
            )
        factor = torch.min(
            torch.tensor(1.0, dtype=eta.dtype, device=eta.device), eps / norm
        )
        eta *= factor
    return eta


def get_or_guess_labels(model, x, **kwargs):
    """
    Get the label to use in generating an adversarial example for x.
    The kwargs are fed directly from the kwargs of the attack.
    If 'y' is in kwargs, then assume it's an untargeted attack and
    use that as the label.
    If 'y_target' is in kwargs and is not none, then assume it's a
    targeted attack and use that as the label.
    Otherwise, use the model's prediction as the label and perform an
    untargeted attack.

    :param model: PyTorch model. Do not add a softmax gate to the output.
    :param x: Tensor, shape (N, d_1, ...).
    :param y: (optional) Tensor, shape (N).
    :param y_target: (optional) Tensor, shape (N).
    """
    if "y" in kwargs and "y_target" in kwargs:
        raise ValueError("Can not set both 'y' and 'y_target'.")
    if "y" in kwargs:
        labels = kwargs["y"]
    elif "y_target" in kwargs and kwargs["y_target"] is not None:
        labels = kwargs["y_target"]
    else:
        _, labels = torch.max(model(x), 1)
    return labels


def optimize_linear(grad, eps, norm=np.inf):
    """
    Solves for the optimal input to a linear function under a norm constraint.

    Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)

    :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
    :param eps: float. Scalar specifying size of constraint region
    :param norm: np.inf, 1, or 2. Order of norm constraint.
    :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
    """

    red_ind = list(range(1, len(grad.size())))
    avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
    if norm == np.inf:
        # Whether signed or not decided by the caller of this function
        optimal_perturbation = grad
    elif norm == 1:
        abs_grad = torch.abs(grad)
        sign = torch.sign(grad)
        red_ind = list(range(1, len(grad.size())))
        abs_grad = torch.abs(grad)
        ori_shape = [1] * len(grad.size())
        ori_shape[0] = grad.size(0)

        max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
        max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
        num_ties = max_mask
        for red_scalar in red_ind:
            num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
        optimal_perturbation = sign * max_mask / num_ties
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
        assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
    elif norm == 2:
        square = torch.max(avoid_zero_div, torch.sum(grad ** 2, red_ind, keepdim=True))
        optimal_perturbation = grad / torch.sqrt(square)
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = (
            optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
        )
        one_mask = (square <= avoid_zero_div).to(torch.float) * opt_pert_norm + (
            square > avoid_zero_div
        ).to(torch.float)
        assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
    else:
        raise NotImplementedError(
            "Only L-inf, L1 and L2 norms are " "currently implemented."
        )

    # Scale perturbation to be the solution for the norm=eps rather than
    # norm=1 problem
    scaled_perturbation = eps * optimal_perturbation
    return scaled_perturbation

def get_no_backprop_grad(net,inputs,criterion,labels):
  with torch.no_grad():
    output,y_grad_by_x = net(inputs,is_out_norm=True)
  output.requires_grad = True
  loss = criterion(output, labels)
  loss.backward()
  grad = torch.reshape(y_grad_by_x*torch.unsqueeze(output.grad,-1),inputs.shape)
  return output,grad

def get_residue_adv_per_batch(net,org_inputs,kwargs):
    kwargs.setdefault('rand_init',True)
    kwargs.setdefault('norm',np.inf)
    kwargs.setdefault('backpropmode','normal')
    norm,criterion,eps,eps_step_size,steps,labels,update_on,backpropmode,residue_vname,rand_init = kwargs['norm'],kwargs['criterion'],kwargs['eps'],kwargs['eps_step_size'],kwargs['steps'],kwargs['labels'],kwargs['update_on'],kwargs['backpropmode'],kwargs['residue_vname'],kwargs["rand_init"]

    relu=nn.ReLU()
    if(residue_vname == 'std'):
        eps_step_size = eps_step_size/eps

    if(labels is None):
        with torch.no_grad():
          labels = net(org_inputs)
          if(len(labels.size())==1 or labels.shape[1]==1):
              labels = torch.squeeze(labels)
          else:
            _, labels = torch.max(labels, 1)
    if(rand_init):
        if(residue_vname == 'eta_growth'):
            inputs = org_inputs + torch.zeros_like(org_inputs).uniform_(-eps_step_size, eps_step_size)
        else:
            inputs = org_inputs + torch.zeros_like(org_inputs).uniform_(-eps, eps)
    else:
        inputs = org_inputs
    if(residue_vname == 'eta_growth'):
        eps_pos = torch.zeros_like(org_inputs)+eps_step_size
        eps_neg = torch.zeros_like(org_inputs)+eps_step_size
    else:    
        eps_pos = torch.zeros_like(org_inputs)+eps
        eps_neg = torch.zeros_like(org_inputs)+eps
    inputs = torch.clamp(inputs,0.0,1.0)
    
    cur_step_size = eps_step_size
    if(residue_vname == 'eta_growth'):
        cur_step_size = 1
    for cs in range(steps):
        inputs = inputs.clone().detach().to(torch.float).requires_grad_(True)
        if(residue_vname == 'eta_growth'):
            eps_neg = relu(inputs-(org_inputs-(cs+1)*eps_step_size))
            eps_pos = relu(org_inputs+(cs+1)*eps_step_size-inputs)
        else:
            eps_neg = relu(inputs-(org_inputs-eps))
            eps_pos = relu(org_inputs+eps-inputs)
        if(backpropmode == 'normal'):
          output = net(inputs)
          if(len(output.size())==1 or output.shape[1]==1):
            labels = labels.type(torch.float32)
            output = torch.squeeze(output,-1)
          loss = criterion(output, labels)
          loss.backward()
          grad = inputs.grad
        else:
          output,grad = get_no_backprop_grad(net,inputs,criterion,labels)

        if(len(output.size())==1 or output.shape[1]==1):
            labels = labels.type(torch.float32)
            output=torch.squeeze(output,-1)
            predicted = torch.where(output>0,1.0,0.0)
        else:
            _, predicted = torch.max(output.data, 1)

        if(update_on=='corr'):
          I = (predicted == labels)
        elif(update_on=='incorr'):
          I = (predicted != labels)
        elif(update_on=='all'):
          I = torch.where(predicted>=0,True,False)

        with torch.no_grad():
            sgngrad = torch.sign(grad)
            relsgngrad = relu(sgngrad)
            if(residue_vname == 'eq'):
                cur_step_size = 1.0/(steps-cs)
            elif(residue_vname == 'max_eps'):
                tmp=max(torch.max(eps_pos).item(),torch.max(eps_neg).item())
                cur_step_size = (eps_step_size*2*eps)/tmp
            elif(residue_vname == 'min_eps'):
                cur_step_size = eps_step_size/1e-10+(min(torch.min(eps_pos).item(),torch.min(eps_neg).item()))
            inputs[I] = (inputs + cur_step_size * (relsgngrad*eps_pos+(1-relsgngrad)*eps_neg) * sgngrad)[I]
            if(residue_vname == 'eta_growth'):
                inputs = torch.clamp(inputs, org_inputs-eps, org_inputs+eps)
            assert ((inputs - (org_inputs-eps) > -1e-5).all() and (org_inputs+eps -inputs > -1e-5).all()), 'cur_step_size:{}cs:{} {},{}::{},{} eps_pos max:{} min:{} eps_neg max:{} min:{}'.format(cur_step_size,cs,torch.max(inputs - (org_inputs-eps)),torch.min(inputs - (org_inputs-eps)),torch.max(org_inputs+eps -inputs),torch.min(org_inputs+eps -inputs),torch.max(eps_pos),torch.min(eps_pos),torch.max(eps_neg),torch.min(eps_neg))
            # inputs = torch.clamp(inputs, org_inputs-eps, org_inputs+eps)
            inputs = torch.clamp(inputs,0.0,1.0)

    return inputs

def get_feature_maps(model):
    if(isinstance(model, torch.nn.DataParallel)):
        conv_outs = model.module.linear_conv_outputs
    else:
        conv_outs = model.linear_conv_outputs
    return conv_outs

def get_gateflip_adv_per_batch(net,inputs,kwargs):
    kwargs.setdefault('rand_init',True)
    kwargs.setdefault('labels',None)
    kwargs.setdefault('update_on','all')
    kwargs.setdefault('criterion',None)
    residue_vname = kwargs.get("residue_vname",None)
    criterion,eps,eps_step_size,steps,labels,update_on,rand_init=kwargs['criterion'],kwargs['eps'],kwargs['eps_step_size'],kwargs['steps'],kwargs['labels'],kwargs['update_on'],kwargs['rand_init']
    # If y_true is not given, assume model's output as y_true
    if(labels is None):
        with torch.no_grad():
          labels = net(inputs)
          if(len(labels.size())==1 or labels.shape[1]==1):
              labels = torch.where(labels>0,1.0,0.0)
              labels = torch.squeeze(labels,1)
          else:
            _, labels = torch.max(labels, 1)
    
    if(criterion is None):
        tmp = net(inputs)
        # Inferring loss function from model output
        if(len(tmp.size())==1 or tmp.shape[1]==1):
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        net(inputs)
        org_fmap = get_feature_maps(net)

    if(rand_init):
      delta = torch.zeros_like(inputs).uniform_(-eps, eps)
    else:
      delta = torch.zeros_like(inputs)
    delta.data = torch.max(torch.min(1-inputs, delta.data), 0-inputs)
    delta.requires_grad = True

    for cs in range(steps):
        output = net(inputs + delta)
        cur_fmap = get_feature_maps(net)
        if(len(output.size())==1 or output.shape[1]==1):
            labels = labels.type(torch.float32)
            output=torch.squeeze(output,1)
            predicted = torch.where(output>0,1.0,0.0)
        else:
            _, predicted = torch.max(output.data, 1)
        
        if(update_on=='corr'):
          I = (predicted == labels)
        elif(update_on=='incorr'):
          I = (predicted != labels)
        elif(update_on=='all'):
          I = torch.where(predicted>=0,True,False)

        loss = 0
        if(residue_vname is not None and "mixed" in residue_vname):
            loss = criterion(output, labels)
        if(residue_vname is not None and (residue_vname == "tanh_gate_flip" or residue_vname == "mixed_tanh_gate_flip")):
            for i in range(len(org_fmap)):
                cur_tmp = -torch.sign(org_fmap[i]) * torch.tanh(cur_fmap[i])
                cur_tmp = torch.log(1 + torch.exp(torch.clamp(cur_tmp, -10, 10)))
                loss += torch.mean(cur_tmp)
        elif(residue_vname is not None and (residue_vname == "full_gate_flip" or residue_vname == "mixed_full_gate_flip")):
            for i in range(len(org_fmap)):
                cur_tmp = -torch.sign(org_fmap[i]) * cur_fmap[i]
                cur_tmp = torch.log(1 + torch.exp(torch.clamp(cur_tmp, -10, 10)))
                loss += torch.mean(cur_tmp)
        elif(residue_vname is not None and (residue_vname == "all_tanh_gate_flip" or residue_vname == "mixed_all_tanh_gate_flip")):
            for i in range(len(org_fmap)):
                cur_tmp = -torch.tanh(org_fmap[i]) * torch.tanh(cur_fmap[i])
                cur_tmp = torch.log(1 + torch.exp(torch.clamp(cur_tmp, -10, 10)))
                loss += torch.mean(cur_tmp)
        
        # print("step:{} loss:{} ".format(cs,loss))
        loss.backward()
        grad = delta.grad
        with torch.no_grad():
          delta.data[I] = torch.clamp(
              delta + eps_step_size * torch.sign(grad), -eps, eps)[I]
          delta.data[I] = torch.max(
              torch.min(1-inputs, delta.data), 0-inputs)[I]
        delta.grad.data.zero_()
    
    # print("Loss: ",loss)
    delta = delta.detach()
    return torch.clamp(inputs + delta, 0, 1)


def get_locuslab_adv_per_batch(net,inputs,kwargs):
    kwargs.setdefault('rand_init',True)
    kwargs.setdefault('labels',None)
    kwargs.setdefault('update_on','all')
    kwargs.setdefault('criterion',None)
    criterion,eps,eps_step_size,steps,labels,update_on,rand_init=kwargs['criterion'],kwargs['eps'],kwargs['eps_step_size'],kwargs['steps'],kwargs['labels'],kwargs['update_on'],kwargs['rand_init']
    # If y_true is not given, assume model's output as y_true
    if(labels is None):
        with torch.no_grad():
          labels = net(inputs)
          if(len(labels.size())==1 or labels.shape[1]==1):
              labels = torch.where(labels>0,1.0,0.0)
              labels = torch.squeeze(labels,1)
          else:
            _, labels = torch.max(labels, 1)
    if(criterion is None):
        tmp = net(inputs)
        # Inferring loss function from model output
        if(len(tmp.size())==1 or tmp.shape[1]==1):
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
    if(rand_init):
      delta = torch.zeros_like(inputs).uniform_(-eps, eps)
    else:
      delta = torch.zeros_like(inputs)
    delta.data = torch.max(torch.min(1-inputs, delta.data), 0-inputs)
    delta.requires_grad = True

    # If it is one step consider it as FGSM method
    if(steps==1):
      eps_step_size=eps
    for _ in range(steps):
        output = net(inputs + delta)
        if(len(output.size())==1 or output.shape[1]==1):
            labels = labels.type(torch.float32)
            output=torch.squeeze(output,1)
            predicted = torch.where(output>0,1.0,0.0)
        else:
            _, predicted = torch.max(output.data, 1)
        if(update_on=='corr'):
          I = (predicted == labels)
        elif(update_on=='incorr'):
          I = (predicted != labels)
        elif(update_on=='all'):
          I = torch.where(predicted>=0,True,False)

        loss = criterion(output, labels)
        loss.backward()
        grad = delta.grad
        with torch.no_grad():
          delta.data[I] = torch.clamp(
              delta + eps_step_size * torch.sign(grad), -eps, eps)[I]
          delta.data[I] = torch.max(
              torch.min(1-inputs, delta.data), 0-inputs)[I]
        delta.grad.data.zero_()
    
    delta = delta.detach()
    return torch.clamp(inputs + delta, 0, 1)

def cleverhans_fast_gradient_method(
    model_fn,
    x,
    kwargs
):
    loss_fn = kwargs.get('criterion',None)
    eps = kwargs['eps']
    norm=kwargs.get('norm',np.inf)
    clip_min=kwargs.get('clip_min',None)
    clip_max=kwargs.get('clip_max',None)
    y=kwargs.get('labels',None)
    targeted=kwargs.get('targeted',False)
    sanity_checks= kwargs.get('sanity_checks',False)
    update_on=kwargs.get('update_on','all')
    residue_vname = kwargs.get("residue_vname",None)
    """
    PyTorch implementation of the Fast Gradient Method.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError(
            "Norm order must be either np.inf, 1, or 2, got {} instead.".format(
                norm)
        )
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(
                eps)
        )
    if eps == 0:
        return x
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # x needs to be a leaf variable, of floating point type and have requires_grad being True for
    # its grad to be computed and stored properly in a backward call
    x = x.clone().detach().to(torch.float).requires_grad_(True)
    if y is None:
        tmp = model_fn(x)
        # Using model predictions as ground truth to avoid label leaking
        if(len(tmp.size())==1 or tmp.shape[1]==1):
            y = torch.where(tmp>0,1,0)
            y = torch.squeeze(y,1)
        else:
            _, y = torch.max(tmp, 1)
    
    if(loss_fn is None):
        tmp = model_fn(x)
        # Inferring loss function from model output
        if(len(tmp.size())==1 or tmp.shape[1]==1):
            loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

    # Compute loss
    if(isinstance(loss_fn,torch.nn.BCEWithLogitsLoss)):
        y = y.type(torch.float32)
        y_pred = torch.squeeze(model_fn(x),1)
        predicted = torch.where(y_pred>0,1.0,0.0)
    elif(isinstance(loss_fn,torch.nn.CrossEntropyLoss)):
        y_pred = model_fn(x)
        _, predicted = torch.max(y_pred.data, 1)
    else:
        y_pred = model_fn(x)
        _, predicted = torch.max(y_pred.data, 1)
    
    loss = loss_fn(y_pred, y)
    
    # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    loss.backward()
    if(residue_vname is not None and "plain_grad_without_sign" == residue_vname):
        cgrad = x.grad
    elif(residue_vname is not None and "mean_grad" == residue_vname):
        cgrad = x.grad / torch.mean(x.grad,[1,2]).unsqueeze(1).unsqueeze(2)
    elif(residue_vname is not None and "L1_norm_grad_scale" == residue_vname):
        cgrad = (x.grad / (torch.norm(x.grad,p=1,dim=[1,2]).unsqueeze(1).unsqueeze(2)+10e-8)) * x[0].numel()
    elif(residue_vname is not None and "L2_norm_grad_scale" == residue_vname):
        cgrad = (x.grad / (torch.norm(x.grad,p=2,dim=[1,2]).unsqueeze(1).unsqueeze(2)+10e-8)) * math.sqrt(x[0].numel())
    elif(residue_vname is not None and "L2_norm_grad_unitnorm" == residue_vname):
        cgrad = (x.grad / (torch.norm(x.grad,p=2,dim=[1,2]).unsqueeze(1).unsqueeze(2)+10e-8))
    elif(residue_vname is not None and "PGD_unit_norm" == residue_vname):
        tmp = torch.sign(x.grad)
        cgrad = tmp / (torch.norm(tmp,p=2,dim=[1,2]).unsqueeze(1).unsqueeze(2) + 10e-8)
    else:
        #Default use sign grad
        cgrad = torch.sign(x.grad)
    
    optimal_perturbation = optimize_linear(cgrad, eps, norm)
    if(update_on=='all'):
      adv_x = x + optimal_perturbation
    elif(update_on=='corr'):
      with torch.no_grad():
        I = (predicted == y)
        adv_x = x
        adv_x[I] = (x + optimal_perturbation)[I]
    else:
        raise ValueError("Incorrect value for update_on")

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
                "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
            )
        adv_x = torch.clamp(adv_x, clip_min, clip_max)
    
    if sanity_checks:
        assert np.all(asserts)
    # print("cgrad max:{} mean:{}".format(torch.max(torch.max(cgrad,-1)[0],-1)[0],torch.mean(cgrad,[1,2])))
    # print("x.grad size:{} max:{} mean:{} norm:{}".format(x.grad.size(),torch.max(torch.max(x.grad,-1)[0],-1)[0],torch.mean(x.grad,[1,2]),torch.norm(x.grad,p=1,dim=[1,2])))
    # print("cgrad:{}".format(cgrad))
    # print("x.grad:{}".format(x.grad))
    return adv_x

def cleverhans_projected_gradient_descent(
    model_fn,
    x,
    kwargs
):
    eps = kwargs['eps']
    loss_fn = kwargs.get('criterion',None)
    lr_sched = kwargs.get("lr_sched",None)
    num_restarts = kwargs.get("num_of_restrts",1)
    residue_vname = kwargs.get("residue_vname",None)
    vname_arr = None
    if(residue_vname is not None):
        vname_arr = residue_vname.split("__")
    eps_iter=kwargs['eps_step_size']
    nb_iter=kwargs['steps']
    kwargs.setdefault('norm',np.inf)
    norm=kwargs['norm']
    kwargs.setdefault('clip_min',None)
    clip_min=kwargs['clip_min']
    kwargs.setdefault('clip_max',None)
    clip_max=kwargs['clip_max']
    kwargs.setdefault('labels',None)
    y=kwargs['labels']
    kwargs.setdefault('targeted',False)
    kwargs.setdefault('rand_init',True)
    rand_init=kwargs['rand_init']
    kwargs.setdefault('rand_minmax',None)
    rand_minmax=kwargs['rand_minmax']
    sanity_checks= kwargs.get('sanity_checks',True)
    kwargs.setdefault('update_on','all')
    """
    This class implements either the Basic Iterative Method
    (Kurakin et al. 2016) when rand_init is set to False. or the
    Madry et al. (2017) method if rand_init is set to True.
    Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
    Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param eps_iter: step size for each attack iteration
    :param nb_iter: Number of attack iterations.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param rand_init: (optional) bool. Whether to start the attack from a randomly perturbed x.
    :param rand_minmax: (optional) bool. Support of the continuous uniform distribution from
              which the random perturbation on x was drawn. Effective only when rand_init is
              True. Default equals to eps.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    
    if norm == 1:
        raise NotImplementedError(
            "It's not clear that FGM is a good inner loop"
            " step for PGD when norm=1, because norm=1 FGM "
            " changes only one pixel at a time. We need "
            " to rigorously test a strong norm=1 PGD "
            "before enabling this feature."
        )
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if eps_iter < 0:
        raise ValueError(
            "eps_iter must be greater than or equal to 0, got {} instead".format(
                eps_iter
            )
        )
    if eps_iter == 0:
        return x

    assert eps_iter <= eps, (eps_iter, eps)
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        ).cpu().numpy()
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        ).cpu().numpy()
        asserts.append(assert_le)

    if(loss_fn is None):
        tmp = model_fn(x)
        # Inferring loss function from model output
        if(len(tmp.size())==1 or tmp.shape[1]==1):
            loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        outs = model_fn(x)
        if(outs.shape[1]==1):
            y = torch.where(outs>0,1,0)
            y = torch.squeeze(y,1)
        elif(len(outs.size())==1):
            y = outs.data.round()
        else:
            _, y = torch.max(outs.data, 1)
        kwargs['labels']=y

    best_adv_x_loss = -float("inf")
    for cur_restart_iter in range(num_restarts):
        # Initialize loop variables
        if rand_init:
            if rand_minmax is None:
                rand_minmax = eps
            eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)

            # Uncomment the below line for starting at a random point on the border of the norm ball
            # eta = eps * torch.where(eta>=0,1,-1)
        else:
            eta = torch.zeros_like(x)
        # print("cleverhans_projected_gradient_descent",kwargs['criterion'],eps,eps_iter,nb_iter,kwargs['labels'],kwargs['update_on'],rand_init)
        # Clip eta
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)
        else:
            adv_x = torch.clamp(adv_x, 0.0,1.0)

        i = 0
        while i < nb_iter:
            kwargs['eps'] = eps_iter
            if(vname_arr and vname_arr[0] == "reach_edge_at_end"):
                if(i == nb_iter-1):
                    kwargs['eps'] = eps
            if(lr_sched is not None):
                for lr_step,lr_step_size in lr_sched:
                    if(i>=lr_step):
                        kwargs['eps'] = lr_step_size

            adv_x = cleverhans_fast_gradient_method(model_fn,adv_x,kwargs)

            # Clipping perturbation eta to norm norm ball
            eta = adv_x - x
            eta = clip_eta(eta, norm, eps)
            adv_x = x + eta

            if(vname_arr and vname_arr[0]=='add_rand_at'):
                if(str(i) in vname_arr):
                    rpert = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
                    eta = adv_x + rpert - x
                    eta = clip_eta(eta, norm, eps)
                    adv_x = x + eta

            # Redo the clipping.
            # FGM already did it, but subtracting and re-adding eta can add some
            # small numerical error.
            if clip_min is not None or clip_max is not None:
                adv_x = torch.clamp(adv_x, clip_min, clip_max)
            else:
                adv_x = torch.clamp(adv_x, 0.0,1.0)
            i += 1

        asserts.append(eps_iter <= eps)
        if norm == np.inf and clip_min is not None:
            # TODO necessary to cast clip_min and clip_max to x.dtype?
            asserts.append(eps + clip_min <= clip_max)

        if sanity_checks:
            assert np.all(asserts)
        # Set back correct value before return
        kwargs['eps'] = eps

        if(num_restarts == 1):
            best_adv_x = adv_x
        else:
            # Just plain loss evaluation doesn't need grads to be switched on
            with torch.no_grad():
                if(isinstance(loss_fn,torch.nn.BCEWithLogitsLoss)):
                    y = y.type(torch.float32)
                    y_pred = torch.squeeze(model_fn(adv_x),1)
                elif(isinstance(loss_fn,torch.nn.CrossEntropyLoss)):
                    y_pred = model_fn(adv_x)
                else:
                    y_pred = model_fn(adv_x)
                    
                adv_x_loss = loss_fn(y_pred, kwargs['labels'])
                if(adv_x_loss.item() > best_adv_x_loss):
                    best_adv_x_loss = adv_x_loss.item()
                    best_adv_x = adv_x
            
            asserts = []
    
    return best_adv_x
