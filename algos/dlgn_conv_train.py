import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import wandb
from datetime import datetime
import traceback
from torch.utils.tensorboard import SummaryWriter
from model.dlgn_conv_nn_model import DLGN_CONV_Network
from utils.generic_utils import create_nested_dir_if_not_exists


def train_one_epoch(model, training_loader, optimizer, loss_fn, epoch_index, params_to_update, tb_writer, verbose=1):
    running_loss = 0.
    last_loss = 0.
    avg_loss = 0.
    running_acc = 0.
    avg_acc = 0.
    step = 0

    predictions, actuals = list(), list()

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    loader = tqdm.tqdm(training_loader, desc='Train data loader')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, data in enumerate(loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(
            device), labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs, verbose)

        # Compute the loss and its gradients
        # print("Labels:", labels)
        # print("Labels size:", labels.size())
        # print("outputs:", outputs)
        # print("outputs size:", outputs.size())
        loss = loss_fn(outputs, labels)

        loader.set_description('Loss %.02f at step %d' % (loss, step))
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        step += 1

        outputs = outputs.softmax(dim=1).max(1).indices
        yhat = outputs.cpu().clone().detach().numpy()
        actual = labels.cpu().numpy()

        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))

        # store
        predictions.append(yhat)
        actuals.append(actual)

        # Gather data and report
        running_loss += loss.item()
        # tb_writer.add_graph(model, inputs)
        # if i % 100 == 99:
        #     last_loss = running_loss / 1000 # loss per batch
        #     print('  batch {} loss: {}'.format(i + 1, last_loss))
        #     tb_x = epoch_index * len(training_loader) + i + 1
        #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        #     running_loss = 0.
    # for p in model.parameters():
    #   print(p.data,p.grad)

    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    avg_loss = running_loss / (i + 1)
    avg_acc = accuracy_score(actuals, predictions)
    return avg_loss, avg_acc


def evaluate_model(model, data_loader, loss_fn):
    # We don't need gradients on to do reporting
    model.train(False)
    running_vloss = 0.0
    running_vacc = 0.
    avg_vacc = 0.

    vpredictions, vactuals = list(), list()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i, vdata in enumerate(data_loader):
        vinputs, vlabels = vdata
        # print("vlabels:",vlabels)
        # print("vinputs:",vinputs)
        vinputs, vlabels = vinputs.to(device), vlabels.to(device)

        voutputs = model(vinputs)
        # print("voutputs:",voutputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss.item()

        voutputs = voutputs.softmax(dim=1).max(1).indices
        vyhat = voutputs.cpu().clone().detach().numpy()
        vactual = vlabels.cpu().numpy()

        vactual = vactual.reshape((len(vactual), 1))
        vyhat = vyhat.reshape((len(vyhat), 1))

        # store
        vpredictions.append(vyhat)
        vactuals.append(vactual)
        # running_vacc += accuracy_score(vlabels, voutputs.numpy())

    vpredictions, vactuals = np.vstack(vpredictions), np.vstack(vactuals)
    avg_vloss = running_vloss / (i + 1)
    avg_vacc = accuracy_score(vactuals, vpredictions)
    return avg_vloss, avg_vacc


def train_DLGN(model, train_data_loader, valid_data_loader, hp, writer, model_path, verbose=1, log_wandb=False):
    loss_fn = hp.loss_fn

    if hp.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hp.lr)
    elif hp.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=hp.lr,
                                    momentum=hp.momentum)
    elif hp.optimizer == 'NSTROV':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=hp.lr,
                                    momentum=hp.momentum, nesterov=True)
    else:
        raise Exception("Optimizer not supported: %s" % hp.optimizer)

    avg_loss = 0
    avg_acc = 0
    avg_vloss = 0
    avg_vacc = 0

    best_vloss = 10000000

    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0

        EPOCHS = hp.epochs

        constant_params = [p.clone().detach()
                           for p in model.linear_conv_net.parameters()]

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            avg_loss, avg_acc = train_one_epoch(
                model, train_data_loader, optimizer, loss_fn, epoch_number, model.parameters(), writer, verbose)

            avg_vloss, avg_vacc = evaluate_model(
                model, valid_data_loader, loss_fn)

            if(log_wandb):
                wandb.log({"current_epoch": epoch_number+1,
                           "train_metric_epoch": {'acc_epoch': round(avg_acc, 5), 'loss_epoch': round(avg_loss, 5)},
                           "valid_metric_epoch": {'acc_epoch': round(avg_vacc, 5), 'loss_epoch': round(avg_vloss, 5)}})

            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            print('Accuracy train {} valid {}'.format(avg_acc, avg_vacc))

            writer.add_scalars('Training vs. Validation Loss',
                               {'Training': avg_loss, 'Validation': avg_vloss},
                               epoch_number + 1)
            writer.add_scalars('Training vs. Validation Accuracy',
                               {'Training': avg_acc, 'Validation': avg_vacc},
                               epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss

            epoch_number += 1

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_fn,
            }, model_path+"_ep_"+str(epoch_number)+".pt")

            target_params = [p for p in model.linear_conv_net.parameters()]

            temp = True
            for indx in range(len(target_params)):
                each_target = target_params[indx]
                each_constant = constant_params[indx]
                temp = temp and torch.all(each_target.eq(each_constant))
            assert temp == False, 'No change after an epoch in gate param bug exists!'

    except Exception as e:
        print("Exiting due to exception: %s" % e)
        traceback.print_exc()
    return avg_loss, avg_acc, avg_vloss, avg_vacc, optimizer


def initialise_DLGN_conv_model(gate_net_conv_info, is_DLGN_all_ones, seed, weight_net_conv_info, gating_activation_func):
    dlgn_net = DLGN_CONV_Network(gate_net_conv_info, weight_net_conv_info, gating_activation_func, is_weight_net_all_ones=is_DLGN_all_ones,
                                 seed=seed, is_enable_weight_net_weight_restore=False, is_enable_gate_net_weight_restore=False)
    return dlgn_net


def construct_train_evaluate_DLGN_model(gate_net_conv_info, weight_net_conv_info, train_data_loader, valid_data_loader, test_data_loader, hp_obj, runs_out_dir, save_out_dir, conf_name, all_param_hash, seed=2022, writer=None, is_DLGN_all_ones=True, verbose=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if(writer is None):
        writer = SummaryWriter(
            str(runs_out_dir)+'/mnist_4_9_trainer_{}_{}'.format(conf_name, timestamp))

    dlgn_net = initialise_DLGN_conv_model(
        gate_net_conv_info, is_DLGN_all_ones, seed, weight_net_conv_info, hp_obj.activ_func)

    if(not(all_param_hash is None)):
        wandb.watch(dlgn_net, log="all", log_freq=100, log_graph=(True))

    dlgn_net.to(device)

    num_params_total = sum(p.numel() for p in dlgn_net.parameters())
    num_params_gate = sum(p.numel()
                          for p in dlgn_net.linear_conv_net.parameters())
    num_params_weight = sum(p.numel()
                            for p in dlgn_net.weight_conv_net.parameters())

    print("Number of params total is:"+str(num_params_total))
    print("Number of params in gate net is:"+str(num_params_gate))
    print("Number of params in weight net is:"+str(num_params_weight))

    model_per_epoch_dir = str(save_out_dir) + "/per_epoch"
    create_nested_dir_if_not_exists(model_per_epoch_dir)
    model_path = str(save_out_dir)+'/model_{}.pt'.format(conf_name)
    full_save_model_path = str(save_out_dir) + \
        '/model_{}.pt'.format(all_param_hash)

    log_wandb = not(all_param_hash is None)

    train_loss, train_acc, valid_loss, valid_acc, optimizer = train_DLGN(
        dlgn_net, train_data_loader, valid_data_loader, hp_obj, writer, str(model_per_epoch_dir)+'/model_{}'.format(conf_name), verbose, log_wandb)
    test_loss, test_acc = evaluate_model(
        dlgn_net, test_data_loader, hp_obj.loss_fn)

    torch.save({
        'model_state_dict': dlgn_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': hp_obj.loss_fn,
    }, full_save_model_path)
    # torch.save(dlgn_net, full_save_model_path)

    return test_acc, train_acc, valid_acc, test_loss, train_loss, valid_loss, dlgn_net
