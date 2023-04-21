import argparse
import os
import random
import time
import torch.distributed as dist
import warnings
from enum import Enum
import wandb
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from conv4_models import get_model_instance_from_dataset, get_model_save_path, vgg16_bn, pad2_vgg16_bn, st1_pad2_vgg16_bn_wo_bias, st1_pad1_vgg16_bn_wo_bias, dnn_vgg16_bn, dnn_st1_pad1_vgg16_bn_wo_bias

temp_names = sorted(name for name in models.__dict__
                    if name.islower() and not name.startswith("__") and "vgg" in name
                    and callable(models.__dict__[name]))
temp_names.append("pad2_vgg16_bn")
temp_names.append("st1_pad2_vgg16_bn_wo_bias")
temp_names.append("st1_pad1_vgg16_bn_wo_bias")
temp_names.append("cvgg16_bn")

prefixes = ["dlgn", "dgn", "dnn"]
model_names = temp_names.copy()
for each_pf in prefixes:
    for each_mname in temp_names:
        model_names.append(each_pf+"__"+each_mname+"__")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg19)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)


class DLGN_VGG_Network_without_BN(nn.Module):
    def __init__(self, num_classes=10):
        super(DLGN_VGG_Network_without_BN, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.g_conv_64_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.initialize_weights(self.g_conv_64_1)
        self.g_conv_64_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.initialize_weights(self.g_conv_64_2)

        self.g_conv_64_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.g_conv_128_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.initialize_weights(self.g_conv_128_1)
        self.g_conv_128_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.initialize_weights(self.g_conv_128_2)

        self.g_conv_128_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.g_conv_256_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.initialize_weights(self.g_conv_256_1)
        self.g_conv_256_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.initialize_weights(self.g_conv_256_2)
        self.g_conv_256_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.initialize_weights(self.g_conv_256_3)

        self.g_conv_256_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.g_conv_512_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.initialize_weights(self.g_conv_512_1)
        self.g_conv_512_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.g_conv_512_2)
        self.g_conv_512_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.g_conv_512_3)

        self.g_conv_512_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.g_conv_512_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.g_conv_512_4)
        self.g_conv_512_5 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.g_conv_512_5)
        self.g_conv_512_6 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.g_conv_512_6)

        self.g_conv_512_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.w_conv_64_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.initialize_weights(self.w_conv_64_1)
        self.w_conv_64_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.initialize_weights(self.w_conv_64_2)
        self.w_conv_64_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.w_conv_128_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.initialize_weights(self.w_conv_128_1)
        self.w_conv_128_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.initialize_weights(self.w_conv_128_2)
        self.w_conv_128_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.w_conv_256_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.initialize_weights(self.w_conv_256_1)
        self.w_conv_256_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.initialize_weights(self.w_conv_256_2)
        self.w_conv_256_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.initialize_weights(self.w_conv_256_3)
        self.w_conv_256_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.w_conv_512_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.initialize_weights(self.w_conv_512_1)
        self.w_conv_512_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.w_conv_512_2)
        self.w_conv_512_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.w_conv_512_3)
        self.w_conv_512_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.w_conv_512_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.w_conv_512_4)
        self.w_conv_512_5 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.w_conv_512_5)
        self.w_conv_512_6 = nn.Conv2d(512, 512, 3, padding=1)
        self.initialize_weights(self.w_conv_512_6)

        self.w_conv_512_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.g_conv_512_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))

        self.w_fc_1 = nn.Linear(512, num_classes)
        self.initialize_weights(self.w_fc_1)
        # self.g_fc_1 = nn.Linear(512 * 7 * 7, 4096)
        # self.initialize_weights(self.g_fc_1)
        # self.w_dp1 = nn.Dropout()
        # self.w_fc_2 = nn.Linear(4096, 4096)
        # self.g_fc_2 = nn.Linear(4096, 4096)
        # self.initialize_weights(self.w_fc_2)
        # self.initialize_weights(self.g_fc_2)
        # self.w_dp2 = nn.Dropout()
        # self.w_fc_3 = nn.Linear(4096, num_classes)
        # self.initialize_weights(self.w_fc_3)

        self.allones = None

    def initialize_weights(self, mod_obj):
        if isinstance(mod_obj, nn.Conv2d):
            nn.init.kaiming_normal_(
                mod_obj.weight, mode='fan_out', nonlinearity='relu')
            if mod_obj.bias is not None:
                nn.init.constant_(mod_obj.bias, 0)
        elif isinstance(mod_obj, nn.BatchNorm2d):
            nn.init.constant_(mod_obj.weight, 1)
            nn.init.constant_(mod_obj.bias, 0)
        elif isinstance(mod_obj, nn.Linear):
            nn.init.normal_(mod_obj.weight, 0, 0.01)
            nn.init.constant_(mod_obj.bias, 0)

    def forward(self, inp, verbose=2):
        self.allones = torch.ones(inp.size(), requires_grad=True,
                                  device=self.device)
        beta = 4
        # conv_g_outs = []
        # 64 blocks *********************************************

        x_g = self.g_conv_64_1(inp)
        x_w = self.w_conv_64_1(self.allones)
        x_w = x_w * nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g1)

        x_g = self.g_conv_64_2(x_g)
        x_w = self.w_conv_64_2(x_w)
        x_w = x_w * nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g2)
        x_g = self.g_conv_64_pool(x_g)
        x_w = self.w_conv_64_pool(x_w)

        # ********************************************************

        # 128 block *********************************************

        x_g = self.g_conv_128_1(x_g)
        x_w = self.w_conv_128_1(x_w)
        x_w = x_w * nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g3)

        x_g = self.g_conv_128_2(x_g)
        x_w = self.w_conv_128_2(x_w)
        x_w = x_w * nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g4)
        x_g = self.g_conv_128_pool(x_g)
        x_w = self.w_conv_128_pool(x_w)

        # **********************************************************

        # 256 blocks ***********************************************

        x_g = self.g_conv_256_1(x_g)
        x_w = self.w_conv_256_1(x_w)
        x_w = x_w * nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g5)

        x_g = self.g_conv_256_2(x_g)
        x_w = self.w_conv_256_2(x_w)
        x_w = x_w * nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g6)

        x_g = self.g_conv_256_3(x_g)
        x_w = self.w_conv_256_3(x_w)
        x_w = x_w * nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g7)

        x_g = self.g_conv_256_pool(x_g)
        x_w = self.w_conv_256_pool(x_w)

        # **********************************************************

        # 512 blocks 1 ***************************************************

        x_g = self.g_conv_512_1(x_g)
        x_w = self.w_conv_512_1(x_w)
        x_w = x_w * nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g8)

        x_g = self.g_conv_512_2(x_g)
        x_w = self.w_conv_512_2(x_w)
        x_w = x_w * nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g9)

        x_g = self.g_conv_512_3(x_g)
        x_w = self.w_conv_512_3(x_w)
        x_w = x_w * nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g10)

        x_g = self.g_conv_512_pool1(x_g)
        x_w = self.w_conv_512_pool1(x_w)

        # **********************************************************

        # 512 blocks 2 ***************************************************

        x_g = self.g_conv_512_4(x_g)
        x_w = self.w_conv_512_4(x_w)
        x_w = x_w * nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g11)

        x_g = self.g_conv_512_5(x_g)
        x_w = self.w_conv_512_5(x_w)
        x_w = x_w * nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g12)

        x_g = self.g_conv_512_6(x_g)
        x_w = self.w_conv_512_6(x_w)
        x_w = x_w * nn.Sigmoid()(beta * x_g)
        # conv_g_outs.append(x_g13)

        # x_g = self.g_conv_512_pool2(x_g)
        x_w = self.w_conv_512_pool2(x_w)

        x_w = self.globalpool(x_w)
        # x_w = self.w_adapt_pool(x_w)
        x_w = torch.flatten(x_w, 1)

        x_w = self.w_fc_1(x_w)
        # x_w = x_w * nn.Sigmoid()(beta * x_g)

        # x_g = self.g_fc_2(x_g)
        # x_w = self.w_fc_2(x_w)
        # x_w = x_w * nn.Sigmoid()(beta * x_g)

        # x_w = self.w_fc_3(x_w)

        return x_w


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


best_acc1 = 0
av_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    dataset = "cifar10"
    wand_project_name = "common_model_init_exps"
    # wand_project_name = None
    global args, best_acc1, av_device
    args = parser.parse_args()
    args.wand_project_name = wand_project_name
    args.start_step = 0

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if(args.arch == "dlgn__vgg16__"):
        model = DLGN_VGG_Network_without_BN(num_classes=10)
    elif(args.arch == "dlgn__vgg16_bn__"):
        allones = np.ones((1, 3, 32, 32)).astype(np.float32)
        model = vgg16_bn(allones)
    elif(args.arch == "dlgn__pad2_vgg16_bn__"):
        model = pad2_vgg16_bn()
    elif(args.arch == "dlgn__st1_pad2_vgg16_bn_wo_bias__"):
        model = st1_pad2_vgg16_bn_wo_bias()
    elif(args.arch == "dlgn__st1_pad1_vgg16_bn_wo_bias__"):
        model = st1_pad1_vgg16_bn_wo_bias()
    elif(args.arch == "dnn__st1_pad2_vgg16_bn_wo_bias__"):
        model = dnn_st1_pad1_vgg16_bn_wo_bias()
    elif(args.arch == "dnn__cvgg16_bn__"):
        model = dnn_vgg16_bn()
    # model = get_model_instance_from_dataset(
    #     dataset=dataset, model_arch_type=args.arch, num_classes=10, pretrained=args.pretrained)

    model = torch.nn.DataParallel(model).cuda()
    print("model", model)
    # model = model.to(av_device)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().to(av_device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            args.start_step = checkpoint['steps']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    final_model_save_path = get_model_save_path(
        args.arch+"_PRET_"+str(args.pretrained), dataset, args.seed)
    model_save_folder = final_model_save_path[0:final_model_save_path.rfind(
        "/")+1]
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    is_log_wandb = not(wand_project_name is None)
    if(is_log_wandb):
        wandb_group_name = "DS_"+str(dataset) + \
            "_MT_"+str(args.arch)+"_PRET_"+str(args.pretrained) + \
            "_SEED_"+str(args.seed)
        wandb_run_name = "MT_" + \
            str(args.arch)+"/SEED_"+str(args.seed)+"/EP_"+str(args.epochs)+"/OPT_"+str(optimizer)+"/LOSS_TYPE_" + \
            str(criterion)+"/BS_"+str(args.batch_size) + \
            "/pretrained"+str(args.pretrained)
        wandb_run_name = wandb_run_name.replace("/", "")

        wandb_config = dict()
        wandb_config["dataset"] = dataset
        wandb_config["model_arch_type"] = args.arch
        wandb_config["epochs"] = args.epochs
        wandb_config["optimizer"] = optimizer
        wandb_config["scheduler"] = scheduler
        wandb_config["criterion"] = criterion
        wandb_config["batch_size"] = args.batch_size
        wandb_config["pretrained"] = args.pretrained

        wandb.init(
            project=f"{wand_project_name}",
            name=f"{wandb_run_name}",
            group=f"{wandb_group_name}",
            config=wandb_config,
        )

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_acc1
        best_acc1 = max(prec1, best_acc1)
        if(is_log_wandb):
            wandb.log({"steps": args.start_step, "test_acc": prec1,
                      "cur_epoch": epoch+1, "best_test_acc": best_acc1})

        save_checkpoint({
            'epoch': epoch + 1,
            'steps': args.start_step,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pretrained': args.pretrained
        }, is_best, final_model_save_path)

    if(is_log_wandb):
        wandb.finish()


def custom_piecewise_lr_decay_scheduler(optimizer, n_iter):
    if n_iter > 48000:
        optimizer.param_groups[0]['lr'] = 0.001
    elif n_iter > 32000:
        optimizer.param_groups[0]['lr'] = 0.01
    elif n_iter > 400:
        optimizer.param_groups[0]['lr'] = 0.1
    else:
        optimizer.param_groups[0]['lr'] = 0.01


def train(train_loader, model, criterion, optimizer, epoch):
    is_log_wandb = not(args.wand_project_name is None)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(av_device, non_blocking=True)
        target = target.to(av_device, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        args.start_step += 1
        # custom_piecewise_lr_decay_scheduler(optimizer, args.start_step)

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if(is_log_wandb):
                wandb.log({"loss_tr": loss.item(),
                          "train_acc": prec1.item()})
            progress.display(i + 1)


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if args.cpu == False:
            input = input.cuda()
            target = target.cuda()

        if args.half:
            input = input.half()

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count],
                             dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
    print("Finished execution!!!")
