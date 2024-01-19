import os, sys, json
# online package
import torch
# optimizer
import torch.optim as optim
# dataloader
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
# utils
from utils.logger import *
from utils.misc import *
from timm.scheduler import CosineLRScheduler

def dataset_builder(args, config):
    dataset = build_dataset_from_cfg(config._base_, config.others)
    shuffle = config.others.subset == 'train'
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.others.bs,
                                            num_workers = int(args.num_workers),
                                            drop_last = config.others.subset == 'train',
                                            worker_init_fn = worker_init_fn,
                                            sampler = sampler)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs,
                                                shuffle = shuffle, 
                                                drop_last = config.others.subset == 'train',
                                                num_workers = int(args.num_workers),
                                                worker_init_fn=worker_init_fn)
    return sampler, dataloader

def model_builder(config):
    model = build_model_from_cfg(config)
    return model

def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("pos_embed"):
        return 0
    elif var_name.startswith("encoder"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def build_opti_sche(base_model, config):
    def add_weight_decay(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
        parameter_group_names = {}
        parameter_group_vars = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            if get_num_layer is not None:
                layer_id = get_num_layer(name)
                group_name = "layer_%d_%s" % (layer_id, group_name)
            else:
                layer_id = None

            if group_name not in parameter_group_names:
                if get_layer_scale is not None:
                    scale = get_layer_scale(layer_id)
                else:
                    scale = 1.

                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }

            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
        print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        return list(parameter_group_vars.values())
    # config.optimizer.kwargs.layer_decay = 0.85
#     print(config.model)
    depth = config.model.depth if not hasattr(config.model, 'transformer_config') else config.model.transformer_config.depth
    
    assigner = LayerDecayValueAssigner(list(0.65 ** (depth + 1 - i) for i in range(depth + 2)))
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay, get_num_layer=assigner.get_layer_id, get_layer_scale=assigner.get_scale)
        # param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'RAdam':
        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.RAdam(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(base_model.parameters(), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(
            base_model.parameters(), nesterov=True, momentum=0.9, **opti_config.kwargs)
    else:
        raise NotImplementedError()

    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs)  # misc.py
    elif sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                t_initial=sche_config.kwargs.epochs,
                cycle_mul=1.,
                # lr_min=1e-6,
                lr_min=1e-6,
                cycle_decay=0.1,
                warmup_lr_init=1e-6,
                # warmup_lr_init=1e-7,
                warmup_t=sche_config.kwargs.initial_epochs,
                cycle_limit=1,
                t_in_epochs=True)
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config.kwargs)
    elif sche_config.type == 'function':
        scheduler = None
    else:
        raise NotImplementedError()
    
    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
        scheduler = [scheduler, bnscheduler]
    
    return optimizer, scheduler

def resume_model(base_model, args, logger=None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger=logger)
        return 0, 0
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger=logger )

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    # parameter resume of base model
    # if args.local_rank == 0:
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt, strict = True)

    # parameter
    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()
    # print(best_metrics)

    print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})', logger=logger)
    return start_epoch, best_metrics

def resume_optimizer(optimizer, args, logger=None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger=logger)
        return 0, 0, 0
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', logger=logger )
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

def save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, skip=False, logger=None):
    if skip:
        print_log(f"Skipped saving checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger=logger)
        return
    if args.local_rank == 0:
        torch.save({
                    'base_model' : base_model.module.state_dict() if args.distributed else base_model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'epoch' : epoch,
                    'metrics' : metrics.state_dict() if metrics is not None else dict(),
                    'best_metrics' : best_metrics.state_dict() if best_metrics is not None else dict(),
                    }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger=logger)

def load_model(base_model, ckpt_path, logger=None):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger=logger)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    base_model.load_state_dict(base_ckpt, strict = True)

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger=logger)
    return 
