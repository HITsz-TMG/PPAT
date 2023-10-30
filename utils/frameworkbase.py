import os
import random
import torch
import numpy as np

class BaseFramework():
    def __init__(self, opt):
        self.opt = opt
        self.set_random(opt.random_seed)

    @staticmethod
    def load_model(logger, model, load_ckpt, optimizer=None, schedule=None):
        logger.info('----------load checkpoint----------')
        if os.path.isfile(load_ckpt):
            checkpoint = torch.load(load_ckpt, map_location=torch.device('cpu'))
        else:
            raise Exception(f"No checkpoint found at {load_ckpt}")
        load_state = checkpoint['model']
        model_state = model.state_dict()
        for name, param in load_state.items():
            if name not in model_state:
                logger.warning(f"[NAME UNMATCH] {name} is not in current model")
                continue
            if param.shape != model_state[name].shape:
                logger.warning(f"[SHAPE UNMATCH] {name} param shape not match current model")
                continue
            model_state[name].copy_(param)
        for name, param in model_state.items():
            if name not in load_state:
                logger.warning(f"[NET UNLOAD] {name} not in checkpoint")
        if optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                logger.warning(f"[OPTIMIZER] cannot load optimizer")
                # assert False
        if schedule is not None:
            try:
                schedule.load_state_dict(checkpoint['schedule'])
            except:
                logger.warning(f"[SCHEDULER] cannot load scheduler")
                # assert False
        logger.info(f'----------load {load_ckpt} successful----------')

    @staticmethod
    def save_model(opt, model, save_ckpt, schedule=None, optimizer=None, epoch=None, tag='best'):
        assert 'best' in tag or 'last' in tag
        model.eval()
        checkpoint = {'opt': opt, 'model': model.state_dict()}
        if tag == 'last':
            assert schedule is not None
            assert optimizer is not None
            assert epoch is not None
            checkpoint['optimizer'] = optimizer.state_dict()
            checkpoint['schedule'] = schedule.state_dict()
            checkpoint['epoch'] = epoch
        torch.save(checkpoint, save_ckpt + tag + '.ckpt')

    @staticmethod
    def set_random(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_opt(opt):
    path = opt.ckpt_dir + 'last.ckpt'
    if os.path.isfile(path):
        checkpoint = torch.load(path)
    else:
        raise Exception(f"No checkpoint found at {path}")
    nopt = checkpoint['opt']
    nopt.start_epoch = checkpoint['epoch'] + 1
    nopt.ckpt_dir = opt.ckpt_dir
    nopt.save_path = opt.save_path
    nopt.device = opt.device
    nopt.run_mode = opt.run_mode
    nopt.notes = opt.notes + '\n' + nopt.notes
    nopt.logger = None
    return nopt