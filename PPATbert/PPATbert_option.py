import argparse
import copy
import json
import os
import time
from pprint import pprint

import torch
from attrdict import AttrDict


def load_opt(opt):
    path = opt.ckpt_dir + f'last{opt.version}.ckpt'
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
    nopt.notes = nopt.notes + '\n' + opt.notes
    return nopt


def load_stack_opt(opt):
    for v in range(opt.stack_num-1,-1,-1):
        path = opt.ckpt_dir + f'last{v}.ckpt'
        if os.path.isfile(path):
            checkpoint = torch.load(path)
        else:
            bst_path = opt.ckpt_dir + f'best{v}.ckpt'
            if os.path.isfile(bst_path):
                return opt
            else:
                continue
        nopt = checkpoint['opt']
        nopt.start_epoch = checkpoint['epoch'] + 1
        nopt.ckpt_dir = opt.ckpt_dir
        nopt.save_path = opt.save_path
        nopt.device = opt.device
        nopt.run_mode = opt.run_mode
        nopt.notes = nopt.notes + '\n' + opt.notes
        return nopt

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(v):
    if isinstance(v, list):
        return v
    if v == 'esc_intra':
        return [2., 6., 0., 0., 0.]
    if v == 'esc_inter':
        return [0., 0., 0.1, 0.3, 1.]
    if v == 'ctb_intra':
        return [0, 1.]
    if v == 'mav_intra':
        return [1., 3., 0., 0., 0.]
    if v == 'mav_inter':
        return [0., 0., 0.1, 0.3, 1.]
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def option():
    parser = argparse.ArgumentParser()

    def _model_config():
        parser.add_argument('--gat_hidden_size', default=768*6, type=int, help='')
        parser.add_argument('--attn_drop', default=0.1, type=float, help='')
        parser.add_argument('--dropout', default=0.1, type=float, help='')
        parser.add_argument('--gat_num_heads', default=48, type=int, help='')
        parser.add_argument('--inter_label_weight', default=8.0, type=float, help='loss wright for label==1')
        parser.add_argument('--intra_label_weight', default=5.0, type=float, help='loss wright for label==1')
        parser.add_argument('--focusing', default=2, type=int, help='')
        parser.add_argument('--loss_mode', default='focal', help='focal/normal')
        parser.add_argument('--event_emb_mul', default=True, type=str2bool)
        parser.add_argument('--need_cls', default=True, type=str2bool)
        parser.add_argument('--addition_mode', default=['intra','pred'],
                            nargs='+', help='intra/pred')
        parser.add_argument('--gat_mode', default='conjugate_cat',
                            help='normal/conjugate_cat/conjugate_add')
        parser.add_argument('--layer_intra_loss_weight', default=[2, 6., 0., 0., 0.], type=str2list,
                            help='a list length = max graph + 1')
        parser.add_argument('--layer_inter_loss_weight', default=[0.,0., 0.1, 0.3, 1.], type=str2list,
                            help='a list length = max graph + 1')

    def _dataset_config():
        parser.add_argument('--maxlen', default=512, type=int)
        parser.add_argument('--tokenlen', default=380, type=int)
        parser.add_argument('--shiftlen', default=190, type=int)
        parser.add_argument('--max_graph', default=3, type=int)
        parser.add_argument('--graph_mode', default='pipeline',
                            help='full/pipeline(5)'
                            )
        parser.add_argument('--data_ckpt', default='')

    def _base_config():
        parser.add_argument('--random_seed', default=233, type=int)
        parser.add_argument('--eval_first', default=True, type=bool)
        parser.add_argument('--ckpt_dir', default='')
        parser.add_argument('--device', default=1, type=int)
        parser.add_argument('--run_mode', default='stack5',
                            help='train#/test#/resume#/stack#/stack#_test/stack#_resume')
        parser.add_argument('--notes', default='esl-BERT',
                            help='experiment notes')
        parser.add_argument('--eval_mode', default='both',
                            help='both/inter/intra')
        parser.add_argument('--early_stop', default=0,type=int,
                            help='epoch for early stop, 0 for not use')

    _model_config()
    _dataset_config()
    _base_config()

    # data source path
    # parser.add_argument('--train_datapath', default='../data/MAVEN/MAVEN_train.json')
    # parser.add_argument('--test_datapath', default='../data/MAVEN/MAVEN_test.json')
    # parser.add_argument('--dev_datapath', default='../data/MAVEN/MAVEN_dev.json')
    # parser.add_argument('--dev_datapath', default='../data/ESL_dev.json')
    # parser.add_argument('--stack_datapath', default='../data/ESL_FOLD5_233/')
    parser.add_argument('--stack_datapath', default='../data/CTB_FOLD10_123/')
    parser.add_argument('--encoder_path', default='../encoder/BERT-base/')
    parser.add_argument('--raw_save_path', default='./runs/')

    # experiment config
    parser.add_argument('--experiment_path', default='./',
                        help='experiment history path')
    parser.add_argument('--experiment_name', default='PPATbert',
                        help='experiment file name')
    parser.add_argument('--store_experiment', default=True, type=bool,
                        help='whether store experiment result')
    parser.add_argument('--error_analysis', default=False, type=bool,
                        help='error analysis')
    parser.add_argument('--save_last_model', default=False,
                        help='save last model')

    # training param
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='learning rate')
    parser.add_argument('--warmup_step', default=0.1, type=float,
                        help='warmup ratio if <= 1 absolute step if > 1')
    parser.add_argument('--eval_step', default=-3, type=int,
                        help='eval times if < 0 absolute eval step if > 0')
    parser.add_argument('--save_step', default=-3, type=int,
                        help='save times if < 0 absolute save step if > 0')
    parser.add_argument('--train_epoch', default=128, type=int,
                        help='train epoch')

    # code backup
    parser.add_argument('--save_codes', default=["PPATbert_dataset.py",
                                                 "PPATbert_framework.py",
                                                 "PPATbert_model.py",
                                                 "PPATbert_option.py",
                                                 "PPATbert_graph_build.py"],
                        help='save code dir list')

    opt = parser.parse_args()

    # debug(opt)

    opt.notes += f' + eval mode {opt.eval_mode}' \
                 f' + running mode {opt.run_mode}' \
                 f' + stack path {opt.stack_datapath}'

    print(f'device : {opt.device}')

    if opt.random_seed is None:
        opt.random_seed = round((time.time() * 1e4) % 1e4)

    opt_dict = copy.copy(opt.__dict__)

    opt.start_epoch = 1

    if 'stack' in opt.run_mode:
        opt.stack_num = eval(opt.run_mode.split('_')[0][5:])
        opt.version = 0
        if len(opt.run_mode.split('_')) != 1:
            opt.run_mode = opt.run_mode.split('_')[0][:5] + '_' + opt.run_mode.split('_')[1]
        else:
            opt.run_mode = opt.run_mode.split('_')[0][:5]
    elif opt.run_mode in ['train','test']:
        opt.stack_num = -1
        opt.version = ''
    else:
        opt.stack_num = -1
        opt.version = eval(opt.run_mode[-1])
        opt.run_mode = opt.run_mode[:-1]

    if opt.ckpt_dir != '':
        opt.notes += f' + {opt.run_mode} {opt.ckpt_dir} '
        opt.ckpt_dir = opt.raw_save_path + opt.ckpt_dir + '/'
        if opt.run_mode == 'resume': opt = load_opt(opt)
        elif opt.run_mode == 'stack_resume': opt = load_stack_opt(opt)

    start_time = time.strftime("%Y_%m-%d-%H-%M-%S", time.localtime()).split('_')[-1]
    opt.model_name = opt.experiment_name + start_time
    opt.save_path = opt.raw_save_path + opt.model_name + '/'

    if not os.path.exists('./runs/'):
        os.mkdir('./runs/')

    os.mkdir(opt.save_path)
    with open(opt.save_path + 'config.json', 'w+') as f:
        json.dump(opt_dict, f, indent=2, ensure_ascii=False)
    pprint(opt_dict)

    opt.device = torch.device("cpu" if opt.device == -1 else opt.device)

    build_model_config(opt)
    build_dataset_config(opt)

    for file in opt.save_codes:
        os.system(f"cp {file} {opt.save_path}")

    return opt


def build_config(opt: argparse.Namespace, attr_list: list):
    opt_dict = opt.__dict__
    config = AttrDict()
    for attr in attr_list:
        config[attr] = opt_dict[attr]
    return config


def build_model_config(opt: argparse.Namespace):
    model_config = build_config(
        opt,
        [
            'gat_hidden_size',
            'attn_drop',
            'dropout',
            'gat_num_heads',
            'inter_label_weight',
            'intra_label_weight',
            'focusing',
            'loss_mode',
            'event_emb_mul',
            'need_cls',
            'addition_mode',
            'gat_mode',
            'encoder_path',
            'layer_intra_loss_weight',
            'layer_inter_loss_weight',
        ]
    )

    with open(opt.encoder_path + 'config.json') as f:
        encoder_config = json.load(f)
        model_config.encoder_dim = encoder_config['hidden_size']

    opt.model_config = model_config
    opt.notes += f'+ {opt.gat_num_heads}Heads-{opt.gat_hidden_size}Hidden {opt.gat_mode} RGAT ' \
                 f'+ {opt.loss_mode} loss ' \
                 f'+ label weight inter-{opt.inter_label_weight} intra-{opt.intra_label_weight} ' \
                 f'+ {opt.focusing} focusing ' \
                 f'+ layer loss weight intra-{str(opt.layer_intra_loss_weight)} inter-{str(opt.layer_inter_loss_weight)} ' \
                 f'+ addition mode {opt.addition_mode} ' \
                 f'+ need cls {opt.need_cls} ' \
                 f'+ event emb mul {opt.event_emb_mul} '


def build_dataset_config(opt: argparse.Namespace):
    dataset_config = build_config(
        opt,
        [
            'maxlen',
            'tokenlen',
            'shiftlen',
            'max_graph',
            'graph_mode',
            'data_ckpt',
            'device',
        ]
    )
    dataset_config.using_result = False
    dataset_config.conjugate = opt.gat_mode not in ['normal']
    if 'stack' in opt.run_mode:
        dataset_config.stack_num = opt.stack_num
    if dataset_config.data_ckpt != '':
        dataset_config.data_ckpt = opt.raw_save_path + dataset_config.data_ckpt + '/'
    dataset_config.save_path = opt.save_path

    opt.dataset_config = dataset_config
    opt.notes += f'+ len {opt.maxlen}max {opt.shiftlen}shift ' \
                 f'+ max graph num {opt.max_graph} ' \
                 f'+ graph mode is {opt.graph_mode} ' \
                 f'+ data ckpt is {opt.data_ckpt if len(opt.data_ckpt) else "none"} '
