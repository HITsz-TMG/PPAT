import csv
import math
import sys

sys.path.append("..")

import numpy as np
import torch
import time
import os
import json
from attrdict import AttrDict
from utils.frameworkbase import BaseFramework
from transformers import AdamW, LongformerTokenizer, get_linear_schedule_with_warmup
from PPATlong_dataset import get_dataloader
from PPATlong_model import PPATlong_model
from PPATlong_option import option
from utils.DocxWriter import docx_writer
from utils.TrainingTimer import TrainingTimer
from utils.CudaApplication import CudaApplication
from utils.LogFile import logfile


class PPATlong_framework(BaseFramework):
    def __init__(self, opt):
        """
        定义全局工具
        """
        super().__init__(opt)
        self.logger = logfile(opt.save_path)
        self.logger.config(0, 1, True)
        self.timer = TrainingTimer()
        self.test_scores = []

    def __model_initial(self):
        self.logger.info("----------start model initial----------")
        opt = self.opt
        model = PPATlong_model(opt.model_config, self.logger)
        self.logger.info("---------- model initial done ----------")
        return model

    def __optimizer_initial(self, model, dataloaders):
        self.logger.info("----------start training initial----------")
        opt = self.opt

        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(parameters_to_optimize, lr=opt.lr, correct_bias=False)

        num_schedule_step = opt.train_epoch * int(math.ceil(len(dataloaders['train'])))
        if opt.warmup_step <= 1:
            opt.warmup_step = round(opt.warmup_step * num_schedule_step)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=opt.warmup_step,
                                                    num_training_steps=num_schedule_step)

        if opt.eval_step < 0:
            opt.eval_step = int(len(dataloaders['train']) // -opt.eval_step)
        if opt.save_step < 0:
            opt.save_step = int(len(dataloaders['train']) // -opt.save_step)

        self.logger.info("---------- training initial done ----------")
        return optimizer, scheduler

    def __dataloader_initial(self):
        self.logger.info("----------start dataloader initial----------")
        opt = self.opt

        if opt.version != '':
            opt.train_datapath = f"{opt.stack_datapath}{opt.version}/train.json"
            opt.test_datapath = f"{opt.stack_datapath}{opt.version}/test.json"
            opt.dev_datapath = opt.test_datapath if 'CTB' in opt.train_datapath else opt.dev_datapath

        # cuda_renter = CudaApplication(10000, opt.device)

        tokenizer = LongformerTokenizer.from_pretrained(opt.encoder_path)

        dataloaders = {'test': get_dataloader(opt.dataset_config, opt.test_datapath, tokenizer, self.logger, f'test{opt.version}')}

        if opt.run_mode in ['stack', 'stack_resume','train', 'train_resume']:
            dataloaders['train'] = get_dataloader(opt.dataset_config, opt.train_datapath, tokenizer, self.logger, mode=f'train{opt.version}')
            dataloaders['dev'] = get_dataloader(opt.dataset_config, opt.dev_datapath, tokenizer, self.logger, mode=f'dev')
        opt.dataset_config.using_result = False

        # cuda_renter.free()

        self.logger.info("---------- dataloader initial done ----------")
        return dataloaders

    def test_initial(self):
        model = self.__model_initial()
        dataloaders = self.__dataloader_initial()
        self.set_random(self.opt.random_seed)
        return model, dataloaders

    def train_initial(self):
        model = self.__model_initial()
        dataloaders = self.__dataloader_initial()
        optimizer, scheduler = self.__optimizer_initial(model, dataloaders)
        self.set_random(self.opt.random_seed)
        return model, optimizer, scheduler, dataloaders

    def train(self, model, optimizer, scheduler, dataloaders):
        best_epoch = (-1, -1)
        opt = self.opt
        trainloader = dataloaders['train']
        devlaoder = dataloaders['dev']
        model.to(opt.device)
        early_stop = False
        self.timer.set(train_epoch=opt.train_epoch, epoch_step=len(dataloaders['train']),
                       eval_epoch=len(dataloaders['dev']))

        if opt.eval_first:
            best_p, best_r, best_f1 = self.evaluate(model, devlaoder)
            eval_log = f"first eval p:{best_p:.6f} r:{best_r:.6f} f1:{best_f1:.6f}"
            self.logger.info("[EVAL] " + eval_log)
            self.timer.ckpt('eval end', eval_log)
        else:
            best_p = best_r = best_f1 = 0

        model.train()
        for epoch in range(opt.start_epoch, opt.train_epoch + 1):
            normal_log1 = f'epoch: {epoch}/{opt.train_epoch}'
            self.timer.ckpt('epoch start', normal_log1)
            step = 0
            for input_ids, atten_mask, pos_ids, event_mask, graph, distan, causal in trainloader:
                step += 1
                loss = model(input_ids, atten_mask, pos_ids, event_mask, graph, distan, causal)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                normal_log2 = f'step: {step}/{len(trainloader)}, loss: {loss.item():.4f}'
                self.timer.ckpt('train step', normal_log2)
                normal_log = '[TRAIN]' + normal_log1 + ', ' + normal_log2
                if step % opt.save_step == 0 or step == len(trainloader):
                    self.logger.info(normal_log)
                    self.save_model(opt, model, opt.save_path, scheduler, optimizer, epoch,tag=f'last{opt.version}')
                elif step % 5 == 0:
                    print(normal_log)

                if step % opt.eval_step == 0 or step == len(trainloader):
                    p, r, f1 = self.evaluate(model, devlaoder)
                    eval_log1 = f"now: ({epoch}, {step}) p:{p:.4f} r:{r:.4f} f1:{f1:.6f}"
                    self.logger.info("[EVAL] " + eval_log1)
                    if f1 >= best_f1:
                        self.save_model(opt, model, opt.save_path, tag=f'best{opt.version}')
                        best_p = p
                        best_r = r
                        best_f1 = f1
                        best_epoch = (epoch, step)
                    else:
                        if opt.early_stop != 0 and epoch - best_epoch[0] > opt.early_stop and epoch > int((opt.train_epoch + 1) / 2):
                            self.logger.info("[EARLY STOP] !!!")
                            early_stop = True
                    eval_log2 = f"best: {best_epoch} p:{best_p:.4f} r:{best_r:.4f} f1:{best_f1:.6f}"
                    self.logger.info("[EVAL] " + eval_log2)
                    self.timer.ckpt('eval end', [eval_log1, eval_log2])

                if early_stop:
                    break

            if early_stop:
                break

            self.timer.ckpt('epoch end')

        if not opt.save_last_model:
            os.system(f"rm {opt.save_path}last{opt.version}.ckpt")

    def __eval(self, model, dataloader, error_analysis, mode='eval'):
        opt = self.opt
        assert mode in ['eval', 'test']

        model = model.to(opt.device)
        model.eval()

        count = AttrDict()
        count.pred_num = count.right_num = count.golden_num = 0
        count.inter_pred = count.inter_right = count.inter_golden = 0
        count.intra_pred = count.intra_right = count.intra_golden = 0

        if mode in ['test'] and error_analysis:
            writer = docx_writer(opt.save_path + f'error_analysis_{opt.version}.docx')
            docs = dataloader.dataset.docs

        with torch.no_grad():
            for data_batch in dataloader:
                if mode == 'eval':
                    input_ids, atten_mask, pos_ids, event_mask, graph, distan, causal = data_batch
                else:
                    input_ids, atten_mask, pos_ids, event_mask, graph, distan, causal, doc_id = data_batch

                metric_score = model.metric(input_ids, atten_mask, pos_ids, event_mask, graph, distan,causal, analysis=True)

                count.right_num += metric_score['right num']
                count.pred_num += metric_score['pred num']
                count.golden_num += metric_score['golden num']

                count.inter_right += metric_score['inter right num']
                count.inter_pred += metric_score['inter pred num']
                count.inter_golden += metric_score['inter golden num']

                count.intra_right += metric_score['intra right num']
                count.intra_pred += metric_score['intra pred num']
                count.intra_golden += metric_score['intra golden num']

                if mode in ['test'] and error_analysis:
                    right = metric_score['right']
                    pred = metric_score['pred']
                    golden = metric_score['golden']
                    doc = docs[doc_id]
                    text = doc['tokens']
                    events = doc['events']
                    sentence = doc['sentence']

                    # 浅灰表示所有事件, 其中靛青为锚事件, 红为预测错误, 绿为预测正确, 蓝为多预测
                    writer.add_heading(doc['id'], level=1)
                    for i in range(pred.shape[0]):
                        main_token = ' '.join(map(lambda x: text[x], events[i]))
                        writer.add_heading(doc['id'].split('/')[-1] + '---' + main_token, level=2)
                        writer.add_content([ch + ' ' for ch in text])
                        # 根据事件设置字体颜色
                        for j, tid in enumerate(events):
                            if i == j:
                                writer.set(pos=tid, color='靛青')
                            elif right[i][j] == 1:
                                writer.set(pos=tid, color='绿')
                            elif pred[i][j] == 1:
                                writer.set(pos=tid, color='蓝')
                            elif golden[i][j] == 1:
                                writer.set(pos=tid, color='红')
                            else:
                                writer.set(pos=tid, color='深灰')
                        # 根据句内单词设置下划线
                        for j in range(len(text)):
                            if sentence[j] == sentence[events[i][0]]:
                                writer.set(pos=j, underline=True)

                    writer.add_page_break()

        if mode in ['test'] and error_analysis:
            writer.close()

        metric_count = self.__eval_count(count, tag=mode)

        if mode == 'eval':
            model.train()
            if opt.eval_mode == 'both':
                return metric_count.P, metric_count.R, metric_count.F1
            elif opt.eval_mode == 'inter':
                return metric_count.interP, metric_count.interR, metric_count.interF1
            elif opt.eval_mode == 'intra':
                return metric_count.intraP, metric_count.intraR, metric_count.intraF1
            else:
                raise KeyError
        else:
            return count, metric_count.P, metric_count.R, metric_count.F1

    def __micro_f1(self, rn, pn, gn):
        P = rn / pn if pn else 0
        R = rn / gn if gn else 0
        F1 = 2 * P * R / (P + R) if P + R else 0
        return P, R, F1

    def __eval_count(self, count, tag):
        metric_count = AttrDict()
        metric_count.P, metric_count.R, metric_count.F1 = self.__micro_f1(count.right_num, count.pred_num, count.golden_num)
        metric_count.interP, metric_count.interR, metric_count.interF1 = self.__micro_f1(count.inter_right, count.inter_pred, count.inter_golden)
        metric_count.intraP, metric_count.intraR, metric_count.intraF1 = self.__micro_f1(count.intra_right, count.intra_pred, count.intra_golden)
        self.logger.info(f"[{tag} INTER] inter p:{metric_count.interP:.4f} r:{metric_count.interR:.4f} f1:{metric_count.interF1:.6f}")
        self.logger.info(f"[{tag} INTRA] intra p:{metric_count.intraP:.4f} r:{metric_count.intraR:.4f} f1:{metric_count.intraF1:.6f}")
        return metric_count

    def evaluate(self, model, dataloader):
        return self.__eval(model, dataloader, False, 'eval')

    def test(self, model, dataloader, error_analysis=False, save=False):
        if save: self.logger.info("----------start test----------")

        count, P, R, F1 = self.__eval(model, dataloader, error_analysis, 'test')

        self.logger.info(f"[TEST] right:{count.right_num} pred:{count.pred_num} gold:{count.golden_num}")
        self.logger.info(f"[TEST] p:{P:.4f} r:{R:.4f} f1:{F1:.6f}")

        if save:
            self.timer.ckpt('train end', f"[TEST] p:{P:.4f} r:{R:.4f} f1:{F1:.6f}")
            self.test_scores.append(count)


    def test_all(self, model, dataloaders, error_analysis=False):
        self.test(model, dataloaders['test'], error_analysis, True)
        del model, dataloaders
        with torch.cuda.device(self.opt.device):
            torch.cuda.empty_cache()

    def score(self):
        self.logger.info('--------- final test score ---------')
        total_count = AttrDict()
        step_count = self.test_scores
        for k in step_count[0].keys():
            total_count[k] = 0
        for sc in step_count:
            for k in sc.keys():
                total_count[k] += sc[k]
        metric_count = self.__eval_count(total_count, tag='final_test')
        self.logger.info(f"[FINAL TEST] right:{total_count.right_num} pred:{total_count.pred_num} gold:{total_count.golden_num}")
        self.logger.info(f"[FINAL TEST] p:{metric_count.P:.4f} r:{metric_count.R:.4f} f1:{metric_count.F1:.6f}")

        result = [f'intra ({metric_count.intraP:.4f},{metric_count.intraR:.4f},{metric_count.intraF1:.4f})',
                  f'inter ({metric_count.interP:.4f},{metric_count.interR:.4f},{metric_count.interF1:.4f})',
                  f'p:{metric_count.P:.4f} r:{metric_count.R:.4f} inter:{metric_count.interF1:.4f} intra:{metric_count.intraF1:.4f}',
                  f'f1:{metric_count.F1:.6f}']
        return result


def main(opt):
    pro_start_time = time.time()

    framework = PPATlong_framework(opt)

    logger = framework.logger
    logger.info(opt.notes.replace('+', '\n+'))

    """
    use framework by `opt.run_mode`
    """
    if opt.run_mode == 'stack':
        # running fresh cross validation
        for v in range(opt.stack_num):
            opt.version = opt.dataset_config.version = opt.model_config.version = v
            framework.set_random(opt.random_seed)
            logger.info(f"**************** stack version {v} ****************")
            model, optimizer, schedule, dataloaders = framework.train_initial()
            framework.train(model, optimizer, schedule, dataloaders)

            framework.set_random(opt.random_seed)
            framework.load_model(logger, model, opt.save_path + f'best{v}.ckpt')
            framework.test_all(model, dataloaders, opt.error_analysis)
        result = framework.score()
    elif opt.run_mode == 'stack_test':
        # testing cross validation
        for v in range(opt.stack_num):
            opt.version = opt.dataset_config.version = opt.model_config.version = v
            framework.set_random(opt.random_seed)
            logger.info(f"**************** stack version {v} ****************")
            model, dataloaders = framework.test_initial()
            framework.load_model(logger, model, opt.ckpt_dir + f'best{v}.ckpt')
            framework.test_all(model, dataloaders, opt.error_analysis)
        result = framework.score()
    elif opt.run_mode == 'stack_resume':
        # resume running cross validation
        for v in range(opt.stack_num):
            opt.version = opt.dataset_config.version = opt.model_config.version = v
            logger.info(f"**************** stack version {v} ****************")
            model, optimizer, schedule, dataloaders = framework.train_initial()
            bst_path = opt.ckpt_dir + f'best{v}.ckpt'
            lst_path = opt.ckpt_dir + f'last{v}.ckpt'
            if os.path.exists(bst_path) and not os.path.exists(lst_path):
                framework.load_model(logger, model, opt.ckpt_dir + f'best{v}.ckpt')
                framework.save_model(opt, model, opt.save_path, tag=f'best{v}')
                framework.test_all(model, dataloaders, opt.error_analysis)
            else:
                if os.path.exists(lst_path):
                    framework.load_model(logger, model, opt.ckpt_dir + f'last{opt.version}.ckpt', optimizer, schedule)
                framework.train(model, optimizer, schedule, dataloaders)
                framework.set_random(opt.random_seed)
                framework.load_model(logger, model, opt.save_path + f'best{v}.ckpt')
                framework.test_all(model, dataloaders)
        result = framework.score()
    elif opt.run_mode == 'train':
        opt.dataset_config.version = opt.model_config.version = opt.version
        framework.set_random(opt.random_seed)
        model, optimizer, schedule, dataloaders = framework.train_initial()
        framework.train(model, optimizer, schedule, dataloaders)

        framework.set_random(opt.random_seed)
        framework.load_model(logger, model, opt.save_path + f'best{opt.version}.ckpt')
        framework.test_all(model, dataloaders, opt.error_analysis)
        result = framework.score()
    elif opt.run_mode == 'test':
        opt.dataset_config.version = opt.model_config.version = opt.version
        framework.set_random(opt.random_seed)
        model, dataloaders = framework.test_initial()
        framework.load_model(logger, model, opt.ckpt_dir + f'best{opt.version}.ckpt')
        framework.test_all(model, dataloaders, opt.error_analysis)
        result = framework.score()
    elif opt.run_mode == 'resume':
        opt.dataset_config.version = opt.model_config.version = opt.version
        model, optimizer, schedule, dataloaders = framework.train_initial()
        framework.load_model(logger, model, opt.ckpt_dir + f'last{opt.version}.ckpt', optimizer, schedule)
        framework.train(model, optimizer, schedule, dataloaders)
        framework.set_random(opt.random_seed)
        framework.load_model(logger, model, opt.save_path + f'best{opt.version}.ckpt')
        framework.test_all(model, dataloaders, opt.error_analysis)
        result = framework.score()
    else:
        raise KeyError

    """
    write the experiment result to CSV 
    """
    if opt.store_experiment:
        fn = opt.experiment_path + opt.experiment_name + '.csv'
        rows = [[opt.random_seed, opt.model_name, opt.notes] + result]
        with open(fn, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    logger.info(f"[END] total time cost : {(time.time() - pro_start_time) / 3600:.2f}h")


if __name__ == '__main__':
    main(option())

"""
"""
