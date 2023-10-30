import math
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import time

"""
用于训练框架下的进度条
基于rich.progress
framework使用样例见main函数
"""


class TrainingTimer:
    def __init__(self, log_end: str = '\n'):
        self.progress = Progress(BarColumn(),
                                 TimeRemainingColumn(),
                                 TimeElapsedColumn(),
                                 TextColumn("[progress.description]{task.description}"),
                                 refresh_per_second=1,
                                 )
        self.progress.start()
        self.start = False
        self.log_end = log_end

    def set(self, train_epoch, epoch_step, eval_epoch):
        self.start = True
        self.epoch_log = self.epoch_title = '[EPOCH]'
        self.step_log = self.step_title = '[STEP]'
        self.eval_log = self.eval_title = '[EVAL]'
        self.epoch_task = self.progress.add_task(description=self.epoch_log, total=train_epoch * epoch_step)
        self.step_task = self.progress.add_task(description=self.step_log, total=epoch_step)
        self.eval_task = self.progress.add_task(description=self.eval_log, total=eval_epoch)

    def ckpt(self, cmd, logs=None):
        if not self.start: return
        if logs is None:
            if cmd in ['epoch start','epoch end']:
                logs = self.epoch_log
            elif cmd in ['train step']:
                logs = self.step_log
            elif cmd in ['eval step', 'eval end']:
                logs = self.eval_log
            else:
                raise KeyError
        else:
            if isinstance(logs, list):
                logs = self.log_end.join(logs)

            if cmd in ['epoch start','epoch end']:
                logs = self.epoch_title + logs
            elif cmd in ['train step']:
                logs = self.step_title + logs
            elif cmd in ['eval step', 'eval end']:
                logs = self.eval_title + logs
            elif cmd in ['train end']:
                logs = self.eval_log + self.log_end + logs
            else:
                raise KeyError

        if cmd == 'epoch start':
            self.progress.update(self.epoch_task, description=logs)
            self.epoch_log = logs

        elif cmd == 'epoch end':
            self.progress.reset(self.step_task)
            self.progress.update(self.epoch_task, advance=1, description=logs)
            self.epoch_log = logs

        elif cmd == 'train step':
            self.progress.update(self.step_task, advance=1, description=logs)
            self.progress.update(self.epoch_task, advance=1)
            self.step_log = logs

        elif cmd == 'eval step':
            self.progress.update(self.eval_task, advance=1, description=logs)
            self.eval_log = logs

        elif cmd == 'eval end':
            self.progress.update(self.eval_task, description=logs)
            self.progress.reset(self.eval_task)
            self.eval_log = logs

        elif cmd == 'train end':
            self.progress.update(self.eval_task, description=logs, refresh=True)
            self.eval_log = logs

        else:
            raise KeyError


if __name__ == '__main__':
    EPOCH = 128
    DATA = 20
    STEP = 3
    EVAL = 5
    tt = TrainingTimer(train_epoch=EPOCH, epoch_step=DATA, eval_epoch=EVAL)
    for epoch in range(1, EPOCH):
        log = f'epoch:{epoch}/{EPOCH}'
        tt.ckpt('epoch start', log)
        for step in range(1, DATA + 1):
            # training
            time.sleep(10)

            # show training
            log = f'step:{step}/{DATA} loss:0.01'
            tt.ckpt('train step', log)
            if step % STEP == 0 or step == DATA:
                # save log
                pass

            # eval
            if step == DATA:
                for e in range(EVAL):
                    tt.ckpt('eval step')
                    time.sleep(10)
                log1 = f"now (4, 2) - p:0.5 r:0.5: F1:0.5"
                log2 = f"best (3, 6) - p:0.5 r:0.5: F1:0.5"
                tt.ckpt('eval end', [log1, log2])

        tt.ckpt('epoch end')
