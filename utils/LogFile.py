import os
import time


class logfile:
    def __init__(self, path='./logFile/', repalce=False):
        self.path = path
        self.log_path = path + 'log.txt'
        self.info_path = path + 'infoLog.txt'
        self.warn_path = path + 'warningLog.txt'
        self.error_path = path + 'errorLog.txt'
        self.debug_path = path + 'debugLog.txt'
        self.impt_path = path + 'imptLog.txt'
        self.log_cnt = 0
        self.info_cnt = 0
        self.warn_cnt = 0
        self.error_cnt = 0
        self.debug_cnt = 0
        self.impt_cnt = 0
        self.showlv = 0
        self.loglv = 0
        self.showtime = True
        if not os.path.exists(path):
            os.mkdir(path)

        for p in [self.log_path,self.info_path,self.warn_path,self.error_path,self.debug_path,self.impt_path]:
            if os.path.exists(p):
                if repalce:
                    os.system(f'rm -rf {p}')
                    print(f"*{p} exist* : deleted previous")
                else:
                    print(f"*{p} exist* : continue")

    def config(self, showlv: int, loglv: int, showtime: bool):
        self.showlv = showlv
        self.loglv = loglv
        self.showtime = showtime

    def log(self, text: str, tm: str):
        log = f"Log({self.log_cnt}) : {text} \n"
        if tm is not None:
            log = f"|{tm}|" + log
        self.log_cnt += 1
        with open(self.log_path, 'a+') as f:
            f.write(log)

    def info(self, text: str, end='\n'):
        log = f"Info({self.info_cnt}) : {text}"
        tm = None
        if self.showtime:
            tm = time.strftime("%Y_%m_%d-%H-%M-%S", time.localtime()).split('_')[-1]
            log = f'|{tm}|' + log
        self.info_cnt += 1
        if self.showlv <= 0:
            print(log, end=end)
        if self.loglv <= 0:
            with open(self.info_path, 'a+') as f:
                f.write(log + '\n')
        self.log(text, tm)

    def warning(self, text: str, end='\n'):
        log = f"Warning({self.warn_cnt}) : {text}"
        tm = None
        if self.showtime:
            tm = time.strftime("%Y_%m_%d-%H-%M-%S", time.localtime()).split('_')[-1]
            log = f'|{tm}|' + log
        self.warn_cnt += 1
        if self.showlv <= 1:
            print(log, end=end)
        if self.loglv <= 1:
            with open(self.warn_path, 'a+') as f:
                f.write(log + '\n')
        self.log(text, tm)

    def error(self, text: str, end='\n'):
        log = f"Error({self.error_cnt}) : {text}"
        tm = None
        if self.showtime:
            tm = time.strftime("%Y_%m_%d-%H-%M-%S", time.localtime()).split('_')[-1]
            log = f'|{tm}|' + log
        self.error_cnt += 1
        if self.showlv <= 2:
            print(log, end=end)
        if self.loglv <= 2:
            with open(self.error_path, 'a+') as f:
                f.write(log + '\n')
        self.log(text, tm)

    def debug(self, text: str, end='\n'):
        log = f"Debug({self.debug_cnt}) : {text}"
        tm = None
        if self.showtime:
            tm = time.strftime("%Y_%m_%d-%H-%M-%S", time.localtime()).split('_')[-1]
            log = f'|{tm}|' + log
        self.debug_cnt += 1
        if self.showlv <= 2:
            print(log, end=end)
        if self.loglv <= 2:
            with open(self.debug_path, 'a+') as f:
                f.write(log + '\n')
        self.log(text, tm)

    def clas(self, text: str, end='\n', tag: str = 'default'):
        log = f"Info({self.info_cnt}) : {text}"
        tm = None
        if self.showtime:
            tm = time.strftime("%Y_%m_%d-%H-%M-%S", time.localtime()).split('_')[-1]
            log = f'|{tm}|' + log
        with open(''.join([self.path, tag, '.txt']), 'a+') as f:
            f.write(log + '\n')
        self.impt(text, end=end, tm=tm)

    def impt(self, text: str, end='\n', tm=None):
        log = f"Impt({self.impt_cnt}) : {text} \n"
        if self.showtime and tm is None:
            tm = time.strftime("%Y_%m_%d-%H-%M-%S", time.localtime()).split('_')[-1]
            log = f'|{tm}|' + log
        self.impt_cnt += 1
        if self.showlv <= 1:
            print(log, end=end)
        if self.loglv <= 1:
            with open(self.impt_path, 'a+') as f:
                f.write(log + '\n')
        self.log(text, tm)