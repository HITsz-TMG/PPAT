import csv
import os
import sys

"""
根据CSV文件来清理文件
del_command: 删除命令
"""

del_command = '/mnt/inspurfs/user-fs/lzy/anaconda3/bin/trash-put'


def clear(csv_path, runs_path, name_col=1, save=None, ddl=None, root='/data/lzy/remote/causal/', do=False):
    """
    :param csv_path: csv文件路径
    :param runs_path: 要清理的文件(夹)的根路径
    :param name_col: 文件夹名所在列
    :param save: 在CSV中的文件夹要保留的文件名或后缀
    :param ddl: 停止删除的文件夹名的前缀
    :param root: csv_path和runs_path的根路径
    :param do: 是否执行删除操作
    :return:
    """

    global del_command

    if save is None: save = ['.py']
    csv_path = root + csv_path
    runs_path = root + runs_path
    if runs_path[-1] != '/': runs_path += '/'

    reader = csv.reader(open(csv_path, 'r'))
    scv_files = [row[name_col] for row in reader]
    exist_file = os.listdir(runs_path)
    exist_file = list(sorted(exist_file))

    if ddl is not None:
        flag = False
        for x, ef in enumerate(exist_file):
            if ef[:len(ddl)] == ddl:
                exist_file = exist_file[:x]
                print(f'stop before {ef}')
                flag = True
                break
        assert flag

    del_file_set = set()
    for ef in exist_file:
        dir_path = runs_path + ef + '/'
        if ef in scv_files:
            files = os.listdir(dir_path)
            for f in files:
                if '.' + f.split('.')[-1] in save: continue
                if f in save: continue
                print(f'[FILE] del file {dir_path + f}')
                del_file_set.add(f)
                if do: os.system(f'{del_command} {dir_path + f}')
        else:
            print(f'[DIR] del dir {dir_path}')
            if do: os.system(f'{del_command} {dir_path}')

    del_file_set = sorted(del_file_set,key=lambda x:x.split('.')[-1])
    del_file_set = '\n'.join(del_file_set)
    print(f"clear file set:\n{del_file_set}")


clear(csv_path='RRGAT/RRGAT.csv',
      runs_path='RRGAT/runs',
      name_col=1,
      save=['.py', 'log.txt'],
      ddl='RRGAT07-14',
      root='/mnt/inspurfs/user-fs/lzy/remote/causal/',
      do=True
      )
