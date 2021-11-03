# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 22:06:49 2021

@author: 陈博华
"""
from shutil import copyfile

import random

import os

def random_select_and_copy_files(dir_name=None, select_number=0,name = 'source',stage='train',folder=''):
    
    try:

        dir_name is None or dir_name not in os.listdir(os.getcwd())

    except:

        print("输入目录名错误")

    dir_path = os.path.join(os.getcwd(), dir_name) # 获取文件目录路径

    files_list = os.listdir(dir_path) # 生成文件名列表

    files_number = len(files_list)

    if select_number < files_number:
        generate_list = random.sample(files_list, select_number) # 随机选取文件
    else:
        generate_list = random.sample(files_list, files_number) # 随机选取文件
        
    new_dir_path = os.getcwd()+'/data/source-target'
    try:
        os.mkdir(new_dir_path)
    except:
        pass

    new_dir_path = os.getcwd()+'/data/source-target/'+stage
    try:
        os.mkdir(new_dir_path)
    except:
        pass
    new_dir_path = os.getcwd()+'/data/source-target/'+stage+'/'+name
    try:
        os.mkdir(new_dir_path)
    except:
        pass

    success_number = 0 # 记录成功数量

    success_list = [] # 记录成功文件

    # 复制文件并记录

    for file_name in generate_list:

        orl_file_path = os.path.join(dir_path, file_name)

        new_file_path = os.path.join(new_dir_path, folder+file_name)
    
        copyfile(orl_file_path, new_file_path) # 复制文件
    
        success_list.append(file_name)
    
        success_number += 1

    if success_number % 100 == 0:

        print("success", success_number)

    # 给出提示信息并给出未成功文件

    if success_number == select_number:

        print("all", select_number, "finish")

    else:

        print("unfinished")

        error_list = []
    
        for file_name in files_list:
    
            if file_name not in success_list:
    
                error_list.append(file_name)
    
        print(error_list, 'error', sep='\n')
        
def makedata(folder,name,stage,number):
    dir_path = os.path.join(os.getcwd(),'data\OfficeHomeDataset_10072016\\'+folder) # 获取文件目录路径

    files_list = os.listdir(dir_path) # 生成文件名列表
    
    for i in files_list:
        random_select_and_copy_files("data\OfficeHomeDataset_10072016\\"+folder+"\\"+i, number,name,stage,i)

if __name__ == "__main__":
    makedata('Clipart','source','train',15)
    makedata('RealWorld','target','train',15)
    makedata('Clipart','source','val',2)
    makedata('RealWorld','target','val',2)
    