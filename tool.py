#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import re
import argparse

def init_dir(output):
    """
    初始化输出路径目录
    :param output:
    :return:
    """
    dir_path = os.path.dirname(output)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def handle_md(filepath, output_dir):
    init_dir(output_dir)
    path, filename = os.path.split(filepath)
    dir_name = filename[:-3]

    text_list = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line[:2] == '{%' and line[-3:-1] == '%}':
                tmp_list = re.split(r'\s+', line.strip())
                new_pic = '![](' + '/' + dir_name + '/' + tmp_list[2] + ')'
                text_list.append(new_pic + '\n')
            else:
                text_list.append(line)
    output_file = output_dir + '/' + filename
    with open(output_file, 'w+', encoding='utf-8') as f:
        for line in text_list:
            f.write(line)
    print('done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="文章图片引用链接转换")
    parser.add_argument('--input', type=str, required=False, help='原文件路径')
    parser.add_argument('--output', type=str, required=False, help='转换后文件保存路径')
    
    args = parser.parse_args()
    if not os.path.isfile(args.input):
        print('输入文件路径无效')
    else:
    	handle_md(args.input, args.output)