import json
import os
import shutil
from opencc import OpenCC
import uuid
import copy

import time
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import pandas as pd
from tqdm.notebook import tqdm
from typing import List
from fastembed.embedding import FlagEmbedding as Embedding


from qdrant_client import QdrantClient
from qdrant_client import models



'''
检查 json 文件格式
从同级路径读取json文件，把检查合格的文件移动到“output_folder“路径中，不合格的文件不移动

checked_keys是需要检查的键及其数据类型，比如：
checked_keys = {
    "author": str, 
    "title": str, 
    "paragraphs": list,
    "id": str
}


文件路径：file_path
文件夹路径：dir_path
current_path：当前（文件夹）路径
input_dir_path：输入文件夹路径
output_dir_path：输出文件夹路径

文件名：filename
文件夹名：dirname
output_filename：输出文件名
output_file_path：输出文件路径

'''
def check_json_files(checked_keys):
    # 当前脚本所在目录
    current_path = os.getcwd()
    # 输入目录：存放待转换的 JSON 文件
    input_dir_path = os.path.join(current_path, "0_original_files")
    # 指定存储无错误文件的目录路径“checked_files”
    output_dir_path = os.path.join(current_path, "1_checked_files")

    # 如果目录checked_files不存在，则创建该目录，用来存储无错误文件的
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    # 用于记录错误文件及其错误
    error_files = {}

    filename_list = [filename for filename in os.listdir(input_dir_path) if filename.endswith('.json')]
    total_num = len(filename_list)
    num = 0

    for filename in filename_list:
        file_path = os.path.join(input_dir_path, filename)
        # 初始化存放文件错误信息的列表
        file_errors = []
        error_found = False

        # print("file_path:", file_path)
        
        # 尝试打开和解析 JSON 文件
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data_list = json.load(file)
                # 遍历json文件中的每一个字典
                for data in data_list:
                    # 检查checked_keys字典中的每一个键，是否也存在于json文件的字典中
                    for key, expected_type in checked_keys.items():
                        if key not in data:
                            # 如果缺少键，记录错误信息
                            error_msg = f"id 为 data['id'] 的数据缺少 {key} 键"
                            file_errors.append(error_msg)
                            error_found = True
                        # 如果经过第一次检查，指定的键存在，再检查这个键对应的值的数据类型
                        elif not isinstance(data[key], expected_type):
                            # 如果值的数据类型不是希望的，记录错误信息
                            error_msg = f"id 为 data['id'] 的数据，键 {key} 的值的数据类型应该是{expected_type}，结果却是 {type(data[key])}"
                            file_errors.append(error_msg)
                            error_found = True
        # 如果json文件格式错误，记录错误信息
        except json.JSONDecodeError as e:
            error_msg = f"无效的json文件格式：{str(e)}"
            file_errors.append(error_msg)
            error_found = True
        # 检查完成后，如果json文件存在错误，把错误信息添加到字典error_files中，键是文件路径，值是错误信息

        print("file_path:", file_path)
        print("output_dir_path:", output_dir_path)

        if error_found:
            error_files[filename] = file_errors
            print(f"{filename} 格式错误")
        else:
            # 如果json文件不存在错误，把该移动文件到目录“output_dir_path”中
            shutil.move(file_path, output_dir_path)
            num += 1
            print(f"{filename} 格式正确，{num}/{total_num}")

    # 输出错误文件列表
    print("错误文件：", error_files)

# 把繁体字转换成简体字
# 简繁体转换的库
def t2s():
    cc = OpenCC('t2s')
    current_path = os.getcwd()

    # 输入目录：存放待转换的 JSON 文件
    input_dir_path = os.path.join(current_path, "1_checked_files")
    # 输出目录：存放转换后的 JSON 文件
    output_dir_path = os.path.join(current_path, "2_simplified_files")

    # 如果目录simplified_files不存在，则创建该目录，用来存储转换成简体字的json文件
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)


    filename_list = [filename for filename in os.listdir(input_dir_path) if filename.endswith('.json')]
    total_num = len(filename_list)
    num = 0

    for filename in filename_list:

        # 构建输入文件的完整路径
        file_path = os.path.join(input_dir_path, filename)
        # print(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            data_list = json.load(file)
            # 遍历文件中的每个数据项，对指定字段进行繁简体转换
            # 需要转换的键：值是字符串
            key_t2s = ["author", "title"]
            for data in data_list:
                for key in key_t2s:
                    if key in data:
                        data[key] = cc.convert(data[key])
                        # print("-"*50)
                        # print(data[key])
                        # print("-"*50)

            # 需要转换的键：值是列表
            key_t2s = ["paragraphs"]
            for data in data_list:
                for key in key_t2s:
                    if key in data:
                        # 把字符串列表合并成字符串
                        str_join = "-".join(data[key])
                        
                        str_t2s = cc.convert(str_join)
                        # 把字符串分割成列表
                        str_list = str_t2s.split("-")
                        # 更新字典中的列表
                        data[key] = str_list

                        # print("-"*50)
                        # print(str_list)
                        # print("-"*50)

            # 构建输出文件的完整路径，添加 'simplified' 后缀名
            base, ext = os.path.splitext(filename)
            output_filename = f"{base}_simplified{ext}"
            output_file_path = os.path.join(output_dir_path, output_filename)

            # 将转换后的数据写入到输出文件
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                json.dump(data_list, outfile, ensure_ascii=False, indent=4)
                num += 1
                print(f"{filename} 繁体字转换成简体字，{num}/{total_num}")


'''
给json文件中的字典，添加键值对
需要添加的键值对，以字典形式输入，比如：key_values = {"type":"唐诗", "dynastic":"唐朝"}
'''
def add_field(key_values):
    current_path = os.getcwd()
    # 输入目录：存放待转换的 JSON 文件
    input_dir_path = os.path.join(current_path, "2_simplified_files")
    # 输出目录：存放转换后的 JSON 文件
    output_dir_path = os.path.join(current_path, "3_add_field_files")

    # 如果目录simplified_files不存在，则创建该目录，用来存储转换成简体字的json文件
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    filename_list = [filename for filename in os.listdir(input_dir_path) if filename.endswith('.json')]
    total_num = len(filename_list)
    num = 0

    for filename in filename_list:
        # 构建输入文件的完整路径
        file_path = os.path.join(input_dir_path, filename)

        # print("file_path", file_path)

        # 判断文件名是否以“.json”结尾
        if filename.endswith(".json"):
            # 读取 JSON 文件
            with open(file_path, 'r', encoding='utf-8') as file:
                data_list = json.load(file)

                # 遍历列表中的每个字典，如果没有该键值对，则添加
                for data in data_list:
                    # 检查需要添加的键值对，是否已经存在
                    for key in key_values.keys():
                        if key not in data:
                            # print("key:", key)
                            # print("data:", data)
                            # print("-"*50)
                            data[key] = key_values[key]


            # 构建输出文件的完整路径，添加 'add' 后缀名
            base, ext = os.path.splitext(filename)
            output_filename = f"{base}_add{ext}"
            output_file_path = os.path.join(output_dir_path, output_filename)

            # print(output_dir_path)

            # 将更新后的数据写回 JSON 文件
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                json.dump(data_list, outfile, ensure_ascii=False, indent=4)
                num += 1
                print(f"{filename} 添加键值对成功，{num}/{total_num}")


'''
拆分json文件字典中，值为列表字符串的字段
比如，如果paragraphs列表中有多个字符串元素，就拆分出来，分别创建新的字典，生成新的id。其他键值对不变

思路：
1. 遍历每个json文件的每个字典
2. 
'''

def split_list_str():
    current_path = os.getcwd()
    # 输入目录：存放待转换的 JSON 文件
    input_dir_path = os.path.join(current_path, "3_add_field_files")
    # 输出目录：存放转换后的 JSON 文件
    output_dir_path = os.path.join(current_path, "4_split_files")

    # 如果目录split_files不存在，则创建该目录，用来存储转换成简体字的json文件
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    filename_list = [filename for filename in os.listdir(input_dir_path) if filename.endswith('.json')]
    total_num = len(filename_list)
    num = 0
    for filename in filename_list:
        file_path = os.path.join(input_dir_path, filename)

        with open(file_path, encoding='utf-8') as file:
            data_list = json.load(file)
            # 初始化新列表，存放处理后的字典
            new_data_list = []
            # 遍历json文件中的每个字典data
            for data in data_list:
                paragraphs_list = data["paragraphs"]
                # 如果p_list是列表，并且列表中的字符串元素数量超过1个，遍历列表
                if isinstance(paragraphs_list, list) and len(paragraphs_list) > 1:
                    # print("ok")
                    for paragraphs_str in paragraphs_list:
                        # 通过深拷贝创建新字典，替换p键值对
                        new_data = copy.deepcopy(data)
                        # 保证数据类型一致
                        new_data["paragraphs"] = [paragraphs_str]
                        new_data["id"] = str(uuid.uuid4())
                        new_data_list.append(new_data)
            # print("new_data_list:", new_data_list)
            # print("-"*50)
            # 根据原文件名生成新的文件名，添加后缀“split”
            base, ext = os.path.splitext(filename)
            output_filename = f"{base}_split{ext}"
            output_file_path = os.path.join(output_dir_path, output_filename)

            # 如果新数据new_data_list不为空，另存为
            if new_data_list:
                with open(output_file_path, 'w', encoding="utf-8") as f:
                    # print(output_file)
                    json.dump(new_data_list, f, ensure_ascii=False, indent=4)
                    num += 1
                    print(f"{filename} 拆分成功，{num}/{total_num}")
            else:
                print(f"{filename} 为空")

'''
功能：合并json文件

'''

def merge_json_files():
    # 合并的目标 JSON 文件
    merged_json = []

    current_path = os.getcwd()
    # 输入目录：存放待转换的 JSON 文件
    input_dir_path = os.path.join(current_path, "4_split_files")
    # 输出目录：存放转换后的 JSON 文件
    output_dir_path = os.path.join(current_path, "6_merge_files")

    # 如果目录split_files不存在，则创建该目录，用来存储转换成简体字的json文件
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    for filename in os.listdir(input_dir_path):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir_path, filename)
            with open(file_path, encoding='utf-8') as file:
                data_list = json.load(file)
                print(f"{filename} 数据数量：{len(data_list)}")
                merged_json.extend(data_list)

    print(f"总数据数量：{len(merged_json)}")
    output_filename = "merged.json"
    output_file_path = os.path.join(output_dir_path, output_filename)

    
    with open(output_file_path, 'w', encoding="utf-8") as f:
        # print(output_file)
        json.dump(merged_json, f, ensure_ascii=False, indent=4)




'''
功能：把json文件批量向量化，每个json文件向量化后文件名不变
输入：
    参数：model_name:str，模型名称
    文件：json文件


输出：
    文件：npy文件

'''
def embedding_json_files(model_name:str):

    # 创建目录
    current_path = os.getcwd()
    # 输入目录：存放待转换的 JSON 文件
    input_dir_path = os.path.join(current_path, "6_merge_files")
    # 输出目录：存放转换后的 npy 文件
    output_dir_path = os.path.join(current_path, "6_merge_files")
    # 如果目录不存在，则创建该目录，用来存储转换成简体字的json文件
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    
    filename = "merged.json"
    file_path = os.path.join(input_dir_path, filename)

    # 向量化json文件
    # 计算向量化时间
    start_time = time.time()
    # 初始化模型（用于向量化）
    
    print("开始向量化")

    embedding_model = Embedding(model_name=model_name, max_length=512)
    # 从 JSON 文件读取数据
    with open(file_path, 'r', encoding='utf-8') as file:
        # 返回字典列表
        documents = json.load(file)

    # 使用模型对文本documents向量化，输出生成器对象，列表化后是二位列表，子列表是documents元素的向量，数据类型是List[np.ndarray]
    # 从字典列表中提取、组合成出字符串，用于向量化
    processed_documents = [doc['title'] + ' ' + ' '.join(doc['paragraphs']) for doc in documents]
    # 向量化
    vectors: List[np.ndarray] = list(embedding_model.embed(processed_documents))

    print(f"向量数量：{len(vectors)}")
    print(f"向量维度：{vectors[0].shape}")
    print(vectors[0].shape)

    # 保存向量化的数据到npy文件
    # 输出文件名+路径

    npy_filename = "merged.npy"
    output_file_path = os.path.join(output_dir_path, npy_filename)
    np.save(output_file_path, vectors, allow_pickle=False)
    print(f"{filename} 向量化成功")

    end_time = time.time()
    run_time = round(end_time - start_time)
    hours, remainder = divmod(run_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"向量化总时间：{hours} h {minutes} m")



'''
功能：批量输入json文件和对应的npy文件，创建向量数据库

输入：
    文件：json文件和对应的npy文件

输出：


文件路径：file_path
文件夹路径：dir_path
文件名：filename
文件夹名：dirname
'''



'''
待优化：
2. 检查向量数据库中向量数量，好像没有全部上传到数据库。测试3个向量数据文档的上传
3. 增加函数说明

'''


if __name__ == "__main__":
    # 1 需要检查的键及其数据类型
    # checked_keys = {
    #     "author": str, 
    #     "title": str, 
    #     "paragraphs": list,
    #     "id": str
    # }
    # check_json_files(checked_keys)
    
    # 2 繁体字转换成简体字
    # t2s()

    # 3 添加字段（字典的键值对）
    # key_values = {"type":"宋词"}
    # add_field(key_values)
    
    # 4 拆分
    # split_list_str()

    # 把多个json文件合并成一个
    # merge_json_files()


    # 5 json文件向量化
    # 该模型向量维度为512
    # model_name = "BAAI/bge-small-zh-v1.5"
    # embedding_json_files(model_name)

    # 6 创建向量数据库
    collection_name = "chinese_poet"
    create_vector_data(collection_name)