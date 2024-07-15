from IPython.display import display, Code, Markdown, Image
from IPython import get_ipython
import time
import openai
import os
import json
from openai import OpenAI
from openai import OpenAIError
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
from pathlib import Path
import oss2
from dotenv import load_dotenv, set_key, find_dotenv
import pymysql
import io
import uuid
import re
import glob
import shutil
import inspect
import requests
import random
import string
import base64
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import base64
from bs4 import BeautifulSoup
import dateutil.parser as parser
import tiktoken
from lxml import etree
import sys
from cryptography.fernet import Fernet
import numpy as np
import pandas as pd
import html2text
import subprocess
import zipfile
import nbconvert
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(f"BASE_DIR: {BASE_DIR}")
dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)
# print(f"dotenv_path: {dotenv_path}")

def clean_and_convert_to_json(raw_str):
    """
    清理并将不规则的字符串转换为满足JSON格式要求的字符串
    :param raw_str: 不规则的原始字符串
    :return: JSON格式的字符串或错误信息
    """
    # 替换未转义的换行符
    cleaned_str = re.sub(r'\\n', r'\\\\n', raw_str)
    # 替换未转义的单引号
    cleaned_str = re.sub(r'(?<!\\)\'', r'\"', cleaned_str)
    # 替换未转义的反斜杠
    cleaned_str = re.sub(r'\\(?=\W)', r'\\\\', cleaned_str)
        
    # 尝试将清理后的字符串转换为JSON对象
    json_obj = json.loads(cleaned_str)
    # 将JSON对象格式化为字符串
    json_str = json.dumps(json_obj, indent=2, ensure_ascii=False)
    return json_str

def clear_folder(path):
    """
    清理指定路径下的所有文件和子文件夹。
    
    :param path: 要清理的文件夹路径。
    """
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        # print(f"The path {path} does not exist.")
        pass

def handle_function_args(function_args):
    """
    处理函数参数，检查并转换为JSON格式
    """
    def is_json(myjson):
        try:
            json_object = json.loads(myjson)
        except ValueError as e:
            return False
        return True

    if not is_json(function_args):
        try:
            function_args = clean_and_convert_to_json(function_args)
        except Exception as e:
            pass
    
    if not is_json(function_args):
        return None
    
    return json.loads(function_args)

def print_code_if_exists(function_args):
    """
    如果存在代码片段，则打印代码
    """
    def convert_to_markdown(code, language):
        return f"```{language}\n{code}\n```"
    
    # 如果是SQL，则按照Markdown中SQL格式打印代码
    if function_args.get('sql_query'):
        code = function_args['sql_query']
        markdown_code = convert_to_markdown(code, 'sql')
        print("即将执行以下代码：")
        display(Markdown(markdown_code))

    # 如果是Python，则按照Markdown中Python格式打印代码
    elif function_args.get('py_code'):
        code = function_args['py_code']
        markdown_code = convert_to_markdown(code, 'python')
        print("即将执行以下代码：")
        display(Markdown(markdown_code))


def extract_run_id(text):
    pattern = r'run_\w+'  # 正则表达式模式，用于匹配以 run_ 开头的字符
    match = re.search(pattern, text)
    if match:
        return match.group(0)  # 返回匹配的字符串
    else:
        return None
    