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
from .getProfile import url1, url2, url3, url4, url5


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print(f"BASE_DIR: {BASE_DIR}")
dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)
# print(f"dotenv_path: {dotenv_path}")

def ensure_file_exists(file_path, timeout=10):
    """
    确保文件存在
    :param file_path: 文件路径
    :param timeout: 等待时间（秒）
    :return: 文件是否存在
    """
    start_time = time.time()
    while not os.path.exists(file_path):
        if time.time() - start_time > timeout:
            print(f"超时 {timeout} 秒，文件 {file_path} 仍不存在。")
            return False
        time.sleep(1)
    return True

def convert_keyword(q):
    """
    将用户输入的问题转化为适合在知乎上进行搜索的关键词
    """
    global client
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你专门负责将用户的问题转化为知乎网站搜索关键词，只返回一个你认为最合适的搜索关键词即可"},
            {"role": "user", "content": "请帮我介绍下Llama3模型基本情况"},
            {"role": "assistant", "content": "Llama3模型介绍"},
            {"role": "user", "content": q}
        ]
    )
    q = completion.choices[0].message.content
    return q


def convert_keyword_github(q):
    """
    将用户输入的问题转化为适合在Github上进行搜索的关键词
    """
    global client
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你专门负责将用户的问题转化为Github上的搜索关键词，只返回一个你认为最合适的搜索关键词即可"},
            {"role": "user", "content": "请问DeepSpeed是什么？"},
            {"role": "assistant", "content": "DeepSpeed"},
            {"role": "user", "content": q}
        ],
    )
    q = completion.choices[0].message.content
    return q

def image_recognition(url_list, question, g='globals()'):
    global client
    """
    根据图片地址，对用户输入的图像进行识别，最终返回用户提问的答案
    :param url_list: 用户输入的图片地址（url）列表，每个图片地址都以字符串形式表示
    :param question: 用户提出的对图片识别的要求
    :param g: g，字符串形式变量，表示环境变量，无需设置，保持默认参数即可
    :return：图片识别的结果
    """
    
    model = os.getenv('VISION_MODEL')
    
    content = []
    content.append({'type': 'text', 'text': question})
    for url in url_list:
        content.append(
            {'type': 'image_url',
             'image_url': {'url': url}
            }
        )
    messages = [
        {'role': 'user', 
         'content':content
        }
    ]

    print("正在对图像内容进行识别...")
    response = client.chat.completions.create(
        model=model,
        messages=messages)
    
    return response.choices[0].message.content


def log_thread_id(thread_id):
    global thread_log_file
    try:
        with open(thread_log_file, "r") as file:
            thread_log = json.load(file)
    except FileNotFoundError:
        thread_log = []

    # 添加新的线程 ID
    thread_log.append(thread_id)

    with open(thread_log_file, "w") as file:
        json.dump(thread_log, file)
        
def get_latest_thread(client):
    global thread_log_file
    try:
        with open(thread_log_file, "r") as file:
            thread_log = json.load(file)
    except FileNotFoundError:
        thread_log = []

    if thread_log:
        # 获取最新的线程 ID 并将其设置为全局变量
        thread_id = thread_log[-1]
        thread = client.beta.threads.retrieve(thread_id=thread_id)
        return thread
    else:
        # 如果没有线程，则创建一个新的线程
        thread = client.beta.threads.create()
        log_thread_id(thread.id)
        return thread
    
def log_token_usage(thread_id, tokens):
    global token_log_file
    try:
        with open(token_log_file, "r") as file:
            token_log = json.load(file)
    except FileNotFoundError:
        token_log = {"total_tokens": 0}

    today = datetime.utcnow().date().isoformat()

    if today not in token_log:
        token_log[today] = {}

    if thread_id not in token_log[today]:
        token_log[today][thread_id] = 0
    token_log[today][thread_id] += tokens

    # 更新累计 token 总数
    if "total_tokens" not in token_log:
        token_log["total_tokens"] = 0
    token_log["total_tokens"] += tokens

    with open(token_log_file, "w") as file:
        json.dump(token_log, file)  

def print_token_usage():
    global token_log_file
    try:
        with open(token_log_file, "r") as file:
            token_log = json.load(file)
    except FileNotFoundError:
        print("目前没有token消耗")
        return
    
    today = datetime.utcnow().date().isoformat()
    
    # 打印今日 token 使用情况
    if today in token_log:
        total_tokens_today = sum(token_log[today].values())
        print(f"今日已消耗的 token 数量：{total_tokens_today}")
    else:
        print("今日没有消耗 token。")
    
    # 打印累计 token 使用情况
    total_tokens = token_log.get("total_tokens", 0)
    print(f"总共消耗的 token 数量：{total_tokens}")

def initialize_agent_info(api_key, agent_type):
    global agent_info_file
    agent_info = {
        "api_key": api_key,
        "initialized": False,
        "agent_type": agent_type,
        'asid': None
    }

    # 检查文件是否存在
    if not os.path.exists(agent_info_file):
        # 如果文件不存在，创建文件并写入信息
        with open(agent_info_file, "w") as file:
            json.dump(agent_info, file)
        # print(f"Agent info file created and initialized at {agent_info_file}")
    else:
        # print(f"Agent info file already exists at {agent_info_file}, initialization skipped.")
        pass
        
def set_agent_initialized(if_initialized=True):
    global agent_info_file
    try:
        with open(agent_info_file, "r") as file:
            agent_info = json.load(file)
    except FileNotFoundError:
        print("Agent信息日志文件不存在。请先初始化Agent信息。")
        return None

    agent_info["initialized"] = if_initialized

    with open(agent_info_file, "w") as file:
        json.dump(agent_info, file)
        
def set_agent_id(asid):
    global agent_info_file
    try:
        with open(agent_info_file, "r") as file:
            agent_info = json.load(file)
    except FileNotFoundError:
        print("Agent信息日志文件不存在。请先初始化Agent信息。")
        return None

    agent_info["asid"] = asid

    with open(agent_info_file, "w") as file:
        json.dump(agent_info, file)
        
def get_agent_info():
    global agent_info_file
    try:
        with open(agent_info_file, "r") as file:
            agent_info = json.load(file)
    except FileNotFoundError:
        print("Agent信息日志文件不存在。")
        return None

    return agent_info

def make_hl():
    global home_dir, log_dir, thread_log_file, token_log_file, agent_info_file
    home_dir = str(Path.home())
    log_dir = os.path.join(home_dir, "._logs")
    os.makedirs(log_dir, exist_ok=True)
    token_log_file = os.path.join(log_dir, "token_usage_log.json")
    thread_log_file = os.path.join(log_dir, "thread_log.json")
    agent_info_file = os.path.join(log_dir, "agent_info_log.json")

    if not os.path.exists(thread_log_file):
        with open(thread_log_file, "w") as f:
            json.dump([], f)  # 创建一个空列表

    if not os.path.exists(token_log_file):
        with open(token_log_file, "w") as f:
            json.dump({"total_tokens": 0}, f) 

def download_files_and_create_kb(client, file_info_json):
    home_dir = str(Path.home())
    kb_name = get_agent_info()['agent_type']
    storage_path = os.path.join(home_dir, f"._logs/{kb_name}")
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    file_info = json.loads(file_info_json)

    print("正在同步Agent基础文件，这会需要一些时间，请耐心等待...")
    for file_name, file_url in file_info.items():
        try:
            response = requests.get(file_url)
            response.raise_for_status()

            file_path = os.path.join(storage_path, file_name)

            with open(file_path, 'wb') as file:
                file.write(response.content)
        except requests.exceptions.RequestException as e:
            print(f"文件 '{file_name}' 下载失败: {e}")
        
    vector_id = create_knowledge_base(client=client, 
                                      knowledge_base_name=kb_name,
                                      folder_path_base=storage_path)

    return vector_id
        
def get_omlf(client):
    vector_id = '-1'
    if get_agent_info()['agent_type'] == 'normal':
        return vector_id
    
    else:
        kb_base_key = os.getenv('KB_BASE_KEY')
        kb_base = decrypt_string(kb_base_key, b'YAboQcXx376HSUKqzkTz8LK1GKs19Skg4JoZH4QUCJc=')
        kb_name = get_agent_info()['agent_type']
        u1 = kb_base + f'/{kb_name}/{kb_name}_kb.py'

        response = requests.get(u1)
        response.raise_for_status()
        file_content = response.text 
        exec_namespace = {}

        exec(file_content, exec_namespace)
        file_info = exec_namespace.get('file_info')
        file_info_json = file_info()
        vector_id = download_files_and_create_kb(client, file_info_json)

        return vector_id

def create_omla(client, vector_id, enhanced_mode):
    
    ag_base_key = os.getenv('AG_BASE_KEY')
    ag_base = decrypt_string(ag_base_key, b'YAboQcXx376HSUKqzkTz8LK1GKs19Skg4JoZH4QUCJc=')
    kb_name = get_agent_info()['agent_type']
    u1 = ag_base+f'/{kb_name}/{kb_name}_agent.py'
    
    response = requests.get(u1)
    response.raise_for_status()
    file_content = response.text
    exec_namespace = {}
    
    exec(file_content, exec_namespace)
    coml = exec_namespace.get('coml')
    if vector_id == '-1':
        asid = coml(client=client, enhanced_mode=enhanced_mode)
    else:
        asid = coml(client=client, vs_id=vector_id, enhanced_mode=enhanced_mode)
    
    return asid

def cre_ct(client, enhanced_mode):
    vector_id = get_omlf(client)
    asid = create_omla(client, vector_id, enhanced_mode)
    return asid   

def function_to_call(run_details, client, thread_id, run_id):
    
    available_functions = {
        "python_inter": python_inter,
        "fig_inter": fig_inter,
        "sql_inter": sql_inter,
        "extract_data": extract_data,
        "image_recognition": image_recognition,
        "get_answer": get_answer,
        "get_answer_github": get_answer_github,
    }
    
    tool_outputs = []
    tool_calls = run_details.required_action.submit_tool_outputs.tool_calls

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments
        
        # 处理多次子调用的情况
        if function_name == 'multi_tool_use.parallel':
            tool_uses = json.loads(function_args).get('tool_uses', [])
            for tool_use in tool_uses:
                recipient_name = tool_use.get('recipient_name')
                parameters = tool_use.get('parameters')
                
                function_to_call = available_functions.get(recipient_name.split('.')[-1])
                
                if function_to_call is None:
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": f"函数 {recipient_name} 不存在"
                    })
                    continue
                
                function_args = json.dumps(parameters)  # 将参数转换为JSON字符串，以便后续处理
                function_args = handle_function_args(function_args)

                if function_args is None:
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": "输入参数不是有效的JSON格式，无法解析"
                    })
                    continue

                # 打印代码
                print_code_if_exists(function_args)

                try:
                    function_args['g'] = globals()
                    # 运行外部函数
                    function_response = function_to_call(**function_args)
                except Exception as e:
                    function_response = "函数运行报错如下:" + str(e)
                
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": function_response
                })

        # 处理单个外部函数调用的情况
        else:
            function_to_call = available_functions.get(function_name)
            
            if function_to_call is None:
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": f"函数 {function_name} 不存在"
                })
                continue

            function_args = handle_function_args(function_args)

            if function_args is None:
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": "输入参数不是有效的JSON格式，无法解析"
                })
                continue

            # 打印代码
            print_code_if_exists(function_args)

            try:
                function_args['g'] = globals()
                # 运行外部函数
                function_response = function_to_call(**function_args)
            except Exception as e:
                function_response = "函数运行报错如下:" + str(e)
            
            tool_outputs.append({
                "tool_call_id": tool_call.id,
                "output": function_response
            })

    return tool_outputs

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

def run_status(assistant_id, client, thread_id, run_id):
    # 创建计数器
    i = 0
    try:
        # 轮询检查运行状态
        while True:
            run_details = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            status = run_details.status

            if status in ['completed', 'expired', 'cancelled']:
                log_token_usage(thread_id, run_details.usage.total_tokens)
                return run_details
            if status in ['requires_action']:
                return run_details
            if status in ['failed']:
                print("当前服务器拥挤，请1分钟后再试。")
                return None

            i += 1
            if i == 30:
                print("响应超时，请稍后再试。")
                return None

            # 等待一秒后再检查状态
            time.sleep(1)
            
    except OpenAIError as e:
        print(f"An error occurred: {e}")
        return None
        
    return None            

def chat_base(user_input, 
              assistant_id, 
              client, 
              thread_id, 
              run_id=None,
              first_input=True, 
              tool_outputs=None):
    
    # 创建消息
    if first_input:
        message = client.beta.threads.messages.create(
          thread_id=thread_id,
          role="user",
          content=user_input
        )
        
    if tool_outputs == None:
        # 执行对话
        run = client.beta.threads.runs.create(
          thread_id=thread_id,
          assistant_id=assistant_id
        )
        
    else:
        # Function calling第二轮对话，更新run的状态
        run = client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run_id,
            tool_outputs=tool_outputs
        )
    
    # 判断运行状态
    run_details = run_status(assistant_id=assistant_id, 
                             client=client, 
                             thread_id=thread_id, 
                             run_id=run.id)
    
    # 若无结果，则打印报错信息
    if run_details == None:
        print('当前应用无法运行，请稍后再试')
        
    # 若消息创建完成，则返回模型返回信息
    elif run_details.status == 'completed':
        messages_check = client.beta.threads.messages.list(thread_id=thread_id)
        chat_res = messages_check.data[0].content[0].text.value
        res = '**MateGen**:' + chat_res
        display(Markdown(res))
        
    # 若外部函数响应超时，则根据用户反馈制定执行流程
    elif run_details.status == 'expired' or run_details.status == 'cancelled' :
        user_res = input('当前编程环境响应超时或此前任务已取消，是否继续？1.继续，2.重新输入需求')
        if user_res == '1':
            print('好的，正在重新创建响应')
            chat_base_auto_cancel(user_input=user_input, 
                                  assistant_id=assistant_id, 
                                  client=client, 
                                  thread_id=thread_id, 
                                  run_id=run_id,
                                  first_input=True, 
                                  tool_outputs=None, 
                                 )

        else:
            user_res1 = input('请输入新的问题：')
            chat_base_auto_cancel(user_input=user_res1, 
                                  assistant_id=assistant_id, 
                                  client=client, 
                                  thread_id=thread_id, 
                                  run_id=run_id,
                                  first_input=True, 
                                  tool_outputs=None, 
                                 )
            
    # 若调用外部函数，则开启Function calling
    elif run_details.status == 'requires_action':
        # 创建外部函数输出结果
        tool_outputs = function_to_call(run_details=run_details, 
                                        client=client, 
                                        thread_id=thread_id, 
                                        run_id=run.id)
        
        chat_base_auto_cancel(user_input=user_input, 
                              assistant_id=assistant_id, 
                              client=client, 
                              thread_id=thread_id, 
                              run_id=run.id,
                              first_input=False, 
                              tool_outputs=tool_outputs)
        
def extract_run_id(text):
    pattern = r'run_\w+'  # 正则表达式模式，用于匹配以 run_ 开头的字符
    match = re.search(pattern, text)
    if match:
        return match.group(0)  # 返回匹配的字符串
    else:
        return None
        
def chat_base_auto_cancel(user_input, 
                          assistant_id, 
                          client, 
                          thread_id, 
                          run_id=None,
                          first_input=True, 
                          tool_outputs=None):
    max_attempt = 3
    now_attempt = 0
    
    while now_attempt < max_attempt:
        try:
            chat_base(user_input=user_input, 
                      assistant_id=assistant_id, 
                      client=client, 
                      thread_id=thread_id, 
                      run_id=run_id,
                      first_input=first_input, 
                      tool_outputs=tool_outputs)
            # print("成功运行")
            break  # 成功运行后退出循环

        except OpenAIError as e:
            run_id_to_cancel = extract_run_id(e.body['message'])
            if run_id_to_cancel:
                client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run_id_to_cancel)
                print(f"已取消运行 {run_id_to_cancel}")
            else:
                print("未找到运行ID，无法取消")
            
        except Exception as e:
            print(f"程序运行错误: {e}")
            
        now_attempt += 1

    if now_attempt == max_attempt:
        print("超过最大尝试次数，操作失败")      

def decrypt_string(encrypted_message: str, key: str) -> str:
    f = Fernet(key)
    decrypted_message = f.decrypt(encrypted_message.encode())
    return decrypted_message.decode()

def move_folder(src_folder, dest_folder):
    """
    将文件夹从 src_folder 剪切到 dest_folder，当目标文件夹存在时覆盖原始文件夹
    """
    try:
        # 确保源文件夹存在
        if not os.path.exists(src_folder):
            print(f"源文件夹不存在: {src_folder}")
            return False
        
        # 如果目标文件夹存在，删除目标文件夹
        if os.path.exists(dest_folder):
            shutil.rmtree(dest_folder)
        
        # 确保目标文件夹的父目录存在，如果不存在则创建
        parent_dir = os.path.dirname(dest_folder)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        # 移动文件夹
        shutil.move(src_folder, dest_folder)
        print(f"本地知识库文件夹已从 {src_folder} 移动到 {dest_folder}")
        return True
    except Exception as e:
        print(f"移动文件夹失败: {e}。或由于目标文件夹正在被读取导致，请重启Jupyter并再次尝试。")
        return False
    
def get_knowledge_base_description(sub_folder_name):
    """
    获取指定知识库的知识库描述内容
    """
    # 获取 KNOWLEDGE_LIBRARY_PATH 环境变量
    knowledge_library_path = os.getenv('KNOWLEDGE_LIBRARY_PATH')
    
    # 检查 KNOWLEDGE_LIBRARY_PATH 是否存在且路径是否有效
    if knowledge_library_path and os.path.exists(knowledge_library_path):
        base_path = os.path.join(knowledge_library_path, 'knowledge_base')
    else:
        # 如果 KNOWLEDGE_LIBRARY_PATH 不存在或路径无效，则在 home 目录下创建文件夹
        home_dir = str(Path.home())
        base_path = os.path.join(home_dir, 'knowledge_base')
    
    # 子文件夹路径
    sub_folder_path = os.path.join(base_path, sub_folder_name)
    sub_json_file = os.path.join(sub_folder_path, f'{sub_folder_name}_vector_id.json')
    # print(sub_json_file)

    # 检查子目录 JSON 文件是否存在
    # if ensure_file_exists(sub_json_file):
    with open(sub_json_file, 'r') as f:
        data = json.load(f)
        description = data.get('knowledge_base_description', "")
        if description:
            return description
        else:
            return False
    # else:
        # print(f"子目录 JSON 文件不存在：{sub_json_file}")
        # return False

def update_knowledge_base_description(sub_folder_name, description):
    """
    更新子目录的 knowledge_base_description 字段
    """
    # 获取 KNOWLEDGE_LIBRARY_PATH 环境变量
    knowledge_library_path = os.getenv('KNOWLEDGE_LIBRARY_PATH')
    
    # 检查 KNOWLEDGE_LIBRARY_PATH 是否存在且路径是否有效
    if knowledge_library_path and os.path.exists(knowledge_library_path):
        base_path = os.path.join(knowledge_library_path, 'knowledge_base')
    else:
        # 如果 KNOWLEDGE_LIBRARY_PATH 不存在或路径无效，则在 home 目录下创建文件夹
        home_dir = str(Path.home())
        base_path = os.path.join(home_dir, 'knowledge_base')
    
    # 子文件夹路径
    sub_folder_path = os.path.join(base_path, sub_folder_name)
    sub_json_file = os.path.join(sub_folder_path, f'{sub_folder_name}_vector_id.json')

    # 检查子目录 JSON 文件是否存在
    # if ensure_file_exists(sub_json_file):
    with open(sub_json_file, 'r+') as f:
        data = json.load(f)
        # 先删除原有的 description 内容
        data['knowledge_base_description'] = ""
        # 再写入新的 description 内容
        data['knowledge_base_description'] = description
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()
    print(f"已更新知识库：{sub_folder_name}的相关描述")
    # else:
        # print(f"子目录 JSON 文件不存在：{sub_json_file}")

def print_and_select_knowledge_base():
    # 获取 KNOWLEDGE_LIBRARY_PATH 环境变量
    knowledge_library_path = os.getenv('KNOWLEDGE_LIBRARY_PATH')
    
    # 检查 KNOWLEDGE_LIBRARY_PATH 是否存在且路径是否有效
    if knowledge_library_path and os.path.exists(knowledge_library_path):
        base_path = os.path.join(knowledge_library_path, 'knowledge_base')
    else:
        # 如果 KNOWLEDGE_LIBRARY_PATH 不存在或路径无效，则在 home 目录下创建文件夹
        home_dir = str(Path.home())
        base_path = os.path.join(home_dir, 'knowledge_base')
    
    # 检查主目录 JSON 文件
    main_json_file = os.path.join(base_path, 'main_vector_db_mapping.json')
    if not os.path.exists(main_json_file):
        print(f"{main_json_file} 不存在。请先创建知识库。")
        return None, None
    
    # 读取主目录 JSON 文件
    with open(main_json_file, 'r') as f:
        main_mapping = json.load(f)
    
    while True:
        # 打印所有知识库名称
        print("知识库列表：")
        knowledge_bases = list(main_mapping.keys())
        for idx, name in enumerate(knowledge_bases, 1):
            print(f"{idx}. {name}")
        
        # 用户选择知识库
        try:
            selection = int(input("请选择一个知识库的序号（或输入0创建新知识库）：")) - 1
            if selection == -1:
                new_knowledge_base = input("请输入新知识库的名称：")
                # 返回新知识库名称和 None 作为 ID
                return new_knowledge_base, None
            elif 0 <= selection < len(knowledge_bases):
                selected_knowledge_base = knowledge_bases[selection]
                vector_db_id = main_mapping[selected_knowledge_base]
                return selected_knowledge_base, vector_db_id
            else:
                print("无效的选择。请再试一次。")
        except ValueError:
            print("请输入一个有效的序号。")
            
def print_and_select_knowledge_base_to_update():
    knowledge_library_path = os.getenv('KNOWLEDGE_LIBRARY_PATH')
    # 检查 KNOWLEDGE_LIBRARY_PATH 是否存在且路径是否有效
    if knowledge_library_path and os.path.exists(knowledge_library_path):
        base_path = os.path.join(knowledge_library_path, 'knowledge_base')
    else:
        # 如果 KNOWLEDGE_LIBRARY_PATH 不存在或路径无效，则在 home 目录下创建文件夹
        home_dir = str(Path.home())
        base_path = os.path.join(home_dir, 'knowledge_base')
            
    main_json_file = os.path.join(base_path, 'main_vector_db_mapping.json')
    if not os.path.exists(main_json_file):
        print(f"{main_json_file} 不存在。请先创建知识库。")
        return None, None
    
    # 读取主目录 JSON 文件
    with open(main_json_file, 'r') as f:
        main_mapping = json.load(f)
    
    while True:
        # 打印所有知识库名称
        print("知识库列表：")
        knowledge_bases = list(main_mapping.keys())
        for idx, name in enumerate(knowledge_bases, 1):
            print(f"{idx}. {name}")
        
        # 用户选择知识库
        try:
            selection = int(input("请选择一个知识库的序号（或输入0退出）：")) - 1
            if selection == -1:
                return None, None
            elif 0 <= selection < len(knowledge_bases):
                selected_knowledge_base = knowledge_bases[selection]
                vector_db_id = main_mapping[selected_knowledge_base]
                return selected_knowledge_base, vector_db_id
            else:
                print("无效的选择。请再试一次。")
        except ValueError:
            print("请输入一个有效的序号。")

        
class MateGenClass:
    def __init__(self, 
                 api_key, 
                 enhanced_mode=False,
                 knowledge_base_chat=False,
                 kaggle_competition_guidance=False):
        
        """
        初始参数解释：
        api_key：必选参数，表示调用OpenAI模型所必须的字符串密钥，没有默认取值，需要用户提前设置才可使用MateGen。api-key获取与token购买：添加客服小可爱：littlelion_1215，回复“MG”详询哦，MateGen测试期间限时免费赠送千亿token，送完即止~
        
        enhanced_mode：可选参数，表示是否开启增强模式，开启增强模式时，MateGen各方面性能都将大幅提高，但同时也将更快速的消耗token额度。
        
        knowledge_base_name：可选参数，表示知识库名称，当输入字符串名称的时候，默认会开启知识库问答模式。需要注意的是，我们需要手动在知识库中放置文档，才能顺利进行知识库问答。需要注意的是，若此处输入一个Kaggle竞赛名字，并在kaggle_competition_guidance输入True，即可开启Kaggle辅导模式。MateGen会自动接入Kaggle API进行赛题信息搜索和热门Kernel搜索，并自动构建同名知识库，并开启知识库问答模式，此时并不需要手动放置文件。
        
        kaggle_competition_guidance：可选参数，表示是否开启Kaggle辅导模式。开启Kaggle辅导模式时，需要在knowledge_base_name参数位置输入正确的Kaggle竞赛名。需要注意，只有能够顺利联网时，才可开启竞赛辅导模式。
        
        """
        make_hl()
        global home_dir, log_dir, thread_log_file, token_log_file, agent_info_file, client
        # 基础属性定义
        self.api_key = api_key
        self.enhanced_mode = enhanced_mode
        self.knowledge_base_chat = knowledge_base_chat
        self.kaggle_competition_guidance = kaggle_competition_guidance
        self.knowledge_base_name = None
        self.competition_name = None
        self.vector_id = None
        self.base_path = None
        self.knowledge_base_description = ''
        
        original_string = decrypt_string(api_key, key=b'YAboQcXx376HSUKqzkTz8LK1GKs19Skg4JoZH4QUCJc=')
        split_strings = original_string.split(' ')
        s1 = split_strings[0]
        s2 = split_strings[1]
        
        initialize_agent_info(api_key=self.api_key, agent_type=s2)
        
        base_url = os.getenv('BASE_URL')
        self.client = OpenAI(api_key = s1, 
                             base_url = base_url)
        client = self.client
        
        print("正在初始化MateGen，请稍后...")
        try:
            self.models = self.client.models.list()
            
            if self.models:
                print("成功连接服务器，API-KEY通过验证！")
                if get_agent_info()['api_key'] != self.api_key:
                    print("检测到API-KEY发生变化，正在重新创建Agent...")
                    shutil.rmtree(log_dir)
                    make_hl()
                    initialize_agent_info(api_key=self.api_key, agent_type=s2)
                    self.s3 = cre_ct(client, enhanced_mode)
                    set_agent_initialized(if_initialized=True)
                    set_agent_id(asid=self.s3)                    
                    
                elif not get_agent_info()['initialized']:
                    print("首次使用MateGen，正在进行Agent基础设置...")
                    self.s3 = cre_ct(client, enhanced_mode)
                    set_agent_initialized(if_initialized=True)
                    set_agent_id(asid=self.s3)
                else:
                    self.s3 = get_agent_info()['asid']
                    
                self.thread = get_latest_thread(self.client)
                self.thread_id = self.thread.id
                log_token_usage(self.thread_id, 0)
                
                if self.kaggle_competition_guidance == True:
                    self.competition_name = input('请输入竞赛名称')
                    self.vector_id = get_vector_db_id(knowledge_base_name=self.competition_name)
                    if self.vector_id != None:
                        if self.get_knowledge_base_vsid(knowledge_base_name=self.competition_name):
                            self.vector_id = self.get_knowledge_base_vsid(knowledge_base_name=self.competition_name)
                            print('已检测到竞赛知识库，正在开启该知识库并进行问答')
                        else:
                            print("知识库已过期，正在重新创建知识库")
                            self.upload_knowledge_base(knowledge_base_name=self.competition_name)
                    else:
                        print('本地并不存在当前竞赛相关知识库，正在开启联网搜索功能，构建竞赛知识库...')
                        self.vector_id = create_competition_knowledge_base(competition_name=self.competition_name, client=self.client)

                elif self.knowledge_base_chat == True:
                    self.knowledge_base_name = input("请输入知识库名称，输入0查询当前知识库列表。") 
                    if self.knowledge_base_name == '0':
                        self.knowledge_base_name, self.vector_id = print_and_select_knowledge_base()
                        if self.knowledge_base_name == None:
                            print('未检测到知识库...')
                            self.knowledge_base_name = input("请重新输入知识库名称，系统将开始创建知识库。")

                    self.vector_id = get_vector_db_id(knowledge_base_name=self.knowledge_base_name)
                        
                    if self.vector_id != None:
                        if self.get_knowledge_base_vsid(knowledge_base_name=self.knowledge_base_name):
                            self.vector_id = self.get_knowledge_base_vsid(knowledge_base_name=self.knowledge_base_name)
                            print('知识库已存在，已启用该知识库')
                        else:
                            print("知识库已过期，正在重新创建知识库")
                            self.upload_knowledge_base(knowledge_base_name=self.knowledge_base_name)
                    else:
                        print('正在创建知识库文件夹')
                        self.base_path = create_knowledge_base_folder(sub_folder_name=self.knowledge_base_name)
                        print(f"当前问答知识库文件夹路径：{self.base_path}，请在文件夹中放置知识库文件。")
                        print("目前支持PDF、Word、PPT、md等格式读取与检索。")
                else:
                    if enhanced_mode:
                        model = 'gpt-4o'
                    else:
                        model = 'gpt-4o-mini'
                    asi = self.client.beta.assistants.retrieve(self.s3)
                    instructions = asi.instructions
                    instructions = remove_knowledge_base_info(instructions)
                    asi = self.client.beta.assistants.update(
                        self.s3,
                        model=model,
                        instructions=instructions
                    )
                    
                if (self.kaggle_competition_guidance == True or self.knowledge_base_chat == True) and self.vector_id != None:
                    if wait_for_vector_store_ready(vs_id=self.vector_id, client=self.client):
                        asi = self.client.beta.assistants.retrieve(self.s3)
                        instructions = asi.instructions
                        instructions = remove_knowledge_base_info(instructions)
                        knowledge_base_description = get_knowledge_base_description(sub_folder_name=self.knowledge_base_name)
                        new_instructions = instructions + knowledge_base_description
                        asi = self.client.beta.assistants.update(
                            self.s3,
                            instructions=new_instructions,
                            tool_resources={"file_search": {"vector_store_ids": [self.vector_id]}}
                        )
                        
                print("已完成初始化，MateGen可随时调用！")

            else:
                print("当前网络环境无法连接服务器，请检查网络并稍后重试...")
                
        except openai.AuthenticationError:
            print("API-KEY未通过验证，请添加客服小可爱微信：littlelion_1215领取限量免费测试API-KEY，或按需购买token。")
        except openai.APIConnectionError:
            print("当前网络环境无法连接服务器，请检查网络并稍后重试...")
        except openai.RateLimitError:
            print("API-KEY账户已达到RateLimit上限，请添加客服小可爱微信：littlelion_1215领取限量免费测试API-KEY，或按需购买token。")
        except openai.OpenAIError as e:
            print(f"An error occurred: {e}")         
            

    def chat(self, question=None):
        if self.knowledge_base_chat == True and self.vector_id == None:
            if not is_folder_not_empty(self.base_path):
                user_input = input(f"知识库文件夹：{self.base_path}为空，请选择1：关闭知识库问答功能并继续对话；2：退出对话，在指定文件夹内放置文件之后再进行知识库问答对话。")
                if user_input == '1':
                    self.knowledge_base_chat = False
                    pass
                else:
                    return None
            else:
                self.upload_knowledge_base(knowledge_base_name=self.knowledge_base_name)
                if self.vector_id != None:
                    if wait_for_vector_store_ready(vs_id=self.vector_id, client=self.client):
                        asi = self.client.beta.assistants.retrieve(self.s3)
                        instructions = asi.instructions
                        instructions = remove_knowledge_base_info(instructions)
                        knowledge_base_description = get_knowledge_base_description(sub_folder_name=self.knowledge_base_name)
                        new_instructions = instructions + knowledge_base_description
                        asi = self.client.beta.assistants.update(
                            self.s3,
                            instructions=new_instructions,
                            tool_resources={"file_search": {"vector_store_ids": [self.vector_id]}}
                        )
                else:
                    print("知识库创建不成功，请稍后再试。")
            
        head_str = "▌ MateGen初始化完成，欢迎使用！"
        display(Markdown(head_str))
        
        if question != None:
            chat_base_auto_cancel(user_input=question, 
                                  assistant_id=self.s3, 
                                  client=self.client, 
                                  thread_id=self.thread_id, 
                                  run_id=None,
                                  first_input=True, 
                                  tool_outputs=None)
        else:
            print("你好，我是MateGen，你的个人交互式编程助理，有任何问题都可以问我哦~")
            while True:
                question = input("请输入您的问题(输入退出以结束对话): ")
                if question == "退出":
                    break        
                chat_base_auto_cancel(user_input=question, 
                                      assistant_id=self.s3, 
                                      client=self.client, 
                                      thread_id=self.thread_id, 
                                      run_id=None,
                                      first_input=True, 
                                      tool_outputs=None)

                
    def upload_knowledge_base(self, knowledge_base_name=None):
        if knowledge_base_name != None:
            self.knowledge_base_name = knowledge_base_name
        elif self.knowledge_base_name == None:
            self.knowledge_base_name = input("请输入需要更新的知识库名称：")
        
        if not is_folder_not_empty(self.knowledge_base_name):
            print(f"知识库文件夹：{self.knowledge_base_name}为空，请先放置文件再更新知识库。")
            return None
        else:
            self.vector_id = create_knowledge_base(self.client, self.knowledge_base_name)
            if self.vector_id != None:
                print(f"已成功更新知识库{self.knowledge_base_name}")
            
    def update_knowledge_base(self):
        knowledge_base_name, vector_id = print_and_select_knowledge_base_to_update()
        if knowledge_base_name != None:
            self.upload_knowledge_base(knowledge_base_name=knowledge_base_name)
                
    def get_knowledge_base_vsid(self, knowledge_base_name=None):
        if knowledge_base_name != None:
            self.knowledge_base_name = knowledge_base_name
        elif self.knowledge_base_name == None:
            self.knowledge_base_name = input("请输入需要获取知识库ID的知识库名称：")     
           
        knowledge_base_name = self.knowledge_base_name + '!!' + self.client.api_key[8: ]
        check_res = check_knowledge_base_name(client=self.client, 
                                              knowledge_base_name=knowledge_base_name)
        
        if check_res == None:
            print("知识库尚未创建或已经过期，请重新创建知识库。")
            return None
        else:
            return check_res
        
    def set_knowledge_base_url(self, base_url):
        if self.is_valid_directory(base_url):
            # knowledge_library_path = os.getenv('KNOWLEDGE_LIBRARY_PATH')
            # 检查 KNOWLEDGE_LIBRARY_PATH 是否存在且路径是否有效
            # if knowledge_library_path and os.path.exists(knowledge_library_path):
                # base_path = os.path.join(knowledge_library_path, 'knowledge_base')
            # else:
                # 如果 KNOWLEDGE_LIBRARY_PATH 不存在或路径无效，则在 home 目录下创建文件夹
                # home_dir = str(Path.home())
                # base_path = os.path.join(home_dir, 'knowledge_base')
            # os.makedirs(base_path, exist_ok=True)
            # res = move_folder(base_path, base_url)
            # if res:
            set_key(dotenv_path, 'KNOWLEDGE_LIBRARY_PATH', base_url)
            print(f"知识库地址修改为：{base_url}")
            # else:
                # pass
        else:
            print(f"无效的知识库地址：{base_url}")
            
    def set_base_url(self, base_url):
        if is_valid_base_url(base_url):
            set_key(dotenv_path, 'BASE_URL', base_url)
            print(f"更新后base_url地址：{base_url}")
        else:
            print(f"无效的base_url地址：{base_url}")    
            
    def is_valid_base_url(self, path):
        original_string = decrypt_string(self.api_key, key=b'YAboQcXx376HSUKqzkTz8LK1GKs19Skg4JoZH4QUCJc=')
        split_strings = original_string.split(' ')
        s1 = split_strings[0]
        client_tmp = OpenAI(api_key = s1, 
                            base_url = path)
        
        models_tmp = client_tmp.models.list()
        return models_tmp
        
    def is_valid_directory(self, path):
        """
        检查路径是否为有效的目录路径
        """
        # 检查路径是否为绝对路径
        if not os.path.isabs(path):
            return False
        
        # 检查路径是否存在且为目录
        if not os.path.isdir(path):
            return False
        
        return True
    
    def write_knowledge_base_description(self, description):
        """
        更新知识库描述
        """
        self.knowledge_base_description = description
        update_knowledge_base_description(self.knowledge_base_name, 
                                          self.knowledge_base_description)
            
    def debug(self):
        res = input('注意：debug功能只能捕获上一个cell运行报错结果，且只有MateGen模块与当前代码环境命名变量一致时，debug功能才能顺利运行。其他情况下请手动复制代码和报错信息，并通过chat方法将信息输入给MateGen，MateGen可以据此进行更加准确的debug。是否继续使用debug功能：1.继续；2.退出')
        if res == '1':
            current_globals = globals()

            ipython = get_ipython()
            history = list(ipython.history_manager.get_range())

            if not history:
                print("没有历史代码记录，无法启动自动debug功能。")
            else:
                last_session, last_line_number, last_cell_code = history[-2]
                try:
                    exec(last_cell_code, current_globals)
                except Exception as e:
                    error_info = str(e)
                    user_input = f"你好，我的代码运行报错了，请你帮我检查代码并为我解释报错原因。代码：{last_cell_code}，报错信息{error_info}"
                    chat_base_auto_cancel(user_input=user_input, 
                                          assistant_id=self.s3, 
                                          client=self.client, 
                                          thread_id=self.thread_id, 
                                          run_id=None,
                                          first_input=True, 
                                          tool_outputs=None)
        else:
            print("请调用MateGen的chat功能，并手动复制报错代码和报错信息，以便进行精准debug哦~。")
                    
    def clear_messages(self):
        client.beta.threads.delete(thread_id=self.thread_id)
        thread = client.beta.threads.create()
        self.thread = thread
        self.thread_id = thread.id
        log_thread_id(self.thread_id)
        print("已经清理历史消息")
        
    def reset(self):
        try:
            home_dir = str(Path.home())
            log_dir = os.path.join(home_dir, "._logs")
            shutil.rmtree(log_dir)
            print("已重置成功！请重新创建MateGen并继续使用。")
            
        except Exception as e:
            print("重置失败，请重启代码环境，并确认API-KEY后再尝试重置。")
            
    def reset_account_info(self):
        res = input("账户重置功能将重置全部知识库在线存储文档、词向量数据库和已创建的Agent，是否继续：1.继续；2.退出。")
        if res == '1':
            print("正在重置账户各项信息...")
            print("正在删除在线知识库中全部文档文档...")
            delete_all_files(self.client)
            print("正在删除知识库的词向量存储...")
            delete_all_vector_stores(self.client)
            print("正在删除已创建的Agent")
            delete_all_assistants(self.client)
            print("正在重置Agent信息")
            self.reset()
            print("已重置成功重置账户！请重新创建MateGen并继续使用。")
        else:
            return None
        
    def print_usage(self):
        print_token_usage()
        print("本地token计数可能有误，token消耗实际情况以服务器数据为准哦~")
    

def get_id(keyword):
    load_dotenv()
    headers_json = os.getenv('HEADERS')
    cookies_json = os.getenv('COOKIES')
        
    headers = json.loads(headers_json)
    cookies = json.loads(cookies_json)
    
    url = "https://www.kaggle.com/api/i/search.SearchWebService/FullSearchWeb"
    data = {
        "query": keyword,
        "page": 1,
        "resultsPerPage": 20,
        "showPrivate": True
    }
    data = json.dumps(data, separators=(',', ':'))
    response = requests.post(url, headers=headers, cookies=cookies, data=data).json()

    # 确保搜索结果不为空
    if "documents" not in response or len(response["documents"]) == 0:
        print(f"竞赛： '{keyword}' 并不存在，请登录Kaggle官网并检查赛题是否正确：https://www.kaggle.com/")
        return None

    document = response["documents"][0]
    document_type = document["documentType"]

    if document_type == "COMPETITION":
        item_id = document["databaseId"]
    elif document_type == "KERNEL":
        item_id = document['kernelInfo']['dataSources'][0]['reference']['sourceId']
    else:
        print(f"竞赛： '{keyword}' 并不存在，请登录Kaggle官网并检查赛题是否正确：https://www.kaggle.com/")
        return None

    return item_id

def create_kaggle_project_directory(competition_name):
    # 创建kaggle知识库目录
    kaggle_dir = create_knowledge_base_folder(sub_folder_name=competition_name)
    
    # 如果 .kaggle 目录不存在，则创建
    # if not os.path.exists(kaggle_dir):
        # os.makedirs(kaggle_dir)
        # print(f"Created directory: {kaggle_dir}")
    
    # 定义项目目录结构
    # base_dir = os.path.join(kaggle_dir, f"{competition_name}_project")
    # directories = [
        # os.path.join(base_dir, 'knowledge_library'),
        # os.path.join(base_dir, 'data'),
        # os.path.join(base_dir, 'submission'),
        # os.path.join(base_dir, 'module')
    # ]
    # task_schedule_file = os.path.join(base_dir, 'task_schedule.json')

    # 创建目录和文件
    # for directory in directories:
        # if not os.path.exists(directory):
            # os.makedirs(directory)

    # if not os.path.exists(task_schedule_file):
        # with open(task_schedule_file, 'w') as f:
            # json.dump({}, f)  
            
    # print("已完成项目创建")
    return kaggle_dir
    
def getOverviewAndDescription(_id):
    load_dotenv()
    headers_json = os.getenv('HEADERS')
    cookies_json = os.getenv('COOKIES')
        
    headers = json.loads(headers_json)
    cookies = json.loads(cookies_json)
    url = "https://www.kaggle.com/api/i/competitions.PageService/ListPages"
    data = {
        "competitionId": _id
    }
    data = json.dumps(data, separators=(',', ':'))
    data = requests.post(url, headers=headers, cookies=cookies, data=data).json()
    overview={}
    data_description={}
    for page in data['pages']:
        # print(page['name'])
        overview[page['name']]=page['content']   
    if 'rules' in overview: del overview['rules']
    if 'data-description' in overview: 
        data_description['data-description']=overview['data-description']
        del overview['data-description']
 
    return overview, data_description

def json_to_markdown(json_obj, level=1):
    markdown_str = ""
    
    for key, value in json_obj.items():
        if isinstance(value, dict):
            markdown_str += f"{'#' * level} {key}\n\n"
            markdown_str += json_to_markdown(value, level + 1)
        else:
            markdown_str += f"{'#' * level} {key}\n\n{value}\n\n"
    
    return markdown_str

def convert_html_to_markdown(html_content):
    """
    将 HTML 内容转换为 Markdown 格式

    :param html_content: 包含 HTML 标签的文本内容
    :return: 转换后的 Markdown 文本
    """
    h = html2text.HTML2Text()
    h.ignore_links = False  # 设置为 False 以保留链接
    markdown_content = h.handle(html_content)
    return markdown_content

def save_markdown(content, competition_name, file_type):
    # home_dir = str(Path.home())
    # directory = os.path.join(os.path.expanduser(home_dir), f'.kaggle./{competition_name}_project/knowledge_library')
    directory = create_kaggle_project_directory(competition_name)
    filename = f'{competition_name}_{file_type}.md'
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
def get_code(_id):
    load_dotenv()
    headers_json = os.getenv('HEADERS')
    cookies_json = os.getenv('COOKIES')
        
    headers = json.loads(headers_json)
    cookies = json.loads(cookies_json)
    data = {
        "kernelFilterCriteria": {
            "search": "",
            "listRequest": {
                "competitionId": _id,
                "sortBy": "VOTE_COUNT",
                "pageSize": 20,
                "group": "EVERYONE",
                "page": 1,
                "modelIds": [],
                "modelInstanceIds": [],
                "excludeKernelIds": [],
                "tagIds": "",
                "excludeResultsFilesOutputs": False,
                "wantOutputFiles": False,
                "excludeNonAccessedDatasources": True
            }
        },
        "detailFilterCriteria": {
            "deletedAccessBehavior": "RETURN_NOTHING",
            "unauthorizedAccessBehavior": "RETURN_NOTHING",
            "excludeResultsFilesOutputs": False,
            "wantOutputFiles": False,
            "kernelIds": [],
            "outputFileTypes": [],
            "includeInvalidDataSources": False
        },
        "readMask": "pinnedKernels"
    }
    url="https://www.kaggle.com/api/i/kernels.KernelsService/ListKernels"
    data = json.dumps(data, separators=(',', ':'))
    kernels = requests.post(url, headers=headers, cookies=cookies, data=data).json()['kernels']

    res=[]
    for kernel in kernels:
        temp={}
        temp['title']=kernel['title']
        temp['scriptUrl']="https://www.kaggle.com"+kernel['scriptUrl']
        res.append(temp)
    res = res[:10]
    return json.dumps(res)

def extract_and_transform_urls(json_urls):

    # 解析 JSON 字符串
    data = json.loads(json_urls)
    
    # 提取并转换 URL
    urls = []
    for item in data:
        url = item.get("scriptUrl", "")
        match = re.search(r"https://www.kaggle.com/code/(.*)", url)
        if match:
            urls.append(match.group(1))
    
    return urls

def download_and_convert_kernels(urls, competition_name):
    # home_dir = str(Path.home())
    # output_dir = os.path.join(os.path.expanduser(home_dir), f'.kaggle/{competition_name}_project/knowledge_library')
    output_dir = create_kaggle_project_directory(competition_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for kernel_path in urls:
        try:
            # 下载 Kernel
            res = subprocess.run(["kaggle", "kernels", "pull", kernel_path, "-p", output_dir], check=True)
            
            # 找到下载的 .ipynb 文件
            ipynb_file = os.path.join(output_dir, f"{os.path.basename(kernel_path)}.ipynb")
            
            if os.path.exists(ipynb_file):
                # 转换 .ipynb 文件为 .md 文件
                try:
                    md_exporter = nbconvert.MarkdownExporter()
                    md_data, resources = md_exporter.from_filename(ipynb_file)
                    
                    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(ipynb_file))[0] + '.md')
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(md_data)
                    #print(f"Converted {ipynb_file} to {output_file}")
                    
                    # 删除原 .ipynb 文件
                    os.remove(ipynb_file)
                    #print(f"Deleted {ipynb_file}")
                except Exception as e:
                    print(f"Error converting {ipynb_file}: {e}")
                    traceback.print_exc()
            else:
                # print(f"{ipynb_file} not found, skipping conversion.")
                pass
            time.sleep(1)  # 避免请求过于频繁

        except subprocess.CalledProcessError as e:
            # print(f"Error downloading kernel {kernel_path}: {e}")
            traceback.print_exc()

    return 'done'

def check_knowledge_base_name(client, knowledge_base_name):
    vector_stores = client.beta.vector_stores.list()
    for vs in vector_stores.data:
        if vs.name == knowledge_base_name:
            return vs.id
    return None


def create_knowledge_base_folder(sub_folder_name=None):
    # 获取 KNOWLEDGE_LIBRARY_PATH 环境变量
    knowledge_library_path = os.getenv('KNOWLEDGE_LIBRARY_PATH')
    
    # 检查 KNOWLEDGE_LIBRARY_PATH 是否存在且路径是否有效
    if knowledge_library_path and os.path.exists(knowledge_library_path):
        base_path = os.path.join(knowledge_library_path, 'knowledge_base')
    else:
        # 如果 KNOWLEDGE_LIBRARY_PATH 不存在或路径无效，则在 home 目录下创建文件夹
        home_dir = str(Path.home())
        base_path = os.path.join(home_dir, 'knowledge_base')

    # 创建 base_path 文件夹
    os.makedirs(base_path, exist_ok=True)

    # 检查并创建主目录 JSON 文件
    main_json_file = os.path.join(base_path, 'main_vector_db_mapping.json')
    if not os.path.exists(main_json_file):
        with open(main_json_file, 'w') as f:
            json.dump({}, f, indent=4)

    # 如果 sub_folder_name 不为空，则在 base_path 内创建子文件夹
    if sub_folder_name:
        sub_folder_path = os.path.join(base_path, sub_folder_name)
        os.makedirs(sub_folder_path, exist_ok=True)

        # 检查并创建子目录 JSON 文件
        sub_json_file = os.path.join(sub_folder_path, f'{sub_folder_name}_vector_id.json')
        if not os.path.exists(sub_json_file):
            with open(sub_json_file, 'w') as f:
                json.dump({"vector_db_id": None, "knowledge_base_description": ""}, f, indent=4)

        return sub_folder_path
    else:
        return base_path

def update_vector_db_mapping(sub_folder_name, vector_db_id):
    # 确保主目录和子目录及其JSON文件存在
    sub_folder_path = create_knowledge_base_folder(sub_folder_name)
    
    # 获取主目录路径和JSON文件路径
    base_path = os.path.dirname(sub_folder_path)
    main_json_file = os.path.join(base_path, 'main_vector_db_mapping.json')
    
    # 更新主目录JSON文件
    with open(main_json_file, 'r') as f:
        main_mapping = json.load(f)
    
    main_mapping[sub_folder_name] = vector_db_id
    
    with open(main_json_file, 'w') as f:
        json.dump(main_mapping, f, indent=4)
    
    # 更新子目录JSON文件
    sub_json_file = os.path.join(sub_folder_path, f'{sub_folder_name}_vector_id.json')
    
    with open(sub_json_file, 'r') as f:
        sub_mapping = json.load(f)
    
    sub_mapping["vector_db_id"] = vector_db_id
    
    with open(sub_json_file, 'w') as f:
        json.dump(sub_mapping, f, indent=4)  

def create_knowledge_base_folder(sub_folder_name=None):
    # 获取 KNOWLEDGE_LIBRARY_PATH 环境变量
    knowledge_library_path = os.getenv('KNOWLEDGE_LIBRARY_PATH')
    
    # 检查 KNOWLEDGE_LIBRARY_PATH 是否存在且路径是否有效
    if knowledge_library_path and os.path.exists(knowledge_library_path):
        base_path = os.path.join(knowledge_library_path, 'knowledge_base')
    else:
        # 如果 KNOWLEDGE_LIBRARY_PATH 不存在或路径无效，则在 home 目录下创建文件夹
        home_dir = str(Path.home())
        base_path = os.path.join(home_dir, 'knowledge_base')

    # 创建 base_path 文件夹
    os.makedirs(base_path, exist_ok=True)

    # 检查并创建主目录 JSON 文件
    main_json_file = os.path.join(base_path, 'main_vector_db_mapping.json')
    if not os.path.exists(main_json_file):
        with open(main_json_file, 'w') as f:
            json.dump({}, f, indent=4)

    # 如果 sub_folder_name 不为空，则在 base_path 内创建子文件夹
    if sub_folder_name:
        sub_folder_path = os.path.join(base_path, sub_folder_name)
        os.makedirs(sub_folder_path, exist_ok=True)

        # 检查并创建子目录 JSON 文件
        sub_json_file = os.path.join(sub_folder_path, f'{sub_folder_name}_vector_id.json')
        if not os.path.exists(sub_json_file):
            with open(sub_json_file, 'w') as f:
                json.dump({"vector_db_id": None, "knowledge_base_description": ""}, f, indent=4)

        return sub_folder_path
    else:
        return base_path
    
def get_specific_files(folder_path):
    # 指定需要过滤的文件扩展名
    file_extensions = ['.md', '.pdf', '.doc', '.docx', '.ppt', '.pptx']
    
    file_paths = [
        os.path.join(folder_path, file) 
        for file in os.listdir(folder_path) 
        if os.path.isfile(os.path.join(folder_path, file)) and any(file.endswith(ext) for ext in file_extensions)
    ]
    return file_paths

def get_formatted_file_list(folder_path):
    # 获取指定文件夹内的特定文件类型的文件路径
    file_paths = get_specific_files(folder_path)
    
    # 提取文件名并去掉扩展名
    file_names = [os.path.splitext(os.path.basename(file_path))[0] for file_path in file_paths]
    
    # 将文件名用顿号分隔并组合成一个字符串
    formatted_file_list = '、'.join(file_names)
    
    # 构建最终输出字符串
    result = f"当前你的知识库包含文档标题如下：{formatted_file_list}。当用户所提出的问题和你的知识库文档内容相关时，请先检索你的知识库再进行回答。"
    
    return result

def remove_knowledge_base_info(text):
    keyword = "当前你的知识库包含文档标题如下："
    
    # 查找关键字的位置
    index = text.find(keyword)
    
    # 如果关键字存在，删除该关键字及其之后的所有字符
    if index != -1:
        return text[:index]
    
    # 如果关键字不存在，返回原始字符串
    return text

def create_knowledge_base(client, knowledge_base_name, folder_path_base = None):
    
    print("正在创建知识库，请稍后...")
    sub_folder_name = knowledge_base_name
    if folder_path_base == None:
        folder_path = create_knowledge_base_folder(sub_folder_name=knowledge_base_name)
    else:
        folder_path = folder_path_base
    
    knowledge_base_name = knowledge_base_name + '!!' + client.api_key[8: ]
    vector_stores = client.beta.vector_stores.list()
    
    vector_id = None
    # expires_after = {
        # "anchor": "last_active_at",  
        # "days": 1  
    # }
    
    for vs in vector_stores.data:
        if vs.name == knowledge_base_name:
            vector_store_files = client.beta.vector_stores.files.list(
                vector_store_id=vs.id
            )
            for file in vector_store_files.data:
                file = client.beta.vector_stores.files.delete(
                    vector_store_id=vs.id,
                    file_id=file.id
                )
                client.files.delete(file.id)
            vector_id = vs.id
       
    print("正在创建知识库的向量存储，请稍后...")
    if vector_id == None:
        vector_store = client.beta.vector_stores.create(name=knowledge_base_name)
        
        vector_id = vector_store.id
        
    try:
        file_paths = get_specific_files(folder_path)
        file_streams = [open(path, "rb") for path in file_paths]
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_id, files=file_streams
        )
        knowledge_base_description = get_formatted_file_list(folder_path)
        update_knowledge_base_description(sub_folder_name=sub_folder_name, description=knowledge_base_description)
        
    except Exception as e:
        print(f"发生错误: {e}")
        print("知识库无法创建，请再次确认知识库文件夹中存在格式合规的文件")
        return None
    
    # dotenv_path = find_dotenv()
    # if not dotenv_path:
        # with open('.env', 'w', encoding='utf-8') as f:
            # pass
        # dotenv_path = find_dotenv()


    # load_dotenv(dotenv_path)
    # specific_base_var = knowledge_base_name + "_vector_id"
    # os.environ[specific_base_var] = vector_id

    # set_key(dotenv_path, specific_base_var, os.environ[specific_base_var])
    if knowledge_base_name != None:
        update_vector_db_mapping(sub_folder_name=sub_folder_name, 
                                 vector_db_id=vector_id)
    print("知识库创建完成！")
    return vector_id

def clear_folder(folder_path):
    """
    删除指定文件夹内的全部内容
    :param folder_path: 要清空的文件夹路径
    """
    # 检查文件夹是否存在
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # 删除文件夹内的全部内容
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或符号链接
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除目录
            except Exception as e:
                pass
                # print(f"Failed to delete {file_path}. Reason: {e}")
        # print(f"清空了文件夹: {folder_path}")
    else:
        # print(f"文件夹不存在或不是一个目录: {folder_path}")
        pass

def create_competition_knowledge_base(competition_name, client):
    try:
        load_dotenv()
        headers_json = os.getenv('HEADERS')
        cookies_json = os.getenv('COOKIES')
        
        headers = json.loads(headers_json)
        cookies = json.loads(cookies_json)
        
        _id = get_id(keyword=competition_name)
        
        if _id:
            folder_path = create_kaggle_project_directory(competition_name)
            print(f"已找到指定竞赛{competition_name}，正在检索支持库是否存在竞赛信息...")
            knowledge_base_name = competition_name + '!!' + client.api_key[8:]
            knowledge_base_check = check_knowledge_base_name(client=client, knowledge_base_name=knowledge_base_name)
            if knowledge_base_check:
                user_input = input('检测到存在该赛题知识库，是否更新知识库（1），或者直接使用该知识库（2）：')
                if user_input == '2':
                    return knowledge_base_check
                else:
                    print("即将更新知识库...")
                    # create_knowledge_base(client=client, knowledge_base_name=competition_name)
                    # client.beta.vector_stores.delete(vector_store_id=knowledge_base_check)
                    clear_folder(create_kaggle_project_directory(competition_name))
            
            print("正在准备构建知识库...")
            create_kaggle_project_directory(competition_name = competition_name)
            overview, data_description = getOverviewAndDescription(_id)
            print("正在获取竞赛说明及数据集说明...")
            overview_md = convert_html_to_markdown(json_to_markdown(overview))
            data_description_md = convert_html_to_markdown(json_to_markdown(data_description))
            save_markdown(content=overview_md, competition_name=competition_name, file_type='overview')
            save_markdown(content=data_description_md, competition_name=competition_name, file_type='data_description')
            print(f"正在获取{competition_name}竞赛热门kernel...")
            json_urls = get_code(_id)
            urls = extract_and_transform_urls(json_urls=json_urls)
            res = download_and_convert_kernels(urls=urls, competition_name=competition_name)
            print("知识文档创建完成，正在进行词向量化处理与存储，请稍后...")
            # home_dir = str(Path.home())
            # folder_path = os.path.join(os.path.expanduser(home_dir), f'.kaggle./{competition_name}_project/knowledge_library')
            
            # vector_store = client.beta.vector_stores.create(name=knowledge_base_name)
            # file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
            # md_files = [file for file in file_paths if file.endswith('.md')]
            # file_streams = [open(path, "rb") for path in md_files]
            # file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                # vector_store_id=vector_store.id, files=file_streams
            # )
            vector_store_id = create_knowledge_base(client=client, knowledge_base_name=competition_name)
            print("已顺利完成Kaggle竞赛知识库创建，后续可调用知识库回答。")
            return vector_store_id
        else:
            print("找不到对应的竞赛，请检查竞赛名称再试。")
            return None
    except Exception as e:
        print("服务器拥挤，请稍后再试...")
        return None
    


def python_inter(py_code, g='globals()'):
    """
    专门用于执行python代码，并获取最终查询或处理结果。
    :param py_code: 字符串形式的Python代码，
    :param g: g，字符串形式变量，表示环境变量，无需设置，保持默认参数即可
    :return：代码运行的最终结果
    """    
    try:
        # 尝试如果是表达式，则返回表达式运行结果
        return str(eval(py_code, g))
    # 若报错，则先测试是否是对相同变量重复赋值
    except Exception as e:
        global_vars_before = set(g.keys())
        try:            
            exec(py_code, g)
        except Exception as e:
            return f"代码执行时报错{e}"
        global_vars_after = set(g.keys())
        new_vars = global_vars_after - global_vars_before
        # 若存在新变量
        if new_vars:
            result = {var: g[var] for var in new_vars}
            return str(result)
        else:
            return "已经顺利执行代码"
             

def sql_inter(sql_query, host, user, password, database, port, g=globals()):
    """
    用于执行一段SQL代码，并最终获取SQL代码执行结果，
    核心功能是将输入的SQL代码传输至MySQL环境中进行运行，
    并最终返回SQL代码运行结果。需要注意的是，本函数是借助pymysql来连接MySQL数据库。
    :param sql_query: 字符串形式的SQL查询语句，用于执行对MySQL中telco_db数据库中各张表进行查询，并获得各表中的各类相关信息
    :param host: MySQL服务器的主机名
    :param user: MySQL服务器的用户名
    :param password: MySQL服务器的密码
    :param database: MySQL服务器的数据库名
    :param port: MySQL服务器的端口
    :param g: g，字符串形式变量，表示环境变量，无需设置，保持默认参数即可
    :return：sql_query在MySQL中的运行结果。
    """   
    connection = pymysql.connect(
        host=host,
        user=user,
        passwd=password,
        db=database,
        port=int(port),
        charset='utf8',
    )
    
    try:
        with connection.cursor() as cursor:
            sql = sql_query
            cursor.execute(sql)
            results = cursor.fetchall()
    finally:
        connection.close()

    return json.dumps(results)

def extract_data(sql_query, df_name, host, user, password, database, port, g=globals()):
    """
    借助pymysql将MySQL中的某张表读取并保存到本地Python环境中。
    :param sql_query: 字符串形式的SQL查询语句，用于提取MySQL中的某张表。
    :param df_name: 将MySQL数据库中提取的表格进行本地保存时的变量名，以字符串形式表示。
    :param host: MySQL服务器的主机名
    :param user: MySQL服务器的用户名
    :param password: MySQL服务器的密码
    :param database: MySQL服务器的数据库名
    :param port: MySQL服务器的端口
    :param g: g，字符串形式变量，表示环境变量，无需设置，保持默认参数即可
    :return：表格读取和保存结果
    """
    connection = pymysql.connect(
        host=host,
        user=user,
        passwd=password,
        db=database,
        port=int(port),
        charset='utf8',
    )
    
    g[df_name] = pd.read_sql(sql_query, connection)
    
    return "已成功完成%s变量创建" % df_name

def generate_object_name(base_name="fig", use_uuid=True):
    """
    生成对象名称，可以选择使用UUID或日期时间。

    :param base_name: 基础名称
    :param use_uuid: 是否使用UUID
    :return: 生成的对象名称
    """
    if use_uuid:
        object_name = f"{base_name}_{uuid.uuid4().hex}.png"
    else:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        object_name = f"{base_name}_{current_time}.png"
    return object_name

def upload_fig_to_oss(fig, object_name=None):
    """
    上传fig对象到阿里云OSS并返回图片的URL

    :param fig: Matplotlib的fig对象
    :param object_name: OSS中的文件路径和名称
    :return: 上传后图片的URL
    """
    if object_name is None:
        object_name = generate_object_name()

    try:  
        access_key_id = os.getenv('ACCESS_KEY_ID')
        access_key_secret = os.getenv('ACCESS_KEY_SECRET')
        endpoint = os.getenv('ENDPOINT')
        bucket_name = os.getenv('BUCKET_NAME')
        
        auth = oss2.Auth(access_key_id, access_key_secret)
        bucket = oss2.Bucket(auth, endpoint, bucket_name)
        
        # 将fig对象保存到内存中的字节流
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)  # 将字节流指针重置到起始位置
        
        # 上传字节流到OSS
        bucket.put_object(object_name, buffer)

        # 构建图片URL
        match = re.search(r'(oss[^/]*)', bucket.endpoint)
        endpoint_url = match.group(1)
        url = f"https://{bucket.bucket_name}.{endpoint_url}/{object_name}"
        # print(f"Figure uploaded to OSS as {object_name}")
        return url
    except Exception as e:
        # print(f"Failed to upload figure: {e}")
        return None

def fig_inter(py_code, g='globals()'):
    """
    用于执行一段包含可视化绘图的Python代码
    :param py_code: 字符串形式的Python代码，用于根据需求进行绘图，代码中必须创建名为fig的Figure对象。
    :param g: g，字符串形式变量，表示环境变量，无需设置，保持默认参数即可
    :return：代码运行的最终结果，若顺利创建图片并上传至oss，则返回图片的url地址
    """    
    # 保存当前的后端
    # current_backend = matplotlib.get_backend()
    
    # 设置为Agg后端
    # matplotlib.use('Agg')
    
    # 创建一个字典，用于存储本地变量
    local_vars = {"plt": plt, "pd": pd, "sns": sns}
    
    try:
        exec(py_code, g, local_vars)
    except Exception as e:
        return f"代码执行时报错: {e}"
    # finally:
        # 恢复默认后端
        # matplotlib.use(current_backend)
    
    # 根据图片名称，获取图片对象
    try:
        fig = local_vars['fig']
    except KeyError:
        return "未找到名为'fig'的Figure对象，请确保py_code中创建了一个名为'fig'的Figure对象。"
    
    
    # 上传图片
    try:
        fig_url = upload_fig_to_oss(fig)
        if fig_url != None:
            markdown_text = f"![Image]({fig_url})"
            # display(Markdown(markdown_text))
            res = f"已经成功运行代码，并已将代码创建的图片存储至：{fig_url}"
        else:
            res = "图像已顺利创建并成功打印，函数已顺利执行。"
        
    except Exception as e:
        res = "图像已顺利创建并成功打印，函数已顺利执行。"
        
    print(res)
    return res

# create_knowledge_base_folder(sub_folder_name=None)

def is_folder_not_empty(knowledge_base_name):
    knowledge_library_path = os.getenv('KNOWLEDGE_LIBRARY_PATH')
    
    # 检查 KNOWLEDGE_LIBRARY_PATH 是否存在且路径是否有效
    if knowledge_library_path and os.path.exists(knowledge_library_path):
        base_path = os.path.join(knowledge_library_path, 'knowledge_base')
    else:
        # 如果 KNOWLEDGE_LIBRARY_PATH 不存在或路径无效，则在 home 目录下创建文件夹
        home_dir = str(Path.home())
        base_path = os.path.join(home_dir, 'knowledge_base')
    
    # 目标文件夹路径
    target_folder_path = os.path.join(base_path, knowledge_base_name)
    
    # 检查目标文件夹是否存在
    if not os.path.exists(target_folder_path) or not os.path.isdir(target_folder_path):
        print(f"目标文件夹 {target_folder_path} 不存在或不是一个文件夹。")
        return False
    
    # 遍历目标文件夹中的文件
    for root, dirs, files in os.walk(target_folder_path):
        for file in files:
            if not file.endswith('.json'):
                return True
    
    return False

def google_search(query, num_results=10, site_url=None):
    api_key = os.getenv('GOOGLE_SEARCH_KEY')
    cse_id = os.getenv('CSE_ID')
    base_url = os.getenv("GOOGLE_SEARCH_BASE_URL")
    
    if base_url:
        url = base_url
    else:
        url = "https://www.googleapis.com/customsearch/v1"
    
    # API 请求参数
    if site_url == None:
        params = {
        'q': query,          
        'api_key': api_key,      
        'cse_id': cse_id,        
        'num_results': num_results   
        }
    else:
        params = {
        'q': query,         
        'api_key': api_key,      
        'cse_id': cse_id,        
        'num_results': num_results,  
        'site_search': site_url
        }

    # 发送请求
    response = requests.get(url, params=params)
    response.raise_for_status()

    # 解析响应
    search_results = response.json().get('items', [])

    # 提取所需信息
    results = [{
        'title': item['title'],
        'link': item['link'],
        # 'snippet': item['snippet']
    } for item in search_results]

    return results

def windows_compatible_name(s, max_length=255):
    """
    将字符串转化为符合Windows文件/文件夹命名规范的名称。
    
    参数:
    - s (str): 输入的字符串。
    - max_length (int): 输出字符串的最大长度，默认为255。
    
    返回:
    - str: 一个可以安全用作Windows文件/文件夹名称的字符串。
    """

    # Windows文件/文件夹名称中不允许的字符列表
    forbidden_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']

    # 使用下划线替换不允许的字符
    for char in forbidden_chars:
        s = s.replace(char, '_')

    # 删除尾部的空格或点
    s = s.rstrip(' .')

    # 检查是否存在以下不允许被用于文档名称的关键词，如果有的话则替换为下划线
    reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", 
                      "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"]
    if s.upper() in reserved_names:
        s += '_'

    # 如果字符串过长，进行截断
    if len(s) > max_length:
        s = s[:max_length]

    return s

def get_search_text(q, url):
    cookie = os.getenv('ZHIHU_SEARCH_COOKIE')
    user_agent = os.getenv('ZHIHU_SEARCH_USER_AGENT')    
    title = None
    
    code_ = False
    headers = {
        'authority': 'www.zhihu.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'cache-control': 'max-age=0',
        'cookie': cookie,
        'upgrade-insecure-requests': '1',
        'user-agent':user_agent,
    }

    # 普通问答地址
    if 'zhihu.com/question' in url:
        res = requests.get(url, headers=headers).text
        res_xpath = etree.HTML(res)
        title = res_xpath.xpath('//div/div[1]/div/h1/text()')[0]
        text_d = res_xpath.xpath('//div/div/div/div[2]/div/div[2]/div/div/div[2]/span[1]/div/div/span/p/text()')
    
    # 专栏地址
    elif 'zhuanlan' in url:
        headers['authority'] = 'zhaunlan.zhihu.com'
        res = requests.get(url, headers=headers).text
        res_xpath = etree.HTML(res)
        title = res_xpath.xpath('//div[1]/div/main/div/article/header/h1/text()')[0]
        text_d = res_xpath.xpath('//div/main/div/article/div[1]/div/div/div/p/text()')
        code_ = res_xpath.xpath('//div/main/div/article/div[1]/div/div/div//pre/code/text()')  
            
    # 特定回答的问答网址
    elif 'answer' in url:
        res = requests.get(url, headers=headers).text
        res_xpath = etree.HTML(res)
        title = res_xpath.xpath('//div/div[1]/div/h1/text()')[0]
        text_d = res_xpath.xpath('//div[1]/div/div[3]/div/div/div/div[2]/span[1]/div/div/span/p/text()')

    if title == None:
        return None
    
    else:
        title = windows_compatible_name(title)

        # 创建问题答案正文
        text = ''
        for t in text_d:
            txt = str(t).replace('\n', ' ')
            text += txt

        # 如果有code，则将code追加到正文的追后面
        if code_:
            for c in code_:
                co = str(c).replace('\n', ' ')    
                text += co

        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")     
        json_data = [
            {
                "link": url,
                "title": title,
                "content": text,
                "tokens": len(encoding.encode(text))
            }
        ]

        path = create_knowledge_base_folder(sub_folder_name='auto_search')
        # 使用 os.path.join 构建文件路径
        file_path = os.path.join(path, q, f"{title}.json")

        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 打开文件并写入数据
        with open(file_path, 'w') as f:
            json.dump(json_data, f)

        return title

def get_answer(q, g='globals()'):
    """
    当你无法回答某个问题时，调用该函数，能够获得答案
    :param q: 必选参数，询问的问题，字符串类型对象
    :return：某问题的答案，以字符串形式呈现
    """
    # 调用转化函数，将用户的问题转化为更适合在知乎上进行搜索的关键词
    q = convert_keyword(q)
    
    # 默认搜索返回10个答案
    print('正在接入谷歌搜索，查找和问题相关的答案...')
    results = google_search(query=q, num_results=10, site_url='https://zhihu.com/')
    
    # 创建对应问题的子文件夹
    path = create_knowledge_base_folder(sub_folder_name='auto_search')
    folder_path = os.path.join(path, q)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    clear_folder(folder_path)
    
    # 单独提取links放在一个list中
    print('正在读取搜索的到的相关答案...')
    num_tokens = 0
    content = ''
    for item in results:
        url = item['link']
        title = get_search_text(q, url)
        file_path = os.path.join(folder_path, f"{title}.json")
        with open(file_path, 'r') as f:
            jd = json.load(f)
        num_tokens += jd[0]['tokens']
        if num_tokens <= 12000:
            content += jd[0]['content']
        else:
            break
    print('正在进行最后的整理...')
    return content

def get_search_text_github(q, dic, path):
    title = dic['owner'] + '_' + dic['repo']
    title = windows_compatible_name(title)

    # 创建问题答案正文
    text = get_github_readme(dic)

    # 写入本地json文件
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    json_data = [
        {
            "title": title,
            "content": text,
            "tokens": len(encoding.encode(text))
        }
    ]

    folder_path = os.path.join(path, q)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, f"{title}.json")
    with open(file_path, 'w') as f:
        json.dump(json_data, f)

    return title


def get_github_readme(dic):
    
    github_token = os.getenv('GITHUB_TOKEN')
    user_agent = os.getenv('ZHIHU_SEARCH_USER_AGENT')
    
    owner = dic['owner']
    repo = dic['repo']

    headers = {
        "Authorization": github_token,
        "User-Agent": user_agent
    }

    response = requests.get(f"https://api.github.com/repos/{owner}/{repo}/readme", headers=headers)

    readme_data = response.json()
    encoded_content = readme_data.get('content', '')
    decoded_content = base64.b64decode(encoded_content).decode('utf-8')
    
    return decoded_content

def extract_github_repos(search_results):
    # 使用列表推导式筛选出项目主页链接
    repo_links = [result['link'] for result in search_results if '/issues/' not in result['link'] and '/blob/' not in result['link'] and 'github.com' in result['link'] and len(result['link'].split('/')) == 5]

    # 从筛选后的链接中提取owner和repo
    repos_info = [{'owner': link.split('/')[3], 'repo': link.split('/')[4]} for link in repo_links]

    return repos_info

def get_answer_github(q, g='globals()'):
    """
    当你无法回答某个问题时，调用该函数，能够获得答案
    :param q: 必选参数，询问的问题，字符串类型对象
    :return：某问题的答案，以字符串形式呈现
    """
    # 调用转化函数，将用户的问题转化为更适合在GitHub上搜索的关键词
    q = convert_keyword_github(q)
    
    # 默认搜索返回10个答案
    print('正在接入谷歌搜索，并在GitHub上搜索相关项目...')
    search_results = google_search(query=q, num_results=10, site_url='https://github.com/')
    results = extract_github_repos(search_results)
    
    # 创建对应问题的子文件夹
    path = create_knowledge_base_folder(sub_folder_name='auto_search')
    folder_path = os.path.join(path, q)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    clear_folder(folder_path)
    print('正在读取相关项目说明文档...')
    num_tokens = 0
    content = ''
    
    for dic in results:
        title = get_search_text_github(q, dic, path)
        file_path = os.path.join(folder_path, f"{title}.json")
        with open(file_path, 'r') as f:
            jd = json.load(f)
        num_tokens += jd[0]['tokens']
        if num_tokens <= 12000:
            content += jd[0]['content']
        else:
            break
    print('正在进行最后的整理...')
    return content
    
def get_vector_db_id(knowledge_base_name):
    # 获取 KNOWLEDGE_LIBRARY_PATH 环境变量
    knowledge_library_path = os.getenv('KNOWLEDGE_LIBRARY_PATH')
    
    # 检查 KNOWLEDGE_LIBRARY_PATH 是否存在且路径是否有效
    if knowledge_library_path and os.path.exists(knowledge_library_path):
        base_path = os.path.join(knowledge_library_path, 'knowledge_base')
    else:
        # 如果 KNOWLEDGE_LIBRARY_PATH 不存在或路径无效，则在 home 目录下创建文件夹
        home_dir = str(Path.home())
        base_path = os.path.join(home_dir, 'knowledge_base')
        
    # 创建 base_path 文件夹
    os.makedirs(base_path, exist_ok=True)
    
    # 检查主目录 JSON 文件
    main_json_file = os.path.join(base_path, 'main_vector_db_mapping.json')
    if not os.path.exists(main_json_file):
        with open(main_json_file, 'w') as f:
            json.dump({}, f, indent=4)
        return None
    
    # 读取主目录 JSON 文件
    with open(main_json_file, 'r') as f:
        main_mapping = json.load(f)
    
    # 检索对应的词向量数据库ID
    vector_db_id = main_mapping.get(knowledge_base_name)
    
    return vector_db_id    

def wait_for_vector_store_ready(vs_id, client, interval=3, max_attempts=20):
    attempt = 0
    while attempt < max_attempts:
        vector_store = client.beta.vector_stores.retrieve(vs_id)
        if vector_store.status == 'completed':
            return True
        time.sleep(interval)
        attempt += 1
    return False    
        
def export_variables():
    return globals()


def check_network_environment():
    print("正在检测MateGen联网环境。MateGen支持连接谷歌搜索、知乎、Github和Kaggle，且可以借助阿里云oss对象存储，将生成的图片存储到oss上。\n\
    默认配置文件下除oss外上述功能可以正常运行，但伴随时间推移，部分cookie可能会失效。\n\
    因此，MateGen也支持用户自定义修改相关配置。即将开启联网功能检测...")
    
    print("即将检测谷歌搜索API能否正常运行")
    if test_google_search_api():
        print("谷歌搜索API可正常运行")
    else:
        print("谷歌搜索API无法正常运行，请调用internet_config_assistant()函数获取帮助文档并进行设置")
            
    print("即将检测能否连接知乎进行问答")
    if test_spider_response():
        print("可以连接知乎进行问答")
    else:
        print("无法连接知乎进行问答，请调用internet_config_assistant()函数获取帮助文档并进行设置")
        
    print("即将检测能否连接Github进行问答")
    if test_github_token():
        print("可以连接Github进行问答")
    else:
        print("无法连接Github进行问答，请调用internet_config_assistant()函数获取帮助文档并进行设置")            

    print("即将检测能否连接Kaggle进行问答")
    if test_kaggle_api():
        print("可以连接Kaggle进行问答")
    else:
        print("无法连接Kaggle进行问答，请调用internet_config_assistant()函数获取帮助文档并进行设置")
        
    print("即将检测阿里云oss对象存储设置是否成功")
    if test_oss_api():
        print("阿里云oss对象存储设置成功")
    else:
        print("阿里云oss对象存储设置不成功，，请调用internet_config_assistant()函数获取帮助文档并进行设置")          
        
def internet_config_assistant():
    user_input = input("你好，欢迎MateGen互联网配置助手，请输入你希望获取的帮助类型：1.获取帮助文档；2.进行互联网参数配置；0.退出")
    if user_input == '1':
        user_input = input("请输入想要获得的帮助文档类型:\
        1.谷歌搜索API获取与配置方法；\n\
        2.知乎搜索cookie获取与配置方法；\n\
        3.Github token获取与配置方法； \n\
        4.Kaggle搜索cookie获取与配置方法；\n\
        5.阿里云oss对象存储配置方法；\n\
        6.返回上一级 \n\
        0.退出")
        if user_input == '1':
            print("以下是《谷歌搜索API获取与配置方法》详情")
            fetch_and_display_markdown(url1)
        elif user_input == '2':
            print("以下是《知乎搜索cookie获取与配置方法》详情")
            fetch_and_display_markdown(url2)
        elif user_input == '3':
            print("以下是《Github token获取与配置方法》详情")
            fetch_and_display_markdown(url3)
        elif user_input == '4':
            print("以下是《Kaggle搜索cookie获取与配置方法》详情")
            fetch_and_display_markdown(url4)
        elif user_input == '5':
            print("以下是《阿里云oss对象存储配置方法》详情")
            fetch_and_display_markdown(url5)
        elif user_input == '6':
            internet_config_assistant()      
        else:
            return None 
    elif user_input == '2':
        user_input = input("想要进行哪方面设置？\
        1.谷歌搜索API配置；\n\
        2.知乎搜索cookie配置；\n\
        3.Github token配置； \n\
        4.Kaggle搜索cookie配置。\n\
        5.阿里云oss对象存储配置；\n\
        6.返回上一级 \n\
        0.退出")
        if user_input == '1':
            google_search_key = input("请输入google_search_key")
            cse_id = input("请输入cse_id")
            write_google_env(google_search_key, cse_id)
            print("即将检测谷歌搜索API设置是否成功")
            if test_google_search_api():
                print("谷歌搜索API设置成功")
            else:
                print("谷歌搜索API设置不成功，请重新尝试")
        elif user_input == '2':
            cookie = input("请输入cookie")
            user_agent = input("请输入user_agent")
            write_zhihu_env(cookie, user_agent)
            print("即将检测知乎cookie设置是否成功")
            if test_spider_response():
                print("知乎cookie设置成功")
            else:
                print("知乎cookie设置不成功，请重新尝试")
        elif user_input == '3':
            github_token = input("请输入github_token")
            write_github_token_to_env(github_token)
            print("即将检测github_token设置是否成功")
            if test_github_token():
                print("github_token设置成功")
            else:
                print("github_token设置不成功，请重新尝试")            
        elif user_input == '4':
            headers = input("请输入headers")
            cookies = input("请输入cookies")
            write_kaggle_env(headers, cookies)
            print("即将检测Kaggle搜索cookie设置是否成功")
            if test_kaggle_api():
                print("Kaggle搜索cookie设置成功")
            else:
                print("Kaggle搜索cookie设置不成功，请重新尝试")
        elif user_input == '5':
            access_key_id = input("请输入access_key_id")
            access_key_secret = input("请输入access_key_secret")
            endpoint = input("请输入endpoint")
            bucket_name = input("请输入bucket_name")
            write_oss_env(access_key_id, access_key_secret, endpoint, bucket_name)
            print("即将检测阿里云oss对象存储设置是否成功")
            if test_oss_api():
                print("阿里云oss对象存储设置成功")
            else:
                print("阿里云oss对象存储设置不成功，请重新尝试")       
        elif user_input == '6':
            internet_config_assistant()
        else:
            return None 

# 谷歌搜索测试
def test_google_search_api(search_term='openai'):
    """
    测试谷歌搜索API是否能够顺利连接并正常获得响应。
    """
    url = "https://www.googleapis.com/customsearch/v1"
    google_search_key = os.getenv('GOOGLE_SEARCH_KEY')
    cse_id = os.getenv('CSE_ID')
    base_url = os.getenv("GOOGLE_SEARCH_BASE_URL")
    
    if base_url:
        url = base_url
    else:
        url = "https://www.googleapis.com/customsearch/v1"

    params = {
        'q': search_term,
        'api_key': google_search_key,
        'cse_id': cse_id
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"请求失败，HTTP状态码：{response.status_code}")
            return False
        
        data = response.json()
        if 'items' in data:
            print("测试成功：API响应包含预期的数据。")
            return True
        else:
            print("测试失败：API响应不包含预期的数据。")
            return False
    
    except Exception as e:
        print(f"测试失败：发生异常 - {e}")
        return False
    
def write_google_env(google_search_key, cse_id):
    """
    将google_search_key和cse_id写入当前项目文件夹内的.env文件中，作为环境变量。
    """
    if load_dotenv(dotenv_path):
    
        set_key(dotenv_path, 'GOOGLE_SEARCH_KEY', google_search_key)
        set_key(dotenv_path, 'CSE_ID', cse_id)
        
        print(f".env 文件已更新: GOOGLE_SEARCH_KEY={google_search_key}, CSE_ID={cse_id}")
    else:
        print("找不到配置文件，请先编写配置文件")
        return None
    

# 知乎爬虫测试  
def test_spider_response(test_url='https://www.zhihu.com/question/589955237'):
    """
    测试某个爬虫响应是否顺利。
    """
    cookie = os.getenv('ZHIHU_SEARCH_COOKIE')
    user_agent = os.getenv('ZHIHU_SEARCH_USER_AGENT')
    
    if not cookie or not user_agent:
        print("环境变量 ZHIHU_SEARCH_COOKIE 或 ZHIHU_SEARCH_USER_AGENT 未设置")
        return False
    
    # 构建请求头
    headers = {
        'authority': 'www.zhihu.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'cache-control': 'max-age=0',
        'cookie': cookie,
        'upgrade-insecure-requests': '1',
        'user-agent': user_agent,
    }

    try:
        # 发送GET请求
        response = requests.get(test_url, headers=headers)
        
        # 检查HTTP状态码是否为200（OK）
        if response.status_code == 200:
            print("测试成功：爬虫响应正常。")
            return True
        else:
            print(f"测试失败，HTTP状态码：{response.status_code}")
            return False
    
    except Exception as e:
        print(f"测试失败：发生异常 - {e}")
        return False

def write_zhihu_env(cookie, user_agent):
    """
    将ZHIHU_SEARCH_COOKIE和ZHIHU_SEARCH_USER_AGENT写入当前项目文件夹内的.env文件中，作为环境变量。
    """
    if load_dotenv(dotenv_path):
        set_key(dotenv_path, 'ZHIHU_SEARCH_COOKIE', cookie)
        set_key(dotenv_path, 'ZHIHU_SEARCH_USER_AGENT', user_agent)      
        print(f".env 文件已更新: ZHIHU_SEARCH_COOKIE={cookie}, ZHIHU_SEARCH_USER_AGENT={user_agent}")
        
    else:
        print("找不到配置文件，请先编写配置文件")
        return None

# Github连接测试
def test_github_token(owner="THUDM", repo="P-tuning-v2"):
    """
    测试 GitHub token 是否能正常使用。
    """
    
    github_token = os.getenv('GITHUB_TOKEN')
    user_agent = os.getenv('ZHIHU_SEARCH_USER_AGENT')
    # 构建请求头
    headers = {
        "Authorization": f"token {github_token}",
        "User-Agent": user_agent
    }

    # 构建请求 URL
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"

    try:
        # 发送 GET 请求
        response = requests.get(url, headers=headers)
        
        # 检查 HTTP 状态码是否为 200（OK）
        if response.status_code == 200:
            print("测试成功：GitHub token 正常工作。")
            return True
        else:
            print(f"测试失败，HTTP 状态码：{response.status_code}")
            return False
    
    except Exception as e:
        print(f"测试失败：发生异常 - {e}")
        return False
    
def write_github_token_to_env(github_token):
    """
    将 GitHub token 写入当前项目文件夹内的 .env 文件中，作为环境变量。
    """

    if load_dotenv(dotenv_path):
        set_key(dotenv_path, 'GITHUB_TOKEN', github_token)

        print(f".env 文件已更新: GITHUB_TOKEN={github_token}")
    else:
        print("找不到配置文件，请先编写配置文件")
        return None
       
# 阿里云配置
def test_oss_api():
    """
    测试阿里云OSS API能否正常调用。
    """
    access_key_id = os.getenv('ACCESS_KEY_ID')
    access_key_secret = os.getenv('ACCESS_KEY_SECRET')
    endpoint = os.getenv('ENDPOINT')
    bucket_name = os.getenv('BUCKET_NAME')
    
    try:
        # 初始化OSS客户端
        
        auth = oss2.Auth(access_key_id, access_key_secret)
        bucket = oss2.Bucket(auth, endpoint, bucket_name)

        # 测试列出存储空间中的所有文件
        file_list = list(oss2.ObjectIterator(bucket))
        
        print(f"测试成功：列出了存储空间中的文件，共计 {len(file_list)} 个文件。")
        return True
    
    except Exception as e:
        print(f"测试失败：OSS API 调用失败 - {e}")
        return False
    
def write_oss_env(access_key_id, access_key_secret, endpoint, bucket_name):
    """
    将阿里云OSS的相关变量写入当前项目文件夹内的.env文件中，作为环境变量。
    """
    
    if load_dotenv(dotenv_path):
        set_key(dotenv_path, 'ACCESS_KEY_ID', access_key_id)
        set_key(dotenv_path, 'ACCESS_KEY_SECRET', access_key_secret)
        set_key(dotenv_path, 'ENDPOINT', endpoint)
        set_key(dotenv_path, 'BUCKET_NAME', bucket_name)

        print(f".env 文件已更新: OSS_ACCESS_KEY_ID={access_key_id}, OSS_ACCESS_KEY_SECRET={access_key_secret}, OSS_ENDPOINT={endpoint}, OSS_BUCKET_NAME={bucket_name}")
    else:
        print("找不到配置文件，请先编写配置文件")
        return None

# Kaggle配置    
def test_kaggle_api():
    """
    测试Kaggle API是否能连接。

    :return: 测试结果 (True表示成功，False表示失败)
    """
    
    headers_json = os.getenv('HEADERS')
    cookies_json = os.getenv('COOKIES')
    
    if not headers_json or not cookies_json:
        print("环境变量 HEADERS 或 COOKIES 未设置")
        return False

    # 解析JSON字符串为字典
    headers = json.loads(headers_json)
    cookies = json.loads(cookies_json)
    
    url = "https://www.kaggle.com/api/i/search.SearchWebService/FullSearchWeb"
    data = {
        "query": 'titanic',
        "page": 1,
        "resultsPerPage": 20,
        "showPrivate": True
    }
    data = json.dumps(data, separators=(',', ':'))

    try:
        # 发送POST请求
        response = requests.post(url, headers=headers, cookies=cookies, data=data)
        
        # 检查响应是否成功
        if response.status_code == 200:
            response_data = response.json()
            print("测试成功：Kaggle API 响应正常。")
            return True
        else:
            print(f"测试失败，HTTP状态码：{response.status_code}")
            return False

    except Exception as e:
        print(f"测试失败：发生异常 - {e}")
        return False    
    
def write_kaggle_env(headers, cookies):
    """
    将HEADERS和COOKIES写入当前项目文件夹内的.env文件中，作为环境变量。
    """
    if load_dotenv(dotenv_path):
        set_key(dotenv_path, 'HEADERS', headers)
        set_key(dotenv_path, 'COOKIES', cookies)

        print(f".env 文件已更新: HEADERS={headers}, COOKIES={cookies}")    
    else:
        print("找不到配置文件，请先编写配置文件")
        return None        
    

# 打印文档   
def fetch_and_display_markdown(url):
    """
    获取指定 URL 的文档内容，并以 Markdown 格式打印。
    """
    try:
        # 发送 GET 请求获取文档内容
        response = requests.get(url)
        
        # 检查 HTTP 状态码是否为 200（OK）
        if response.status_code == 200:
            response.encoding = 'utf-8'
            content = response.text
            
            # 以 Markdown 格式打印内容
            display(Markdown(content))
        else:
            print(f"获取文档失败，HTTP 状态码：{response.status_code}")
    
    except Exception as e:
        print(f"获取文档失败：发生异常 - {e}")        
        
def reset_base_url(api_key, base_url):
    if is_base_url_valid(api_key, base_url):
        set_key(dotenv_path, 'BASE_URL', base_url)
        print(f"更新后base_url地址：{base_url}")
    else:
        print(f"无效的base_url地址：{base_url}")    
            
def is_base_url_valid(api_key, path):
    original_string = decrypt_string(api_key, key=b'YAboQcXx376HSUKqzkTz8LK1GKs19Skg4JoZH4QUCJc=')
    split_strings = original_string.split(' ')
    s1 = split_strings[0]
    client_tmp = OpenAI(api_key = s1, 
                        base_url = path)        
    models_tmp = client_tmp.models.list()
    return models_tmp    

def set_google_search_base_url(base_url):
    if is_google_base_url_valid(base_url):
        set_key(dotenv_path, 'GOOGLE_SEARCH_KEY', base_url)
        print(f"更新后google_search_base_url地址：{base_url}")
    else:
        print(f"无效的google_search_base_url地址：{base_url}")  
        
def is_google_base_url_valid(base_url):
    api_key = os.getenv("GOOGLE_SEARCH_KEY")
    cse_id = os.getenv("CSE_ID")
    
    if api_key is None or cse_id is None:
        print("需要先设置谷歌搜索api_key和cse_id")
        return False
    else:
        try:
            PROXY_URL = base_url
            query = 'OpenAI GPT-4'
            response = requests.get(PROXY_URL, params={'api_key': api_key, 'cse_id': cse_id, 'q': query, 'num_results': 1})
            if response.status_code == 200:
                return True
            else:
                return False
        except Exception as e:
            print(f"检测中转地址时出错: {e}")
            return False
        
def delete_all_files(client):
    # 获取所有文件的列表
    files = client.files.list()

    # 逐个删除文件
    for file in files.data:
        file_id = file.id
        client.files.delete(file_id)
        # print(f"Deleted file: {file_id}")

def delete_all_vector_stores(client):
    # 获取所有词向量库的列表
    vector_stores = client.beta.vector_stores.list()

    # 逐个删除词向量库
    for vector_store in vector_stores.data:
        vector_store_id = vector_store.id
        client.beta.vector_stores.delete(vector_store_id)
        # print(f"Deleted vector store: {vector_store_id}")
        
def delete_all_assistants(client):
    assistants = client.beta.assistants.list()
    for assistant in assistants.data:
        try:
            client.beta.assistants.delete(assistant_id=assistant.id)
            # print(f"Assistant {assistant.id} deleted successfully.")
        except OpenAIError as e:
            print(f"An error occurred while deleting assistant {assistant.id}: {e}")
        
def main():
    import argparse

    parser = argparse.ArgumentParser(description="MateGen Assistant")
    parser.add_argument('--api_key', required=True, help="MateGen API KEY")
    args = parser.parse_args()
    assistant = MateGenClass(api_key=args.api_key)
    print("成功实例化MateGen Agent")
    
if __name__ == "__main__":
    main()