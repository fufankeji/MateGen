from fastapi import FastAPI, Request, HTTPException, Depends, Body
import os
from MateGen.mateGenClass import MateGenClass
from init_interface import get_mate_gen
from pathlib import Path
import json

from pytanic_router import ChatRequest


def initialize_mate_gen(mate_gen: MateGenClass = Depends(get_mate_gen)):
    """
    初始化MetaGen实例
    :param mate_gen:
    :return:
    """
    try:
        global global_instance
        global_instance = mate_gen
        # 这里根据初始化结果返回相应的信息
        return {"status": 200, "data": "MateGen 实例初始化成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def custom_chat(request: ChatRequest):
    try:
        response = global_instance.chat(request.question)
        return {"status": 200, "data": response['data']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_knowledge_bases():
    # 获取 KNOWLEDGE_LIBRARY_PATH 环境变量
    knowledge_library_path = os.getenv('KNOWLEDGE_LIBRARY_PATH')
    if knowledge_library_path and os.path.exists(knowledge_library_path):
        base_path = os.path.join(knowledge_library_path, 'knowledge_base')
    else:
        home_dir = str(Path.home())
        base_path = os.path.join(home_dir, 'knowledge_base')

    main_json_file = os.path.join(base_path, 'main_vector_db_mapping.json')
    if not os.path.exists(main_json_file):
        return {"error": f"{main_json_file} 不存在。请先创建知识库。"}

    with open(main_json_file, 'r') as f:
        main_mapping = json.load(f)

    return list(main_mapping.keys())
