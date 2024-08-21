#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, HTTPException, Depends, Body, status, Security
from fastapi.responses import JSONResponse
import uvicorn
import json
import argparse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import Depends, Query
from fastapi.security import APIKeyHeader
from openai import OpenAI

from MateGen.mateGenClass import (MateGenClass,
                                  get_vector_db_id,
                                  create_knowledge_base,
                                  create_knowledge_base_folder,
                                  print_and_select_knowledge_base,
                                  update_knowledge_base_description,
                                  delete_all_files,
                                  get_latest_thread,
                                  make_hl,
                                  )
from utils import get_mate_gen, get_openai_instance
from func_router import get_knowledge_bases
from pytanic_router import KbNameRequest

# 全局变量来存储MateGenClass实例
global_instance = None
global_openai_instance = None


class ChatRequest(BaseModel):
    question: str


class UrlModel(BaseModel):
    url: str


class KnowledgeBaseCreateRequest(BaseModel):
    knowledge_base_name: str
    folder_path_base: str = None  # 可选字段


class KnowledgeBaseDescriptionUpdateRequest(BaseModel):
    sub_folder_name: str
    description: str


def create_app():
    app = FastAPI(
        title="MateGen API Server",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载路由
    mount_app_routes(app)

    # 挂载 前端 项目构建的前端静态文件夹 (对接前端静态文件的入口)
    # app.mount("/", StaticFiles(directory="static/dist"), name="static")
    return app


def mount_app_routes(app: FastAPI):
    """
    这里定义所有 RestFul API interfence
    待做：
    1. 获取用户全部的历史对话会话框
    2. 获取某个会话框内的全部对话内容
    3.
    """

    # 初始化MetaGen实例，保存在全局变量中，用于后续的子方法调用
    @app.post("/api/initialize", tags=["Initialization"],
              summary="用于普通问答的 MateGen 实例初始化")
    def initialize_mate_gen(mate_gen: MateGenClass = Depends(get_mate_gen),
                            openai_ins: OpenAI = Depends(get_openai_instance)):
        try:
            global global_instance, global_openai_instance
            global_instance = mate_gen
            global_openai_instance = openai_ins

            # 这里根据初始化结果返回相应的信息
            return {"status": 200, "data": "MateGen 实例初始化成功"}
        except Exception as e:

            raise HTTPException(status_code=500, detail=str(e))

    # 定义知识库对话（即如果勾选了知识库对话按钮后，重新实例化 MateGen 实例）
    @app.post("/api/knowledge_initialize", tags=["Initialization"],
              summary="用于开启知识库问答的 MateGen 实例初始化")
    def initialize_knowledge_mate_gen(
            api_key: str = Body(..., description="API key required for operation"),
            knowledge_base_chat: bool = Body(..., description="Enable knowledge base chat"),
            knowledge_base_name: str = Body(..., description="Name of the knowledge base if chat is enabled"),
            openai_ins: OpenAI = Depends(get_openai_instance)
    ):
        try:
            global global_instance, global_openai_instance
            mate_gen = get_mate_gen(api_key, False, knowledge_base_chat, False, None, knowledge_base_name)
            global_instance = mate_gen
            global_openai_instance = openai_ins
            # 这里根据初始化结果返回相应的信息
            return {"status": 200, "data": "MateGen 实例初始化成功"}
        except Exception as e:

            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/enhanced_initialize", tags=["Initialization"],
              summary="用于开启增强模式的 MateGen 实例初始化")
    def initialize_knowledge_mate_gen(
            api_key: str = Body(..., description="API key required for operation"),
            enhanced_mode: bool = Body(True, description="Enable enhanced mode"),
            knowledge_base_chat: bool = Body(False, description="Enable knowledge base chat"),
            kaggle_competition_guidance: bool = Body(False, description="Enable Kaggle competition guidance"),
            competition_name: str = Body(None, description="Name of the Kaggle competition if guidance is enabled"),
            knowledge_base_name: str = Body(None, description="Name of the knowledge_base_chat is enabled"),
            openai_ins: OpenAI = Depends(get_openai_instance)
    ):
        try:
            global global_instance, global_openai_instance
            mate_gen = get_mate_gen(api_key, enhanced_mode, knowledge_base_chat, kaggle_competition_guidance,
                                    competition_name, knowledge_base_name)
            global_instance = mate_gen
            global_openai_instance = openai_ins
            # 这里根据初始化结果返回相应的信息
            return {"status": 200, "data": "MateGen 实例初始化成功"}
        except Exception as e:

            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/set_knowledge_base_url", tags=["Knowledge"],
              summary="设置本地知识库的根目录")
    def set_base_url(url_data: UrlModel):
        # 因为Json会转义 \ , 这里手动进行转换
        corrected_path = url_data.url.replace('\\', '\\\\')
        if global_instance.set_knowledge_base_url(corrected_path):
            return {"status": 200, "data": {"knowledge_base_url": url_data.url}}
        else:
            raise HTTPException(status_code=400, detail="无效的知识库地址，正确的路径实例：E:\\work")

    @app.post("/api/create_knowledge_base_folder", tags=["Knowledge"],
              summary="在本地知识库的根目录下，创建知识库文件夹")
    def create_knowledge_folder(sub_folder_name: str = Body(..., embed=True)):
        try:
            folder_path = create_knowledge_base_folder(sub_folder_name)
            return {"status": 200, "data": {"folder_path": folder_path}}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/create_knowledge", tags=["Knowledge"],
              summary="新建一个本地知识库，并执行向量化操作")
    def create_knowledge(request: KnowledgeBaseCreateRequest):
        corrected_path = request.folder_path_base.replace('\\', '\\\\')
        # 注意： 这里使用 openai 的 实例，而不是 MetaGen的实例
        vector_id = create_knowledge_base(global_openai_instance, request.knowledge_base_name, corrected_path)
        if vector_id is not None:
            return {"status": 200, "data": {"vector_id": vector_id}}
        else:
            raise HTTPException(status_code=400, detail="知识库无法创建，请再次确认知识库文件夹中存在格式合规的文件")

    @app.get("/api/get_all_knowledge", tags=["Knowledge"],
             summary="获取所有的本地知识库列表")
    def get_all_knowledge():
        try:
            knowledge_bases = print_and_select_knowledge_base()
            if not knowledge_bases:
                return {"status": 404, "data": [], "message": "没有找到知识库，请先创建。"}
            return {"status": 200, "data": knowledge_bases}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/get_vector_db_id", tags=["Knowledge"],
             summary="根据知识库的名称，获取其对应的向量数据库ID")
    def api_get_vector_db_id(request: KbNameRequest):
        vector_db_id = get_vector_db_id(request.knowledge_base_name)
        if vector_db_id is None:
            raise HTTPException(status_code=404,
                                detail="Vector database ID not found for the given knowledge base name.")
        return {"status": 200, "vector_db_id": vector_db_id}

    @app.post("/api/update_knowledge_base_description", tags=["Knowledge"], summary="更新某个知识库的描述")
    def api_update_knowledge_base_description(request: KnowledgeBaseDescriptionUpdateRequest):
        try:
            # 调用之前定义的函数
            if update_knowledge_base_description(request.sub_folder_name, request.description):
                return {"status": 200, "data": {"message": f"《{request.sub_folder_name}》知识库的描述已更新"}}
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="找不到指定的 JSON 文件")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")

    @app.get("/api/all_knowledge_base", tags=["Knowledge"], summary="根据向量数据库id获取到上传的所有本地文件")
    def get_knowledge_base_all(request: KbNameRequest):
        # 根据输入的 知识库名称，先获取到对应的 向量库id
        vector_store_files = global_openai_instance.beta.vector_stores.files.list(
            vector_store_id=get_vector_db_id(request.knowledge_base_name)
        )

        # 遍历列表，提取并格式化所需信息
        formatted_files = [
            {
                "id": file.id,
                "created_at": file.created_at,
                "vector_store_id": file.vector_store_id
            }
            for file in vector_store_files.data
        ]

        return {"status": 200, "data": formatted_files}

    @app.delete("/api/delete_all_files", tags=["Files"], summary="删除所有文件")
    def api_delete_all_files():
        vector_stores = global_openai_instance.beta.vector_stores.list()
        # TODO
        # if delete_all_files(global_openai_instance):
        #     return {"message": "所有文件已被成功删除。"}
        # else:
        #     raise HTTPException(status_code=500, detail="无法删除文件，请检查日志了解更多信息。")

    @app.post("/api/chat", tags=["Chat"], summary="问答的通用对话接口")
    def chat(request: ChatRequest):
        try:
            response = global_instance.chat(request.question)
            return {"status": 200, "data": response['data']}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/messages", tags=["Chat"], summary="根据thread_id获取指定的会话历史信息")
    def get_messages():

        import os
        from pathlib import Path

        home_dir = str(Path.home())
        log_dir = os.path.join(home_dir, "._logs")
        os.makedirs(log_dir, exist_ok=True)
        thread_log_file = os.path.join(log_dir, "thread_log.json")
        try:
            with open(thread_log_file, "r") as file:
                thread_log = json.load(file)

            # 用于存储每个 thread_id 对应的对话
            all_threads_dialogues = []

            # 遍历所有线程
            if thread_log:
                for thread in thread_log:
                    # 获取指定 thread_id 的消息列表
                    thread_messages = global_openai_instance.beta.threads.messages.list(thread).data

                    dialogues = []  # 用于存储当前线程的对话内容

                    # 遍历消息，按 role 提取文本内容
                    for message in reversed(thread_messages):  # 反转列表处理，直接在循环中反转
                        content_value = next((cb.text.value for cb in message.content if cb.type == 'text'), None)
                        if content_value:
                            if message.role == "assistant":
                                dialogue = {"assistant": content_value}
                            elif message.role == "user":
                                dialogue = {"user": content_value}
                            dialogues.append(dialogue)

                    # 将当前线程的对话添加到总列表
                    all_threads_dialogues.append({
                        "thread_id": thread,
                        "data": dialogues
                    })

            return {"status": 200, "data": all_threads_dialogues}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


def run_api(host, port, **kwargs):
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(app,
                    host=host,
                    port=port,
                    ssl_keyfile=kwargs.get("ssl_keyfile"),
                    ssl_certfile=kwargs.get("ssl_certfile"),
                    )
    else:
        uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--host", type=str, default="192.168.110.131")
    # parser.add_argument("--port", type=int, default=8000)
    # parser.add_argument("--ssl_keyfile", type=str)
    # parser.add_argument("--ssl_certfile", type=str)
    # # 初始化消息
    # args = parser.parse_args()
    # args_dict = vars(args)
    #
    # app = create_app()
    #
    # run_api(host=args.host,
    #         port=args.port,
    #         ssl_keyfile=args.ssl_keyfile,
    #         ssl_certfile=args.ssl_certfile,
    #         )

    app = create_app()
    run_api(host="localhost",
            port=9000,
            ssl_keyfile=None,
            ssl_certfile=None,
            )
