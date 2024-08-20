#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, HTTPException, Depends, Body
from fastapi.responses import JSONResponse
import uvicorn
import argparse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import Depends, Query

# 全局变量来存储MateGenClass实例
global_instance = None


class ChatRequest(BaseModel):
    question: str


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
    """

    from MateGen.mateGenClass import MateGenClass, create_knowledge_base_folder, get_vector_db_id
    from utils import get_mate_gen
    from func_router import get_knowledge_bases
    from pytanic_router import KbNameRequest

    # 初始化MetaGen实例，保存在全局变量中，用于后续的子方法调用
    @app.post("/api/initialize", tags=["Initialization"], summary="用于普通问答的 MateGen 实例初始化")
    def initialize_mate_gen(mate_gen: MateGenClass = Depends(get_mate_gen)):
        try:
            global global_instance
            global_instance = mate_gen
            # 这里根据初始化结果返回相应的信息
            return {"status": 200, "data": "MateGen 实例初始化成功"}
        except Exception as e:

            raise HTTPException(status_code=500, detail=str(e))

    # 定义知识库对话（即如果勾选了知识库对话按钮后，重新实例化 MateGen 实例）
    @app.post("/api/knowledge_initialize", tags=["Initialization"], summary="用于开启知识库问答的 MateGen 实例初始化")
    def initialize_knowledge_mate_gen(
            api_key: str = Body(..., description="API key required for operation"),
            knowledge_base_chat: bool = Body(..., description="Enable knowledge base chat"),
            knowledge_base_name: str = Body(..., description="Name of the knowledge base if chat is enabled")
    ):
        try:
            global global_instance
            mate_gen = get_mate_gen(api_key, False, knowledge_base_chat, False, None, knowledge_base_name)
            global_instance = mate_gen
            # 这里根据初始化结果返回相应的信息
            return {"status": 200, "data": "MateGen 实例初始化成功"}
        except Exception as e:

            raise HTTPException(status_code=500, detail=str(e))

    # 初始化本地知识库的存储地址
    @app.post("/api/create_knowledge_base")
    def api_create_knowledge_base(sub_folder_name: str = Body(..., embed=True)):
        try:
            folder_path = create_knowledge_base_folder(sub_folder_name)
            return {"status": 200, "data": {"folder_path": folder_path}}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 根据知识库的名称获取对应的 知识库 id
    @app.get("/api/get_vector_db_id")
    def api_get_vector_db_id(request: KbNameRequest):
        print(request.knowledge_base_name)
        vector_db_id = get_vector_db_id(request.knowledge_base_name)
        if vector_db_id is None:
            raise HTTPException(status_code=404,
                                detail="Vector database ID not found for the given knowledge base name.")
        return {"status": 200, "vector_db_id": vector_db_id}

    # 定义对话接口
    @app.post("/api/chat", tags=["Chat"], summary="普通问答的对话接口")
    def chat(request: ChatRequest):
        try:
            response = global_instance.chat(request.question)
            return {"status": 200, "data": response['data']}
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
