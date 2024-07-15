from openai import OpenAI
import os
from dotenv import load_dotenv, set_key, find_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(f"BASE_DIR: {BASE_DIR}")
dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)
# print(f"dotenv_path: {dotenv_path}")

# 删除所有文件
def delete_all_files(client):
    # 获取所有文件的列表
    files = client.files.list()

    # 逐个删除文件
    for file in files.data:
        file_id = file.id
        client.files.delete(file_id)
        print(f"Deleted file: {file_id}")

# 删除所有词向量库
def delete_all_vector_stores(client):
    # 获取所有词向量库的列表
    vector_stores = client.beta.vector_stores.list()

    # 逐个删除词向量库
    for vector_store in vector_stores.data:
        vector_store_id = vector_store.id
        client.beta.vector_stores.delete(vector_store_id)
        print(f"Deleted vector store: {vector_store_id}")
        
url1 = 'https://mategen.oss-cn-beijing.aliyuncs.com/mategen-illustration/%E8%B0%B7%E6%AD%8C%E6%90%9C%E7%B4%A2API%E9%85%8D%E7%BD%AE%E6%96%B9%E6%B3%95.md'
url2 = 'https://mategen.oss-cn-beijing.aliyuncs.com/mategen-illustration/%E7%9F%A5%E4%B9%8ECookie%E8%8E%B7%E5%8F%96%E6%96%B9%E6%B3%95.md'
url3 = 'https://mategen.oss-cn-beijing.aliyuncs.com/mategen-illustration/Github%20Token%E8%8E%B7%E5%8F%96.md'
url4 = 'https://mategen.oss-cn-beijing.aliyuncs.com/mategen-illustration/Kaggle%20headers%E5%92%8Ccookies%E8%8E%B7%E5%8F%96%E6%96%B9%E6%B3%95.md'
url5 = 'https://mategen.oss-cn-beijing.aliyuncs.com/mategen-illustration/%E9%98%BF%E9%87%8C%E4%BA%91oss%E8%AE%BE%E7%BD%AE.md'