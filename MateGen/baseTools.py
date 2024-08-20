import os
import requests
import json
from dotenv import load_dotenv, set_key, find_dotenv
from IPython.display import display, Code, Markdown, Image
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)

# 谷歌搜索测试
def test_google_search_api(search_term='openai'):
    """
    测试谷歌搜索API是否能够顺利连接并正常获得响应。
    """
    url = "https://www.googleapis.com/customsearch/v1"
    google_search_key = os.getenv('GOOGLE_SEARCH_KEY')
    cse_id = os.getenv('CSE_ID')
    
    params = {
        'q': search_term,
        'key': google_search_key,
        'cx': cse_id
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
    # 加载.env文件
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dotenv_path = os.path.join(BASE_DIR, '.env')
    # 加载.env文件，如果不存在则创建一个
    if not os.path.exists(dotenv_path):
        with open(dotenv_path, 'w') as f:
            pass  # 创建一个空的 .env 文件
        
    load_dotenv(dotenv_path)
    
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
    # 加载.env文件
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dotenv_path = os.path.join(BASE_DIR, '.env')
    # 加载.env文件，如果不存在则创建一个
    if not os.path.exists(dotenv_path):
        with open(dotenv_path, 'w') as f:
            pass  # 创建一个空的 .env 文件
        
    load_dotenv(dotenv_path)
    
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
    # 加载 .env 文件
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dotenv_path = os.path.join(BASE_DIR, '.env')
    # 加载.env文件，如果不存在则创建一个
    if not os.path.exists(dotenv_path):
        with open(dotenv_path, 'w') as f:
            pass  # 创建一个空的 .env 文件
        
    load_dotenv(dotenv_path)
    
    # 如果 .env 文件不存在，创建一个
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
    
    if access_key_id or access_key_secret or endpoint or bucket_name == None:
        print("请先设置相关变量取值")
        return False
    
    try:
        # 初始化OSS客户端
        
        auth = oss2.Auth(access_key_id, access_key_secret)
        bucket = oss2.Bucket(auth, endpoint, bucket_name)

        # 测试列出存储空间中的所有文件
        file_list = list(oss2.ObjectIterator(bucket))
        
        print(f"测试成功：列出了存储空间中的文件，共计 {len(file_list)} 个文件。")
        return True
    
    except OssError as e:
        print(f"测试失败：OSS API 调用失败 - {e}")
        return False
    
def write_oss_env(access_key_id, access_key_secret, endpoint, bucket_name):
    """
    将阿里云OSS的相关变量写入当前项目文件夹内的.env文件中，作为环境变量。
    """
    # 加载.env文件
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dotenv_path = os.path.join(BASE_DIR, '.env')
    # 加载.env文件，如果不存在则创建一个
    if not os.path.exists(dotenv_path):
        with open(dotenv_path, 'w') as f:
            pass  # 创建一个空的 .env 文件

    load_dotenv(dotenv_path)
    
    if load_dotenv(dotenv_path):
        set_key(dotenv_path, 'OSS_ACCESS_KEY_ID', access_key_id)
        set_key(dotenv_path, 'OSS_ACCESS_KEY_SECRET', access_key_secret)
        set_key(dotenv_path, 'OSS_ENDPOINT', endpoint)
        set_key(dotenv_path, 'OSS_BUCKET_NAME', bucket_name)

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
    # 加载环境变量
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    
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
    # 加载.env文件
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dotenv_path = os.path.join(BASE_DIR, '.env')
    # 加载.env文件，如果不存在则创建一个
    if not os.path.exists(dotenv_path):
        with open(dotenv_path, 'w') as f:
            pass  
    
    load_dotenv(dotenv_path)
    
    # 如果.env文件不存在，创建一个
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