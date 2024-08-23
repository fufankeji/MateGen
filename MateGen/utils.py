from sqlalchemy import create_engine
from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session
from typing import List, Dict
from db.thread_model import ThreadModel, SecretModel

from sqlalchemy import desc

from config.config import SQLALCHEMY_DATABASE_URI

engine = create_engine(
    SQLALCHEMY_DATABASE_URI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# 依赖项：获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def insert_agent_with_fixed_id(session: Session, api_key: str):
    """
    向数据库中插入一条新的代理记录，其中agent_id固定为-1。
    :param session: SQLAlchemy Session 对象，用于数据库操作
    :param api_key: 新代理的API密钥
    :return: 插入操作的结果，成功或失败
    """
    try:
        # 创建新的代理实例，id固定为"-1"
        new_agent = SecretModel(id="-1", api_key=api_key)
        session.add(new_agent)
        session.commit()
        return
    except Exception as e:
        session.rollback()  # 出错时回滚以避免数据不一致
        return


def store_agent_info(session: Session, assis_id: str):
    """
    如果ID为-1，则更新代理的ID。如果传入的ID已存在且不为-1，则不执行任何操作。
    """
    # 查询数据库中ID为-1的代理
    agent_with_id_negative_one = session.query(SecretModel).filter(SecretModel.id == "-1").one_or_none()
    # 如果找到了ID为-1的代理，则更新其ID
    if agent_with_id_negative_one:
        agent_with_id_negative_one.id = assis_id
        session.commit()
        return


def get_thread_from_db(session: Session, thread_id: str):
    """
    从数据库中检索与给定 thread_id 相匹配的 ThreadModel 实例。
    :param session: SQLAlchemy Session 对象，用于数据库操作
    :param thread_id: 要检索的线程的 ID
    :return: 如果找到相应的 ThreadModel 实例则返回它，否则返回 None
    """

    thread = session.query(ThreadModel).filter(ThreadModel.id == thread_id).one_or_none()
    return thread


def store_thread_info(session: Session, agent_id: str, thread_id: str, conversation_name: str, run_mode: str):
    # 检查数据库中是否已存在该 thread_id
    existing_thread = session.query(ThreadModel).filter(ThreadModel.id == thread_id).first()
    if existing_thread:
        return existing_thread  # 或者更新信息，取决于需求

    # 创建新的 ThreadModel 实例并存储到数据库
    new_thread = ThreadModel(id=thread_id, agent_id=agent_id, conversation_name=conversation_name, run_mode=run_mode)
    session.add(new_thread)
    session.commit()
    return new_thread


def update_conversation_name(session: Session, thread_id: str, new_conversation_name: str):
    """
    更新数据库中指定线程的 conversation_name。
    :param session: SQLAlchemy Session 对象，用于数据库操作
    :param thread_id: 要更新的线程ID
    :param new_conversation_name: 新的会话名称
    :return: None
    """
    # 如果提供的新会话名称超过7个字符，截断它
    new_conversation_name = new_conversation_name[:7] if len(new_conversation_name) > 7 else new_conversation_name

    # 查找数据库中的线程
    thread = session.query(ThreadModel).filter(ThreadModel.id == thread_id).first()
    if thread:
        # 更新 conversation_name
        thread.conversation_name = new_conversation_name
        session.commit()
        print(f"Updated thread {thread_id} with new conversation name: {new_conversation_name}")
    else:
        print("No thread found with the given ID.")


def fetch_threads_by_agent(session: Session, agent_id: str) -> Dict[str, List[Dict[str, str]]]:
    """
    根据给定的agent_id从数据库中检索所有线程的ID和对应的会话名称。
    :param session: SQLAlchemy Session 对象，用于数据库操作
    :param agent_id: 用于筛选线程的代理ID
    :return: 包含所有相关线程信息的列表，每个元素都是一个包含线程ID和会话名称的字典
    """
    # 根据agent_id查询所有相关的线程
    threads = session.query(ThreadModel.id, ThreadModel.conversation_name).filter(ThreadModel.agent_id == agent_id).all()

    # 创建包含所有相关线程信息的列表
    threads_list = [{"id": thread.id, "conversation_name": thread.conversation_name or ""} for thread in threads]

    # 将结果打包成JSON格式
    return threads_list


def fetch_threads_mode(session: Session, thread_id: str) -> Dict[str, List[Dict[str, str]]]:
    """
    根据给定的agent_id从数据库中检索所有线程的ID和对应的会话名称。
    :param session: SQLAlchemy Session 对象，用于数据库操作
    :param agent_id: 用于筛选线程的代理ID
    :return: 包含所有相关线程信息的列表，每个元素都是一个包含线程ID和会话名称的字典
    """
    # 根据thread_id查询对应的模式
    threads = session.query(ThreadModel.run_mode).filter(ThreadModel.id == thread_id).all()
    return threads.run_mode


def fetch_latest_agent_id(session: Session) -> str:
    """
    从数据库中检索最新代理的api_key。
    :param session: SQLAlchemy Session 对象，用于数据库操作
    :return: 如果找到代理则返回其api_key，否则返回空字符串
    """
    # 查询代理，按照创建时间降序排序，获取第一个结果
    # 假设你的模型中有一个创建时间字段名为 'created_at'
    # 如果没有，则按照 id 或其他可用字段降序排序
    agent = session.query(SecretModel).order_by(desc(SecretModel.created_at)).first()

    # 如果找到代理，则返回其id，否则返回空字符串
    return agent.id if agent else ""

def fetch_latest_api_key(session: Session) -> str:
    """
    从数据库中检索最新代理的api_key。
    :param session: SQLAlchemy Session 对象，用于数据库操作
    :return: 如果找到代理则返回其api_key，否则返回空字符串
    """
    # 查询代理，按照创建时间降序排序，获取第一个结果
    # 假设你的模型中有一个创建时间字段名为 'created_at'
    # 如果没有，则按照 id 或其他可用字段降序排序
    agent = session.query(SecretModel).order_by(desc(SecretModel.created_at)).first()

    # 如果找到代理，则返回其api_key，否则返回空字符串
    return agent.api_key if agent else ""


def fetch_run_mode_by_thread_id(session: Session, thread_id: str) -> str:
    """
    根据线程ID从数据库中检索对应的运行模式（run_mode）。
    :param session: SQLAlchemy Session 对象，用于数据库操作
    :param thread_id: 线程的ID
    :return: 返回找到的运行模式，如果没有找到则返回空字符串
    """
    thread = session.query(ThreadModel).filter(ThreadModel.id == thread_id).one_or_none()
    return thread.run_mode if thread else ""
