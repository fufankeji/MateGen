from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

from fastapi import HTTPException, Body
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel
from sqlalchemy import create_engine, text

"""
官方文档：https://docs.sqlalchemy.org/en/20/dialects/mysql.html
"""

# 用于创建一个基类，该基类将为后续定义的所有模型类提供 SQLAlchemy ORM 功能的基础。
Base = declarative_base()


class DBConfig(BaseModel):
    username: str
    password: str
    hostname: str
    database_name: str


def get_engine(db_config: DBConfig):
    uri = f"mysql+pymysql://{db_config.username}:{db_config.password}@{db_config.hostname}/{db_config.database_name}?charset=utf8mb4"
    engine = create_engine(uri, echo=True)
    # Base.metadata.create_all(engine)  # 创建所有表
    return engine


SessionLocal = sessionmaker(autocommit=False, autoflush=False)


def create_database_connection(db_config: DBConfig):
    """
    创建数据库连接，如果连接成功，返回该连接下所有数据库名称及对应的表名（不包含系统预设的数据库）
    :param db_config:
    :return:
    """
    engine = get_engine(db_config)
    SessionLocal.configure(bind=engine)
    session = SessionLocal()

    # 要排除的数据库集
    excluded_databases = {'information_schema', 'mysql', 'performance_schema', 'sys'}

    try:
        all_databases_and_tables = {}
        with engine.connect() as conn:
            # 获取所有数据库并过滤掉系统数据库
            databases = conn.execute(text("SHOW DATABASES;"))
            databases_list = [db[0] for db in databases if db[0] not in excluded_databases]

            for db in databases_list:
                # 获取每个数据库中的所有表，使用反引号来避免特殊字符的问题
                conn.execute(text(f"USE `{db}`;"))
                tables = conn.execute(text("SHOW TABLES;"))
                table_list = [table[0] for table in tables]
                all_databases_and_tables[db] = table_list

        session.close()
        return {"status": 200, "data": {"all_databases_and_tables": all_databases_and_tables}}
    except SQLAlchemyError as e:
        session.close()
        return {"status": 400, "data": f"连接失败，错误: {str(e)}"}
    except Exception as e:
        session.close()
        raise Exception(f"连接失败，错误: {str(e)}")




