from sqlalchemy import Column, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from sqlalchemy.orm import declarative_base

from sqlalchemy import create_engine

# 用于创建一个基类，该基类将为后续定义的所有模型类提供 SQLAlchemy ORM 功能的基础。
Base = declarative_base()


class SecretModel(Base):
    __tablename__ = 'agents'
    id = Column(String(255), primary_key=True)  # 考虑将 'id' 重命名为 'assis_id'，如果它直接存储 'assis_id'
    api_key = Column(String(255), unique=True, nullable=False)  # 确保 api_key 是唯一的
    created_at = Column(DateTime(timezone=True), server_default=func.now())  # 自动生成创建时间

    threads = relationship("ThreadModel", back_populates="agent", cascade="all, delete-orphan")


class ThreadModel(Base):
    __tablename__ = 'threads'
    id = Column(String(255), primary_key=True)  # 这作为 'thread_id'
    agent_id = Column(String(255), ForeignKey('agents.id'))
    conversation_name = Column(String(255))
    run_mode = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())  # 自动生成创建时间

    agent = relationship("SecretModel", back_populates="threads")


if __name__ == '__main__':
    if __name__ == '__main__':
        uri = f"mysql+pymysql://root:snowball950123@192.168.110.131/mategen?charset=utf8mb4"
        engine = create_engine(uri, echo=True)
        Base.metadata.create_all(engine)
