import numpy as np
from typing import List, Dict, Any, Optional, Callable
import os

from src.common.logger import get_logger
from src.llm_models.utils_model import LLMRequest
from src.config.config import model_config
from .faiss_store import FaissVectorStore

logger = get_logger("vector_manager")

class VectorStoreManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.store_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "vector_store")
        self.store = FaissVectorStore(self.store_dir)
        
        # Use embedding model from config
        self.embedding_model = LLMRequest(
            model_set=model_config.model_task_config.embedding, 
            request_type="embedding"
        )
        
        self._initialized = True
        logger.info("VectorStoreManager 初始化完成")

    async def get_embedding(self, text: str) -> np.ndarray:
        """获取文本向量"""
        try:
            embedding_list, _ = await self.embedding_model.get_embedding(text)
            return np.array([embedding_list], dtype=np.float32)
        except Exception as e:
            logger.error(f"获取Embedding失败: {e}")
            raise

    async def add_memory(self, text: str, metadata: Dict[str, Any]) -> str:
        """添加记忆向量"""
        vector = await self.get_embedding(text)
        metadata["type"] = "memory"
        metadata["timestamp"] = metadata.get("timestamp") or metadata.get("created_at")
        ids = await self.store.add([text], vector, [metadata])
        return ids[0] if ids else ""

    async def add_jargon(self, text: str, metadata: Dict[str, Any]) -> str:
        """添加黑话向量"""
        vector = await self.get_embedding(text)
        metadata["type"] = "jargon"
        ids = await self.store.add([text], vector, [metadata])
        return ids[0] if ids else ""
    
    async def add_relation(self, text: str, metadata: Dict[str, Any]) -> str:
        """添加关系向量"""
        vector = await self.get_embedding(text)
        metadata["type"] = "relation"
        ids = await self.store.add([text], vector, [metadata])
        return ids[0] if ids else ""

    async def add_emoji(self, text: str, metadata: Dict[str, Any]) -> str:
        """添加表情向量"""
        vector = await self.get_embedding(text)
        meta = metadata.copy()
        meta["type"] = "emoji"
        if "emoji_hash" in meta and "id" not in meta:
            meta["id"] = meta["emoji_hash"]
        ids = await self.store.add([text], vector, [meta])
        return ids[0] if ids else ""

    async def search_memory(self, query: str, k: int = 5, filter_fn: Optional[Callable] = None, include_relation: bool = True) -> List[Dict[str, Any]]:
        """搜索记忆（默认同时搜索 memory 和 relation 类型）"""
        vector = await self.get_embedding(query)
        
        valid_types = {"memory"}
        if include_relation:
            valid_types.add("relation")
        
        def combined_filter(meta):
            if meta.get("type") not in valid_types:
                return False
            if filter_fn:
                return filter_fn(meta)
            return True
            
        return await self.store.search(vector, k, combined_filter)

    async def search_emoji(self, query: str, k: int = 5, filter_fn: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """搜索表情向量"""
        vector = await self.get_embedding(query)

        def combined_filter(meta):
            if meta.get("type") != "emoji":
                return False
            if filter_fn:
                return filter_fn(meta)
            return True

        return await self.store.search(vector, k, combined_filter)

    async def search_jargon(self, query: str, k: int = 5, filter_fn: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """搜索黑话"""
        vector = await self.get_embedding(query)
        
        def combined_filter(meta):
            if meta.get("type") != "jargon":
                return False
            if filter_fn:
                return filter_fn(meta)
            return True
            
        return await self.store.search(vector, k, combined_filter)

    async def search_relation(self, query: str, k: int = 5, filter_fn: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """搜索关系"""
        vector = await self.get_embedding(query)
        
        def combined_filter(meta):
            if meta.get("type") != "relation":
                return False
            if filter_fn:
                return filter_fn(meta)
            return True
            
        return await self.store.search(vector, k, combined_filter)

    async def delete(self, ids: List[str]) -> bool:
        """删除向量"""
        return await self.store.delete(ids)

vector_manager = VectorStoreManager()
