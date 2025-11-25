from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np

class VectorStoreBase(ABC):
    """向量存储抽象基类"""

    @abstractmethod
    async def add(self, texts: List[str], vectors: np.ndarray, metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """添加向量和文本"""
        pass

    @abstractmethod
    async def search(self, query_vector: np.ndarray, k: int = 5, filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        pass

    @abstractmethod
    async def delete(self, ids: List[str]) -> bool:
        """删除向量"""
        pass

    @abstractmethod
    def save(self):
        """保存索引到磁盘"""
        pass

    @abstractmethod
    def load(self):
        """从磁盘加载索引"""
        pass
