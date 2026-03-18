import os
import json
import uuid
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Callable
from src.common.logger import get_logger
from .base import VectorStoreBase

logger = get_logger("faiss_store")


class FaissVectorStore(VectorStoreBase):
    def __init__(self, storage_dir: str, dimension: int = 1536):
        self.storage_dir = storage_dir
        self.dimension = dimension
        self.index_path = os.path.join(storage_dir, "vector_index.faiss")
        self.metadata_path = os.path.join(storage_dir, "vector_metadata.json")

        self.index = None
        self.metadata: Dict[str, Dict[str, Any]] = {}  # id -> metadata (includes text)
        self._int_id_to_uid: Dict[int, str] = {}  # Cache for reverse lookup

        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

        self.load()

    def _init_index(self):
        # Use IDMap to support add_with_ids and remove_ids
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIDMap(quantizer)

    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        prepared = np.asarray(vectors, dtype=np.float32)
        if prepared.ndim == 1:
            prepared = prepared.reshape(1, -1)
        if prepared.ndim != 2:
            raise ValueError(f"向量必须是二维数组，当前 shape={prepared.shape}")
        if prepared.shape[0] == 0:
            raise ValueError("向量数组不能为空")
        return prepared

    def _ensure_index_dimension(self, vector_dimension: int) -> None:
        if self.index is None:
            self.dimension = vector_dimension
            self._init_index()
            return

        current_dimension = self.index.d
        if current_dimension == vector_dimension:
            return

        if self.index.ntotal == 0:
            logger.warning(
                f"检测到空向量索引维度与当前向量不一致，自动重建索引: {current_dimension} -> {vector_dimension}"
            )
            self.dimension = vector_dimension
            self._init_index()
            self.metadata = {}
            self._int_id_to_uid = {}
            return

        raise ValueError(
            f"向量维度不匹配: 索引维度={current_dimension}, 输入维度={vector_dimension}。"
            f"请保持 embedding 模型维度一致，或清理旧索引文件后重建: {self.index_path}"
        )

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                self.dimension = self.index.d
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                # Rebuild cache
                self._int_id_to_uid = {meta["_int_id"]: uid for uid, meta in self.metadata.items()}
                logger.info(f"已加载向量索引，包含 {self.index.ntotal} 条数据")
            except Exception as e:
                logger.error(f"加载向量索引失败: {e}，将重新初始化")
                self._init_index()
                self.metadata = {}
                self._int_id_to_uid = {}
        else:
            self._init_index()
            self.metadata = {}
            self._int_id_to_uid = {}

    def save(self):
        try:
            if self.index:
                faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.debug("向量索引已保存")
        except Exception as e:
            logger.error(f"保存向量索引失败: {e}")

    async def add(
        self, texts: List[str], vectors: np.ndarray, metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        prepared_vectors = self._prepare_vectors(vectors)
        self._ensure_index_dimension(prepared_vectors.shape[1])

        count = len(texts)
        if count == 0:
            return []

        ids = []
        int_ids = []

        # Generate IDs
        for i in range(count):
            meta = metadatas[i].copy() if metadatas and i < len(metadatas) else {}

            # Prefer external id (e.g. memory_id) as stable key when provided
            raw_id = meta.get("id")
            if raw_id is not None:
                uid = str(raw_id)
            else:
                uid = str(uuid.uuid4())

            # If this uid already exists, remove old vector entry to avoid duplicates
            old_meta = self.metadata.get(uid)
            if old_meta is not None:
                old_int_id = old_meta.get("_int_id")
                if old_int_id is not None and self.index is not None and self.index.ntotal > 0:
                    try:
                        self.index.remove_ids(np.array([old_int_id], dtype=np.int64))
                    except Exception as e:
                        logger.warning(f"删除旧向量失败 (id={uid}): {e}")

            # Generate a new int64 id for FAISS
            try:
                # If uid is a valid UUID string, derive int_id from it for stability
                int_id = int(uuid.UUID(uid).int & (2**63 - 1))
            except (ValueError, AttributeError):
                # Fallback: hash-based id
                int_id = abs(hash(uid)) & (2**63 - 1)

            ids.append(uid)
            int_ids.append(int_id)

            meta["text"] = texts[i]
            meta["_int_id"] = int_id  # Store mapping
            self.metadata[uid] = meta
            self._int_id_to_uid[int_id] = uid

        self.index.add_with_ids(prepared_vectors, np.array(int_ids, dtype=np.int64))
        self.save()
        return ids

    async def search(
        self, query_vector: np.ndarray, k: int = 5, filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        prepared_query = self._prepare_vectors(query_vector)
        if prepared_query.shape[1] != self.index.d:
            raise ValueError(
                f"查询向量维度不匹配: 索引维度={self.index.d}, 输入维度={prepared_query.shape[1]}。"
                f"请检查 embedding 模型是否与现有索引一致。"
            )

        # If filter provided, we might need to fetch more candidates
        search_k = k
        if filter_fn:
            search_k = min(k * 5, self.index.ntotal)  # Fetch more for filtering

        distances_matrix, indices_matrix = self.index.search(prepared_query, search_k)

        results = []

        # Flatten results
        distances = distances_matrix[0]
        indices = indices_matrix[0]

        for dist, int_id in zip(distances, indices, strict=False):
            if int_id == -1:
                continue

            found_uid = self._int_id_to_uid.get(int_id)
            if not found_uid:
                continue

            found_meta = self.metadata.get(found_uid)

            if found_meta:
                if filter_fn and not filter_fn(found_meta):
                    continue

                result_item = found_meta.copy()
                result_item["id"] = found_uid
                result_item["score"] = float(dist)  # L2 distance
                result_item["similarity"] = 1.0 / (1.0 + float(dist))

                results.append(result_item)
                if len(results) >= k:
                    break

        return results

    async def delete(self, ids: List[str]) -> bool:
        if self.index is None:
            return False

        int_ids_to_remove = []

        for uid in ids:
            if uid in self.metadata:
                int_id = self.metadata[uid]["_int_id"]
                int_ids_to_remove.append(int_id)
                del self.metadata[uid]
                if int_id in self._int_id_to_uid:
                    del self._int_id_to_uid[int_id]

        if int_ids_to_remove:
            self.index.remove_ids(np.array(int_ids_to_remove, dtype=np.int64))
            self.save()
            return True

        return False
