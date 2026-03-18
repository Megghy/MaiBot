from __future__ import annotations

import numpy as np
import pytest

from src.common.vector_store.faiss_store import FaissVectorStore


@pytest.mark.asyncio
async def test_add_reinitializes_empty_index_to_match_first_vector_dimension(tmp_path):
    store = FaissVectorStore(str(tmp_path))

    assert store.index is not None
    assert store.index.d == 1536

    ids = await store.add(
        ["emoji vector"],
        np.zeros((1, 1024), dtype=np.float32),
        [{"id": "emoji-1"}],
    )

    assert ids == ["emoji-1"]
    assert store.index is not None
    assert store.index.d == 1024
    assert store.index.ntotal == 1


@pytest.mark.asyncio
async def test_add_raises_clear_error_when_non_empty_index_dimension_mismatch(tmp_path):
    store = FaissVectorStore(str(tmp_path))

    await store.add(
        ["emoji vector"],
        np.zeros((1, 1024), dtype=np.float32),
        [{"id": "emoji-1"}],
    )

    with pytest.raises(ValueError, match="向量维度不匹配"):
        await store.add(
            ["memory vector"],
            np.zeros((1, 1536), dtype=np.float32),
            [{"id": "memory-1"}],
        )
