"""
Vector Store for Document Retrieval
====================================
Stores and retrieves document embeddings for RAG.
"""

from __future__ import annotations

import logging
import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None


@dataclass
class SearchResult:
    """Result from vector search."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    page_number: int
    section: Optional[str]


class VectorStore:
    """
    Vector store for document embeddings with metadata.
    
    Uses FAISS for efficient similarity search with SQLite for metadata.
    """
    
    def __init__(
        self, 
        dimension: int = 384,
        index_path: Optional[str] = None,
        db_path: Optional[str] = None,
    ):
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        self.db_path = Path(db_path) if db_path else None
        
        self.index = None
        self.chunk_id_to_index = {}  # chunk_id -> faiss index
        self.index_to_chunk_id = {}  # faiss index -> chunk_id
        self._next_index = 0
        # Tombstones for deleted chunks. FAISS IndexFlatIP has no efficient
        # remove; SQLite is source of truth, and search filters these IDs so
        # stale hits never return (prod PGVector path untouched).
        self._deleted_chunk_ids: set[str] = set()
        
        # Initialize storage
        self._init_storage()
    
    def _init_storage(self) -> None:
        """Initialize FAISS index and SQLite database."""
        # Load or create FAISS index
        if self.index_path and self.index_path.exists() and FAISS_AVAILABLE:
            self.index = faiss.read_index(str(self.index_path))
            self._load_index_mapping()
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            if FAISS_AVAILABLE:
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            else:
                self.index = None
                logger.warning("FAISS not available, using fallback search")
        
        # Initialize SQLite for metadata
        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
    
    def _init_db(self) -> None:
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    page_number INTEGER,
                    section TEXT,
                    source_type TEXT,
                    token_count INTEGER,
                    metadata TEXT,  -- JSON
                    faiss_index INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_id ON chunks(document_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_faiss_index ON chunks(faiss_index)
            """)
            conn.commit()
    
    def _load_index_mapping(self) -> None:
        """Load chunk_id to FAISS index mapping from database."""
        if not self.db_path:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT chunk_id, faiss_index FROM chunks WHERE faiss_index IS NOT NULL"
            ).fetchall()
            
            for row in rows:
                chunk_id = row["chunk_id"]
                faiss_idx = row["faiss_index"]
                self.chunk_id_to_index[chunk_id] = faiss_idx
                self.index_to_chunk_id[faiss_idx] = chunk_id
                self._next_index = max(self._next_index, faiss_idx + 1)
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of dicts with keys: chunk_id, document_id, content, 
                   embedding, page_number, section, source_type, token_count, metadata
        
        Returns:
            List of added chunk_ids
        """
        if not chunks:
            return []
        
        if not FAISS_AVAILABLE or self.index is None:
            logger.warning("FAISS not available, skipping vector indexing")
            return [c["chunk_id"] for c in chunks]
        
        # Prepare embeddings
        embeddings = []
        chunk_ids = []
        metadatas = []
        
        for chunk in chunks:
            embedding = chunk.get("embedding")
            if embedding is None:
                logger.warning(f"Chunk {chunk.get('chunk_id')} has no embedding, skipping")
                continue
            
            embeddings.append(embedding)
            chunk_ids.append(chunk["chunk_id"])
            metadatas.append(chunk)
        
        if not embeddings:
            return []
        
        # Convert to numpy array
        vectors = np.array(embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)
        
        # Add to FAISS index
        start_idx = self._next_index
        self.index.add(vectors)
        
        # Update mappings (re-added ids clear any prior tombstone)
        for i, chunk_id in enumerate(chunk_ids):
            faiss_idx = start_idx + i
            self.chunk_id_to_index[chunk_id] = faiss_idx
            self.index_to_chunk_id[faiss_idx] = chunk_id
            self._deleted_chunk_ids.discard(chunk_id)
        
        self._next_index += len(chunk_ids)
        
        # Save to SQLite
        if self.db_path:
            self._save_chunks_to_db(chunk_ids, metadatas, start_idx)
        
        # Persist index
        self.persist()
        
        logger.info(f"Added {len(chunk_ids)} chunks to vector store")
        return chunk_ids
    
    def _save_chunks_to_db(
        self, 
        chunk_ids: List[str], 
        metadatas: List[Dict], 
        start_idx: int
    ) -> None:
        """Save chunk metadata to SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            for i, (chunk_id, meta) in enumerate(zip(chunk_ids, metadatas)):
                faiss_idx = start_idx + i
                conn.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (chunk_id, document_id, content, page_number, section, 
                     source_type, token_count, metadata, faiss_index)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk_id,
                    meta.get("document_id", ""),
                    meta.get("content", ""),
                    meta.get("page_number", 0),
                    meta.get("section"),
                    meta.get("source_type", "digital"),
                    meta.get("token_count", 0),
                    json.dumps(meta.get("metadata", {})),
                    faiss_idx,
                ))
            conn.commit()
    
    def search(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        document_ids: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            document_ids: Optional filter by document IDs
            min_score: Minimum similarity score (0-1)
        
        Returns:
            List of SearchResult
        """
        if not FAISS_AVAILABLE or self.index is None or self.index.ntotal == 0:
            return self._fallback_search(query_embedding, k, document_ids, min_score)
        
        # Normalize query
        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)
        
        # Search
        # If filtering by document_ids, we need to search more and filter.
        # Over-fetch when tombstones exist so deleted vectors consumed from
        # top-k never shrink the visible result set.
        over_fetch = len(self._deleted_chunk_ids)
        search_k = k * 3 if document_ids else k
        if over_fetch:
            search_k = max(search_k, k + over_fetch)
        scores, indices = self.index.search(query_vec, min(search_k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < min_score:
                continue
            
            chunk_id = self.index_to_chunk_id.get(idx)
            if not chunk_id:
                continue
            # Filter tombstoned chunks (deleted from SQLite but still resident
            # in the FAISS index until rebuilt).
            if chunk_id in self._deleted_chunk_ids:
                continue

            # Get metadata from DB (missing row == deleted -> skip stale hit)
            meta = self._get_chunk_metadata(chunk_id)
            if not meta:
                continue
            
            # Filter by document_ids if provided
            if document_ids and meta.get("document_id") not in document_ids:
                continue
            
            results.append(SearchResult(
                chunk_id=chunk_id,
                document_id=meta["document_id"],
                content=meta["content"],
                score=float(score),
                metadata=meta.get("metadata", {}),
                page_number=meta.get("page_number", 0),
                section=meta.get("section"),
            ))
            
            if len(results) >= k:
                break
        
        return results
    
    def _fallback_search(
        self, 
        query_embedding: List[float], 
        k: int,
        document_ids: Optional[List[str]],
        min_score: float,
    ) -> List[SearchResult]:
        """Fallback search using SQLite + numpy when FAISS unavailable."""
        if not self.db_path:
            return []
        
        results = []
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            return []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM chunks WHERE 1=1"
            params = []
            
            if document_ids:
                placeholders = ",".join("?" * len(document_ids))
                query += f" AND document_id IN ({placeholders})"
                params.extend(document_ids)
            
            rows = conn.execute(query, params).fetchall()

            for row in rows:
                if row["chunk_id"] in self._deleted_chunk_ids:
                    continue
                # Fallback: construct SearchResult from DB metadata only (no vector similarity)
                results.append(SearchResult(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    content=row["content"],
                    score=0.0,
                    page_number=row["page_number"],
                    section=row["section"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                ))
        
        return results
    
    def _get_chunk_metadata(self, chunk_id: str) -> Optional[Dict]:
        """Get chunk metadata from database."""
        if not self.db_path:
            return None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
            
            if row:
                meta = dict(row)
                meta["metadata"] = json.loads(meta["metadata"]) if meta["metadata"] else {}
                return meta
        return None
    
    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document.

        SQLite is the source of truth; FAISS IndexFlatIP has no efficient
        deletion, so deleted chunk_ids are tombstoned in-memory (mappings
        dropped) and filtered in :meth:`search`. When the index supports
        removal/reconstruction we additionally rebuild without the deleted
        vectors; otherwise filtering guarantees stale hits never return.
        """
        if not self.db_path:
            return 0

        # Get chunk IDs to delete
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT chunk_id, faiss_index FROM chunks WHERE document_id = ?",
                (document_id,)
            ).fetchall()

            chunk_ids = [r["chunk_id"] for r in rows]
            faiss_indices = [r["faiss_index"] for r in rows if r["faiss_index"] is not None]

            # Delete from SQLite
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.commit()

        # Tombstone + drop in-memory mappings so stale FAISS hits are skipped.
        for cid in chunk_ids:
            self._deleted_chunk_ids.add(cid)
            self.chunk_id_to_index.pop(cid, None)
        for fi in faiss_indices:
            self.index_to_chunk_id.pop(fi, None)

        # Best-effort FAISS cleanup: native remove_ids when available,
        # else rebuild a fresh IndexFlatIP from surviving vectors.
        if FAISS_AVAILABLE and self.index is not None and faiss_indices:
            try:
                if hasattr(self.index, "remove_ids"):
                    try:
                        import numpy as _np

                        ids = _np.array(sorted(set(faiss_indices)), dtype=_np.int64)
                        self.index.remove_ids(ids)  # type: ignore[attr-defined]
                    except Exception:
                        self._rebuild_index_without(set(faiss_indices))
                else:
                    self._rebuild_index_without(set(faiss_indices))
                self.persist()
            except Exception as exc:
                logger.warning("FAISS index cleanup skipped (%s); tombstone filter active.", exc)

        logger.info(f"Deleted {len(chunk_ids)} chunks for document {document_id}")

        return len(chunk_ids)

    def _rebuild_index_without(self, condemned: set[int]) -> None:
        """Rebuild IndexFlatIP excluding condemned positions (best-effort).

        Reconstructs surviving vectors, builds a fresh index, and reassigns
        dense faiss positions; mapping dicts and ``_next_index`` are updated.
        No-op when reconstruction is unsupported.
        """
        if not FAISS_AVAILABLE or self.index is None:
            return
        if not hasattr(self.index, "reconstruct"):
            return
        survivors: list[tuple[str, Any]] = []
        for pos in sorted(self.index_to_chunk_id):
            if pos in condemned:
                continue
            cid = self.index_to_chunk_id.get(pos)
            if not cid or cid in self._deleted_chunk_ids:
                continue
            try:
                vec = self.index.reconstruct(int(pos))
            except Exception:
                continue
            survivors.append((cid, vec))
        try:
            new_index = faiss.IndexFlatIP(self.dimension)
            if survivors:
                import numpy as _np

                mat = _np.stack([v for _, v in survivors]).astype("float32")
                new_index.add(mat)
            self.index = new_index
            self.chunk_id_to_index = {cid: i for i, (cid, _) in enumerate(survivors)}
            self.index_to_chunk_id = {i: cid for i, (cid, _) in enumerate(survivors)}
            self._next_index = len(survivors)
        except Exception as exc:
            logger.warning("FAISS rebuild skipped (%s); tombstone filter active.", exc)
    
    def get_document_chunks(self, document_id: str) -> List[Dict]:
        """Get all chunks for a document."""
        if not self.db_path:
            return []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY page_number, faiss_index",
                (document_id,)
            ).fetchall()
            
            return [dict(r) for r in rows]
    
    def persist(self) -> None:
        """Persist FAISS index to disk."""
        if self.index_path and FAISS_AVAILABLE and self.index:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))
            logger.debug(f"Persisted FAISS index to {self.index_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        stats = {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_path": str(self.index_path) if self.index_path else None,
            "db_path": str(self.db_path) if self.db_path else None,
        }
        
        if self.db_path:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) as count FROM chunks")
                stats["chunks_in_db"] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(DISTINCT document_id) as count FROM chunks")
                stats["documents"] = cursor.fetchone()[0]
        
        return stats


class SimpleVectorStore:
    """
    Simple in-memory vector store for testing/development.
    Uses numpy for similarity search.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors = []  # List of (chunk_id, vector, metadata)
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Add chunks to store."""
        added = []
        for chunk in chunks:
            embedding = chunk.get("embedding")
            if embedding is None:
                continue
            
            vector = np.array(embedding, dtype=np.float32)
            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            self.vectors.append({
                "chunk_id": chunk["chunk_id"],
                "vector": vector,
                "metadata": chunk,
            })
            added.append(chunk["chunk_id"])
        
        return added
    
    def search(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        document_ids: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """Search using cosine similarity."""
        if not self.vectors:
            return []
        
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_vec = query_vec / query_norm
        
        results = []
        for item in self.vectors:
            meta = item["metadata"]
            
            if document_ids and meta.get("document_id") not in document_ids:
                continue
            
            # Cosine similarity
            score = float(np.dot(query_vec, item["vector"]))
            if score < min_score:
                continue
            
            results.append(SearchResult(
                chunk_id=item["chunk_id"],
                document_id=meta.get("document_id", ""),
                content=meta.get("content", ""),
                score=score,
                metadata=meta.get("metadata", {}),
                page_number=meta.get("page_number", 0),
                section=meta.get("section"),
            ))
        
        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
    
    def delete_document(self, document_id: str) -> int:
        """Delete document chunks."""
        original_len = len(self.vectors)
        self.vectors = [v for v in self.vectors if v["metadata"].get("document_id") != document_id]
        return original_len - len(self.vectors)
    
    def get_stats(self) -> Dict[str, Any]:
        doc_ids = set(v["metadata"].get("document_id") for v in self.vectors)
        return {
            "total_vectors": len(self.vectors),
            "dimension": self.dimension,
            "documents": len(doc_ids),
            "type": "simple_in_memory",
        }


def create_vector_store(
    dimension: int = 384,
    index_path: Optional[str] = None,
    db_path: Optional[str] = None,
    use_simple: bool = False,
) -> VectorStore:
    """Factory function to create vector store."""
    if use_simple or not FAISS_AVAILABLE:
        return SimpleVectorStore(dimension)
    return VectorStore(dimension, index_path, db_path)