"""
Vector Memory Store for HyperCortex-AI
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import structlog

from ..core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class MemoryEntry:
    """Represents a memory entry with vector embedding"""
    
    def __init__(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        embedding: Optional[np.ndarray] = None,
        entry_id: Optional[str] = None
    ):
        self.id = entry_id or str(uuid.uuid4())
        self.content = content
        self.metadata = metadata or {}
        self.embedding = embedding
        self.timestamp = datetime.utcnow()
        
        # Add default metadata
        self.metadata.update({
            "created_at": self.timestamp.isoformat(),
            "content_length": len(content)
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary"""
        entry = cls(
            content=data["content"],
            metadata=data.get("metadata", {}),
            entry_id=data["id"]
        )
        
        if data.get("embedding"):
            entry.embedding = np.array(data["embedding"])
        
        if data.get("timestamp"):
            entry.timestamp = datetime.fromisoformat(data["timestamp"])
        
        return entry


class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    async def add_memory(self, entry: MemoryEntry) -> str:
        """Add a memory entry"""
        pass
    
    @abstractmethod
    async def search_similar(
        self, 
        query: str, 
        k: int = 5, 
        threshold: float = 0.7
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search for similar memories"""
        pass
    
    @abstractmethod
    async def get_memory(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID"""
        pass
    
    @abstractmethod
    async def delete_memory(self, entry_id: str) -> bool:
        """Delete a memory entry"""
        pass
    
    @abstractmethod
    async def clear_all(self) -> bool:
        """Clear all memories"""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store implementation"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        import faiss
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Memory storage
        self.memories: Dict[str, MemoryEntry] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        
        logger.info(f"FAISS Vector Store initialized with dimension {self.dimension}")
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to vector embedding"""
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding.astype(np.float32)
    
    async def add_memory(self, entry: MemoryEntry) -> str:
        """Add a memory entry to the vector store"""
        
        # Generate embedding if not provided
        if entry.embedding is None:
            entry.embedding = self._encode_text(entry.content)
        
        # Add to FAISS index
        current_index = self.index.ntotal
        self.index.add(entry.embedding.reshape(1, -1))
        
        # Update mappings
        self.id_to_index[entry.id] = current_index
        self.index_to_id[current_index] = entry.id
        self.memories[entry.id] = entry
        
        logger.info(f"Added memory entry {entry.id} to vector store")
        return entry.id
    
    async def search_similar(
        self, 
        query: str, 
        k: int = 5, 
        threshold: float = 0.7
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search for similar memories"""
        
        if self.index.ntotal == 0:
            return []
        
        # Encode query
        query_embedding = self._encode_text(query)
        
        # Search in FAISS
        scores, indices = self.index.search(query_embedding.reshape(1, -1), min(k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx in self.index_to_id:
                memory_id = self.index_to_id[idx]
                memory = self.memories[memory_id]
                results.append((memory, float(score)))
        
        logger.info(f"Found {len(results)} similar memories for query")
        return results
    
    async def get_memory(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID"""
        return self.memories.get(entry_id)
    
    async def delete_memory(self, entry_id: str) -> bool:
        """Delete a memory entry"""
        if entry_id not in self.memories:
            return False
        
        # Note: FAISS doesn't support deletion, so we just remove from our mappings
        # In production, consider rebuilding the index periodically
        index_pos = self.id_to_index[entry_id]
        
        del self.memories[entry_id]
        del self.id_to_index[entry_id]
        del self.index_to_id[index_pos]
        
        logger.info(f"Deleted memory entry {entry_id}")
        return True
    
    async def clear_all(self) -> bool:
        """Clear all memories"""
        self.index.reset()
        self.memories.clear()
        self.id_to_index.clear()
        self.index_to_id.clear()
        
        logger.info("Cleared all memories from vector store")
        return True


class MemoryManager:
    """High-level memory management interface"""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or FAISSVectorStore()
        self.settings = settings
        
        logger.info("Memory Manager initialized")
    
    async def store_memory(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        category: str = "general"
    ) -> str:
        """Store a new memory"""
        
        # Add category to metadata
        metadata = metadata or {}
        metadata["category"] = category
        
        entry = MemoryEntry(content=content, metadata=metadata)
        memory_id = await self.vector_store.add_memory(entry)
        
        logger.info(f"Stored memory: {memory_id}")
        return memory_id
    
    async def recall_memories(
        self,
        query: str,
        k: int = 5,
        category: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> List[Tuple[MemoryEntry, float]]:
        """Recall relevant memories"""
        
        threshold = threshold or self.settings.memory_similarity_threshold
        
        # Search for similar memories
        results = await self.vector_store.search_similar(query, k * 2, threshold)
        
        # Filter by category if specified
        if category:
            results = [
                (memory, score) for memory, score in results
                if memory.metadata.get("category") == category
            ]
        
        # Limit results
        results = results[:k]
        
        logger.info(f"Recalled {len(results)} memories for query: {query}")
        return results
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID"""
        return await self.vector_store.get_memory(memory_id)
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory"""
        return await self.vector_store.delete_memory(memory_id)
    
    async def clear_category(self, category: str) -> int:
        """Clear all memories in a category"""
        # This is a simplified implementation
        # In production, you'd want a more efficient approach
        count = 0
        for memory_id, memory in list(self.vector_store.memories.items()):
            if memory.metadata.get("category") == category:
                await self.delete_memory(memory_id)
                count += 1
        
        logger.info(f"Cleared {count} memories from category: {category}")
        return count


# Global memory manager instance
memory_manager = MemoryManager()


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance"""
    return memory_manager