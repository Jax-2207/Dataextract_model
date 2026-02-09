"""
Learned Answers Storage - MongoDB collection for self-learning RAG
Stores high-confidence answers from internet searches for future reuse
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from app.config import MONGODB_URI, MONGODB_DB_NAME

# Collection name for learned answers
LEARNED_ANSWERS_COLLECTION = "learned_answers"

_client = None
_db = None


def get_db():
    """Get MongoDB database connection"""
    global _client, _db
    
    if _db is None:
        if not MONGODB_URI:
            return None
        try:
            _client = MongoClient(MONGODB_URI)
            _db = _client[MONGODB_DB_NAME]
        except ConnectionFailure:
            return None
    
    return _db


def get_learned_collection():
    """Get the learned answers collection"""
    db = get_db()
    if db is None:
        return None
    return db[LEARNED_ANSWERS_COLLECTION]


def search_learned_answer(question: str) -> Optional[Dict[str, Any]]:
    """
    Search for a previously learned answer to a similar question.
    
    Args:
        question: The user's question
    
    Returns:
        Learned answer document if found, None otherwise
    """
    collection = get_learned_collection()
    if collection is None:
        return None
    
    # Simple text search - look for similar questions
    # In production, you might use text search or embeddings for better matching
    try:
        # First try exact match
        result = collection.find_one({"question_lower": question.lower().strip()})
        if result:
            # Update access count and last accessed time
            collection.update_one(
                {"_id": result["_id"]},
                {
                    "$inc": {"access_count": 1},
                    "$set": {"last_accessed": datetime.utcnow()}
                }
            )
            return result
        
        return None
    except Exception as e:
        print(f"Error searching learned answers: {e}")
        return None


def save_learned_answer(
    question: str,
    answer: str,
    confidence_score: int,
    source_query: str = None
) -> bool:
    """
    Save a high-confidence internet answer to the database.
    
    Args:
        question: The original question
        answer: The answer from internet search
        confidence_score: The confidence score (should be >= 90)
        source_query: The search query used
    
    Returns:
        True if saved successfully, False otherwise
    """
    collection = get_learned_collection()
    if collection is None:
        return False
    
    try:
        doc = {
            "question": question,
            "question_lower": question.lower().strip(),
            "answer": answer,
            "confidence_score": confidence_score,
            "source": "internet",
            "source_query": source_query,
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow(),
            "access_count": 1
        }
        
        # Upsert - update if exists, insert if not
        collection.update_one(
            {"question_lower": question.lower().strip()},
            {"$set": doc},
            upsert=True
        )
        
        print(f"✅ Saved learned answer with {confidence_score}% confidence")
        return True
        
    except Exception as e:
        print(f"Error saving learned answer: {e}")
        return False


def get_all_learned_answers(limit: int = 100) -> List[Dict[str, Any]]:
    """Get all learned answers for admin view"""
    collection = get_learned_collection()
    if collection is None:
        return []
    
    try:
        return list(collection.find().sort("created_at", -1).limit(limit))
    except Exception:
        return []


def delete_learned_answer(question: str) -> bool:
    """Delete a learned answer"""
    collection = get_learned_collection()
    if collection is None:
        return False
    
    try:
        result = collection.delete_one({"question_lower": question.lower().strip()})
        return result.deleted_count > 0
    except Exception:
        return False


def get_learned_stats() -> Dict[str, Any]:
    """Get statistics about learned answers"""
    collection = get_learned_collection()
    if collection is None:
        return {"total": 0, "avg_confidence": 0}
    
    try:
        total = collection.count_documents({})
        
        if total == 0:
            return {"total": 0, "avg_confidence": 0}
        
        # Get average confidence
        pipeline = [
            {"$group": {"_id": None, "avg_confidence": {"$avg": "$confidence_score"}}}
        ]
        result = list(collection.aggregate(pipeline))
        avg_confidence = result[0]["avg_confidence"] if result else 0
        
        return {
            "total": total,
            "avg_confidence": round(avg_confidence, 1)
        }
    except Exception:
        return {"total": 0, "avg_confidence": 0}
