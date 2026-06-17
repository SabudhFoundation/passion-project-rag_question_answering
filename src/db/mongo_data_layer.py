"""
src/db/mongo_data_layer.py
==========================
Custom Chainlit Data Layer using MongoDB (Motor).
Implements BaseDataLayer for ChatGPT-like persistent chat history.
"""
import uuid
import logging
from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass
from typing import Dict, List, Optional

import chainlit as cl
import chainlit.data as cl_data
from chainlit.user import PersistedUser
from chainlit.types import (
    Feedback,
    FeedbackDict,
    ThreadDict,
    Pagination,
    ThreadFilter,
    PaginatedResponse,
    PageInfo,
)
from chainlit.step import StepDict
from chainlit.element import Element, ElementDict
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MongoDataLayer(cl_data.BaseDataLayer):
    """
    Chainlit Data Layer backed by MongoDB (Motor async driver).
    Provides full ChatGPT-style persistent chat history.
    """

    def __init__(self, connection_string: str, database_name: str):
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client[database_name]

        # Collections
        self.users     = self.db["users"]
        self.threads   = self.db["threads"]
        self.steps     = self.db["steps"]
        self.elements  = self.db["elements"]
        self.feedback  = self.db["feedback"]
        logger.info("MongoDataLayer: connected to database '%s'", database_name)

    # ── Users ────────────────────────────────────────────────────────────────

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        doc = await self.users.find_one({"identifier": identifier})
        if not doc or "id" not in doc:
            # No doc or legacy doc missing the id field → let Chainlit call create_user
            return None
        return PersistedUser(
            id=doc["id"],
            identifier=doc["identifier"],
            metadata=doc.get("metadata", {}),
            createdAt=doc.get("createdAt", _now_iso()),
        )

    async def create_user(self, user: cl.User) -> Optional[PersistedUser]:
        # Generate a stable id from identifier so it is always the same for this user
        user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, user.identifier))
        now = _now_iso()
        await self.users.update_one(
            {"identifier": user.identifier},
            {"$setOnInsert": {
                "id": user_id,
                "identifier": user.identifier,
                "metadata": user.metadata,
                "createdAt": now,
            }},
            upsert=True,
        )
        # Fetch back to return current state
        doc = await self.users.find_one({"identifier": user.identifier})
        return PersistedUser(
            id=doc["id"],
            identifier=doc["identifier"],
            metadata=doc.get("metadata", {}),
            createdAt=doc.get("createdAt", now),
        )

    # ── Steps ────────────────────────────────────────────────────────────────

    async def create_step(self, step_dict: StepDict):
        # Use upsert so duplicate create calls don't explode
        sid = step_dict.get("id")
        if sid:
            await self.steps.update_one(
                {"id": sid},
                {"$setOnInsert": dict(step_dict)},
                upsert=True,
            )
        else:
            await self.steps.insert_one(dict(step_dict))

    async def update_step(self, step_dict: StepDict):
        sid = step_dict.get("id")
        if sid:
            await self.steps.update_one(
                {"id": sid},
                {"$set": dict(step_dict)},
                upsert=True,
            )

    async def delete_step(self, step_id: str):
        await self.steps.delete_one({"id": step_id})

    # ── Elements ─────────────────────────────────────────────────────────────

    async def create_element(self, element: Element):
        element_dict = element.to_dict()
        eid = element_dict.get("id")
        if eid:
            await self.elements.update_one(
                {"id": eid},
                {"$set": element_dict},
                upsert=True,
            )
        else:
            await self.elements.insert_one(element_dict)

    async def get_element(
        self, thread_id: str, element_id: str
    ) -> Optional[ElementDict]:
        doc = await self.elements.find_one(
            {"threadId": thread_id, "id": element_id}
        )
        if doc:
            doc.pop("_id", None)
            return doc
        return None

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        query: Dict = {"id": element_id}
        if thread_id:
            query["threadId"] = thread_id
        await self.elements.delete_many(query)

    # ── Feedback ─────────────────────────────────────────────────────────────

    async def upsert_feedback(self, feedback: Feedback) -> str:
        fid = feedback.id or str(uuid.uuid4())
        if is_dataclass(feedback):
            fdict = asdict(feedback)
        else:
            fdict = dict(feedback.__dict__) if hasattr(feedback, "__dict__") else {}
        fdict["id"] = fid
        await self.feedback.update_one(
            {"id": fid}, {"$set": fdict}, upsert=True
        )
        return fid

    async def delete_feedback(self, feedback_id: str) -> bool:
        res = await self.feedback.delete_one({"id": feedback_id})
        return res.deleted_count > 0

    # ── Threads ──────────────────────────────────────────────────────────────

    async def get_thread_author(self, thread_id: str) -> str:
        thread_doc = await self.threads.find_one(
            {"id": thread_id}, {"userIdentifier": 1, "userId": 1}
        )
        if not thread_doc:
            return ""
            
        uid = thread_doc.get("userIdentifier") or thread_doc.get("userId")
        if not uid:
            return ""
            
        # Chainlit's ACL compares this against current_user.identifier (e.g. "local_user")
        # so we must look up the user's readable identifier from the users collection
        user_doc = await self.users.find_one({"id": uid}, {"identifier": 1})
        if user_doc and "identifier" in user_doc:
            return user_doc["identifier"]
            
        return uid

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        thread_doc = await self.threads.find_one({"id": thread_id})
        if not thread_doc:
            return None
        thread_doc.pop("_id", None)

        # Fix userIdentifier for socket.py ACL checks
        uid = thread_doc.get("userIdentifier") or thread_doc.get("userId")
        if uid:
            user_doc = await self.users.find_one({"id": uid}, {"identifier": 1})
            if user_doc and "identifier" in user_doc:
                thread_doc["userIdentifier"] = user_doc["identifier"]


        # Attach steps (sorted oldest → newest)
        steps = []
        async for s in self.steps.find({"threadId": thread_id}).sort("createdAt", 1):
            s.pop("_id", None)
            steps.append(s)

        # Attach elements
        elements = []
        async for e in self.elements.find({"threadId": thread_id}):
            e.pop("_id", None)
            elements.append(e)

        thread_doc["steps"] = steps
        thread_doc["elements"] = elements
        return thread_doc

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Called by Chainlit whenever a thread is created or updated.
        We always ensure all required ThreadDict fields exist.
        """
        # Build the fields that are changing
        set_fields: Dict = {}
        if name is not None:
            set_fields["name"] = name
        if user_id is not None:
            set_fields["userIdentifier"] = user_id
        if metadata is not None:
            set_fields["metadata"] = metadata
        if tags is not None:
            set_fields["tags"] = tags

        # On insert, ensure mandatory ThreadDict fields are present
        set_on_insert: Dict = {
            "id": thread_id,
            "createdAt": _now_iso(),
        }

        await self.threads.update_one(
            {"id": thread_id},
            {
                "$set": set_fields,
                "$setOnInsert": set_on_insert,
            },
            upsert=True,
        )

    async def delete_thread(self, thread_id: str):
        await self.threads.delete_one({"id": thread_id})
        await self.steps.delete_many({"threadId": thread_id})
        await self.elements.delete_many({"threadId": thread_id})

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        """
        Return paginated threads for the sidebar.
        `pagination.first` is the page SIZE (not skip offset).
        `pagination.cursor` is an opaque cursor (we use the last createdAt).
        """
        query: Dict = {}
        if filters.userId:
            query["userIdentifier"] = filters.userId
        if filters.search:
            query["name"] = {"$regex": filters.search, "$options": "i"}

        limit = pagination.first or 20

        # Cursor-based pagination: cursor encodes the createdAt of the last item
        if pagination.cursor:
            query["createdAt"] = {"$lt": pagination.cursor}

        cursor = self.threads.find(query).sort("createdAt", -1).limit(limit + 1)
        docs = []
        async for doc in cursor:
            doc.pop("_id", None)
            # Ensure required ThreadDict fields are present
            doc.setdefault("steps", [])
            doc.setdefault("elements", [])
            docs.append(doc)

        has_next = len(docs) > limit
        if has_next:
            docs = docs[:limit]

        end_cursor = docs[-1]["createdAt"] if (has_next and docs) else None
        start_cursor = docs[0]["createdAt"] if docs else None

        return PaginatedResponse(
            data=docs,
            pageInfo=PageInfo(
                hasNextPage=has_next,
                startCursor=start_cursor,
                endCursor=end_cursor,
            ),
        )

    # ── Misc ─────────────────────────────────────────────────────────────────

    async def build_debug_url(self) -> str:
        return ""

    async def get_favorite_steps(self, user_id: str) -> List[StepDict]:
        return []

    async def close(self) -> None:
        if self.client:
            self.client.close()
            logger.info("MongoDataLayer: connection closed.")
