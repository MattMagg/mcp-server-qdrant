import logging
import uuid
from typing import Any
from collections import defaultdict

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.settings import METADATA_PATH

logger = logging.getLogger(__name__)

# Import for optimal search
try:
    import voyageai
    VOYAGEAI_AVAILABLE = True
except ImportError:
    VOYAGEAI_AVAILABLE = False
    logger.warning("voyageai not installed - optimal search disabled")

try:
    from sentence_transformers import SparseEncoder
    SPARSE_ENCODER_AVAILABLE = True
except ImportError:
    SPARSE_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers not installed - optimal search disabled")

Metadata = dict[str, Any]
ArbitraryFilter = dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Metadata | None = None


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    :param voyage_api_key: VoyageAI API key for optimal search.
    :param use_multi_query: Enable multi-query expansion for search.
    :param use_reranking: Enable VoyageAI reranking for search.
    :param first_stage_k: Number of candidates before reranking.
    :param hnsw_ef: HNSW search parameter for accuracy.
    """

    def __init__(
        self,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        collection_name: str | None,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: str | None = None,
        field_indexes: dict[str, models.PayloadSchemaType] | None = None,
        voyage_api_key: str | None = None,
        use_multi_query: bool = True,
        use_reranking: bool = True,
        first_stage_k: int = 50,
        hnsw_ef: int = 128,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )
        self._field_indexes = field_indexes

        # Optimal search configuration
        self._voyage_api_key = voyage_api_key
        self._use_multi_query = use_multi_query
        self._use_reranking = use_reranking
        self._first_stage_k = first_stage_k
        self._hnsw_ef = hnsw_ef

        # Initialize optimal search clients
        self._voyage_client = None
        self._sparse_encoder = None
        self._sparse_encoder_initialized = False
        if VOYAGEAI_AVAILABLE and voyage_api_key:
            self._voyage_client = voyageai.Client(api_key=voyage_api_key)

    def _ensure_sparse_encoder(self):
        """Lazy-load SPLADE encoder on first use to avoid blocking startup."""
        if not self._sparse_encoder_initialized and SPARSE_ENCODER_AVAILABLE:
            logger.info("Loading SPLADE encoder (first use)...")
            self._sparse_encoder = SparseEncoder("naver/splade-cocondenser-ensembledistil", device="cpu")
            self._sparse_encoder_initialized = True
            logger.info("SPLADE encoder loaded successfully")

    def _generate_query_variations(self, query: str, num_variations: int = 3) -> list[str]:
        """Generate multiple query perspectives for multi-query expansion."""
        queries = [query]
        templates = [
            f"anatomical aspects: {query}",
            f"energetic qualities: {query}",
            f"traditional teachings: {query}"
        ]
        return queries + templates[:num_variations]

    async def _embed_query_dense(self, query: str) -> list[float]:
        """Embed query with voyage-context-3 (dense vector)."""
        if not self._voyage_client:
            raise ValueError("VoyageAI client not initialized")

        # VoyageAI is sync, run in executor to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()

        def _embed():
            import requests
            url = "https://api.voyageai.com/v1/contextualizedembeddings"
            headers = {
                "Authorization": f"Bearer {self._voyage_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "inputs": [[query]],
                "model": "voyage-context-3",
                "input_type": "query",
                "output_dimension": 2048
            }
            response = requests.post(url, headers=headers, json=data)
            if not response.ok:
                raise ValueError(f"Failed to embed query: {response.text}")
            result = response.json()
            return result["data"][0]["data"][0]["embedding"]

        return await loop.run_in_executor(None, _embed)

    def _embed_query_sparse(self, query: str) -> dict[str, list]:
        """Embed query with SPLADE (sparse vector)."""
        self._ensure_sparse_encoder()
        if not self._sparse_encoder:
            raise ValueError("Sparse encoder not available")

        sparse_embeds = self._sparse_encoder.encode_document([query])
        sparse_vec = sparse_embeds[0].coalesce()

        return {
            "indices": sparse_vec.indices()[0].tolist(),
            "values": sparse_vec.values().tolist()
        }

    def _reciprocal_rank_fusion(self, results_lists: list[list[dict]], k: int = 60) -> list[dict]:
        """Fuse multiple result lists using Reciprocal Rank Fusion."""
        scores = defaultdict(float)
        doc_map = {}

        for results in results_lists:
            for rank, doc in enumerate(results, start=1):
                doc_id = doc["id"]
                scores[doc_id] += 1.0 / (k + rank)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        fused_results = []
        for doc_id in ranked_ids:
            doc = doc_map[doc_id].copy()
            doc["rrf_score"] = scores[doc_id]
            fused_results.append(doc)

        return fused_results

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    async def store(self, entry: Entry, *, collection_name: str | None = None):
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        Uses hybrid vectors (dense + sparse) when VoyageAI is configured.
        :param entry: The entry to store in the Qdrant collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        await self._ensure_collection_exists(collection_name)

        payload = {"document": entry.content, METADATA_PATH: entry.metadata}

        # Check if we should use hybrid embedding
        use_hybrid = (self._voyage_client is not None and SPARSE_ENCODER_AVAILABLE)

        if use_hybrid:
            # Embed with VoyageAI (dense) and SPLADE (sparse)
            import asyncio

            # Dense embedding
            dense_vec = await self._embed_query_dense(entry.content)

            # Sparse embedding
            sparse_vec = await asyncio.get_event_loop().run_in_executor(
                None, self._embed_query_sparse, entry.content
            )

            await self._client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=uuid.uuid4().hex,
                        vector={
                            "dense": dense_vec,
                            "sparse": models.SparseVector(
                                indices=sparse_vec["indices"],
                                values=sparse_vec["values"]
                            )
                        },
                        payload=payload,
                    )
                ],
            )
        else:
            # Use simple embedding provider
            embeddings = await self._embedding_provider.embed_documents([entry.content])
            vector_name = self._embedding_provider.get_vector_name()

            await self._client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=uuid.uuid4().hex,
                        vector={vector_name: embeddings[0]},
                        payload=payload,
                    )
                ],
            )

    async def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection using optimal multi-stage retrieval pipeline.

        Pipeline:
        1. Multi-query expansion (if enabled)
        2. Hybrid search (dense + sparse with RRF fusion)
        3. VoyageAI reranking (if enabled)

        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :param query_filter: The filter to apply to the query, if any.

        :return: A list of entries found.
        """
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        # Check if optimal search is available
        use_optimal = (self._voyage_client is not None and self._sparse_encoder is not None)

        if not use_optimal:
            logger.info("Falling back to simple search - VoyageAI or sparse encoder not available")
            # Fallback to simple search
            query_vector = await self._embedding_provider.embed_query(query)
            vector_name = self._embedding_provider.get_vector_name()

            search_results = await self._client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using=vector_name,
                limit=limit,
                query_filter=query_filter,
            )

            return [
                Entry(
                    content=result.payload["document"],
                    metadata=result.payload.get("metadata"),
                )
                for result in search_results.points
            ]

        # Optimal search pipeline
        logger.info("Using optimal search pipeline")

        # Stage 1: Multi-query expansion
        if self._use_multi_query:
            query_variations = self._generate_query_variations(query)
        else:
            query_variations = [query]

        # Stage 2: Hybrid search for each query variation
        import asyncio
        all_results = []

        for query_var in query_variations:
            # Embed query (dense + sparse)
            dense_vec = await self._embed_query_dense(query_var)
            sparse_vec = await asyncio.get_event_loop().run_in_executor(
                None, self._embed_query_sparse, query_var
            )

            # Hybrid search with RRF fusion
            search_result = await self._client.query_points(
                collection_name=collection_name,
                prefetch=[
                    # Dense vector search
                    models.Prefetch(
                        query=dense_vec,
                        using="dense",
                        limit=self._first_stage_k,
                        params=models.SearchParams(
                            hnsw_ef=self._hnsw_ef,
                            quantization=models.QuantizationSearchParams(
                                rescore=True,
                                oversampling=2.0
                            )
                        )
                    ),
                    # Sparse vector search
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_vec["indices"],
                            values=sparse_vec["values"]
                        ),
                        using="sparse",
                        limit=self._first_stage_k
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=self._first_stage_k if not self._use_multi_query else limit * 2,
                query_filter=query_filter,
                with_payload=True
            )

            # Convert to dict format for RRF fusion
            results = []
            for point in search_result.points:
                results.append({
                    "id": point.id,
                    "payload": point.payload,
                    "score": point.score if hasattr(point, 'score') else 0,
                })

            all_results.append(results)

        # Stage 2b: Fuse results from multiple queries
        if self._use_multi_query and len(all_results) > 1:
            candidates = self._reciprocal_rank_fusion(all_results)[:self._first_stage_k]
        else:
            candidates = all_results[0]

        # Stage 3: VoyageAI reranking
        if self._use_reranking and len(candidates) > 0:
            documents = [c["payload"]["document"] for c in candidates]

            # Run reranking in executor (sync operation)
            def _rerank():
                return self._voyage_client.rerank(
                    query=query,
                    documents=documents,
                    model="rerank-2.5",
                    top_k=min(limit, len(documents))
                )

            reranking = await asyncio.get_event_loop().run_in_executor(None, _rerank)

            # Map reranked results back
            final_results = []
            for result in reranking.results:
                original = candidates[result.index]
                final_results.append(
                    Entry(
                        content=original["payload"]["document"],
                        metadata=original["payload"].get("metadata"),
                    )
                )
        else:
            final_results = [
                Entry(
                    content=c["payload"]["document"],
                    metadata=c["payload"].get("metadata"),
                )
                for c in candidates[:limit]
            ]

        return final_results

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        Supports both simple (single vector) and hybrid (dense + sparse) configurations.
        :param collection_name: The name of the collection to ensure exists.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            # Check if we should create hybrid collection
            use_hybrid = (self._voyage_client is not None and self._sparse_encoder is not None)

            if use_hybrid:
                logger.info(f"Creating hybrid collection '{collection_name}' with dense + sparse vectors")
                # Create collection with hybrid vectors (dense + sparse)
                await self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=2048,  # voyage-context-3 dimension
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            index=models.SparseIndexParams()
                        )
                    }
                )
            else:
                logger.info(f"Creating simple collection '{collection_name}' with single vector")
                # Create simple collection with single vector
                vector_size = self._embedding_provider.get_vector_size()
                vector_name = self._embedding_provider.get_vector_name()
                await self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        vector_name: models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE,
                        )
                    },
                )

            # Create payload indexes if configured
            if self._field_indexes:
                for field_name, field_type in self._field_indexes.items():
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )
