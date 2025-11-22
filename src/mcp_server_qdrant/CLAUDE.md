# RAG Query Execution for Agent Tasks

## When to Use RAG

Use RAG queries when you need authoritative yoga knowledge from traditional texts to ground your work in authentic and sourced information.

## How to Execute Queries

### Setup

```bash
# Environment variables are in .env.local
export $(cat .env.local | grep -E "(VOYAGE|QDRANT)" | xargs)
export TOKENIZERS_PARALLELISM=false


python3 src/embeddings/qdrant/query_optimal.py "your query here" \
  --top-k 5 \
  --collection light-on-yoga \
  --verbose
```

```python
import sys

sys.path.insert(0, 'src/embeddings')
from qdrant.query_optimal import search_optimal

results = search_optimal(
    query="your query text",
    top_k=3,
    collection_name="light-on-yoga",
    filters={"doc_id": "light_on_yoga"},  # or yoga_sequencing, yoga_anatomy_notes, etc.
    multi_query=True,
    rerank=True
)

for r in results['results']:
    print(f"Score: {r['score']:.4f}")
    print(f"Text: {r['text'][:200]}...")
```


## Available Sources (doc_id filters)

light_on_yoga - B.K.S. Iyengar

yoga_sequencing - Mark Stephens

yoga_anatomy_notes - Anatomy & physiology

teaching_hatha_yoga - Teaching methodology

yin_yoga_main - Yin yoga specific

power_yoga - Baron Baptiste

kundalini_yoga_sivananda - Swami Sivananda

yoga_teaching - General teaching

yoga_200hr_manual - Teacher training



Collection Name

Always use: light-on-yoga (all documents are in this one collection, filtered by doc_id)



Now let me revise the article based on the actual RAG results. The key findings were:

1. **Query 3 (0.76 score)**: Iyengar's explicit cardiovascular contraindications

2. **Query 4 (0.70 score)**: Iyengar's pregnancy guidance with specific poses
