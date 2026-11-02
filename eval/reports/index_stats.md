# Index statistics

Documents indexed: 7956 (with extraction: 7956, with full text: 2529). Extraction model: `/models/gemma-4-26B-A4B-it`. Vector index: hnsw.

| Strategy | Chunks | Section chunks | Tokens/section chunk (mean, p50, p95, max) | Embed s | Index s |
|---|---|---|---|---|---|
| sliding_window | 69637 | 53302 | 360.5, 400, 401, 500 | 6355.1 | 41.6 |

Chunks by kind: sliding_window: {'abstract': 7956, 'section': 53302, 'table': 8379}

Table sizes: chunks 347 MB, chunks_doc 4096 kB, chunks_filters 4096 kB, chunks_hnsw_recursive 628 MB, chunks_hnsw_sliding_window 544 MB, chunks_hnsw_token_aware 562 MB, chunks_pkey 14 MB, chunks_pop 3920 kB, chunks_topics 3584 kB, chunks_tsv 98 MB

Total build time: 6470.3 s
