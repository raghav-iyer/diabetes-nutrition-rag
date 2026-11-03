# Index statistics

Documents indexed: 7956 (with extraction: 7956, with full text: 2529). Extraction model: `/models/gemma-4-26B-A4B-it`. Vector index: hnsw.

| Strategy | Chunks | Section chunks | Tokens/section chunk (mean, p50, p95, max) | Embed s | Index s |
|---|---|---|---|---|---|
| hierarchical | 145527 | 129192 | 132.8, 139, 160, 225 | 466.3 | 173.3 |

Chunks by kind: hierarchical: {'abstract': 7956, 'section': 129192, 'table': 8379}

Table sizes: chunks 606 MB, chunks_doc 4096 kB, chunks_filters 4096 kB, chunks_hnsw_hierarchical 1137 MB, chunks_hnsw_recursive 628 MB, chunks_hnsw_sliding_window 544 MB, chunks_hnsw_token_aware 562 MB, chunks_pkey 26 MB, chunks_pop 5056 kB, chunks_topics 4760 kB, chunks_tsv 130 MB

Total build time: 752.9 s
