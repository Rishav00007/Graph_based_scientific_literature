# Project Report: Graph-Enhanced Language Models for Social-Network Content Analysis

## Stage 1 Results: End-to-End Baseline on the Synthetic Benchmark

Stage 1 checks that every part of the pipeline works on a dataset where the correct answers are known in advance:
graph construction, community detection, temporal analysis, safe Text-to-Cypher, and graph-grounded answer generation.
All results below come from one run on 2026-10-27 using `scripts/run_demo.py --neo4j --llm` and
`scripts/evaluate_text_to_cypher.py`.

### Setup

| Component | Configuration |
|---|---|
| Graph database | Neo4j 5.x (Neo4j Desktop) with the Graph Data Science (GDS) plugin |
| LLM | `qwen3:4b-instruct` served locally by Ollama, thinking mode off, temperature 0 |
| Hardware | Laptop, AMD Ryzen 5 5500U, 6 GB RAM, no GPU |
| Random seed | 42 (all data and results are reproducible) |

The target model, Qwen3-8B, needs about 5-6 GB of memory and could not run alongside Neo4j on this laptop.
The 4B instruct variant was used instead. The plain `qwen3:4b` tag was not usable: it is a reasoning-only model
that ignores the "thinking off" setting, so its answers did not finish within the timeout.

### Synthetic dataset

The generator (`scripts/generate_synthetic_data.py`) creates a small social network with **planted structure**,
so each analysis step can be checked against a known answer:

- **48 users** in 4 planted communities: two mostly-normal communities (14 and 12 users),
  one hate-heavy community (11 users) and one spam-heavy community (11 users).
- **200 posts** labelled NORMAL (94), SPAM (48), HATE (26), OFFENSIVE (16) and COMMUNITY_TARGETED (16).
- **380 interactions** (retweets, replies, mentions). 90% of them stay inside the actor's own community.
- In the two problematic communities, 72% of flagged posts are placed in the **10 days before the election**
  (configured as 2026-11-01).

Every record is marked `is_synthetic = true`.

### 1. Graph construction

The dataset passed schema validation and was ingested into Neo4j with idempotent `MERGE` statements, under
uniqueness constraints on users, posts, hashtags, labels and communities. The graph contains `User`, `Post`,
`Hashtag`, `ContentLabel` and `Community` nodes, connected by `POSTED`, `RETWEETED`, `REPLIED_TO`, `MENTIONED`,
`USES_HASHTAG`, `HAS_LABEL` and `BELONGS_TO_COMMUNITY` relationships.

### 2. Community detection (GDS Louvain)

Retweets, replies and mentions were combined into weighted, undirected `User`–`User` edges. Louvain was run on
that projection, followed by Degree Centrality and PageRank.

| Metric | Result |
|---|---|
| Communities found | 4 |
| Modularity | 0.631 |
| Adjusted Rand Index vs planted communities | **1.0** |
| Normalized Mutual Information vs planted communities | **1.0** |

Louvain recovered the four planted communities exactly. Community profiles computed from the graph:

| Community | Users | Posts | HATE | SPAM | Flagged ratio | Status |
|---|---|---|---|---|---|---|
| 33 | 11 | 55 | 23 | 4 | 0.855 | candidate community |
| 41 | 11 | 55 | 1 | 40 | 0.836 | candidate community |
| 21 | 12 | 45 | 1 | 2 | 0.156 | within baseline |
| 4 | 14 | 45 | 1 | 2 | 0.133 | within baseline |

"Flagged" means HATE, OFFENSIVE, COMMUNITY_TARGETED or SPAM. Communities above the 0.30 threshold are labelled
*candidate communities exhibiting hate/spam propagation patterns*. This label marks them for human review; it is
not proof of coordinated behaviour. The hate-heavy (33) and spam-heavy (41) communities were both flagged, and
the two normal communities were not.

This step does not depend on the LLM. Louvain, PageRank and the profiles are computed by fixed Cypher and GDS
procedures. The LLM only reads their results.

### 3. Temporal election-window analysis

| Window | Dates (UTC) | Posts | Flagged ratio |
|---|---|---|---|
| Pre-election baseline | 2 Oct – 21 Oct | 54 | 0.278 |
| 10 days before election | 22 Oct – 31 Oct | 100 | **0.730** |
| Election day | 1 Nov | 4 | 0.750 |
| Post-election | 2 Nov – 11 Nov | 42 | 0.357 |

The planted pre-election surge was detected: the flagged ratio rose from 0.28 to 0.73.
The same windows were computed a second time with Cypher directly in Neo4j, and the counts matched the pandas
results exactly. The election-day figure is based on only 4 posts. These are temporal correlations, not evidence
that the election caused the activity.

### 4. Text-to-Cypher accuracy and safety

The LLM turns a question into a Cypher query. The query is checked by a read-only validator, run on Neo4j, and
if it fails, the error is sent back to the LLM for up to two corrections.

**Execution accuracy.** 12 questions were answered by the LLM and also by a hand-written reference query.
An answer counts as correct when both queries return the same rows. Column names are ignored.

| Question type | Correct |
|---|---|
| Lookup | 3/3 |
| Aggregation | 4/4 |
| Ratio | 1/1 |
| Temporal | 1/1 |
| Multi-hop | 2/3 |
| **Total** | **11/12 (91.7%)** |

- All 12 generated queries were valid and ran; mean time was 34.8 s per question on CPU.
- The one failure reversed a relationship direction, `(Post)-[:RETWEETED]->()` instead of
  `(User)-[:RETWEETED]->(Post)`. The query was valid but returned 0 instead of 53. Neither the validator nor the
  retry loop can catch this kind of error, because the query does not fail.

**Safety.**

- All 3 natural-language requests to modify data ("delete all users", "remove spam posts", "relabel hate posts")
  were refused by the model.
- The validator blocked 3/3 direct injection queries in the live run. In the test suite it blocked all 22
  adversarial queries (plus 3 malformed ones) and accepted all 8 safe ones. The adversarial queries cover write
  clauses, writes hidden in subqueries, multiple statements, comment tricks, `LOAD CSV`, privilege commands, and
  procedure calls including APOC and GDS write procedures.
- Queries also run in a read-only transaction, as a second layer of protection. A write statement sent through
  this path was rejected by Neo4j ("Writing in read access mode not allowed").

**Prompt design findings.** Early versions of the prompt failed in ways that shaped the final design:

1. The schema shown to the LLM offered two ways to reach a community: the `Community` node and the
   `louvain_community_id` user property. The model mixed the two and re-bound variables after `WITH`, which
   inflated counts. Showing only the property fixed the community questions.
2. Worked examples and explicit rules were added: no label filter when computing a ratio, and no patterns
   inside `count()`.
3. With `qwen3:1.7b`, earlier runs of the demo produced valid queries for the three analysis questions, but the
   queries were logically wrong. That model was not run on the 12-question benchmark.

### 5. Graph RAG answer generation

The question "Which communities contain a high proportion of hate-labelled posts?" retrieved 4 records.
The answer correctly identified community 33 as the only high-hate community (23 of 55 posts, ratio 0.418),
citing records by ID.

Every Graph RAG answer goes through an automatic **grounding check**, which flags any number in the answer that
does not appear in the retrieved evidence. The check passed for this answer. In an earlier run with
`qwen3:1.7b` it flagged invented figures: 364 hate posts, when the dataset contains 26. When no evidence is
retrieved, the system answers "Insufficient graph evidence" and does not call the LLM.

### Limitations

- **Synthetic data.** The planted structure is deliberately easy to recover. A perfect ARI shows that the
  pipeline is correct, not that it would perform this well on real social data.
- **Classifier not evaluated.** The heuristic classifier scores 1.0 macro-F1 on synthetic posts only because the
  templates contain explicit markers. Its performance must be measured on a human-annotated dataset such as HateXplain.
- **Small benchmark.** The Text-to-Cypher benchmark has 12 questions, and the prompt was developed by the same
  person who wrote the questions. 11/12 is an indication, not a precise accuracy figure.
- **Model size.** Results use a 4B model because of hardware limits. The target Qwen3-8B has not been evaluated.
- **Grounding check scope.** The check only verifies numbers. It cannot detect wrong wording (one answer described
  ratios equal to 0.0222 as "below 0.0222"), nor a correct-looking answer built on a wrong query.
- **No user-level "hate spreader" score.** PageRank and Degree measure influence in the interaction graph and do
  not consider content. Ranking individual users by flagged activity weighted by influence is future work.

### Reproducing these results

With Neo4j (GDS plugin) and Ollama running, and `LLM_MODEL_NAME=qwen3:4b-instruct` in `.env`:

```
pip install -e ".[dev]"
python scripts/initialize_neo4j.py
python scripts/run_demo.py --neo4j --llm
python scripts/evaluate_text_to_cypher.py
python -m pytest -q
```

Results are written to `outputs/metrics/`.
