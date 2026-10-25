# Public Dataset Research & Selection Report

## 1. Candidate Dataset Comparison Table

| Dataset Name | Official Source / Repository | Primary Paper | Text Included? | User IDs & Timestamps? | Network Edges (Retweet/Reply/Follow)? | Label Taxonomy | Suitability Summary |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HateXplain** | `github.com/hate-alert/HateXplain` / Hugging Face `hatexplain` | Mathew et al. (AAAI 2021), *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection* | Yes (tokenized post text from Twitter & Gab) | No stable user social graph or continuous timestamps | No interaction edges | `hatespeech`, `offensive`, `normal` + target community rationales | **Best for training/evaluating the Content Classifier** (HATE, OFFENSIVE, NORMAL, COMMUNITY_TARGETED). Unsuitable alone for Neo4j graph topology or temporal election windows. |
| **HASOC (2019–2023)** | `hasocfire.github.io` (FIRE Shared Tasks) | Mandl et al. (FIRE 2019–2023), *Overview of the HASOC Track at FIRE* | Yes (English, Hindi, German, code-mixed social posts) | Partial post IDs; no full user timeline timestamps | Minimal/No full network edges | Task 1: `HOF` vs `NOT`; Task 2: `HATE`, `OFFN`, `PRFN`, `UNT`/`TIN` (targeted) | **Excellent for multilingual/Indian context content classification** (especially English/Hindi). Lacks graph interaction edges. |
| **TwiBot-22** | `github.com/LuoUndergradXJTU/TwiBot-22` (requires dataset access agreement) | Feng et al. (NeurIPS 2022 Datasets & Benchmarks), *TwiBot-22: Towards Graph-Based Twitter Bot Detection* | Yes (user tweets + descriptions) | Yes (user IDs, creation dates, post timestamps) | Yes (14+ relation types including follow, retweet, reply, mention, quote) | `human` vs `bot` | **Best public dataset for heterogeneous Neo4j graph construction and GDS scalability**, but labels represent bot automation—**NOT** hate speech. |
| **Synthetic Benchmark (`synthetic_v1`)** | Generated locally via `scripts/generate_synthetic_data.py` | Project Baseline Specification | Yes (synthetic posts) | Yes (`syn_user_*`, ISO-8601 timestamps around configurable election date) | Yes (`POSTED`, `RETWEETED`, `REPLIED_TO`, `MENTIONED`) | `NORMAL`, `HATE`, `OFFENSIVE`, `COMMUNITY_TARGETED`, `SPAM` + planted community IDs | **Required for zero-dependency development, unit testing, Text-to-Cypher verification, and GDS Louvain validation.** |

## 2. Recommended Multi-Stage Dataset Strategy
1. **Stage 1 (Immediate End-to-End Baseline):** Use the built-in reproducible **Synthetic Benchmark (`data/synthetic/`)** to verify Neo4j schema creation, GDS Louvain community detection, temporal election-window slicing, Text-to-Cypher safety, and Qwen3-8B Graph RAG.
2. **Stage 2 (Content Classifier Training/Validation):** Download **HateXplain** (or **HASOC**) using `scripts/download_dataset.py` and evaluate the RoBERTa/Transformer classifier (`scripts/evaluate_classifier.py`).
3. **Stage 3 (Social Graph Application):** Apply the trained classifier to score posts in an approved social-interaction dataset (such as a subset of TwiBot-22 or a mentor-provided election hydrate CSV/JSONL), storing `label_source = 'model_prediction'`, `classifier_model`, and `classification_confidence` on every `:Post` node.

## 3. Prohibited Practices (Methodological Guardrails)
* **Never join HateXplain and TwiBot-22** by row index or fabricated IDs—they come from completely different distributions and time periods.
* **Never relabel `bot` in TwiBot-22 as `HATE` or `SPAM`** without running content classification on the post text.

## 4. Checklist to Confirm with Mentor
- [ ] Does the research group already hold a hydrated X election dataset (CSV/JSONL) under institutional terms?
- [ ] Should the content classifier focus on English only (HateXplain) or English + Hindi code-mixed posts (HASOC)?
- [ ] What specific election date(s) should be configured in `configs/project.yaml` for the final evaluation?