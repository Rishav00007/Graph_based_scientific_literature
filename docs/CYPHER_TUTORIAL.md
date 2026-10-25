// ============================================================================
// cypher/01_constraints.cypher & cypher/02_indexes.cypher
// ============================================================================

// [Query 01] Uniqueness constraint on User ID
CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE;

// [Query 02] Uniqueness constraint on Post ID
CREATE CONSTRAINT post_id_unique IF NOT EXISTS FOR (p:Post) REQUIRE p.post_id IS UNIQUE;

// [Query 03] Range index on Post timestamp and content_label for fast temporal/category filtering
CREATE INDEX post_timestamp_idx IF NOT EXISTS FOR (p:Post) ON (p.timestamp);
CREATE INDEX post_content_label_idx IF NOT EXISTS FOR (p:Post) ON (p.content_label);

// ============================================================================
// cypher/03_basic_queries.cypher (Beginner Pattern Matching & Aggregations)
// ============================================================================

// [Query 04] List users with their dataset provenance and synthetic flag
MATCH (u:User)
RETURN u.user_id AS user_id, u.username AS username, u.dataset_source AS source, u.is_synthetic AS is_synthetic
ORDER BY u.user_id ASC
LIMIT 25;

// [Query 05] List recent posts with classification metadata
MATCH (p:Post)
RETURN p.post_id AS post_id, p.text AS text, p.content_label AS label,
       p.classification_confidence AS confidence, p.timestamp_str AS timestamp
ORDER BY p.timestamp DESC
LIMIT 25;

// [Query 06] Find all hate-labelled posts above a confidence threshold
MATCH (u:User)-[:POSTED]->(p:Post)
WHERE p.content_label = 'HATE' AND p.classification_confidence >= $min_confidence
RETURN u.user_id AS author_id, p.post_id AS post_id, p.text AS text, p.classification_confidence AS confidence
ORDER BY p.classification_confidence DESC
LIMIT 50;

// [Query 07] Find all spam-labelled posts and their associated hashtags
MATCH (u:User)-[:POSTED]->(p:Post)
WHERE p.content_label = 'SPAM'
OPTIONAL MATCH (p)-[:USES_HASHTAG]->(h:Hashtag)
RETURN u.user_id AS author_id, p.post_id AS post_id, collect(h.tag) AS hashtags
LIMIT 50;

// [Query 08] Count total posts per user and breakdown by content category
MATCH (u:User)-[:POSTED]->(p:Post)
RETURN u.user_id AS user_id,
       count(p) AS total_posts,
       sum(CASE WHEN p.content_label = 'HATE' THEN 1 ELSE 0 END) AS hate_count,
       sum(CASE WHEN p.content_label = 'SPAM' THEN 1 ELSE 0 END) AS spam_count
ORDER BY total_posts DESC
LIMIT 25;

// [Query 09] Find the most active users across all interaction types (posts + retweets + replies)
MATCH (u:User)
OPTIONAL MATCH (u)-[r:POSTED|RETWEETED|REPLIED_TO]->()
RETURN u.user_id AS user_id, count(r) AS total_activity
ORDER BY total_activity DESC
LIMIT 20;

// [Query 10] Find users whose posts receive the most retweets
MATCH (author:User)-[:POSTED]->(p:Post)<-[rt:RETWEETED]-(retweeter:User)
RETURN author.user_id AS author_id,
       count(rt) AS total_retweets_received,
       count(DISTINCT retweeter) AS unique_retweeters
ORDER BY total_retweets_received DESC
LIMIT 20;

// ============================================================================
// cypher/04_temporal_queries.cypher (Date/Time Filtering & Election Windows)
// ============================================================================

// [Query 11] Find posts within a parameterized ISO-8601 date range
MATCH (u:User)-[:POSTED]->(p:Post)
WHERE p.timestamp >= datetime($start_date) AND p.timestamp <= datetime($end_date)
RETURN u.user_id AS user_id, p.post_id AS post_id, p.content_label AS label, p.timestamp_str AS timestamp
ORDER BY p.timestamp ASC
LIMIT 50;

// [Query 12] Find users who posted HATE or COMMUNITY_TARGETED content in the 10 days before an election
MATCH (u:User)-[:POSTED]->(p:Post)
WHERE p.timestamp >= datetime($election_date) - duration({days: 10})
  AND p.timestamp < datetime($election_date)
  AND p.content_label IN ['HATE', 'COMMUNITY_TARGETED']
RETURN u.user_id AS user_id, u.louvain_community_id AS community_id,
       count(p) AS pre_election_flagged_posts, collect(p.post_id) AS supporting_post_ids
ORDER BY pre_election_flagged_posts DESC
LIMIT 25;

// [Query 13] Daily post volume and flagged content proportion around an election window
MATCH (p:Post)
WHERE p.timestamp >= datetime($start_date) AND p.timestamp <= datetime($end_date)
WITH date(p.timestamp) AS activity_date,
     count(p) AS daily_total,
     sum(CASE WHEN p.content_label IN ['HATE', 'OFFENSIVE', 'COMMUNITY_TARGETED', 'SPAM'] THEN 1 ELSE 0 END) AS daily_flagged
RETURN toString(activity_date) AS day, daily_total, daily_flagged,
       round(toFloat(daily_flagged) / daily_total, 4) AS flagged_proportion
ORDER BY day ASC;

// [Query 14] Compare community activity before vs. during the 10-day pre-election window
MATCH (u:User)-[:POSTED]->(p:Post)
WHERE u.louvain_community_id IS NOT NULL
WITH u.louvain_community_id AS community_id,
     sum(CASE WHEN p.timestamp >= datetime($election_date) - duration({days: 30})
               AND p.timestamp < datetime($election_date) - duration({days: 10})
               AND p.content_label IN ['HATE', 'SPAM', 'OFFENSIVE', 'COMMUNITY_TARGETED'] THEN 1 ELSE 0 END) AS baseline_flagged,
     sum(CASE WHEN p.timestamp >= datetime($election_date) - duration({days: 10})
               AND p.timestamp < datetime($election_date)
               AND p.content_label IN ['HATE', 'SPAM', 'OFFENSIVE', 'COMMUNITY_TARGETED'] THEN 1 ELSE 0 END) AS surge_window_flagged
RETURN community_id, baseline_flagged, surge_window_flagged,
       (surge_window_flagged - baseline_flagged) AS flagged_delta
ORDER BY flagged_delta DESC;

// ============================================================================
// cypher/05_hate_spam_queries.cypher & cypher/06_subgraph_queries.cypher
// ============================================================================

// [Query 15] Find users who repeatedly amplified (retweeted) the exact same author's flagged posts
MATCH (retweeter:User)-[rt:RETWEETED]->(p:Post)<-[:POSTED]-(author:User)
WHERE p.content_label IN ['HATE', 'OFFENSIVE', 'COMMUNITY_TARGETED', 'SPAM']
  AND retweeter.user_id <> author.user_id
WITH retweeter, author, count(rt) AS amplification_count, collect(p.post_id) AS amplified_posts
WHERE amplification_count >= $min_amplifications
RETURN retweeter.user_id AS amplifier_id, author.user_id AS original_author_id,
       amplification_count, amplified_posts
ORDER BY amplification_count DESC
LIMIT 25;

// [Query 16] Extract a content-filtered subgraph of HATE and SPAM interactions for visualization
MATCH (u1:User)-[r:POSTED|RETWEETED|REPLIED_TO]->(p:Post)
WHERE p.content_label IN ['HATE', 'SPAM']
  AND p.timestamp >= datetime($start_date) AND p.timestamp <= datetime($end_date)
RETURN u1.user_id AS source_user, type(r) AS rel_type, p.post_id AS post_id, p.content_label AS label
LIMIT 100;

// [Query 17] Find 2-hop amplification chains (User A -> Post <- Retweeted by User B -> Mentions User C)
MATCH path = (author:User)-[:POSTED]->(p:Post)<-[:RETWEETED]-(amp:User)-[:MENTIONED]->(target:User)
WHERE p.content_label IN ['HATE', 'COMMUNITY_TARGETED']
RETURN author.user_id AS author, p.post_id AS post_id, amp.user_id AS amplifier, target.user_id AS mentioned_user
LIMIT 25;

// [Query 18] Find co-retweeting user pairs (users who retweeted at least 2 of the same flagged posts)
MATCH (u1:User)-[:RETWEETED]->(p:Post)<-[:RETWEETED]-(u2:User)
WHERE u1.user_id < u2.user_id AND p.content_label IN ['HATE', 'SPAM']
WITH u1, u2, count(DISTINCT p) AS shared_retweets, collect(p.post_id) AS shared_post_ids
WHERE shared_retweets >= 2
RETURN u1.user_id AS user_a, u2.user_id AS user_b, shared_retweets, shared_post_ids
ORDER BY shared_retweets DESC
LIMIT 25;

// [Query 19] Hashtag co-occurrence with hate/offensive posts
MATCH (p:Post)-[:USES_HASHTAG]->(h:Hashtag)
WHERE p.content_label IN ['HATE', 'OFFENSIVE', 'COMMUNITY_TARGETED']
RETURN h.tag AS hashtag, count(p) AS flagged_post_count
ORDER BY flagged_post_count DESC
LIMIT 20;

// ============================================================================
// cypher/07_community_queries.cypher (GDS Community Profiling & Evidence)
// ============================================================================

// [Query 20] Compute comprehensive statistics per Louvain community
MATCH (u:User)
WHERE u.louvain_community_id IS NOT NULL
OPTIONAL MATCH (u)-[:POSTED]->(p:Post)
RETURN u.louvain_community_id AS community_id,
       count(DISTINCT u) AS num_users,
       count(DISTINCT p) AS num_posts,
       sum(CASE WHEN p.content_label = 'HATE' THEN 1 ELSE 0 END) AS hate_posts,
       sum(CASE WHEN p.content_label = 'SPAM' THEN 1 ELSE 0 END) AS spam_posts,
       round(toFloat(sum(CASE WHEN p.content_label IN ['HATE', 'OFFENSIVE', 'COMMUNITY_TARGETED', 'SPAM'] THEN 1 ELSE 0 END)) / CASE WHEN count(p) = 0 THEN 1 ELSE count(p) END, 4) AS flagged_ratio
ORDER BY flagged_ratio DESC;

// [Query 21] Find top structural nodes (by PageRank and Degree) inside a specific Louvain community
MATCH (u:User)
WHERE u.louvain_community_id = $community_id
RETURN u.user_id AS user_id, u.username AS username,
       u.degree_centrality AS degree, u.pagerank_score AS pagerank
ORDER BY u.pagerank_score DESC, u.degree_centrality DESC
LIMIT 15;

// [Query 22] Retrieve supporting graph evidence (users, posts, interactions) for a specific community
MATCH (u:User)-[:POSTED]->(p:Post)
WHERE u.louvain_community_id = $community_id
  AND p.content_label IN ['HATE', 'OFFENSIVE', 'COMMUNITY_TARGETED', 'SPAM']
OPTIONAL MATCH (amp:User)-[r:RETWEETED|REPLIED_TO]->(p)
RETURN u.user_id AS author_id, p.post_id AS post_id, p.content_label AS label,
       p.classification_confidence AS confidence, p.timestamp_str AS timestamp,
       collect(DISTINCT amp.user_id) AS amplifying_users
ORDER BY p.timestamp DESC
LIMIT 30;

// [Query 23] Idempotent incremental upsert of a new classified post (MERGE pattern)
MERGE (u:User {user_id: $user_id})
MERGE (p:Post {post_id: $post_id})
SET p.text = $text,
    p.timestamp = datetime($timestamp),
    p.content_label = $content_label,
    p.classifier_model = $classifier_model,
    p.is_synthetic = toBoolean($is_synthetic)
MERGE (u)-[:POSTED]->(p)
RETURN u.user_id AS user_id, p.post_id AS post_id;

// [Query 24] Profile query execution plan (Teaching query optimization with PROFILE)
PROFILE MATCH (u:User)-[:POSTED]->(p:Post)
WHERE p.content_label = 'HATE' AND p.timestamp >= datetime('2026-10-20T00:00:00Z')
RETURN u.user_id, count(p) AS cnt;

// [Query 25] Safe deletion of synthetic test records ONLY (Never run unbounded DETACH DELETE in production!)
MATCH (n)
WHERE n.is_synthetic = true AND n.dataset_source = 'synthetic_temp_test'
DETACH DELETE n;