# Ontology, RDF Triples, and Neo4j Property Graphs

## 1. What Is an Ontology?
In knowledge representation, an **ontology** is a formal, explicit specification of a shared conceptualization within a domain. It defines:
* **Classes (Concepts):** e.g., `soc:User`, `soc:SocialPost`, `soc:Hashtag`, `soc:AnalyticalCommunity`, `soc:ContentCategory`.
* **Object Properties (Relationships between entities):** e.g., `soc:posted`, `soc:retweeted`, `soc:belongsToCommunity`.
* **Datatype Properties (Attributes of an entity):** e.g., `soc:hasTimestamp` (`xsd:dateTime`), `soc:hasConfidence` (`xsd:float`).

## 2. What Are RDF Triples?
The **Resource Description Framework (RDF)** represents knowledge as atomic **Subject – Predicate – Object** triples:
* **Subject:** The resource being described (a URI, e.g., `ex:user/syn_user_001`).
* **Predicate:** The property or relationship URI (e.g., `soc:posted` or `rdf:type`).
* **Object:** Another resource URI (`ex:post/syn_post_0001`) or a literal value (`"HATE"^^xsd:string`).

## 3. RDF Graph vs. Neo4j Labeled Property Graph (LPG)

| Feature | RDF Triple Store (e.g., GraphDB, Fuseki) | Neo4j Labeled Property Graph (LPG) |
| :--- | :--- | :--- |
| **Atomic Unit** | `(Subject, Predicate, Object)` triple | Nodes (with labels + key-value map) and Relationships (with type + key-value map) |
| **Properties on Edges** | Requires RDF-star (`<<s p o>>`) or reification nodes | Native support (e.g., `[:RETWEETED {timestamp: ..., confidence: 0.91}]`) |
| **Graph Algorithms** | Limited native centrality/clustering | Native **Neo4j GDS** (Louvain, PageRank, Degree, Betweenness in RAM) |
| **Query Language** | SPARQL | **Cypher** |

## 4. Conceptual Mapping & Optional RDF/Turtle Serialization
Our Neo4j property graph maps cleanly onto an ontology without requiring a separate RDF triple store. Below is an example **RDF/Turtle (`.ttl`)** representation of a record from our schema:

```turtle
@prefix ex:  [http://example.org/social-graph/](http://example.org/social-graph/) .
@prefix soc: [http://example.org/ontology/social#](http://example.org/ontology/social#) .
@prefix xsd: [http://www.w3.org/2001/XMLSchema#](http://www.w3.org/2001/XMLSchema#) .

ex:user_syn_user_001 a soc:User ;
    soc:userId "syn_user_001"^^xsd:string ;
    soc:isSynthetic true ;
    soc:belongsToCommunity ex:community_2 ;
    soc:posted ex:post_syn_post_0010 .

ex:post_syn_post_0010 a soc:Post ;
    soc:postId "syn_post_0010"^^xsd:string ;
    soc:postedAt "2026-10-25T14:30:00Z"^^xsd:dateTime ;
    soc:hasContentLabel soc:Label_HATE ;
    soc:classificationConfidence "0.89"^^xsd:float ;
    soc:usesHashtag ex:hashtag_election2026 .