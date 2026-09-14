// Queries over the end-to-end example's graph, in the order the README walks
// through them. tests/test_e2e.py runs every one against a live Neo4j, and
// checks that the README quotes them exactly.

// 1. Provenance: every fact one source gave, and the characters it was read from.
MATCH (s)-[r]->(o)
WHERE 'corpus/notes/halden-robotics.md' IN r.evidence_doc_ids
RETURN s.label AS subject, type(r) AS predicate, coalesce(o.label, o.value) AS object,
       r.verdict AS verdict, r.evidence_starts AS starts, r.evidence_ends AS ends
ORDER BY predicate, object;

// 2. Why is this value here? Two sources disagree on a head office; both are
//    kept, and the loser says why it lost.
MATCH (c:Company {key: 'Company:halden robotics ltd'})-[r:headquarters]->(v:Claim)
RETURN v.value AS headquarters, r.evidence_doc_ids AS sources, r.evidence_tiers AS tiers,
       r.confidence AS confidence, r.`odke.conflict` AS conflict
ORDER BY confidence DESC;

// 3. A DIFFERENT link: one name, two registration numbers, two companies.
MATCH (a:Company)-[l:DIFFERENT]->(b:Company)
RETURN a.label AS company, b.label AS other, l.score AS name_similarity, l.reason AS reason;

// 4. Cardinality: a company with more than one head office. This is the check
//    openodke compiles from the ontology (`headquarters` is single-valued).
MATCH (s)-[r:`headquarters`]->(o)
WHERE r.polarity = 'asserted' AND r.valid_to IS NULL
WITH s, collect(DISTINCT coalesce(o.key, o.value)) AS objects
WHERE size(objects) > 1
RETURN labels(s) AS labels, s.key AS subject, objects;

// 5. What changed: every fact written or confirmed in the last seven days.
MATCH (s)-[r]->(o)
WHERE r.signature IS NOT NULL AND r.extracted_at >= datetime() - duration('P7D')
RETURN type(r) AS predicate, count(*) AS facts
ORDER BY predicate;
