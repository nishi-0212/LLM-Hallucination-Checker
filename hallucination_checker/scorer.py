from hallucination_checker.claim_splitter import split_into_claims, normalize_claim
from hallucination_checker.retriever import fetch_wiki, chunk_text, VectorSearch
from hallucination_checker.nli_verifier import verify_claim

def hallucination_score(results: list[dict]) -> float:
    total = 0.0
    for r in results:
        label = r["label"]
        confidence = r["confidence"]
        if label == "ENTAILMENT":
            total += 0
        elif label == "NEUTRAL":
            total += 0.5 * confidence
        elif label == "CONTRADICTION":
            total += 1.0 * confidence
    return min(total / len(results), 1.0) if results else 1.0

def verdict_from_score(score: float) -> str:
    if score < 0.3:
        return "Grounded"
    elif score < 0.6:
        return "Partially Grounded"
    else:
        return "Hallucinated"

def check_answer(answer: str, subject: str) -> tuple[str, float, list[dict]]:
    """
    Main pipeline entry point.
    Returns (verdict, score, claim_results)
    """
    claims = split_into_claims(answer)
    if not claims:
        return "Unknown", 1.0, []

    wiki_text = fetch_wiki(subject)
    if wiki_text == "No information found":
        return "Unknown", 1.0, []

    chunks = chunk_text(wiki_text)
    if not chunks:
        return "Unknown", 1.0, []

    search_engine = VectorSearch(chunks)
    claim_results = []

    for claim in claims:
        clean_claim = normalize_claim(claim, subject)
        evidence_chunks = search_engine.search(clean_claim, k=2)
        evidence = " ".join(evidence_chunks)
        label, confidence = verify_claim(clean_claim, evidence)

        claim_results.append({
            "claim": claim,
            "clean_claim": clean_claim,
            "label": label,
            "confidence": round(confidence, 3),
            "evidence": evidence
        })

    score = hallucination_score(claim_results)
    verdict = verdict_from_score(score)

    return verdict, round(score, 3), claim_results