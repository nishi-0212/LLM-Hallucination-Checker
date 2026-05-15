from transformers import pipeline

_nli_model = None

def get_nli_model():
    global _nli_model
    if _nli_model is None:
        _nli_model = pipeline(
            "text-classification",
            model="facebook/bart-large-mnli",
            device=-1  # CPU; change to 0 for GPU
        )
    return _nli_model

def verify_claim(claim: str, evidence: str) -> tuple[str, float]:
    """
    Returns (label, confidence) where label is one of:
    ENTAILMENT, NEUTRAL, CONTRADICTION
    """
    nli = get_nli_model()
    text = f"{evidence} </s></s> {claim}"
    result = nli(text, truncation=True, max_length=1024)[0]
    return result["label"].upper(), result["score"]