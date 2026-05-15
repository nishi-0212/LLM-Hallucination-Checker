import nltk

def ensure_nltk():
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

def split_into_claims(answer: str) -> list[str]:
    ensure_nltk()
    from nltk.tokenize import sent_tokenize
    return [s.strip() for s in sent_tokenize(answer) if s.strip()]

def normalize_claim(claim: str, subject: str) -> str:
    pronouns = ["He", "She", "They", "he", "she", "they", "It", "it"]
    words = claim.split()
    if words and words[0] in pronouns:
        return subject + " " + " ".join(words[1:])
    return claim