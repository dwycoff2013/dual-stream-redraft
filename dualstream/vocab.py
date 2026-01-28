# Default concept vocabulary (illustrative).
# The whitepaper explicitly treats the example vocabulary as non-normative.

DEFAULT_CONCEPT_VOCAB = {
    1001: "USER_INTENT:CONFIRMATION_REQUEST",
    2001: "FACTUALITY_CONCERN:PREMISE_LIKELY_FALSE",
    3001: "POLICY_TENSION:HONESTY_vs_PLEASING_USER",
}

def concept_label(concept_id: int) -> str:
    return DEFAULT_CONCEPT_VOCAB.get(concept_id, f"CONCEPT_{concept_id}")
