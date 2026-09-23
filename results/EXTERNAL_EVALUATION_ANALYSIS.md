# External Evaluation: Error Analysis

Source: `results/external_conflicts_metrics.json` / `results/external_conflicts_results.csv`,
produced by `app/evaluation/external_conflicts_runner.py` against the 60-case
sample in `data/test_cases/external_conflicts_sample.json`
(from [google-research-datasets/rag_conflicts](https://github.com/google-research-datasets/rag_conflicts)).

## Headline numbers

| Metric | Internal unseen benchmark (42 cases) | External benchmark (55 scored / 60 cases) |
|---|---|---|
| Conflict-type accuracy | 97.62% | 25.45% |
| Macro F1 | 0.976 | 0.182 |
| Average latency | 22.2s | 51.6s |

| External per-category accuracy | Cases | Accuracy |
|---|---|---|
| Conflicting opinions and research outcomes → `direct_factual_conflict` | 12 | 66.67% |
| Conflict due to outdated information → `temporal_conflict` | 12 | 25.00% |
| Complementary information → `ambiguous_question` / `conditional_conflict` | 19 | 15.79% |
| No conflict → `no_conflict` | 12 | 0.00% |

This drop is real and expected — the internal benchmark was built to match
ConflictGuard's own document format (explicit `Publication Year:` /
`Policy Version:` / `Source Authority:` headers, single unambiguous factual
claims per document). The external set is real web search results: long,
unstructured, written by different authors for different purposes, with no
such metadata. Evaluating on it is exactly what the proposal's methodology
calls this step for — measuring generalization, not confirming the internal
number. Below is why the gap is this large, with specific cases.

## Root cause 1: the "no conflict" check requires near-identical text

`ConflictDetectionAgent._relevant_claims_agree()` only marks two sources as
agreeing if their claim text is identical after normalization. That works
when your own documents are written in the same style, but real web sources
never phrase the same fact the same way.

**EXT001** — *"what is the most viewed video on youtube ever?"* Both
sources agree: Wikipedia says the top video is "Baby Shark Dance", a second
source says "Baby Shark Dance... has become the highest-watched video on
YouTube with 15.17 billion views." Predicted `direct_factual_conflict`
(expected `no_conflict`) purely because the sentences don't match
character-for-character — but the final answer was still correct:
*"the most viewed video on YouTube is 'Baby Shark Dance' with 15.17 billion
views."* The conflict-type label was wrong; the actual output wasn't.

This pattern (label wrong, answer still right) recurred in several `no_conflict`
and `Conflicting opinions` cases — worth keeping in mind when reading the
25.45% headline number: it measures whether the *reasoning path* matched
expectations, not always whether the *final answer* was wrong.

**Fix worth trying**: replace the exact-text-match check with a semantic
similarity check using the `SentenceTransformer` embedding model already in
the project (`app/rag/embeddings.py`) — e.g. cosine similarity above a
threshold counts as agreement, not just string equality.

## Root cause 2: differing publication dates are treated as a conflict signal

`ConflictDetectionAgent._has_temporal_evidence()` flags a temporal conflict
whenever sources carry different years. For your own policy documents this
is a reliable signal (a 2025 policy year reliably means it supersedes a 2023
one). For general web content it isn't — two articles from different years
can report the same still-true fact.

**EXT004** — *"Can gene therapy reverse the aging process?"* Sources include
a 2023 bioRxiv preprint, a Harvard/Fortune article, and a Harvard Stem Cell
Institute piece, each with different dates, none of them actually
contradicting the others — they describe different specific findings that
all support "yes, in some forms." The system split on the year difference
alone and predicted `temporal_conflict`. The final answer it gave ("Harvard
scientists have identified a drug combo that may reverse aging in just one
week") is defensible and on-topic, just narrower than the full picture.

This heuristic assumes "newer supersedes older" is universal. It should
likely require an explicit signal (like the internal documents' "supersedes"
language) rather than triggering on any year mismatch.

## Root cause 3: "conflicting opinions/research" doesn't map cleanly to "direct factual conflict"

The external dataset's `Conflicting opinions and research outcomes` label
was mapped to `direct_factual_conflict` for scoring (the closest available
category), but several of these cases are really about research findings
applying under different conditions — which is closer, in spirit, to
ConflictGuard's own `conditional_conflict`.

**EXT040** — *"Does Brain Training actually improve cognitive function?"*
Predicted `conditional_conflict`; scored as wrong (expected
`direct_factual_conflict`) but the answer produced — "informal mentally
stimulating activities... may lower the risk... some types of cognitive
training... " — is a genuinely well-hedged, conditional answer to a question
where the honest answer *is* conditional. This is arguably the taxonomy
mapping being too strict, not the system being wrong.

This is the category with the best raw accuracy anyway (66.67%), so it's
the one place the mapping still mostly held up.

## Secondary issue: claim relevance can pick a related-but-wrong claim

**EXT036** — *"when was the first temple built in jerusalem?"* — the system
answered with the temple's *destruction* date (587 BCE) instead of its
construction date. The retrieved passage discussed both events; the
token-overlap relevance scoring in `ConflictDetectionAgent`/`ResolutionAgent`
picked the wrong sentence. This is a genuine extraction/relevance bug
independent of the conflict-taxonomy mismatch, and would be worth a look on
multi-fact passages generally, not just this external set.

## Secondary issue: a small amount of scraped content is not real content

**EXT023** — *"who sings gone gone gone she been gone so long?"* — two of
three search results are YouTube bot-detection pages ("Our systems have
detected unusual traffic...") rather than real page content, and their
scrape dates got read as `Publication Year:` values, triggering a spurious
temporal conflict on pure noise. Checked across the full 60-case sample,
this affects only 1 case (2 of 155 passages) — not a systemic problem, but
worth a content sanity check (e.g. skip passages under ~50 words or
containing known bot-check phrases) if this dataset is used more heavily.

## Takeaways

1. **The internal 97.6% is a legitimate result for the domain ConflictGuard
   was designed for**: documents with explicit, structured metadata. It
   isn't inflated by an easy benchmark — the structure is doing real work.
2. **The 25.45% external number mostly reflects two specific, nameable rule
   thresholds** (exact-text-match for agreement, any-year-difference for
   temporal conflict) that were tuned for structured documents and don't
   hold on free-form web text — not a general failure of the claim
   extraction → detection → resolution → verification pipeline. Multiple
   "wrong label" cases still produced correct or reasonable final answers.
3. **Concrete next steps**, roughly in order of expected impact: (a)
   semantic-similarity agreement check instead of exact text match, (b)
   require an explicit supersession/currency signal for temporal conflicts
   rather than any year mismatch, (c) fix claim-relevance scoring for
   multi-fact passages, (d) build a proper LLM-judge-based answer-accuracy
   metric (this analysis used manual inspection and a rough text-containment
   check, not a rigorous automated one) to separate "wrong conflict label"
   from "wrong final answer" at scale.
