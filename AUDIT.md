# Plan-to-implementation audit

Scope: the supplied **Video Moment Retrieval — Revised Architecture**, plus the accepted review changes (coverage windows, subject/actor/recipient evidence, separate retrieval evaluation, and honest enumeration). This audit distinguishes implemented code from tested model behavior.

## Substantive discrepancies corrected

| Plan requirement | Before audit | Fix | Regression evidence |
|---|---|---|---|
| Independent speech detection (§2.3) | Transcript intervals impersonated VAD, so missing ASR also removed audio candidates | WebRTC VAD adapter, native dependency installed locally, optional diarization attached by interval; unknown/overlapping speakers retained | Native silent/partial-frame test; VAD runs despite ASR failure; exercised real audio from all three downloads |
| Transcription + alignment (§2.2) | Remote estimated timestamps or manually prepared sidecar only | Optional full-track WhisperX transcription, word alignment, and diarization adapter; CLI selection and cache identity | Adapter test exercises ASR → alignment sequence with controlled outputs; full-track integration called once |
| Rolling context (§2.4) | Entirely omitted | Bounded previous-window ending observations passed to extractor, with context in cache identity; failure resets state; ablation flag | Changing prior cached ending state recomputes the following window |
| Separate reconciliation (§2.4) | Exact-string deduplication was called reconciliation | Separate cached LLM pass proposes evidence-linked event/person continuity across neighboring windows; hypothesis status enforced | Different-window IDs remain linked and searchable; invented references rejected; description-only identity cannot be promoted |
| Subject relationships (§2.6) | Schema fields existed, but no producer populated them | Extractor emits speaker/person proposals; validator requires known subjects/speakers, interval, provenance, and affirmative evidence for support; relational table persists links | Co-occurrence cannot become supported; stored relationship retrieval; same-person verifier guard |
| Parallel retrieval (§2.8/§3.2) | Three sequential calls | Concurrent independent readers plus dense query encoding; explicit channel errors | Barrier test fails if channels run sequentially |
| Merge and LLM assessment (§3.3) | Rank fusion only; assessment deliberately deferred | Bounded text-only LLM assessment with per-candidate evidence, relationship checks, caching, and reranking; unknowns retained | Reordering test; missing same-person links downgrade to unresolved; ablation flag |
| Bounded exhaustive enumeration (§3.5) | Every record visited, but only its beginning inspected; one result per clip | Long records split into inspection tasks covering the tail; multiple verdicts per clip; explicit incomplete flags | Tail-only match recovered; two separate matches from one slice returned |
| Pagination completeness (§3.5) | No cross-page deduplication or continuation state | Persisted state, no skipped offsets, settings-aware snapshot, idempotent page replay, accumulated unique results | Cross-page duplicates suppressed; changed geometry rejects cursor |
| Evidence validation (§0/§3.4) | Some protections only existed in the OpenRouter adapter; cache hits bypassed validation | Central verification validation for all adapters and cached data, modality checks, explicit same-person binding, conflicting criteria rejected | Invalid cached verdict recomputed; visual-only evidence cannot establish audio behavior |
| Candidate recall (§1) | One-to-one final-detection metric reused for retrieval | Coverage-based candidate recall; one-to-one matching reserved for final predictions | One broad window correctly retrieves two short labeled events |
| OCR correctness (§1/§2.5) | Whole-frame blur ranking could discard a clear crop; exact returned characters were not evaluable | Independent detection on nearby frames, crop-level sharpness ordering, content-addressed crops; label `details` supports exact text matching | Original-resolution unreadable crop test; correct timestamp with wrong plate string scores false |
| Independently usable indexes (§2.8) | Embedding failure prevented publishing all extracted evidence | Structured/lexical publication proceeds with missing vectors reported as a failed stage | Searchable lexical evidence after an embedding failure |
| Complete coverage accounting (§1) | Some future tasks absent, and whole-stage pending rows could remain after success | Tasks registered at appropriate intervals; explicit VAD/reconciliation status; no stale pending rows after successful run | Successful pipeline leaves no pending coverage; failed stages remain distinguishable |
| Cache dependencies (§2.9) | Rolling/reconciliation dependencies absent; some malformed model outputs could remain cached | Previous-state and reconciliation inputs in keys, materialization validation before caching, cached embeddings/verdicts validated, immutable crop paths | Upstream correction invalidates downstream work; malformed references cannot become permanent successes |
| Fair ablations (§1) | Five policies lacked the missing assessment stage | Seven variants isolate retrieval, assessment, and verification; equal candidate cap; query-level cache/cost reporting | CLI evaluation round trip, metric tests, synthetic seven-policy run |
| Original timeline preservation (§2.1/§2.2) | Demux could move delayed audio to the start of the extracted WAV | Explicit timestamp-based silence padding preserves initial delay and requested duration | Generated video with delayed audio retains leading silence after extraction |
| Component replacement (§4/§2.9) | A single aggregate model identity invalidated unrelated stages | Stage-specific identities; changing an audio model preserves visual/assessment caches while invalidating audio-dependent work | Component-identity isolation test |

## Follow-up review corrections

- Preserve extractor speaker assignments through materialization, storage, and structured retrieval; require references to supplied transcript speakers.
- Normalize numeric-string segment and word timestamps. Shift nested words with their ASR chunk, then clip and rebase both segments and words for visual-window excerpts. Cached segments are not mutated by rebasing.
- Separate temporal candidate recall from detail-aware candidate recall; report both, including interval-scoped OCR readings in candidate evidence.
- Read-only database opens no longer create parent directories. CLI help includes provider calls during live evaluation.
- Reject duplicate download filenames. Validate response lengths and video streams before publishing, and verify saved SHA-256 transfer receipts before reusing files. Interrupted downloads preserve existing files.

Existing indexes need reindexing to materialize corrected speaker and word metadata. Reindexing reuses valid extraction caches; transcript excerpts with corrected word times invalidate their dependent visual caches. These corrections do not constitute a live provider or real-accuracy benchmark.

## Intentional choices retained

- Lexical search handles words; semantic embeddings handle paraphrases. The query planner does not fabricate vectors: the encoder supplies them.
- Fixed coverage windows coexist with precise events. A missing event extraction cannot remove its whole interval from search.
- Structured fields generate candidates, not hard exclusions based on uncertain observations.
- Reconciliation creates an evidence-linked graph of hypotheses. It does not silently assign persistent identities or convert related but distinct events into one asserted action.
- Named-person linking remains deferred, exactly as revised §2.7 specifies. No foundation model is trained.
- The OCR detector/reader is a separate high-resolution VLM route behind an OCR interface; it is not a specialist plate-recognition model. The plan permits a scoped high-resolution VLM route. Sampling may miss brief plate visibility.
- Live verification is enabled by default for precision; `--unverified` returns explicitly labeled candidates. Verification is bounded per request, while an exhaustive job's total work may grow across pages.
- Synthetic fixtures remain explicitly synthetic. Their verifier reads authored assertions and does not demonstrate visual/audio perception.

## What is still not established

1. **Real OpenRouter compatibility and accuracy:** A key is now configured and loads locally; authenticated requests remain deliberately untested until the requested live test run. Text planning, VLM extraction, remote ASR, assessment, reconciliation, embeddings, and media verification have controlled-response contract tests, not a successful live end-to-end run.
2. **WhisperX model behavior:** the optional adapter and integration are tested with controlled outputs. Its heavyweight dependencies/model weights have not been installed or run in this workspace; it may need a compatible Python/device environment and authorized speaker-model access.
3. **Ground truth and empirical tuning:** no reviewed real-corpus labels have been supplied/created. Real precision, recall, localization error, latency, and API cost remain unmeasured. Configuration and retrieval-policy choices remain hypotheses.
4. **Provider sampling:** `--fps` changes uploaded frames. It is not a guarantee of the provider's internal sampling rate. Adaptive/agentic provider routing has not been exercised. Claims about its quality or latency require live experiments, not an untested parameter.
5. **Submission:** GitHub publication and Loom recording are not performed by this audit.

These are empirical/environment or submission gaps, not claims that the implementation has already fulfilled them.

## Reproduction

```bash
source .venv/bin/activate
python -m unittest discover -v
python -m video_moment_retrieval --data-dir data/demo demo
python -m video_moment_retrieval --data-dir data/demo --demo evaluate \
  data/demo/demo-labels.json --output reports/audited-demo-evaluation.json
```

Real VAD smoke results are in the local ignored `reports/vad-smoke.json`. It ran on five-second audio samples from `video_18`, `video_26`, and `video_29`; this demonstrates execution on the actual recording format, not a measured VAD accuracy score.


## Preflight check — 2026-09-24

- The configured API key loads through the CLI and remains ignored by Git; its contents were not displayed. No authenticated requests or corpus inference were performed.
- Local configuration selects `google/gemini-3.5-flash` for video, text, and audio and `openai/text-embedding-3-small` for embeddings. The [OpenRouter model page](https://openrouter.ai/google/gemini-3.5-flash) lists the required input modalities. Request formats were checked against the official video, audio, and embedding documentation; account access and live responses remain untested.
- Fixed invalid index bounds (including NaN expanding a short test to a whole video), empty-index searches spending requests, empty completion responses, interrupted HTTP bodies, and missing/null usage reporting. Corrected the silence-transcription prompt to preserve its object schema.
- Fixed local virtualenv activation paths after the project folder rename. The three downloaded MP4s have readable video/audio metadata. WhisperX is not installed; the configured default ASR uses OpenRouter.
- All 83 automated tests pass with the optional speech dependency installed. Compilation and Git whitespace checks pass. The suite includes synthetic media integration and CLI demo/evaluation tests; those are not live model accuracy measurements.
- Next step awaits the user's explicit live-test instructions. Authentication, credits, provider routing, response quality, timing, and cost can only be established during that run.


## Correctness review fixes — 2026-09-24

The seven follow-up findings now have regression coverage:

1. Reindexing stages its full snapshot in a separate database. Failed extraction/embedding and interrupted runs preserve existing searchable data; publication is transactional and rolls back on write failure. Attempt reports remain inspectable independently of published coverage.
2. Model-output validators check nested object/list types before accessing fields, producing explicit stage/verification failures instead of uncaught attribute errors.
3. Search exposes operational errors and returns a failing CLI exit status for them. Evaluation excludes failed queries from aggregate correctness and does not count failures or abstentions as correct negative answers.
4. Subject-binding evidence references are validated against actual supplied media, including nested references and required explanations.
5. Visual subject continuity is separate from visual/speaker binding. Visual-only conjunctions do not acquire audio or speech requirements.
6. Reconciliation coverage comes from scheduled neighboring-window tasks; failed neighbors leave skipped coverage.
7. OCR verification requires a normalized box associated with a cited supplied frame inside the returned result interval. Shared validation applies to the provider adapter and cached/custom adapter outputs.

Enumeration now freezes its retrieval queue once and enriches only the current page. A persisted index revision invalidates stale continuations. Its JSON queue/state still grows with job size; no claim of constant total job I/O is made.

Ingestion now reserves bounded HTTP attempts across enabled stages sharing a client, including embeddings. Unused attempts roll forward without increasing the user's total cap; this does not guarantee that a small request budget can finish an entire video.

No live API calls or corpus inference were performed for these fixes. Real manually reviewed evaluation remains part of the separately authorized live test.
