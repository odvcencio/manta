# Production Embedding Candidate

Use `scripts/train_manta_embed_v1_candidate.fw` to create a release-grade `manta-embed-v1` candidate. The workflow wraps the current Manta CLI primitives with production guardrails:

- refuses temporary input paths unless explicitly overridden
- refuses dirty repositories unless explicitly overridden
- records repo commit, Go version, selected environment, dataset SHA256, artifact SHA256, and run config
- trains from either a raw corpus or prepared JSONL
- runs eval-only verification on a copied package and requires `optimizer_updates=0`
- runs a separate hard holdout eval by default
- exports and verifies a sealed MLL
- supports metric gates through environment variables

## Prepared JSONL Path

Prepared JSONL is the preferred path for production because train/eval splits are fixed before training starts.

```bash
MANTA_RUN_ROOT=/data/manta/runs \
MANTA_REPO_ROOT=$PWD \
MANTA_RUN_ID=manta-embed-v1-20260412-a \
MANTA_TRAIN_JSONL=/data/manta/datasets/manta-embed-v1/train.jsonl \
MANTA_EVAL_JSONL=/data/manta/datasets/manta-embed-v1/eval.jsonl \
MANTA_HARD_EVAL_JSONL=/data/manta/datasets/manta-embed-v1/hard-eval.jsonl \
MANTA_EPOCHS=3 \
MANTA_BATCH_SIZE=1024 \
MANTA_LR=0.005 \
MANTA_TEMPERATURE=0.05 \
MANTA_SELECT_METRIC=mrr \
MANTA_MIN_MRR=0.45 \
MANTA_MIN_PAIR_ACCURACY=0.70 \
ferrous-wheel run scripts/train_manta_embed_v1_candidate.fw
```

Set `MANTA_TOKENIZER=/path/to/tokenizer.mll` when `train.jsonl`, `eval.jsonl`, or `hard-eval.jsonl` are text-pair JSONL instead of token JSONL.

## Corpus Path

Use corpus mode only when you intentionally want this run to mine the train/eval pairs:

```bash
MANTA_RUN_ROOT=/data/manta/runs \
MANTA_REPO_ROOT=$PWD \
MANTA_RUN_ID=manta-embed-v1-20260412-corpus-a \
MANTA_CORPUS=/data/manta/corpus/prod-corpus.txt \
MANTA_HARD_EVAL_JSONL=/data/manta/datasets/manta-embed-v1/hard-eval.jsonl \
MANTA_EPOCHS=3 \
MANTA_BATCH_SIZE=1024 \
MANTA_MAX_PAIRS=0 \
MANTA_EVAL_PAIRS=512 \
ferrous-wheel run scripts/train_manta_embed_v1_candidate.fw
```

Corpus mode writes mined pairs and the tokenizer into the run directory, then records their SHA256 values in `datasets.sha256`.

## JSONL Formats

Token contrastive JSONL:

```json
{"query_tokens":[1,2,3],"positive_tokens":[1,2,3],"query_mask":[1,1,1],"positive_mask":[1,1,1]}
```

Text-pair JSONL:

```json
{"query":"how to reset password","document":"reset your password from account settings","label":1}
{"left":"how to reset password","right":"billing invoice export","label":0}
```

Positive-only text pairs can train contrastively. Mixed positive/negative text pairs are valid for eval-only gates.

## Release Gate

A candidate is releasable only when:

- `manifest.json` status is `success`
- `logs/final-eval.log` and `logs/hard-eval.log` report `optimizer_updates=0`
- configured metric gates pass on hard eval
- `logs/inspect-package.log` reports `package verify: OK`
- `logs/inspect-sealed.log` reports `package verify: OK`
- `artifacts.sha256` contains the sealed MLL hash

The release artifact is:

```text
<run-dir>/manta-embed-v1.sealed.mll
```

Keep the full run directory with the released artifact. It is the audit trail for reproducing or rejecting the candidate later.
