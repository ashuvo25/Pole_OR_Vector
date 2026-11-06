# BanglaCV-Probe

**Pole or Vector? Encoders Capture Bangla Compound Verbs, but LLMs Read Them Literally**

BanglaCV-Probe is a benchmark for testing how language models handle **Bangla compound verbs**: a main verb followed by a *vector verb* that loses its own meaning and adds a shade of meaning, such as completion.

> 📄 Paper: *coming soon on arXiv*
> 💾 Data and code: this repository

---

## Table of contents

1. [What is a compound verb?](#what-is-a-compound-verb)
2. [What the benchmark tests](#what-the-benchmark-tests)
3. [Main findings](#main-findings)
4. [Dataset](#dataset)
5. [The 17 vector verbs](#the-17-vector-verbs)
6. [Models](#models)
7. [Results](#results)
8. [Repository structure](#repository-structure)
9. [How to reproduce](#how-to-reproduce)
10. [Quick start: load the data](#quick-start-load-the-data)
11. [Annotation and quality](#annotation-and-quality)
12. [Limitations](#limitations)
13. [License](#license)
14. [Citation](#citation)

---

## What is a compound verb?

In Bangla (and Hindi, Urdu and other South Asian languages), a main verb is often followed by a second verb:

| Part | Name | Example | Role |
|---|---|---|---|
| V1 | **pole** | লিখে (write) | carries the main meaning |
| V2 | **vector** | ফেলল (throw) | loses its own meaning and adds a shade, here *completion* |

**লিখে ফেলল** means *finished writing*. Nothing is thrown.

But the same two verbs can also keep their own meanings and describe two separate events:

| Sentence | Use | Meaning |
|---|---|---|
| ছেলেটা পুরো ভাতটা খেয়ে **ফেলল**। | vector | The boy ate up all the rice. |
| ছেলেটা কলাটা খেয়ে খোসাটা রাস্তায় **ফেলল**। | full verb | The boy ate the banana and threw the peel on the road. |

The context must show which use is meant. This makes compound verbs a hard test of how language models handle multiword expressions.

---

## What the benchmark tests

We ask four research questions (RQs):

| RQ | Question | Method | Data |
|---|---|---|---|
| **RQ1** | Do models tell the two uses apart internally? | Linear probes on hidden states | 1,044 minimal pairs |
| **RQ2** | Do models understand what a vector adds? | Zero-shot entailment (NLI) | 204 entailment items |
| **RQ3** | Do models prefer correct pole–vector combinations? | Sentence probability (surprisal) | 197 selectional pairs |
| **RQ4** | Do models keep the vector's meaning in translation? | Bangla → English translation | 2,088 sentences (both sides of every pair) |

For RQ1 we do not trust probe accuracy alone. We compare it with:
- the same model with **random weights**,
- a one-feature **surface cue** (is the word before V2 a conjunctive participle?),
- the **majority class**,
- **hard pairs** where V1 comes right before V2 in both sentences, so the surface cue does not help.

---

## Main findings

- **Encoders capture the difference.** Probes trained on cue-free pairs reach up to **93%** accuracy, while random-weight models stay near chance (49–60%).
- **But a simple cue explains a lot.** On real-text pairs, one surface cue alone reaches **99.5%**. Benchmarks without hard pairs can give models credit for knowledge they do not use.
- **Prompted models rarely use the difference.** Entailment accuracy stays between **59% and 81%**. Several LLMs answer *Yes* or *No* to almost every item, and nine of ten models miss that someone who *ate up the rice* left none.
- **LLMs translate vectors literally.** Most instruction-tuned LLMs translate vectors literally in **15–20%** of vector sentences, and up to **55%** for মরা (die). Dedicated MT systems do much better (2.8–9.6%).
- **Bangla data often matters more than size.** On correct combinations, the small Bangla-only BanglaBERT generator (88%) matches BLOOM-7B1 (87%), and both beat Qwen2.5-7B (74%).

> **Note:** we probe the hidden states of encoders only. The behavioural failures come mostly from instruction-tuned LLMs. Our claim is therefore across model types, not within one model. See [Limitations](#limitations).

---

## Dataset

### Overview

| Part | File | Items | Used for |
|---|---|---|---|
| Minimal pairs (vector vs. full verb) | `minimal_pairs.csv` | 1,044 pairs | RQ1, RQ4 |
| ↳ hard / adjacent (cue-free) | | 74 | RQ1 |
| ↳ from real text (corpus) | | 103 | RQ1, RQ4 |
| Entailment items | `semantic_items.csv` | 204 | RQ2 |
| Selectional pairs (correct vs. incorrect) | `selectional_items.csv` | 197 | RQ3 |
| Vector verb list | `vector_verbs.csv` | 17 | all |
| Probing splits | `split.csv` | 1,044 rows | RQ1 |
| **Total items** | | **1,445** | |

Every part covers all 17 vector verbs (64–95 items per verb).

---

### 1. `minimal_pairs.csv` (1,044 rows)

Each row is one pair: a sentence where V2 is a **vector**, and a matched sentence where the same V2 is a **full verb**.

| Column | Description |
|---|---|
| `id` | Pair ID (`MP001` …) |
| `data_source` | `authored` (941) or `corpus` (103) |
| `vector_id` | Link to `vector_verbs.csv` (`V01` … `V17`) |
| `vector_verb` | Vector verb in dictionary form (e.g. ফেলা) |
| `v1` | The pole (V1), e.g. খেয়ে |
| `v2_form_vector` | Inflected V2 in the vector sentence |
| `v2_form_full` | Inflected V2 in the full-verb sentence |
| `sentence_vector` | Bangla sentence, vector use |
| `sentence_full_verb` | Bangla sentence, full-verb use |
| `translation_vector_en` | Reference English translation (vector sentence) |
| `translation_full_verb_en` | Reference English translation (full-verb sentence) |
| `pair_type` | `separated` (819), `same_v2` (151) or `adjacent` (74) |
| `source` | Where the pair came from |
| `notes` | Optional notes |
| `source_vector`, `source_full` | Source text for corpus pairs |
| `register_vector`, `register_full` | `chalit` or `sadhu` (corpus pairs only) |

**Pair types**

| Type | Meaning | Count |
|---|---|---|
| `separated` | In the full-verb sentence, other words come between V1 and V2 | 819 |
| `same_v2` | V2 is a plain main verb in the full-verb sentence | 151 |
| `adjacent` | V1 comes right before V2 in **both** sentences, with V2 in the same position (hard, cue-free pairs) | 74 |

**Corpus pairs.** The 103 real-text pairs were chosen and paired by hand from 260,063 sentences of public-domain fiction (Rabindranath Tagore, Sarat Chandra Chattopadhyay) and news (bdnews24, Ei Samay). They are 36 news–news, 28 literature–literature and 39 mixed pairs.

The pairs contain 552 different V1s and 932 different V1+V2 combinations; sentences have 6.6 words on average.

---

### 2. `semantic_items.csv` (204 rows)

Premise–hypothesis pairs that test what a vector adds.

| Column | Description |
|---|---|
| `item_id` | Item ID (`N001` …) |
| `vector_id`, `vector_bn` | The vector verb |
| `premise_bn` | Bangla premise |
| `hypothesis_bn` | Bangla hypothesis |
| `label` | `entailment` (95) or `not_entailment` (109) |
| `tested_meaning` | What the item tests (e.g. `completion`, `literal die lost`) |
| `premise_type` | `vector` (145), `full_verb` (40) or `simple_verb` (19) |
| `source` | `authored` |

**Item categories**

| Category | n | Gold |
|---|---|---|
| Vector meaning (stated) | 56 | entailment |
| Vector meaning (contradicted) | 25 | not entailment |
| **Literal trap**: the hypothesis states V2's own meaning | 64 | not entailment |
| Full-verb control | 40 | entailment (39) |
| Simple-verb control | 19 | not entailment |

Example:

| Premise | Hypothesis | Label |
|---|---|---|
| সে পুরো ভাতটা খেয়ে ফেলল। (He ate up all the rice.) | থালায় আর কোনো ভাত নেই। (No rice is left on the plate.) | entailment |
| শিশুটা হঠাৎ কেঁদে উঠল। (The child suddenly burst into tears.) | শিশুটা কোথাও উপরে উঠল। (The child climbed up somewhere.) | not entailment (literal trap) |

Twenty-one hypotheses appear after both a vector premise and a control premise; we use these for **pair consistency**.

---

### 3. `selectional_items.csv` (197 rows)

Pairs of a correct and an incorrect sentence that differ in one word or phrase.

| Column | Description |
|---|---|
| `item_id` | Item ID (`S001` …) |
| `vector_id`, `vector_bn` | The vector verb |
| `valid_sentence_bn` | Correct sentence |
| `invalid_sentence_bn` | Incorrect sentence |
| `reason` | Why the incorrect sentence is ruled out |
| `source` | `Paul 2003` (6), `authored (rule from Paul 2003)` (69), `authored (aspectual/semantic test)` (122) |

Example:

| Correct | Incorrect |
|---|---|
| ঋতু এক ঘণ্টার **মধ্যে** ছবিটা এঁকে ফেলল। (finished the picture within an hour) | \*ঋতু এক ঘণ্টা **ধরে** ছবিটা এঁকে ফেলল। (for an hour) |

**Rule types**

| Rule type | n |
|---|---|
| Duration / telicity | 73 |
| Intentionality / futility | 30 |
| V1 verb class | 28 |
| Agentivity of V1 | 28 |
| Benefactive direction | 23 |
| Arguments / animacy | 10 |
| হয়ে frame | 5 |
| **Total** | **197** |

---

### 4. `vector_verbs.csv` (17 rows)

| Column | Description |
|---|---|
| `vector_id` | `V01` … `V17` |
| `vector_bn` | Bangla form |
| `roman` | Romanisation |
| `lexical_meaning_en` | Meaning as a full verb |
| `vector_meaning_en`, `vector_meaning_bn` | What it adds as a vector |
| `sense_type` | aspectual and/or attitudinal |
| `inflected_forms` | Forms used for matching (`|`-separated) |
| `in_paul_2003`, `in_ghosh_2014` | Listed in Paul (2003) / Ghosh (2014) |
| `source_example_bn`, `source_example_en`, `example_source` | Example from the literature |

---

### 5. `split.csv` (1,044 rows)

Probing splits for RQ1.

| Column | Description |
|---|---|
| `id` | Pair ID (links to `minimal_pairs.csv`) |
| `split` | `train` (609), `dev` (82), `test` (176), `hard_test` (74), `corpus_test` (103) |
| `v1_seen_in_train` | Whether the V1 occurs in the training split |
| `fold` | 5-fold assignment for the hard pairs (1–5) |

Train, dev and test are split so that **no V1 appears in more than one set**. Hard and corpus pairs are used only for testing.

---

## The 17 vector verbs

| ID | Vector | Own meaning | Adds as a vector |
|---|---|---|---|
| V01 | ফেলা | throw | completion; doing without hesitation |
| V02 | দেওয়া | give | action done for someone else |
| V03 | নেওয়া | take | action done for oneself |
| V04 | বসা | sit | sudden, inappropriate action |
| V05 | ওঠা | rise | sudden onset |
| V06 | যাওয়া | go | change of state; continuation |
| V07 | পড়া | fall | suddenness; resulting state |
| V08 | রাখা | keep | done in advance |
| V09 | আসা | come | long duration up to now |
| V10 | আনা | bring | approaching completion |
| V11 | তোলা | lift | attainment after effort |
| V12 | চলা | walk, go on | continuous action |
| V13 | পাঠানো | send | action through someone else |
| V14 | বেড়ানো | roam | repeated, aimless action |
| V15 | মরা | die | futility |
| V16 | দাঁড়ানো | stand | entering a state |
| V17 | ছাড়া | leave | persistence until the goal |

Sources: Paul (2003) and Ghosh (2014). See `vector_verbs.csv` for which source lists each verb.

---

## Models

We test **26 public models**. No model is fine-tuned on our data; only the RQ1 probes are trained. Large models run in 4-bit NF4 precision on Kaggle T4 GPUs.

| # | Model | Hugging Face ID | RQ |
|---|---|---|---|
| 1 | BanglaBERT | `csebuetnlp/banglabert` | RQ1 |
| 2 | BanglaBERT generator | `csebuetnlp/banglabert_generator` | RQ3 |
| 3 | mBERT | `bert-base-multilingual-cased` | RQ1, RQ3 |
| 4 | MuRIL | `google/muril-base-cased` | RQ1, RQ3 |
| 5 | XLM-R base | `xlm-roberta-base` | RQ1, RQ3 |
| 6 | XLM-R large | `xlm-roberta-large` | RQ1, RQ3 |
| 7 | mDeBERTa-v3 NLI (2mil7) | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | RQ2 |
| 8 | mDeBERTa-v3 NLI (XNLI) | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | RQ2 |
| 9 | XLM-R large XNLI | `joeddav/xlm-roberta-large-xnli` | RQ2 |
| 10–13 | Qwen2.5 base 0.5B / 1.5B / 3B / 7B | `Qwen/Qwen2.5-{0.5B,1.5B,3B,7B}` | RQ3 |
| 14–17 | Qwen2.5 Instruct 0.5B / 1.5B / 3B / 7B | `Qwen/Qwen2.5-{0.5B,1.5B,3B,7B}-Instruct` | RQ2 (all), RQ4 (7B) |
| 18 | BLOOM-7B1 | `bigscience/bloom-7b1` | RQ3 |
| 19 | BLOOMZ-7B1 | `bigscience/bloomz-7b1` | RQ2, RQ4 |
| 20 | Gemma-2-2B | `google/gemma-2-2b` | RQ3 |
| 21 | Gemma-2-2B-IT | `google/gemma-2-2b-it` | RQ2, RQ4 |
| 22 | Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | RQ2, RQ4 |
| 23 | NLLB-200 600M | `facebook/nllb-200-distilled-600M` | RQ4 |
| 24 | NLLB-200 1.3B | `facebook/nllb-200-distilled-1.3B` | RQ4 |
| 25 | NLLB-200 3.3B | `facebook/nllb-200-3.3B` | RQ4 |
| 26 | BanglaT5 NMT | `csebuetnlp/banglat5_nmt_bn_en` | RQ4 |

Gemma and Llama are gated models: accept their licences on Hugging Face before running.

---

## Results

### RQ1: Probing (accuracy)

| Model | Layer | Test | Corpus | Hard (transfer) | Hard (5-fold) | Random (5-fold) |
|---|---|---|---|---|---|---|
| BanglaBERT | 4 | **.97** | **.87** | **.64** | **.93** | .51 |
| mBERT | 9 | .93 | .67 | .59 | .80 | .51 |
| MuRIL | 2 | .96 | .82 | .54 | .91 | .49 |
| XLM-R base | 6 | .95 | .79 | .61 | .85 | .56 |
| XLM-R large | 9 | .95 | .78 | .55 | .86 | .60 |
| Surface cue | – | .66 | .995 | .50 | .50 | – |
| Majority | – | .50 | .50 | .50 | .50 | – |

*Transfer*: probe trained on ordinary pairs, tested on hard pairs. *5-fold*: trained and tested within the hard pairs.

### RQ2: Entailment

| Model | Acc | AUC | Yes rate | Trap acc |
|---|---|---|---|---|
| mDeBERTa-NLI (2mil7) | .77 | .85 | .45 | .72 |
| mDeBERTa-NLI (XNLI) | .77 | .84 | .53 | .59 |
| XLM-R-large-XNLI | .75 | .86 | .61 | .50 |
| Qwen2.5-0.5B-Inst | .62 | .66 | .48 | .73 |
| Qwen2.5-1.5B-Inst | .75 | .81 | .53 | .73 |
| Qwen2.5-3B-Inst | .59 | .79 | .05 | 1.00 |
| Qwen2.5-7B-Inst | **.81** | **.91** | .33 | .97 |
| BLOOMZ-7B1 | .61 | .80 | .78 | .47 |
| Gemma-2-2B-IT | .59 | .74 | .83 | .28 |
| Llama-3.1-8B-Inst | .68 | .88 | .16 | .98 |
| always *No* | .53 | .50 | .00 | 1.00 |

LLM scores average an English and a Bangla prompt. Gold *Yes* rate: 0.47.

### RQ3: Selectional preferences (accuracy, 197 pairs)

| Model | Acc | Fewer-tokens baseline |
|---|---|---|
| BanglaBERT generator | **.88** | .55 |
| BLOOM-7B1 | .87 | .55 |
| MuRIL | .80 | .55 |
| Qwen2.5-7B | .74 | .51 |
| XLM-R large | .73 | .55 |
| Gemma-2-2B | .71 | .52 |
| XLM-R base | .68 | .55 |
| Qwen2.5-1.5B | .64 | .51 |
| mBERT | .64 | .55 |
| Qwen2.5-0.5B | .60 | .51 |
| Qwen2.5-3B | .59 | .51 |

### RQ4: Translation (%)

| System | Literal error ↓ | Meaning lost ↓ | chrF++ (vector) | chrF++ (full) |
|---|---|---|---|---|
| NLLB-200 600M | 9.6 | 27.1 | 45.9 | 50.9 |
| NLLB-200 1.3B | 6.5 | 27.9 | 48.9 | 54.6 |
| NLLB-200 3.3B | 6.5 | 26.4 | **50.3** | 55.7 |
| BanglaT5-NMT | **2.8** | 25.6 | 49.9 | **59.7** |
| Gemma-2-2B-IT | 18.0 | 34.7 | 41.6 | 41.1 |
| Qwen2.5-7B-Inst | 19.7 | **19.2** | 47.4 | 51.3 |
| Llama-3.1-8B-Inst | 15.4 | 27.6 | 49.5 | 52.5 |
| BLOOMZ-7B1 | 4.0 | 43.6 | 44.8 | 44.3 |

*Literal error*: vector sentences whose translation has a literal word (e.g. *threw*) that the reference does not have (n = 948; পাঠানো is excluded). *Meaning lost*: full-verb sentences whose translation misses the literal meaning.

---

## Repository structure

```
BanglaCV-Probe/
├── data/
│   ├── minimal_pairs.csv
│   ├── semantic_items.csv
│   ├── selectional_items.csv
│   ├── vector_verbs.csv
│   └── split.csv
├── notebooks/
│   ├── rq1_probing.ipynb
│   ├── rq2_nli.ipynb
│   ├── rq3_surprisal.ipynb
│   └── rq4_translation.ipynb
├── results/
│   ├── rq1/   # summary tables, per-layer scores, predictions
│   ├── rq2/   # per-model P(Yes), summary, by category, by vector
│   ├── rq3/   # per-model scores, summary, by rule type, by vector
│   └── rq4/   # per-model translations, summary, by subset, by vector
├── figures/
├── LICENSE
└── README.md
```

---

## How to reproduce

### Requirements

- Python 3.10+
- A GPU with about 15 GB memory (we used Kaggle T4 GPUs). Models above 3B run in 4-bit.

```bash
pip install torch transformers accelerate bitsandbytes \
            scikit-learn pandas numpy sacrebleu matplotlib sentencepiece
```

### Hugging Face token

Gemma and Llama need a Hugging Face token.

- **On Kaggle:** add it under *Add-ons → Secrets* as `HF_TOKEN` and tick the box for the notebook.
- **Locally:** run `huggingface-cli login`, or set the `HF_TOKEN` environment variable.

⚠️ **Never write your token into a notebook or commit it to GitHub.**

```python
# Kaggle
from kaggle_secrets import UserSecretsClient
import os
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

### Steps

Each notebook is self-contained and resumable: it saves results per model, so you can stop and continue.

| Step | Notebook | What it does | Output |
|---|---|---|---|
| 1 | `rq1_probing.ipynb` | Mean-pools V2 hidden states at every layer, trains logistic-regression probes, and runs random-weight, surface-cue and 5-fold hard-pair controls | `results/rq1/` |
| 2 | `rq2_nli.ipynb` | Scores 204 items with NLI classifiers and zero-shot LLMs (English and Bangla prompts) | `results/rq2/` |
| 3 | `rq3_surprisal.ipynb` | Compares sentence probabilities (PLL-word-l2r for masked models, summed log-probability for causal models) | `results/rq3/` |
| 4 | `rq4_translation.ipynb` | Translates 2,088 sentences, detects literal translations, computes chrF++ | `results/rq4/` |

The full RQ4 run (8 systems, 2,088 sentences each) took about 39 minutes on Kaggle T4 GPUs.

### Key settings

| Setting | Value |
|---|---|
| Probe | L2 logistic regression, C ∈ {0.01, 0.1, 1, 10}, layer and C chosen on dev |
| Hard-pair CV | 5 folds, both sentences of a pair in the same fold, layer chosen inside each fold |
| NLI decision (LLMs) | P(*Yes*) vs. P(*No*) for the first answer token, averaged over an English and a Bangla prompt, *Yes* if > 0.5 |
| Masked-LM scoring | PLL-word-l2r (Kauf & Ivanova, 2023) |
| MT decoding | Beam search, 4 beams, up to 96 new tokens; NLLB `ben_Beng → eng_Latn` |
| LLM decoding | Greedy, up to 80 new tokens |
| Confidence intervals | Bootstrap over items (2,000 resamples for RQ1; 5,000 for RQ3 comparisons) |

---

## Quick start: load the data

```python
import pandas as pd

pairs  = pd.read_csv("data/minimal_pairs.csv", encoding="utf-8-sig")
nli    = pd.read_csv("data/semantic_items.csv", encoding="utf-8-sig")
sel    = pd.read_csv("data/selectional_items.csv", encoding="utf-8-sig")
verbs  = pd.read_csv("data/vector_verbs.csv", encoding="utf-8-sig")
split  = pd.read_csv("data/split.csv", encoding="utf-8-sig")

print(len(pairs), len(nli), len(sel), len(verbs))   # 1044 204 197 17

# hard (cue-free) pairs
hard = pairs.merge(split, on="id").query("split == 'hard_test'")
print(hard[["sentence_vector", "sentence_full_verb"]].head())

# literal traps
traps = nli[nli.tested_meaning.str.contains("literal") & (nli.premise_type == "vector")]
print(len(traps))   # 64
```

**Tip:** Bangla text is NFC-normalised. If you add your own sentences, normalise them too, because letters such as ড় and য় have two Unicode forms:

```python
import unicodedata
text = unicodedata.normalize("NFC", text)
```

---

## Annotation and quality

- Three native Bangla speakers checked every item on their own.
- They judged naturalness and the intended use for minimal pairs, the label for entailment items, and correctness for selectional pairs.
- Agreement: **Fleiss' κ = 0.92**. Disagreements were resolved by discussion.
- An early version of the adjacent pairs had a position cue (V2 more often sentence-final in the vector sentence; the cue scored 0.84). We rewrote 54 of the 74 pairs so that V2 has the same position in both sentences. Both cues now score 0.50.

---

## Limitations

- **Probing covers encoders only.** We do not probe the LLMs whose behaviour we test, so we cannot say whether a given LLM knows the difference and fails to use it.
- **Mostly written data.** 941 of the 1,044 pairs were written by us. All items use standard colloquial Bangla (*chalit*); dialects and the formal *sadhu* style are almost absent.
- **Pragmatic labels.** Some entailment items rest on a strong default inference rather than strict entailment.
- **Evaluation choices.** Zero-shot prompts, greedy decoding and 4-bit quantisation; other settings could change single scores.
- **Literal-translation detector.** Word lists can miss paraphrases and sometimes flag a correct idiom.
- **Small groups.** Some rule types and per-verb subsets are small, so we do not read much into differences within them.

---

## License

- **Authored data** (written by us): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Corpus sentences:** literature sentences are from public-domain works. News sentences are released only as far as their original licences permit.
- **Code:** MIT License.
- **Models** are used under their own licences.

---

## Citation

If you use BanglaCV-Probe, please cite:

```bibtex
@misc{banglacvprobe2026,
  title  = {Pole or Vector? Encoders Capture Bangla Compound Verbs, but LLMs Read Them Literally},
  author = {AUTHOR NAMES},
  year   = {2026},
  note   = {arXiv preprint, ID to be added}
}
```

---

## Contact

For questions or issues, please open a GitHub issue or contact: AUTHOR EMAIL
