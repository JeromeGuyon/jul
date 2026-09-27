<div align="center">

# JuL — Juste un LLM

**Typed decisions on your machine, with the model of your choice.<br>No training, no API, no task learned by heart.**

[![PyPI](https://img.shields.io/pypi/v/jul?color=blue&label=PyPI)](https://pypi.org/project/jul/)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-3776AB?logo=python&logoColor=white)](https://pypi.org/project/jul/)
[![CI](https://img.shields.io/github/actions/workflow/status/usejul/jul/ci.yml?branch=main&label=CI)](https://github.com/usejul/jul/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://github.com/usejul/jul/blob/main/LICENSE)
<br>
[![Backends](https://img.shields.io/badge/backends-MLX%20%C2%B7%20PyTorch%20%C2%B7%20ONNX-orange)](https://github.com/usejul/jul/blob/main/docs/installation.md)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20models-usejul-yellow)](https://huggingface.co/usejul)
[![Website](https://img.shields.io/badge/site-usejul.github.io%2Fjul-black)](https://usejul.github.io/jul/)

[Website](https://usejul.github.io/jul/) · [Quickstart](#quickstart) · [Showcases](#showcases) · [Results](#results) · [Docs](#documentation)

</div>

JuL is a Python library that answers typed questions about a piece of text: pick an option
(`Choice`), say yes or no (`Noul`), give a score (`Score`). Each answer comes with a probability. It
has the same API as the SDK of Jev, TypeSafe's hosted decision model.

The model is stopped one step before its first syllable and the answer is read straight out of its
hidden states: no monologue, no reasoning trace, no opinion on the matter. It has nothing to say, and
it says it in 55 milliseconds.

It runs on your own machines, from a Mac to a Linux server, so you can keep it running 24/7 on your own
infrastructure. Your text never goes to a third party and there is no per-call bill. The model weights
are downloaded once from the Hugging Face Hub, or loaded from a local path; after that it runs with no
network at all.

## Why JuL

- Zero-shot: the default model scores 0.857 on Jev's public benchmark (Jev: 0.753) without one example
  of its tasks. Its settings were fitted on dev sets the benchmark never touches. On AG News, the one
  task embedding models haven't trained on, the default gets 0.90, `wemm-4b` 0.95 and Jev 0.91.
- Many options: option vectors are computed once and cached, so adding options barely changes the cost
  of a call. On Banking77's 72 intents the default model gets 0.87, on a task its training data
  (MTEB) contains.
- Probabilities you can use: ECE 0.084, so you can automate above a confidence threshold and send the
  rest to a human.
- Small and fast: 2.6 GB and 55 ms per decision on a Mac. `f2llm-1.7b` (converted
  locally, not published yet) fits in 1 GB and answers in 24 ms.
- Replaceable model: we measured 18, and `jul models add` wires in a new one without touching your code.
- Tunable when you have labels: `autotune(...)` takes the default model from 0.857 to 0.897 with 1000
  examples, and keeps the head only if it beats zero-shot.
- Drop-in for the Jev SDK: change the import and the same code runs on your machines. `jul serve`
  speaks the Jev HTTP protocol for other languages.

## Quickstart

```python
# from typesafe_sdk import TypeSafeClient, Choice, Noul, Score
from jul import TypeSafeClient, Choice, Noul, Score

client = TypeSafeClient()                             # wemm-4b-4bit, or model="minicpm5-2b"

response = client.system_one(
    state={"ticket": "I was charged twice for my subscription this month."},
    questions={
        "team": Choice(instructions="Which team should handle this ticket?",
                       criteria={"billing": "payments, invoices, refunds",
                                 "technical": "bugs, errors, crashes",
                                 "sales": "pricing, plans, demos"}),
        "is_bug": Noul(instructions="Does the message report a software bug?"),
        "frustration": Score(instructions="How frustrated is the customer?",
                             criteria=["Calm", "Frustrated but civil", "Very angry"]),
    },
)

response.choices["team"].choice           # "billing"
response.choices["team"].probabilities    # {"billing": 0.88, "technical": 0.11, "sales": 0.01}
response.nouls["is_bug"].noul             # 0.12
response.scores["frustration"].score      # 1.15
```

The model compares your text with each option's *description*, so write descriptions a colleague
would understand; the key is only the name you get back. `noul` is the probability of yes, and `score`
is the expected level on your scale, from 0 ("Calm") to 2 ("Very angry"). `AsyncTypeSafeClient` has the same API.
Arguments that only make sense for a remote API (`api_key`, `retry`, …) are accepted and ignored, so
code written for Jev runs unchanged.
From the shell, use `jul ask`. Non-Python callers can use `jul serve`, which speaks the Jev HTTP
protocol ([docs/serve.md](https://github.com/usejul/jul/blob/main/docs/serve.md)).

## Showcases

Eight demos from [jul-showcases](https://github.com/usejul/jul-showcases), each an idea from
[jevable.com](https://jevable.com/) running entirely on device, for $0.

<table>
<tr>
<td width="50%" valign="top">
<img src="https://raw.githubusercontent.com/usejul/jul/main/docs/assets/showcase-sncf.gif" alt="A browser agent booking a Lyon to Toulouse train on SNCF Connect"><br>
<b><a href="https://github.com/usejul/jul-showcases/tree/main/browser-agent">Browser agent on SNCF Connect</a>.</b>
JuL picks each action from the page's accessibility tree and the Apple Foundation Model types the
city names (macOS 26). Six steps from the homepage to priced results, about 130 ms per decision.
</td>
<td width="50%" valign="top">
<img src="https://raw.githubusercontent.com/usejul/jul/main/docs/assets/showcase-triage.svg" alt="Terminal output: 50,000 support tickets triaged in 668 s at 82.9% accuracy for $0"><br>
<b><a href="https://github.com/usejul/jul-showcases/tree/main/ticket-triage-scale">Ticket triage at scale</a>.</b>
50,000 real support tickets routed in 668 s, 82.9% accurate, for $0, with the small
`qwen3-embedding-0.6b` on an Apple Silicon Mac. At
that rate a million take about 3.7 hours.
</td>
</tr>
</table>

- [Browser agent](https://github.com/usejul/jul-showcases/tree/main/browser-agent): books a train on SNCF Connect from its accessibility tree.
- [QA from the ticket](https://github.com/usejul/jul-showcases/tree/main/qa-browser): runs a plain-language acceptance test in a real browser.
- [Ticket triage at scale](https://github.com/usejul/jul-showcases/tree/main/ticket-triage-scale): 50,000 real support tickets in 668 s, $0.
- [Ticket triage with autotune](https://github.com/usejul/jul-showcases/tree/main/ticket-triage-autoscale): a fast model goes from 82.0% to 96.5% with a head trained in 6.2 s.
- [Self-branching form](https://github.com/usejul/jul-showcases/tree/main/julform): picks the next question from the answers so far.
- [Intent re-ranker](https://github.com/usejul/jul-showcases/tree/main/intent-reranker): sorts a list by a plain-language intent.
- [Notification triage](https://github.com/usejul/jul-showcases/tree/main/notification-triage): mutes marketing, keeps OTPs and appointments.
- [Prompt difficulty](https://github.com/usejul/jul-showcases/tree/main/prompt-difficulty): routes a prompt to fast mode or the full model before it is sent.

## Install

```bash
pip install jul
jul setup          # picks MLX or PyTorch, installs it, downloads the default model, runs one decision
```

Python ≥ 3.10. The default model weighs 2.6 GB and is downloaded once from the Hugging Face Hub.
To pick the backend yourself (`jul[mlx]`, `jul[torch]`, `jul[onnx]`), see
[docs/installation.md](https://github.com/usejul/jul/blob/main/docs/installation.md).

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/usejul/jul/main/docs/assets/how-it-works-dark.svg">
  <img src="https://raw.githubusercontent.com/usejul/jul/main/docs/assets/how-it-works.svg" alt="The text and the option descriptions go through the same model; the answer is the closest option vector" width="760">
</picture>

JuL reads the hidden state the model built for your text and compares it with the vectors of your
option descriptions. Your task never reaches the weights, so you can change the options between two calls, or swap the model, without retraining
anything. When it is off on your data, `client.autotune(...)` fits a small head on labeled examples
and keeps it only if it beats zero-shot in cross-validation
([docs/tuning.md](https://github.com/usejul/jul/blob/main/docs/tuning.md)).

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/usejul/jul/main/docs/assets/results-dark.svg">
  <img src="https://raw.githubusercontent.com/usejul/jul/main/docs/assets/results.svg" alt="Mean accuracy on Jev's benchmark per model" width="760">
</picture>

Jev's published benchmark, 300 examples, zero-shot for every JuL model, run with
`scripts/bench_jul.py`. Each mean is ±3 points. Banking77 and Emotion are in MTEB, which the
`wemm-*` and `f2llm-*` models trained on, so AG News is the clean comparison: 0.95 for `wemm-4b`,
0.91 for Jev, and 0.90 for the default model. The default model's ECE is 0.084 against Jev's 0.156, which is what lets you automate
above a confidence threshold. Every setting was fitted on English, and we haven't tested domains far
from this kind of text (sensor logs, chemistry and so on). Per-task scores, the 18 models measured and
the tuned results are in
[docs/benchmarks.md](https://github.com/usejul/jul/blob/main/docs/benchmarks.md).

### The baseline worth remembering

A TF-IDF + linear SVM, trained on 1000 labeled examples with no LLM at all, scores 0.88 on AG News,
0.76 on Banking77 and 0.43 on Emotion: a mean of 0.690 at 0.17 ms per prediction
(`scripts/bench_tfidf.py`). It is 11 points behind the default model on Banking77 and far behind on
Emotion, where telling feelings apart takes meaning, not vocabulary. If you have labels and your problem is
sorting by topic or intent, try it first. CI re-measures these numbers on every PR
([numbers.yml](https://github.com/usejul/jul/blob/main/.github/workflows/numbers.yml), `CLAIMED`),
so update both together.

## Models

| Preset | Size | Jev bench, zero-shot | + autotune, 1000 labels | Notes |
| --- | ---: | ---: | ---: | --- |
| `wemm-4b-4bit` (default, alias `accurate`) | 2.6 GB | 0.857 | 0.897 | built in |
| `minicpm5-2b` (alias `fast`) | 2.7 GB | 0.617 | 0.757 | built in, 64 ms on an M4 Pro |
| `minicpm5-2b-decision` | 1.3 GB | see [benchmarks](https://github.com/usejul/jul/blob/main/docs/benchmarks.md) | — | trained decision model, `jul models add` |
| `e5-small` (ONNX, 8-bit) | 0.09 GB | 0.543 | 0.713 (0.790 hybrid head) | encoder, 6 ms per text on an M4 Pro; needs an ONNX export first, see [models](https://github.com/usejul/jul/blob/main/docs/models.md#micro-models-encoders) |

To use another model, run `jul models add <name> --repo <hf-repo>`. It fits the layer, center and
temperature on the dev sets. The 18 models we measured, encoders, decision models and every setting
are listed in [docs/models.md](https://github.com/usejul/jul/blob/main/docs/models.md).

## Documentation

- [Installation](https://github.com/usejul/jul/blob/main/docs/installation.md): backends, devices, batching, `jul setup`
- [Models](https://github.com/usejul/jul/blob/main/docs/models.md): presets, the 18 models measured, `jul models add`
- [Adapting to your data](https://github.com/usejul/jul/blob/main/docs/tuning.md): `Context`, `autotune(...)`, hybrid heads, `jul synth`
- [Deployment](https://github.com/usejul/jul/blob/main/docs/deployment.md): `jul pack`, ONNX bundles, AWS Lambda
- [Serving over HTTP](https://github.com/usejul/jul/blob/main/docs/serve.md): `jul serve`
- [Command line](https://github.com/usejul/jul/blob/main/docs/cli.md): every command and file format
- [Benchmarks](https://github.com/usejul/jul/blob/main/docs/benchmarks.md): full results and how to reproduce them
- [Development](https://github.com/usejul/jul/blob/main/docs/development.md) and [Publishing](https://github.com/usejul/jul/blob/main/docs/publishing.md)

## Contributing

Issues, measurements on your own data and pull requests are welcome on
[GitHub](https://github.com/usejul/jul/issues). `pip install -e ".[dev]"`, then `pytest tests` (no
model, about 30 s). Tune on the dev datasets and run the benchmark once at the end
([how](https://github.com/usejul/jul/blob/main/docs/benchmarks.md#reproducing-the-measurements)).

## License

Apache 2.0, see [LICENSE](https://github.com/usejul/jul/blob/main/LICENSE) and
[NOTICE](https://github.com/usejul/jul/blob/main/NOTICE). The decision-model format comes from
[Kev](https://github.com/jaredpalmer/kev) (Jared Palmer, Apache 2.0) and `minicpm5-2b-decision` is
[MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) (OpenBMB, Apache 2.0) trained with Kev's
code. JuL follows TypeSafe's public System One API and uses no TypeSafe or Jev code, weights or
outputs.
