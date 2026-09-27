<div align="center">

# JuL, Juste un LLM

**Typed decisions on your machine, with the model of your choice.<br>No training, no API, no task learned by heart.**

[![PyPI](https://img.shields.io/pypi/v/jul?color=blue&label=PyPI)](https://pypi.org/project/jul/)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-3776AB?logo=python&logoColor=white)](https://pypi.org/project/jul/)
[![CI](https://img.shields.io/github/actions/workflow/status/usejul/jul/ci.yml?branch=main&label=CI)](https://github.com/usejul/jul/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://github.com/usejul/jul/blob/main/LICENSE)
<br>
[![Backends](https://img.shields.io/badge/backends-MLX%20%C2%B7%20PyTorch%20%C2%B7%20ONNX-orange)](https://github.com/usejul/jul/blob/main/docs/installation.md)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20models-usejul-yellow)](https://huggingface.co/usejul)
[![Website](https://img.shields.io/badge/site-usejul.github.io%2Fjul-black)](https://usejul.github.io/jul/)

[Website](https://usejul.github.io/jul/) · [Showcases](#showcases) · [Quickstart](#quickstart) · [Results](#results) · [Docs](#documentation)

</div>

JuL answers typed questions about a piece of text: pick an option, say yes or no, give a score. Each
answer comes with a probability, it takes about 55 ms on a Mac, and nothing leaves your machine. It is
a drop-in for the Jev SDK: change the import and your code runs locally.

## Showcases

[jul-showcases](https://github.com/guyon-it-consulting/jul-showcases), by
[Jérôme Guyon](https://github.com/JeromeGuyon), takes ideas from [jevable.com](https://jevable.com/)
and runs them with JuL, on device, for $0.

<table>
<tr>
<td width="50%" valign="top">
<img src="https://raw.githubusercontent.com/usejul/jul/main/docs/assets/showcase-sncf.gif" alt="A browser agent booking a Lyon to Toulouse train on SNCF Connect"><br>
<b>A browser agent on SNCF Connect.</b> JuL picks each action from the page's accessibility tree
and the Apple Foundation Model types the city names. Six steps from the homepage to priced
results, about 130 ms per decision.
</td>
<td width="50%" valign="top">
<img src="https://raw.githubusercontent.com/usejul/jul/main/docs/assets/showcase-qa.gif" alt="JuL running a plain-language QA ticket on vistaprint.com"><br>
<b>QA written by the product manager.</b> The test is a ticket in plain French or English. JuL
runs it in a real browser and checks each acceptance criterion. Played here at 3× speed.
</td>
</tr>
</table>

The others run in a terminal:

- [Ticket triage at scale](https://github.com/guyon-it-consulting/jul-showcases/tree/main/ticket-triage-scale): 50,000 real support tickets in 668 s, $0.
- [autotune](https://github.com/guyon-it-consulting/jul-showcases/tree/main/ticket-triage-autoscale) takes a fast model from 82.0% to 96.5% with a head trained in 6.2 s.
- [A form that branches on its own answers](https://github.com/guyon-it-consulting/jul-showcases/tree/main/julform), [re-ranking by intent](https://github.com/guyon-it-consulting/jul-showcases/tree/main/intent-reranker), [notification triage](https://github.com/guyon-it-consulting/jul-showcases/tree/main/notification-triage) and [prompt-difficulty routing](https://github.com/guyon-it-consulting/jul-showcases/tree/main/prompt-difficulty).

## Install

```bash
pip install jul
jul setup          # picks MLX or PyTorch, installs it, downloads the default model, runs one decision
```

Python ≥ 3.10. The default model weighs 2.6 GB and is downloaded once from the Hugging Face Hub.
To pick the backend yourself (`jul[mlx]`, `jul[torch]`, `jul[onnx]`), see
[docs/installation.md](https://github.com/usejul/jul/blob/main/docs/installation.md).

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
would understand; the key is only the name you get back. `AsyncTypeSafeClient` has the same API.
From the shell, use `jul ask`. Non-Python callers can use `jul serve`, which speaks the Jev HTTP
protocol ([docs/serve.md](https://github.com/usejul/jul/blob/main/docs/serve.md)).

## How it works

<img src="https://raw.githubusercontent.com/usejul/jul/main/docs/assets/how-it-works.svg" alt="The text and the option descriptions go through the same model; the answer is the closest option vector" width="760">

The model is stopped one step before it would start writing. JuL reads the hidden state it built for
your text and compares it with the vectors of your option descriptions. Your task never reaches the
weights, so you can change the options between two calls, or swap the model, without retraining
anything. When it is off on your data, `client.autotune(...)` fits a small head on labeled examples
and keeps it only if it beats zero-shot in cross-validation
([docs/tuning.md](https://github.com/usejul/jul/blob/main/docs/tuning.md)).

## Results

<img src="https://raw.githubusercontent.com/usejul/jul/main/docs/assets/results.svg" alt="Mean accuracy on Jev's benchmark per model" width="760">

Jev's published benchmark, 300 examples, zero-shot for every JuL model, run with
`scripts/bench_jul.py`. Each mean is ±3 points. Banking77 and Emotion are in MTEB, which the
`wemm-*` and `f2llm-*` models trained on, so AG News is the clean comparison: 0.95 for `wemm-4b`,
0.91 for Jev. The default model's ECE is 0.084 against Jev's 0.156, which is what lets you automate
above a confidence threshold. We have only measured English text so far. Per-task scores, the 17
models measured and the tuned results are in
[docs/benchmarks.md](https://github.com/usejul/jul/blob/main/docs/benchmarks.md).

### The baseline worth remembering

A TF-IDF + linear SVM, trained on 1000 labeled examples with no LLM at all, scores 0.88 on AG News,
0.76 on Banking77 and 0.43 on Emotion: a mean of 0.690 at 0.17 ms per prediction
(`scripts/bench_tfidf.py`). It only clearly loses on Emotion. If you have labels and your problem is
sorting by topic or intent, try it first. CI re-measures these numbers on every PR
([numbers.yml](https://github.com/usejul/jul/blob/main/.github/workflows/numbers.yml), `CLAIMED`),
so update both together.

## Documentation

- [Installation](https://github.com/usejul/jul/blob/main/docs/installation.md): backends, devices, batching, `jul setup`
- [Models](https://github.com/usejul/jul/blob/main/docs/models.md): presets, the 17 models measured, `jul models add`
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
