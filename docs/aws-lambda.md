# Running a decision in AWS Lambda

JuL's micro model, `jul-decision-e5-small`, runs inside an AWS Lambda function: no GPU, no server, no
container, the model in the function's own package. Jev is a hosted API and CLM-8B needs a GPU server;
to our knowledge no other typed-decision library puts its model in a Lambda.

**What this is for.** Not a general-purpose decision endpoint that answers any question you send it:
that is what the larger models are for. It is a model embedded in a function that decides one precise
thing, always the same, on every event it receives — which team a ticket goes to, whether a message is
angry, whether a request needs a human — at a few milliseconds and well under a dollar per million
calls. The questions are fixed when you build the function; the event only carries the text.

**Tune it.** The model is small (21 M parameters outside its embedding). Zero-shot it is a starting
point, not the answer: on the Jev bench it scores 0.557 alone and 0.780 with a head trained on your
labeled examples (`autotune`, hybrid features). For a function that decides one thing, labeled examples
of that thing are the best investment: a few hundred per question, more for many options. No labels
yet? `jul synth` writes synthetic ones from the questions (see [tuning](tuning.md)).

We do not publish the code of our own deployment yet. What follows is everything it does, step by step,
deployed with the AWS CLI or with CDK. It needs a jul release with cross models (after 0.2.0).

## What was measured

Support-message triage, two questions per message (intent among 72, emotion among 6), hybrid heads,
multilingual-e5-small (the same architecture and size), arm64, eu-west-1 prices:

| Lambda memory | Cold start | Per message (billed p50) | Per million messages |
| --- | ---: | ---: | ---: |
| 1,024 MB | 2.4 s | 36 ms | $0.67 |
| **1,769 MB (1 vCPU)** | 2.4 s | **17 ms** | **$0.59** |
| 3,538 MB | 3.0 s | 17 ms | $0.98 |

Accuracy of that function: 0.787 (emotion) / 0.890 (intent). The package, model included, is 225 MB.
The cross model that reads Noul and Score (see [models](models.md#cross-models-reading-the-question-and-the-text-together))
answers a yes/no in 4–7 ms on an M4 Pro; it has not been timed in Lambda yet.

## 1. Install and add the model (on your machine)

```bash
pip install "jul[onnx,tune,yaml,calibrate]"
jul models add jul-decision-e5-small --repo usejul/jul-decision-e5-small-onnx --backend onnx
```

`jul models add` fits the vector reading on the calibration sets and attaches the cross model the repo
carries in `cross/`.

## 2. Write the questions

`questions.yaml`: the fixed questions the function answers.

```yaml
team:
  type: choice
  instructions: Which team should handle this ticket?
  criteria:
    billing: payments, invoices, refunds
    tech: bugs, errors, crashes
    sales: plans, pricing, upgrades
angry:
  type: noul
  instructions: Is the customer angry?
```

Keep it to what the function decides. Each Choice adds its options' reading, each Noul or Score a pass
of the cross model.

## 3. Label examples and tune

`labeled.jsonl`, one message per line, with the answer to each question:

```json
{"state": "I was charged twice for order 4411, fix it now", "answers": {"team": "billing", "angry": true}}
{"state": "the export button does nothing", "answers": {"team": "tech", "angry": false}}
```

```bash
jul autotune tickets --questions questions.yaml --labeled labeled.jsonl --features hybrid \
    --model jul-decision-e5-small --backend onnx
```

`autotune` prints, per question, whether the head beat zero-shot in cross-validation; a head that does
not is not used. For a Noul or a Score, "zero-shot" is the cross model: the head must beat it on your
examples to replace it.

## 4. Pack

```bash
jul pack bundle/ --questions questions.yaml --context tickets --model jul-decision-e5-small --backend onnx
```

The last part of the first line says which models the bundle needs: `vector`, `cross` or both. Choice
questions and tuned heads read the vector model; untuned Noul and Score questions read the cross model.
Only those go into the package: a function that asks only yes/no and score questions ships the cross
model alone. If a pack says `vector + cross`, split the questions into two bundles (and two functions,
see below), or tune the yes/no questions so their heads read the vectors.

## 5. Build the package

A Lambda zip holds at most 250 MB once unzipped, and must be built for the Lambda's CPU (arm64 here) and
Python (3.12), whatever machine you build on:

```bash
mkdir -p build
PLAT="--platform manylinux2014_aarch64 --platform manylinux_2_28_aarch64 --implementation cp --python-version 3.12 --only-binary=:all:"
pip install -q --target build $PLAT "onnxruntime>=1.18" "jinja2>=3.1" "numpy>=1.24"
pip install -q --target build $PLAT --no-deps "tokenizers>=0.20"   # its Hub dependencies only download models
pip install -q --target build --no-deps jul
# not used at run time: ONNX Runtime's model tools, numpy's tests; boto3 comes with the Lambda runtime
rm -rf build/onnxruntime/{transformers,quantization,tools} build/numpy/_core/tests build/numpy/tests build/bin
find build -name __pycache__ -type d -prune -exec rm -rf {} +

cp -r bundle build/bundle
huggingface-cli download usejul/jul-decision-e5-small-onnx --local-dir models/jul-decision-e5-small
cp -r models/jul-decision-e5-small build/model                 # if the bundle needs "vector"
rm -rf build/model/cross build/model/.cache
cp -r models/jul-decision-e5-small/cross build/model-cross     # if the bundle needs "cross"
du -sh build                                                   # must stay under 250 MB
```

**One model per function.** One model fits the 250 MB (about 225 MB with the dependencies); both do
not. If your questions need both (Choice and untuned yes/no), prefer two functions, each packed with
its own questions and shipping its own model: one for the Choice questions, one for the yes/no and
score questions. Each keeps the 2.4 s cold start.

Other ways, not recommended:
- a container image (up to 10 GB) holds both models; its cold start was not measured;
- reading a graph from S3 at start (`JUL_ONNX_MODEL`, `JUL_ONNX_CROSS_MODEL` = `s3://bucket/key`, read
  into memory) lengthens every cold start by the download: a 1.1 GB graph from S3 took the cold start
  to ~17 s, against 2.4 s for a model in the package.

`build/handler.py`:

```python
import os
import time

from jul import Bundle

HERE = os.path.dirname(__file__)
_cross = os.path.join(HERE, "model-cross")
bundle = Bundle.load(os.path.join(HERE, "bundle"), backend="onnx", model=os.path.join(HERE, "model"),
                     cross_model=_cross if os.path.isdir(_cross) else None)
bundle.system_one("warm-up")   # at init, not on the first call


def handler(event, context):
    messages = event.get("messages") or [event["message"]]
    t = time.perf_counter()
    responses = bundle.system_one_batch(messages)
    return {"results": [{name: answer.as_dict() for name, answer in r.answers.items()} for r in responses],
            "ms": round((time.perf_counter() - t) * 1000, 1)}
```

The bundle names its models; `model=` and `cross_model=` only say where they are in the package, so
nothing is downloaded at run time.

## 6a. Deploy with the AWS CLI

A zip over 50 MB is uploaded through S3 (the package only, the model stays inside it). The role needs
`AWSLambdaBasicExecutionRole`.

```bash
(cd build && zip -qr ../function.zip .)
aws s3 cp function.zip s3://<bucket>/jul/function.zip
aws lambda create-function --function-name tickets --runtime python3.12 --architectures arm64 \
    --memory-size 1769 --timeout 30 --handler handler.handler --role <role arn> \
    --code S3Bucket=<bucket>,S3Key=jul/function.zip
aws lambda invoke --function-name tickets --cli-binary-format raw-in-base64-out \
    --payload '{"message": "I was charged twice, fix this now"}' /dev/stdout
```

## 6b. Deploy with CDK

The same function as a CDK stack (TypeScript), next to `build/`. CDK zips the directory, uploads it and
creates the role.

```bash
mkdir infra && cd infra
npm init -y && npm install aws-cdk-lib constructs && npm install -D aws-cdk tsx typescript
echo '{"app": "npx tsx app.ts"}' > cdk.json
```

`infra/app.ts`:

```typescript
import * as path from "node:path";
import { App, CfnOutput, Duration, RemovalPolicy, Stack } from "aws-cdk-lib";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as logs from "aws-cdk-lib/aws-logs";

const BUILD = path.join(__dirname, "..", "build");
const MEMORY_MB = 1769; // one full vCPU

const app = new App();
const stack = new Stack(app, "JulTickets", {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
});

const fn = new lambda.Function(stack, "Tickets", {
  runtime: lambda.Runtime.PYTHON_3_12,
  architecture: lambda.Architecture.ARM_64,
  handler: "handler.handler",
  code: lambda.Code.fromAsset(BUILD),
  memorySize: MEMORY_MB,
  timeout: Duration.seconds(30),
  environment: { JUL_ONNX_THREADS: String(Math.max(1, Math.floor(MEMORY_MB / 1769))) },
  logGroup: new logs.LogGroup(stack, "TicketsLogs", {
    retention: logs.RetentionDays.ONE_WEEK,
    removalPolicy: RemovalPolicy.DESTROY,
  }),
});

new CfnOutput(stack, "FunctionName", { value: fn.functionName });
```

Two functions (one per model): build two directories (`build-choice/`, `build-yesno/`) from two bundles and
declare two `lambda.Function`s, one per directory.

```bash
npx cdk bootstrap          # once per account and region
npx cdk deploy
aws lambda invoke --function-name <FunctionName from the output> --cli-binary-format raw-in-base64-out \
    --payload '{"message": "I was charged twice, fix this now"}' /dev/stdout
```

## Sizing

1,769 MB is one full vCPU, the best price per call in the table above; `JUL_ONNX_THREADS` gives ONNX
Runtime one thread per whole vCPU. Batching several messages per event (`"messages": [...]`) lowers the
price further. With the model in the package, a cold start is ~2.4 s; SnapStart brings nothing at that size.

## Pitfalls

- **Tune on your own messages.** Heads trained on another domain transfer badly: emotion heads trained on
  tweets read a fraud report as "joy" or "anger". A few hundred labeled messages from the real inbox
  beat thousands from elsewhere.
- **Pack on the backend you deploy on** (onnx): the option vectors and heads are computed with it.
- **Long texts are cut** to the model's 512 positions (the end of the message goes first); the cross
  model reads 256 tokens, question included.
