# Code of the method proposed in "Adaptive Probe-based Steering for Robust LLM Jailbreaking"

This is a repository containing our method only. The probe-based steering is implemented by inserting MLP adapters (see ```ProbeManager.py```) rather than hooks, which we believe is more elegant.

## Generating Probes

You can generate probes by running ```iterSCAV.py```. For example, run:
```commandline
CUDA_VISIBLE_DEVICES="0" python iterSCAV.py --judge srf --maxIter 20 --trainL 256 --layer -2 --model 'GraySwanAI/Mistral-7B-Instruct-RR' --saveDir ./iterSCAVWeight --thres 0.05 0.6 --gpuLR --evalPT "1" --pt "abs0" --embType last --val
```
Below are the arguments of ```iterSCAV.py```:

| Argument | Type | Required           | Default | Choices | Description |
|--------|------|--------------------|---------|---------|-------------|
| `--model` | `str` | ✅ Yes              | — | — | The repository name on Hugging Face. |
| `--tokenizer` | `str` | ❌ No               | — | — | Some repositories do not provide a tokenizer; specify the repository containing the tokenizer here. |
| `--pt` | `str` | ✅ Yes              | — | — | If using percentile-based target logits, provide a float between `0` and `1` (e.g., `"1.0"`). To specify exact target logits, use `"abs[float]"` (e.g., `"abs0"`). |
| `--evalPT` | `str` | ✅ Yes if val==True | — | — | If using percentile-based target logits, provide a float between `0` and `1` (e.g., `"1.0"`). To specify exact target logits, use `"abs[float]"` (e.g., `"abs0"`). |
| `--thres` | `float` (multi) | ✅ Yes               | — | — | Lower and upper thresholds for the annotator (e.g., `0.05 0.6`). |
| `--maxIter` | `int` | ✅ Yes               | — | — | Maximum number of iterations for adaptive retraining. |
| `--trainL` | `int` | ✅ Yes               | — | — | Maximum number of new tokens allowed during adaptive retraining. |
| `--embType` | `str` | ✅ Yes               | — | `last`, `response`, `all`, `prompt` | Type of embedding to use. |
| `--saveDir` | `str` | ✅ Yes               | — | — | Root directory for storing trained probes. |
| `--judge` | `str` | ✅ Yes               | — | — | Annotator model used for evaluation or labeling. |
| `--layer` | `int` (multi) | ❌ No               | `[-2]` | — | Indices of transformer layers to use. Accepts one or two values defining an interval. |
| `--gpuLR` | flag | ❌ No               | `False` | — | Whether to use cuML to accelerate probe training (may fail with LBFGS). |
| `--val` | flag | ❌ No               | `False` | — | Whether to perform validation during adaptive retraining. |
| `--normReg` | flag | ❌ No               | `False` | — | Whether to dynamically adjust regularization strength based on input norms. |
| `--filter` | flag | ❌ No               | `False` | — | Whether to use StrongReject’s fine-tuned judge to filter benign prompts that the model refuses. |

## Evaluating Probe-based Steering

After training the probes, you can evaluate the probe-based steering by running ```eval.py```. For example, run
```commandline
CUDA_VISIBLE_DEVICES="0" python eval.py --evalData harm --maxL 512 --model 'GraySwanAI/Mistral-7B-Instruct-RR' --evalPT "1" --clfP "path_to_probes" --csvP myRes.csv --evalClfr 'best' --judge "srf" "hb" "qwen-plus https://dashscope.aliyuncs.com/compatible-mode/v1 sk-xxxxxxx"
```

Below are the arguments of ```eval.py```:

| Argument | Type | Required                   | Default | Choices | Description |
|:--------|:-----|:---------------------------|:--------|:--------|:------------|
| `--model` | `str` | ✅ Yes                      | — | — | The repository name on Hugging Face. |
| `--tokenizer` | `str` | ❌ No                       | — | — | Some repositories do not provide a tokenizer; specify the repository containing the tokenizer here. |
| `--evalPT` | `str` | ✅ Yes if clfP is specified | — | — | If using percentile-based target logits, provide a float in `[0, 1]` (e.g., `"1.0"`). To specify exact target logits, use `"abs[float]"` (e.g., `"abs0"`). |
| `--csvP` | `str` | ✅ Yes                       | — | — | Path to store evaluation results. |
| `--clfP` | `str` | ❌ No                       | `None` | — | Path to trained probe classifiers. |
| `--evalClfr` | `str` | ✅ Yes if clfP is specified                       | — | `all`, `first`, `last`, `best` | Which probe classifier to use during evaluation (`best` is recommended). |
| `--maxL` | `int` | ✅ Yes                      | — | — | Maximum number of new tokens allowed during generation. |
| `--doSample` | flag | ❌ No                       | `False` | — | Whether to enable sampling during generation. |
| `--answerOnly` | flag | ❌ No                       | `False` | — | Whether to discard Chain-of-Thought (CoT) outputs and keep only final answers. |
| `--judge` | `str` (multi) | ✅ Yes                       | — | — | Judge models used for evaluation. Multiple values can be provided. |
| `--evalData` | `str` | ✅ Yes                       | — | `sr`, `harmbench`, `harm` | Benchmark dataset to evaluate on. |

## What to Improve

Which component is important for a model extraction (or active learning) algorithm?

First, the judge to be approximated should be accurate. Without an accurate judge, how can the probe, which  be accurate? You can add some new powerful judges into ```myJudge.py```. Of course, you can also utilize commercial big LLMs if you are ok with their randomness induced by the lack of batch invariance. We use SRF, which is finetuned from an old gemma-2b, by default because it is deterministic and save token charge.

Second, you may develop some better and adaptive steering strength schemes for sampling hidden states for the next iteration. We set strength="abs0", which steers the hidden state to the boundry, because classic model extraction (or active learning) algorithms do so. We have proven that increasing the strength to "0.5" can slightly improve our method in Table 7. So, don't let the default setting limits you (Yet, it is ok to use the default setting if you want to beat our method in your new paper).

Third, do think which part of the hidden state can best reflect the LLM's text output. We use the hidden state located at the first response token position by default because baselines do so and because we do not want to bargain with reviewers on the fairness of comparison. We have also proven that using the averaged hidden states from all response token positions can improve our method against some LLMs in Table 7. So, if you do not give a shit about the troubling peer review, do explore the alignment between the hidden states and the text output.
