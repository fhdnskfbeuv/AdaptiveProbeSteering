# Code of the method proposed in "Adaptive Probe-based Steering for Robust LLM Jailbreaking"

This is a repository containing our method only.

## Pretrained Probes at HF

We provide some probes at [HuggingFace][https://huggingface.co/collections/FTK11558/aps-jailbreak]. You can directly load them with ```from_pretrained```. You can also load them with vllm. 

## Generating Probes

You can generate probes by running ```iterSCAV.py```. For example, run:
```commandline
CUDA_VISIBLE_DEVICES="0" python iterSCAV.py --judge srf --bs 50 --maxIter 50 --trainL 512 --layer -2 --model 'GraySwanAI/Mistral-7B-Instruct-RR' --saveDir ./iterSCAVWeight --thres 0.05 0.8 --linearC cuSVC --evalPT "1" --pt "mean" --embType last --train "rd" --val "rdVal"
```
## Evaluating Probe-based Steering

After training the probes, you can evaluate the probe-based steering by running ```eval.py```. For example, run
```commandline
CUDA_VISIBLE_DEVICES="7" python eval.py --evalData sr --bs 25 --maxL 512 --model 'GraySwanAI/Mistral-7B-Instruct-RR' --evalPT "1" --clfP "path_to_probe --csvP myRes.csv --evalClfr 'best' --judge "srf" "hb" "qwen-plus https://dashscope.aliyuncs.com/compatible-mode/v1 sk-xxxxx"
```

## What to Improve

Which component is important for a model extraction (or active learning) algorithm?

First, the judge should be accurate. Without an accurate judge, how can the probe, which approximates the judge, be accurate? You can add some new powerful judges into ```myJudge.py```. Of course, you can also utilize commercial big LLMs if you are ok with their randomness induced by the lack of batch invariance. We use SRF, which is finetuned from an old gemma-2b, by default because it is deterministic and saves token charge.

Second, you may develop some better and adaptive steering strength schemes for sampling hidden states for the next iteration. We set strength="abs0", which steers the hidden state to the boundary, because classic model extraction (or active learning) algorithms do so. We have proven that increasing the strength to "50%" can slightly improve our method, as shown in Table 7. So, don't let the default setting limit you (Yet, it is ok to use the default setting if you want to beat our method in your new paper).

Third, explore which part of the hidden state can best reflect the LLM's text output. We use the hidden state located at the first response token position by default because baselines do so and because we do not want to bargain with reviewers about the fairness of comparison. We have also proven that using the averaged hidden states from all response token positions can improve our method against some LLMs in Table 7. So, if you do not give a shit about the troubling peer review, do explore the alignment between the hidden states and the text output.

Fourth, why not use more prompts? We include 50 harmful prompts during the model extraction because we have limited computation resources and have to benchmark the experiment, again, to deal with the peer review. If you get plenty of daily inference that can be annotated, you may store the hidden states to train the probe iteratively. I don't believe there will be some awful reviewers jumping out of nowhere and arguing that this scheme is not fair.

	

