# harm fortified
CUDA_VISIBLE_DEVICES="0" python eval.py --evalData harm --maxL 512 --model 'GraySwanAI/Mistral-7B-Instruct-RR' --evalPT "1" --clfP "path_to_probes" --csvP myRes.csv --evalClfr 'best' --judge "srf" "hb" "qwen-plus https://dashscope.aliyuncs.com/compatible-mode/v1 sk-xxxxxxx"
