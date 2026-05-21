CUDA_VISIBLE_DEVICES="4" python iterSCAV.py --judge srf --maxIter 20 --trainL 256 --layer -2 --model 'skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal' --saveDir ./iterSCAVWeight --thres 0.05 0.6 --gpuLR --evalPT "1" --pt "abs0" --embType response --val
CUDA_VISIBLE_DEVICES="5" python iterSCAV.py --filter --judge srf --maxIter 20 --trainL 256 --layer -2 --model 'skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal' --saveDir ./iterSCAVWeight --thres 0.05 0.6 --gpuLR --evalPT "1" --pt "abs0" --embType response --val
CUDA_VISIBLE_DEVICES="5" python iterSCAV.py --judge srf --maxIter 20 --trainL 256 --layer -2 --model 'GraySwanAI/Mistral-7B-Instruct-RR' --saveDir ./iterSCAVWeight --thres 0.05 0.6 --gpuLR --evalPT "1" --pt "abs0" --embType last --val
CUDA_VISIBLE_DEVICES="6,7" python iterSCAV.py --judge srf --maxIter 20 --trainL 2048 --layer -2 --model 'openai/gpt-oss-20b' --saveDir ./iterSCAVWeight --thres 0.05 0.6 --gpuLR --evalPT "1" --pt "abs0" --embType response --val


