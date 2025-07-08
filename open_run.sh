


MODEL=qwen-32B
TASK=end2end_SP

echo $MODEL
echo $TASK

conda activate vllm
python open_inference.py --model_nickname $MODEL --generation_mode vllm --task $TASK
conda deactivate

conda activate hug4.29
python open_evaluate.py --model_nickname $MODEL --task $TASK


