from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os, json, tqdm, random, time, re
from vllm import LLM
from vllm import SamplingParams
import numpy as np
import argparse
from collections import Counter

args = None
NICKNAME2NAME = {"llama-8B": "meta-llama/Llama-3.1-8B-Instruct",
                 "llama-70B": "meta-llama/Llama-3.1-70B-Instruct",
                 "qwen-32B": "Qwen/Qwen2.5-32B-Instruct",
                 "qwen-7B": "Qwen/Qwen2.5-7B-Instruct"}

def parse_args():
    ## python open_inference.py --model_nickname qwen-7B --generation_mode vllm --task end2end_SP
    
    parser = argparse.ArgumentParser(description="QG test")
    parser.add_argument('--data_file', type=str, default="/home/seonjeongh/DCAQG/box/data/difficulty_attribute/ReCo.dcqg.test.json")
    parser.add_argument('--model_nickname', type=str, required=True, help='model nickname')
    parser.add_argument('--generation_mode', type=str, default="naive")
    parser.add_argument('--task', type=str)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    return args

def set_random_seed(seed):
    torch.manual_seed(seed)  # CPU
    np.random.seed(seed)     # NumPy
    random.seed(seed)        # Python Random
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # 모든 GPU

class Generator:
    def __init__(self, model_name, mode="naive"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        self.top_k = 20         ##self.model.generation_config.top_k
        self.top_p = 0.8        ##self.model.generation_config.top_p
        self.temperature = 0.7  ##self.model.generation_config.temperature
            
        if mode == "naive":
            print("Naive Generation Mode")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
            self.model = torch.nn.DataParallel(self.model)
            
        elif mode == "vllm":
            print("VLLM Generation Mode")
            self.model = LLM(model=model_name,
                             dtype="auto",
                             trust_remote_code="True",
                             tensor_parallel_size=torch.cuda.device_count(),
                             max_model_len=3000,
                             disable_cascade_attn=True)
            
    def get_examples(self, data_file):
        data_list = json.load(open(data_file))
        prompt_template = open(f"prompts/{args.task}.txt", 'r', encoding="utf-8").read()
        
        id_list, input_prompts = [], []
        for data in tqdm.tqdm(data_list):
            id = data["id"]
            
            if data["reasoning_complexity"] == "NEI":
                reasoning_type = "Not Enough Information"
                evidence_scope = "Insufficient"
            else:
                reasoning_type = data["reasoning_complexity"].split("_")[1]
                evidence_scope = "Single" if "single" in data["reasoning_complexity"] else "Inter"
            
            prompt = prompt_template.replace("{ document }", data["document"])
            prompt = prompt.replace("{ passage_length }", data["passage_length"])
            prompt = prompt.replace("{ sentence_length }", data["sentence_length"])
            prompt = prompt.replace("{ vocab_level }", data["vocab_level"])
            prompt = prompt.replace("{ statement_propositions }", str(data["statement_propositions"]))
            prompt = prompt.replace("{ reasoning_type }", reasoning_type)
            prompt = prompt.replace("{ evidence_scope }", evidence_scope)
            
            id_list.append(id)
            input_prompts.append(prompt)
        
        examples = []
        for input_prompt in input_prompts:      
            messages = [
                {"role": "user", "content": input_prompt}
            ]
            example = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            examples.append(example)
            
        return id_list, examples
            
    def generate(self, id_list, examples, do_greedy=False):
        
        def extract_passage_and_statement(text):
            match = re.search(r'\{\s*"passage"\s*:\s*".+?",\s*"statement"\s*:\s*".+?"\s*\}', text, re.DOTALL)

            if match:
                json_str = match.group(0)
                try:
                    result = json.loads(json_str)
                    return result
                except json.JSONDecodeError:
                    return None
            else:
                return None
        
        if do_greedy:
            print("DO GREEDY")
        
        start = time.time()
        
        id2prediction = dict()
        id2response = dict()
        if args.generation_mode == "naive":
            
            max_new_tokens = 1000
            
            batch_offsets = list(range(0,len(examples), args.batch_size))
            for batch_start_offset in tqdm.tqdm(batch_offsets, total=len(batch_offsets)):
                mini_id_list = id_list[batch_start_offset:batch_start_offset+args.batch_size]
                mini_batch = examples[batch_start_offset:batch_start_offset+args.batch_size]
                
                inputs = self.tokenizer(mini_batch, padding=True, truncation=False, return_tensors='pt').to("cuda")
                input_length = inputs["input_ids"].shape[-1]
                
                if do_greedy:
                    res_ = self.model.module.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=False,
                        temperature=None,
                        top_k=None,
                        top_p=None,
                        num_return_sequences=1,
                        use_cache=True
                    )
                
                with torch.no_grad():
                    res_ = self.model.module.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        top_k=self.top_k,
                        top_p=self.top_p,
                        temperature=self.temperature,
                        num_return_sequences=1,
                        use_cache=True
                    )
                    
                res = self.tokenizer.batch_decode(res_[:,input_length:], skip_special_tokens=True)
                
                for id, r in zip(mini_id_list, res):
                    id2response[id] = r
                    output = extract_passage_and_statement(r)
                    if output is None:
                        id2prediction[id] = {"passage": "EMPTY",
                                             "statement": "EMPTY"}
                    else:
                        id2prediction[id] = {"passage": output["passage"],
                                             "statement": output["statement"]}
                
        elif args.generation_mode == "vllm":
            sampling_params = SamplingParams(seed=args.seed,
                                             top_p=self.top_p,
                                             top_k=self.top_k,
                                             temperature=self.temperature,
                                             max_tokens=3000)
                
            res = self.model.generate(examples, sampling_params)
            
            for id, r in zip(id_list, res):
                r = r.outputs[0].text
                id2response[id] = r
                output = extract_passage_and_statement(r)
                if output is None:
                    id2prediction[id] = {"passage": "EMPTY",
                                         "statement": "EMPTY"}
                else:
                    id2prediction[id] = {"passage": output["passage"],
                                         "statement": output["statement"]}                
            
        with open(f"outputs/{args.model_nickname}-{args.task}/predictions.json", "w") as fout:
            json.dump(id2prediction, fout, indent=3)
        with open(f"outputs/{args.model_nickname}-{args.task}/responses.json", "w") as fout:
            json.dump(id2response, fout, indent=3)
            
        return id2prediction
        
def main():
    global args
    args = parse_args()
    print(args)
    set_random_seed(args.seed)
    model_name = NICKNAME2NAME[args.model_nickname]
    print(f'{model_name} is used')
    
    os.makedirs(f"outputs/{args.model_nickname}-{args.task}", exist_ok=True)
        
    generator = Generator(model_name, mode=args.generation_mode)
    id_list, examples = generator.get_examples(args.data_file)
            
    now = time.localtime()
    print("current time:", time.strftime("%Y-%m-%d %H:%M:%S", now))
    
    generator.generate(id_list, examples)
        
if __name__ == "__main__":
    main()