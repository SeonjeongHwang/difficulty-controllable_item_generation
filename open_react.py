from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os, json, tqdm, random, time, re
from vllm import LLM
from vllm import SamplingParams
import numpy as np
import argparse
from collections import Counter

import sys
sys.path.append("/home/seonjeongh/DCAQG/box/dcqg/difficulty_eval")
from evaluation import Propositionalizer, get_values, observe

args = None
NICKNAME2NAME = {"llama-8B": "meta-llama/Llama-3.1-8B-Instruct",
                 "llama-70B": "meta-llama/Llama-3.1-70B-Instruct",
                 "qwen-32B": "Qwen/Qwen2.5-32B-Instruct",
                 "qwen-7B": "Qwen/Qwen2.5-7B-Instruct"}

TOP_K = {"mixtral": None,
         "qwen": 20,
         "llama": 20,
         "phi": None,
         "test": 20}
TOP_P = {"mixtral": None,
         "qwen": 0.8,
         "llama": 0.9,
         "phi": None,
         "test": 0.8}
TEMPERATURE = {"mixtral": 0.15,
               "qwen": 0.7,
               "llama": 0.6,
               "phi": 0.5,
               "test": 0.7}

C_TYPE2TEMPLATE = {"4c": "prompts/react_4c.txt",
                   "fullc": "prompts/react_fullc.txt"}
C_TYPE2DATASET = {"4c": "/home/seonjeongh/DCAQG/box/data/Brown.True.dcqg-4c.test_small.json",
                  "fullc": "/home/seonjeongh/DCAQG/box/data/Brown.True.dcqg.test.json"}

def parse_args():
    ## python open_react.py --model_nickname qwen-7B --c_type 4c
    
    parser = argparse.ArgumentParser(description="QG test")
    parser.add_argument('--model_nickname', type=str, required=True, help='model nickname')
    parser.add_argument('--c_type', choices=["4c", "fullc"], help="Constraint Type")
    parser.add_argument('--seed', type=int, default=2025)
    args = parser.parse_args()
    return args

def set_random_seed(seed):
    torch.manual_seed(seed)  # CPU
    np.random.seed(seed)     # NumPy
    random.seed(seed)        # Python Random
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # 모든 GPU

class Generator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        self.top_k = TOP_K[args.model_nickname.split("-")[0]]
        self.top_p = TOP_P[args.model_nickname.split("-")[0]]
        self.temperature = TEMPERATURE[args.model_nickname.split("-")[0]]
        
        self.propositionalizer = Propositionalizer()
            
        print("VLLM Generation Mode")
        self.model = LLM(model=model_name,
                            dtype="auto",
                            trust_remote_code="True",
                            device=["cuda:0", "cuda:1"],
                            tensor_parallel_size=torch.cuda.device_count()-2,
                            max_model_len=30000,
                            gpu_memory_utilization=0.5,
                            max_num_seqs=128,
                            disable_cascade_attn=True)
                
    def get_examples(self, data_file):
        data_list = json.load(open(data_file))
        prompt_template = open(C_TYPE2TEMPLATE[args.c_type], 'r', encoding="utf-8").read()
        
        id2examples = dict()
        for data in tqdm.tqdm(data_list):
            combinations = data["combinations"]
            np.random.shuffle(combinations)
            for i, combs in enumerate(combinations[:1]):
                if not combs["usable"]:
                    continue
                
                id = f'{data["id"]}_{i}'
                constraints = combs
                
                input_prompt = prompt_template.replace("{ source_text }", data["document"])
                input_prompt = input_prompt.replace("{ passage_length }", combs["passage_length"])
                input_prompt = input_prompt.replace("{ sentence_length }", combs["sentence_length"])
                input_prompt = input_prompt.replace("{ vocab_level }", combs["vocab_level"])
                input_prompt = input_prompt.replace("{ statement_propositions }", str(combs["statement_propositions"]))
                if args.c_type == "fullc":
                    input_prompt = input_prompt.replace("{ evidence_scope }", combs["evidence_scope"])
                    input_prompt = input_prompt.replace("{ reasoning_type }", combs["reasoning_type"])                    
                
                id2examples[id] = {"step": 1,
                                   "constraints": constraints,
                                   "input_prompt": input_prompt,
                                   "passage": [],
                                   "statement": "",
                                   "observations": dict(),
                                   "finish": False}
            
        return id2examples
            
    def generate(self, id2examples):
        def parsing_action(action_text):
            action_parsing_pattern = r'(\w+_\w+)\[\s*((?:.|\n)*?)\]'
            matches = re.findall(action_parsing_pattern, action_text)
            
            actions = []
            for action_name, action_result in matches:
                if action_name == "Write_Passage":
                    passage_parsing_pattern = r'\(\d+\)\s(.*?)\n'
                    sentences = re.findall(passage_parsing_pattern, action_result+"\n")
                    actions.append({"action": action_name,
                                    "passage": sentences})
                elif action_name in ["Insert_Sentence", "Revise_Sentence"]:
                    match = re.match(r'\((\d+)\)\s+(.*)', action_result)
                    number, sentence = int(match.group(1)), match.group(2)
                    actions.append({"action": action_name,
                                    "index": number,
                                    "sentence": sentence})
                elif action_name == "Remove_Sentence":
                    actions.append({"action": action_name,
                                    "index": int(action_result)})
                elif action_name in ["Write_Statement", "Revise_Statement"]:
                    actions.append({"action": action_name,
                                    "statement": action_result})
            return actions
        
        def get_thought_action(response, step):
            thought, action = None, None
            if response.count(f"Action {step}:") == 1:
                thought, action = response.split(f"Action {step}:")
            if thought is not None and f"Thought {step}:" in thought:
                thought = thought.split(f"Thought {step}:")[-1]
                
            if thought is None:
                return None, None
            
            return thought.strip(), action.strip()
        
        def step(actions, example, propositionalizer):
            for action in actions:
                if action["action"] == "Write_Passage":
                    example["passage"] = action["passage"]
                elif action["action"] == "Insert_Sentence":
                    example["passage"] = example["passage"][:action["index"]-1] + [action["sentence"]] + example["passage"][action["index"]-1:]
                elif action["action"] == "Revise_Sentence":
                    example["passage"][action["index"]-1] = action["sentence"]
                elif action["action"] == "Remove_Sentence":
                    example["passage"] = example["passage"][:action["index"]-1] + example["passage"][action["index"]:]
                elif action["action"] in ["Write_Statement", "Revise_Statement"]:
                    example["statement"] = action["statement"]
            
            _, detail = get_values("\n".join(example["passage"]), example["statement"])
            if example["statement"] != "":
                props = propositionalizer.get_propositions([example["statement"]])[0]
                detail["statement_propositions"] = props
            else:
                detail["statement_propositions"] = None              
            
            detail["evidence_scope"] = None
            detail["reasoning_type"] = None
            
            observation, finish = observe(example["constraints"], detail, args.c_type == "fullc")
            example["observations"][example["step"]] = observation
            example["finish"] = finish
            
            return observation, finish, example
        
        print(f"Top-k: {self.top_k} | Top-p: {self.top_p} | Temperature: {self.temperature}")
                    
        for _ in range(20):
            print(f"### Round {_+1} ###")
            id_list, examples = [], []
            for id, example in id2examples.items():
                if not example["finish"]:
                    id_list.append(id)
                    messages = [
                        {"role": "user", "content": example["input_prompt"] + f"\nThought {id2examples[id]['step']}: "}
                    ]
                    chat_example = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    examples.append(chat_example)
                    
            sampling_params = SamplingParams(seed=args.seed,
                                    top_k=self.top_k,
                                    top_p=self.top_p,
                                    temperature=self.temperature,
                                    stop=f"Observation {id2examples[id]['step']}:",
                                    max_tokens=30000,
                                    n=3)              

            if len(examples) == 0:
                print("All Done")
                break

            res = self.model.generate(examples, sampling_params)
            for id, r_ in zip(id_list, res):
                for output in r_.outputs:
                    r = output.text
                    
                    print(r)
                    print("-"*30)
                    
                    thought, action_sequence = get_thought_action(r, id2examples[id]["step"])
                    
                    if thought == None:
                        continue
                    actions = parsing_action(action_sequence)
                    if len(actions) == 0:
                        continue
                    
                    observation, finish, id2examples[id] = step(actions, id2examples[id], self.propositionalizer)
                    
                    print(observation)
                    print("="*30)
                    
                    current_step = id2examples[id]["step"]
                    new_prompt = ""
                    new_prompt += f"\nThought {current_step}: {thought}\n"
                    new_prompt += f"\nAction {current_step}: {action_sequence}\n"
                    new_prompt += f"\nObservation {current_step}: {observation}\n"
                    id2examples[id]["input_prompt"] += new_prompt
                    
                    if finish:
                        id2examples[id]["finish"] = True
                        break                    
                    
                    id2examples[id]["step"] += 1
                    
                    break
                
        print("Not Finish:", len(examples))
            
        return id2examples
        
def main():
    global args
    args = parse_args()
    print(args)
    set_random_seed(args.seed)
    model_name = NICKNAME2NAME[args.model_nickname]
    print(f'{model_name} is used')
    
    os.makedirs(f"outputs/{args.model_nickname}-{args.c_type}", exist_ok=True)
        
    generator = Generator(model_name)
    id2examples = generator.get_examples(C_TYPE2DATASET[args.c_type])
            
    now = time.localtime()
    print("current time:", time.strftime("%Y-%m-%d %H:%M:%S", now))
    
    id2examples = generator.generate(id2examples)
    
    simple_results, detail_results = [], []
    whether_success, iteration_num = [], []
    for id, examples in id2examples.items():
        simple_result = {"id": id,
                         "constraint": examples["constraints"],
                         "passage": examples["passage"],
                         "statement": examples["statement"],
                         "final_observation": examples["observations"][examples["step"]-1],
                         "final_step": examples["step"],
                         "is_success": examples["finish"]}
        detail_result = {"id": id,
                         "constraint": examples["constraints"],
                         "passage": examples["passage"],
                         "statement": examples["statement"],
                         "input_prompt": examples["input_prompt"],
                         "observations": examples["observations"],
                         "final_step": examples["step"],
                         "is_success": examples["finish"]}
        
        whether_success.append(examples["finish"])
        if examples["finish"]:
            iteration_num.append(examples["step"])
        
        simple_results.append(simple_result)
        detail_results.append(detail_result)
        
    success_rate = round(whether_success.count(True)/len(whether_success)*100, 2)
    avg_iteration = round(np.mean(iteration_num), 2)
    
    with open(f"outputs/{args.model_nickname}-{args.c_type}/score.json", "w") as fout:
        json.dump({"success_rate": success_rate, "avg_iteration": avg_iteration}, fout, indent=3)    
    with open(f"outputs/{args.model_nickname}-{args.c_type}/simple.json", "w") as fout:
        json.dump(simple_results, fout, indent=3)
    with open(f"outputs/{args.model_nickname}-{args.c_type}/detail.json", "w") as fout:
        json.dump(detail_results, fout, indent=3)
        
if __name__ == "__main__":
    main()