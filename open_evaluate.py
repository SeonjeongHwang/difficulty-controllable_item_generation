import sys, argparse, tqdm, json
sys.path.append("/home/seonjeongh/DCAQG/box/dcqg/difficulty_eval")
from evaluation import Propositionalizer, eval, get_values

args = None
NICKNAME2NAME = {"llama-8B": "meta-llama/Llama-3.1-8B-Instruct",
                 "qwen-32B": "Qwen/Qwen2.5-32B-Instruct",
                 "qwen-7B": "Qwen/Qwen2.5-7B-Instruct"}

def parse_args():
    ## python open_evaluate.py --model_nickname qwen-8B --task end2end_SP
    
    parser = argparse.ArgumentParser(description="QG test")
    parser.add_argument('--data_file', type=str, default="/home/seonjeongh/DCAQG/box/data/difficulty_attribute/ReCo.dcqg.test.json")
    parser.add_argument('--model_nickname', type=str, required=True, help='model nickname')
    parser.add_argument('--task', type=str)
    args = parser.parse_args()
    return args

def main():
    global args
    args = parse_args()
    evaluator = Evaluator()
    id2prediction = json.load(open(f"outputs/{args.model_nickname}-{args.task}/predictions.json"))
    
    passages, statements = [], []
    for id, pred_dict in tqdm.tqdm(id2prediction.items(), total=len(id2prediction.keys())):
        passages.append(pred_dict["passage"])
        statements.append(pred_dict["statement"])
        
    predictions = evaluator.get_values(passages, statements)

    id2cal_prediction = dict()
    for id, prediction in tqdm.tqdm(zip(id2prediction.keys(), predictions), total=len(id2prediction.keys())):
        id2cal_prediction[id] = prediction
        
    with open(f"outputs/{args.model_nickname}-{args.task}/prediction_values.json", "w") as fout:
        json.dump(id2cal_prediction, fout, indent=3)
        
    scores = eval(args.data_file, id2cal_prediction)
    with open(f"outputs/{args.model_nickname}-{args.task}/performances.json", "w") as fout:
        json.dump(scores, fout, indent=3)
        
if __name__ == "__main__":
    main()