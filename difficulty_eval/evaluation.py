import spacy
import nltk
import json
import os
import tqdm
import numpy as np
import evaluate
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

nlp = spacy.load("en_core_web_sm")

dictionary = json.load(open("/home/seonjeongh/DCAQG/box/data/difficulty_attribute/vocab_level/Oxford/Dictionary.json"))
level2num = {"A": 1, "B": 2, "C": 3}
CEFR2threelevels = {"A1": "A", "A2": "A", "B1": "B", "B2": "B", "C1": "C"}
stop_words = nlp.Defaults.stop_words

start_marker = '<s>'
end_marker = '</s>'
separator = '\n'

class Propositionalizer:
    def __init__(self, gpus=None):
        self.tokenizer = AutoTokenizer.from_pretrained('google/gemma-7b-aps-it')
        self.model = AutoModelForCausalLM.from_pretrained('google/gemma-7b-aps-it').to("cuda:2") #, torch_dtype=torch.float16, device_map="auto")
        #self.model = torch.nn.DataParallel(self.model)
        
    def get_propositions(self, sents):
        
        def create_propositions_input(sent: str) -> str:
            input_sents = [sent]
            propositions_input = ''
            for sent in input_sents:
                propositions_input += f'{start_marker} ' + sent + f' {end_marker}{separator}'
            propositions_input = propositions_input.strip(f'{separator}')
            return propositions_input

        def process_propositions_output(text):
            pattern = re.compile(f'{re.escape(start_marker)}(.*?){re.escape(end_marker)}', re.DOTALL)
            output_grouped_strs = re.findall(pattern, text)
            predicted_grouped_propositions = []
            for grouped_str in output_grouped_strs:
                grouped_str = grouped_str.strip(separator)
                props = [x[2:] for x in grouped_str.split(separator)]
                predicted_grouped_propositions.append(props)
            return predicted_grouped_propositions

        examples = []
        for sent in sents:
            messages = [
                {'role': 'user', 'content': create_propositions_input(sent)}
            ]
            example = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            examples.append(example)
            
        results = []
        batch_size = 8
        batch_offsets = list(range(0,len(examples), batch_size))
        for batch_start_offset in batch_offsets:
            mini_batch = examples[batch_start_offset:batch_start_offset+batch_size]
            
            inputs = self.tokenizer(mini_batch, padding=True, truncation=False, return_tensors='pt')
            inputs = {k: v.to("cuda:2") for k, v in inputs.items()}
            input_length = inputs["input_ids"].shape[-1]
            
            with torch.no_grad():
                res_ = self.model.generate(  ## module
                    **inputs,
                    max_new_tokens=100,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                    num_return_sequences=1,
                    use_cache=True
                )
                
            outputs = self.tokenizer.batch_decode(res_[:,input_length:], skip_special_tokens=True) 
        
            for output in outputs:
                result = process_propositions_output(output)
                if len(result) == 0:
                    results.append(["ERROR"])
                else:
                    results.append(result[0])
        return results
    
def get_values(passage, statement):
    prediction =  {"passage_length": None,
                    "sentence_length": None,
                    "passage_vocab_level": None,
                    "statement_vocab_level": None,
                    "vocab_level": None}
    
    detail = {"passage_length": None,
              "sentence_length_list": [],
              "passage_vocab_levels": [],
              "statement_vocab_levels": []}        
    
    if passage != "":
        ## Passage Length
        sents = nltk.sent_tokenize(passage)
        passage_length = len(sents)
        
        if 6 <= passage_length <= 15:
            passage_length_level = "short"
        elif 15 < passage_length <= 20:
            passage_length_level = "medium"
        elif passage_length <= 30:
            passage_length_level = "long"
        else:
            passage_length_level = "out_of_range"
            
        prediction["passage_length"] = passage_length_level
        detail["passage_length"] = passage_length
        
        ## Sentence Length
        sentence_length_list = []
        passage_words = []
        for sent in sents:
            doc = nlp(sent)
            sentence_length_list.append(len([token.text for token in doc if token.pos_ not in ["PUNCT", "SYM"]]))
            
            for token in doc:
                if token.pos_ in ["PROPN", 'PUNCT', "SYM", "NUM"]:
                    continue
                if token.lemma_.lower() in stop_words:
                    continue
                passage_words.append((token.lemma_.lower(), token.pos_, token.text))
        sentence_length = np.mean(sentence_length_list)
        
        if sentence_length <= 15:
            sentence_length_level = "short"
        elif 15 < sentence_length <= 20:
            sentence_length_level = "medium"
        else:
            sentence_length_level = "long"
            
        prediction["sentence_length"] = sentence_length_level
        detail["sentence_length_list"] = sentence_length_list
            
        ## Vocab Level
        levels = []
        passage_vocab_levels = []
        for lemma, pos, word in passage_words:
            level = None
            if lemma in dictionary:
                if len(dictionary[lemma].keys()) == 1:
                    level = CEFR2threelevels[list(dictionary[lemma].values())[0]]
                else:
                    if pos in dictionary[lemma]:
                        level = CEFR2threelevels[dictionary[lemma][pos]]
                    else:
                        doc = nlp(lemma)
                        pos = [token.pos_ for token in doc][0]
                        if pos in dictionary[lemma]:
                            level = CEFR2threelevels[dictionary[lemma][pos]]
                        else:
                            continue
            if level:
                levels.append(level)
                passage_vocab_levels.append((word, pos, level))
                    
        passage_vocab_levels = list(set(passage_vocab_levels)) 
        if len(levels) == 0:
            passage_vocab_max_level = "C"
        else:
            passage_vocab_max_level = sorted(list(set(levels)), reverse=True)[0]
        
        prediction["passage_vocab_level"] = passage_vocab_max_level
        detail["passage_vocab_levels"] = passage_vocab_levels                
        
    if statement != "":
        ## Vocab Level
        statement_words = []
        doc = nlp(statement)
        for token in doc:
            if token.pos_ in ["PROPN", 'PUNCT', "SYM", "NUM"]:
                continue
            if token.lemma_.lower() in stop_words:
                continue
            statement_words.append((token.lemma_.lower(), token.pos_, token.text))
            
        levels = []
        statement_vocab_levels = []
        for lemma, pos, word in statement_words:
            level = None
            if lemma in dictionary:
                if len(dictionary[lemma].keys()) == 1:
                    level = CEFR2threelevels[list(dictionary[lemma].values())[0]]
                else:
                    if pos in dictionary[lemma]:
                        level = CEFR2threelevels[dictionary[lemma][pos]]
                    else:
                        doc = nlp(lemma)
                        pos = [token.pos_ for token in doc][0]
                        if pos in dictionary[lemma]:
                            level = CEFR2threelevels[dictionary[lemma][pos]]
                        else:
                            continue
            if level:
                levels.append(level)
                statement_vocab_levels.append((word, pos, level))
                        
        statement_vocab_levels = list(set(statement_vocab_levels))
        if len(levels) == 0:
            statement_vocab_max_level = "C"
        else:
            statement_vocab_max_level = sorted(list(set(levels)), reverse=True)[0]
            
        prediction["statement_vocab_level"] = statement_vocab_max_level
        detail["statement_vocab_levels"] = statement_vocab_levels              
            
    max_vocabs = list(set([prediction["passage_vocab_level"], prediction["statement_vocab_level"]])-set([None]))
    if len(max_vocabs) == 0:
        prediction["vocab_level"] = None
    else:
        prediction["vocab_level"] = sorted(max_vocabs)[-1]
        
    return prediction, detail

def observe(constraints, detail, is_full_constraints=False):
    finish = True
    output = ""
    
    constraints_satisfactions = dict()
    for k in constraints.keys():
        constraints_satisfactions[k] = True
    
    ### Passage Length
    if detail["passage_length"] != None:
        passage_length_satisfaction = "unsatisfied"
        if constraints["passage_length"] == "short" and 6 <= detail["passage_length"] <= 15:
            passage_length_satisfaction = "satisfied"
        elif constraints["passage_length"] == "medium" and 16 <= detail["passage_length"] <= 20:
            passage_length_satisfaction = "satisfied"
        elif constraints["passage_length"] == "long" and 21 <= detail["passage_length"] <= 30:
            passage_length_satisfaction = "satisfied"
        
        output += f'- The number of sentences in the passage: {detail["passage_length"]} -> ({passage_length_satisfaction})\n'
        if passage_length_satisfaction == "unsatisfied":
            finish = False
            constraints_satisfactions["passage_length"] = False
    else:
        output += '- The number of sentences in the passage: -\n'
        finish = False
        constraints_satisfactions["passage_length"] = False
    
    ### Sentence Length
    if detail["sentence_length_list"] != None:
        output += "- The number of words per sentence in the passage:\n"
        for i, num in enumerate(detail["sentence_length_list"]):
            sentence_length_satisfaction = "unsatisfied"
            if constraints["sentence_length"] == "short" and num <= 15:
                sentence_length_satisfaction = "satisfied"
            elif constraints["sentence_length"] == "medium" and 15 < num <= 20:
                sentence_length_satisfaction = "satisfied"
            elif constraints["sentence_length"] == "long" and 20 <= num:
                sentence_length_satisfaction = "satisfied"
            output += f'sentence ({i+1}) - {num} words -> ({sentence_length_satisfaction})\n'
            if sentence_length_satisfaction == "unsatisfied":
                finish = False
                constraints_satisfactions["sentence_length"] = False
        
    else:
        output += "- The number of words per sentence in the passage: -\n"
        finish = False
        constraints_satisfactions["sentence_length"] = False
        
    ### Passage Vocab Levels
    if len(detail["passage_vocab_levels"]) > 0:
        levels = {"A": [], "B": [], "C": []}
        for w, p, l in detail["passage_vocab_levels"]:
            levels[l].append(w)
            
        passage_vocab_satisfaction = "satisfied"
        output += "- Vocabulary Levels used in the passage:\n"
        output += f"A1-A2: {levels['A']}\n"
        if constraints["vocab_level"] == "A" and len(levels["A"]) == 0:
            passage_vocab_satisfaction = "unsatisfied"
            
        output += f"B1-B2: {levels['B']}\n"
        if constraints["vocab_level"] == "B" and len(levels["B"]) == 0:
            passage_vocab_satisfaction = "unsatisfied"
        elif constraints["vocab_level"] == "A" and len(levels["B"]) > 0:
            passage_vocab_satisfaction = "unsatisfied"
        
        output += f"C1-C2: {levels['C']}\n"
        if constraints["vocab_level"] == "C" and len(levels["C"]) == 0:
            passage_vocab_satisfaction = "unsatisfied"
        elif constraints["vocab_level"] in ["A", "B"] and len(levels["C"]) > 0:
            passage_vocab_satisfaction = "unsatisfied"
        output += f"-> ({passage_vocab_satisfaction})\n"
        if passage_vocab_satisfaction == "unsatisfied":
            finish = False
            constraints_satisfactions["vocab_level"] = False
        
    else:
        output += "- Vocabulary Levels used in the passage: -\n"
        finish = False
        constraints_satisfactions["vocab_level"] = False
        
    ### Sentence Vocab Levels
    if len(detail["statement_vocab_levels"]) > 0:
        levels = {"A": [], "B": [], "C": []}
        for w, p, l in detail["statement_vocab_levels"]:
            levels[l].append(w)
            
        statement_vocab_satisfaction = "satisfied"
        output += "- Vocabulary Levels used in the statement:\n"
        output += f"A1-A2: {levels['A']}\n"
        if constraints["vocab_level"] == "A" and len(levels["A"]) == 0:
            statement_vocab_satisfaction = "unsatisfied"
            
        output += f"B1-B2: {levels['B']}\n"
        if constraints["vocab_level"] == "B" and len(levels["B"]) == 0:
            statement_vocab_satisfaction = "unsatisfied"
        elif constraints["vocab_level"] == "A" and len(levels["B"]) > 0:
            statement_vocab_satisfaction = "unsatisfied"
        
        output += f"C1-C2: {levels['C']}\n"
        if constraints["vocab_level"] == "C" and len(levels["C"]) == 0:
            statement_vocab_satisfaction = "unsatisfied"
        elif constraints["vocab_level"] in ["A", "B"] and len(levels["C"]) > 0:
            statement_vocab_satisfaction = "unsatisfied"
        output += f"-> ({statement_vocab_satisfaction})\n"
        if statement_vocab_satisfaction == "unsatisfied":
            finish = False
            constraints_satisfactions["vocab_level"] = False
        
    else:
        output += "- Vocabulary Levels used in the statement: -\n"
        finish = False
        constraints_satisfactions["vocab_level"] = False
        
    ### Statement Propositions
    if detail["statement_propositions"] != None:
        proposition_satisfaction = "satisfied"
        if constraints["statement_propositions"] != len(detail["statement_propositions"]):
            proposition_satisfaction = "unsatisfied"
        output += f'- Statement Propositions: {detail["statement_propositions"]} -> ({proposition_satisfaction})\n'
        if proposition_satisfaction == "unsatisfied":
            finish = False
            constraints_satisfactions["statement_propositions"] = False
                
    else:
        output += "- Statement Propositions: -\n"
        finish = False
        constraints_satisfactions["statement_propositions"] = False
        
    if is_full_constraints:
        ### Evidence Scope
        if detail["evidence_scope"] != None:
            es_satisfaction = "satisfied"
            if constraints["evidence_scope"] != detail["evidence_scope"]:
                es_satisfaction = "unsatisfied"
            output += f'- Evidence Scope: {detail["evidence_scope"]} -> ({es_satisfaction})\n'
            if es_satisfaction == "unsatisfied":
                finish = False
                constraints_satisfactions["evidence_scope"] = False
        
        else:
            output += "- Evidence Scope: -\n" 
            finish = False
            constraints_satisfactions["evidence_scope"] = False
        
        ### Reasoning Type
        if detail["reasoning_type"] != None:
            rt_satisfaction = "satisfied"
            if constraints["reasoning_type"] != detail["reasoning_type"]:
                rt_satisfaction = "unsatisfied"
            output += f'- Reasoning Type: {detail["reasoning_type"]} -> ({rt_satisfaction})\n'
            if rt_satisfaction == "unsatisfied":
                finish = False
                constraints_satisfactions["reasoning_type"] = False
        
        else:
            output += "- Reasoning Type: -\n"
            finish = False
            constraints_satisfactions["reasoning_type"] = False
        
    return output, finish, constraints_satisfactions
    
def eval(data_file, id2prediction):
    data_list = json.load(open(data_file))
    id2reference = dict()
    for data in data_list:
        id = data["id"]
        
        if data["reasoning_complexity"] == "NEI":
            reasoning_type = "Not Enough Information"
            evidence_scope = "Insufficient"
        else:
            reasoning_type = data["reasoning_complexity"].split("_")[1]
            evidence_scope = "Single" if "single" in data["reasoning_complexity"] else "Inter"
        
        gold = {"passage_length": data["passage_length"],
                "sentence_length": data["sentence_length"],
                "vocab_level": data["vocab_level"],
                "statement_propositions": data["statement_propositions"],
                "reasoning_type": reasoning_type,
                "evidence_scope": evidence_scope}
        id2reference[id] = gold
        
    scores = {"passage_length": None,
              "sentence_length": None,
              "vocab_level": None,
              "statement_propositions": None,
              "reasoning_type": None,
              "evidence_scope": None}
    
    mapping = {"short": 0,
               "medium": 1,
               "long": 2,
               "A": 0,
               "B": 1,
               "C": 2,
               "Insufficient": 0,
               "Single": 1,
               "Inter": 2,
               "Not Enough Information": 0,
               "Word Matching": 1,
               "Transformed Word Matching": 2,
               "Paraphrasing": 3,
               "Transformed Paraphrasing": 4,
               "Inference": 5}
    
    for dimension in scores.keys():
        em_scores = []
        for id, pred_dict in id2prediction.items():
            if dimension not in pred_dict:
                break
            
            pred = pred_dict[dimension]
            gold = id2reference[id][dimension]
            
            if dimension != "statement_propositions":
                pred = mapping[pred]
                gold = mapping[gold]
                
            em_scores.append(int(pred == gold))
            
        if len(em_scores) > 0:
            scores[dimension] = round(np.mean(em_scores)*100,2) 
        
    print(json.dumps(scores, indent=3))
    return scores
                
if __name__ == "__main__":
    propositionalizer = Propositionalizer("1,2")
    
    constraints = {"passage_length": "medium",
                   "sentence_length": "medium",
                   "vocab_level": "B",
                   "statement_propositions": 2,
                   "evidence_scope": "Inter",
                   "reasoning_type": "Inference"}
    
    passage = """(1) In recent years, China’s food industry has faced growing public concern due to repeated food safety scandals.
(2) These scandals have led people to examine the honesty and quality of the food they regularly consume."""

    statement = ""
    
    passage = [re.sub(r'\(\d+\)', '', text).strip() for text in passage.split("\n")]
    passage = " ".join(passage)
    prediction, detail = get_values(passage, statement)
    
    if statement != "":
        props = propositionalizer.get_propositions([statement])[0]
        detail["statement_propositions"] = props
    else:
        detail["statement_propositions"] = None
        
    detail["evidence_scope"] = None
    detail["reasoning_type"] = None
    
    observation, finish, constraints_state = observe(constraints, detail)
    
    print(observation)
    print(constraints_state)
    print("Finish:", finish)
    