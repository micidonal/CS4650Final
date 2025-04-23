import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftConfig, PeftModel

torch.set_default_device('cuda')
config = PeftConfig.from_pretrained('VityaVitalich/TaxoLLaMA')
# Do not forget your token for Llama2 models
model = LlamaForCausalLM.from_pretrained(config.base_model_name_or_path, load_in_4bit=True, torch_dtype=torch.bfloat16)
tokenizer = LlamaTokenizer.from_pretrained(config.base_model_name_or_path)
inference_model = PeftModel.from_pretrained(model, 'VityaVitalich/TaxoLLaMA')

processed_term = "hyponym: tiger | hypernyms:"

system_prompt = """<s>[INST] <<SYS>> You are a helpfull assistant. List all the possible words divided with a coma. Your answer should not include anything except the words divided by a coma<</SYS>>"""
processed_term = system_prompt + '\n' + processed_term + '[/INST]'

input_ids = tokenizer(processed_term, return_tensors='pt')

gen_conf = {
            "no_repeat_ngram_size": 3,
            "do_sample": True,
            "num_beams": 8,
            "num_return_sequences": 2,
            "max_new_tokens": 32,
            "top_k": 20,
        }

out = inference_model.generate(inputs=input_ids['input_ids'].to('cuda'), **gen_conf)

text = tokenizer.batch_decode(out)[0][len(system_prompt):]
print(text)
