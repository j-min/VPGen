"""Wrapper around HuggingFace APIs."""
import torch

from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig

DEFAULT_REPO_ID = "gpt2"
VALID_TASKS = ("text2text-generation", "text-generation")


class LlamaHuggingFace:

    def __init__(self, 
                 base_model,
                 lora_model,
                 task='text-generation',
                 device='cpu',
                 max_new_tokens=512,
                 temperature=0.1,
                 top_p=0.75,
                 top_k=40,
                 num_beams=1):
        self.task = task
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.num_beams = num_beams
        self.tokenizer = LlamaTokenizer.from_pretrained(
            base_model, use_fast=False)
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16)
        self.model = PeftModel.from_pretrained(
            model,
            lora_model,
            torch_dtype=torch.float16)
        self.model.to(device)

        self.tokenizer.pad_token_id = 0
        self.model.config.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        if device == "cpu":
            self.model.float()
        else:
            self.model.half()
        self.model.eval()
    
    @torch.no_grad()
    def __call__(self, inputs, params=None):
        if inputs.endswith('Thought:'):
            inputs = inputs[:-len('Thought:')]
        inputs = inputs.replace('Observation:\n\nObservation:', 'Observation:')
        inputs = inputs + '### ASSISTANT:\n'
        input_ids = self.tokenizer(inputs, return_tensors="pt").to(self.device).input_ids

        generation_config = GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            num_beams=self.num_beams)

        generate_ids = self.model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            max_new_tokens=self.max_new_tokens)
        response = self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        
        # # Start from the ### ASSISTANT:
        # response = [res.split('### ASSISTANT:')[-1].strip() for res in response]

        print('raw response')
        print(response)

        response = [res.split('### ASSISTANT:')[-1].strip() for res in response]

        print('response')
        print(response)

        # Remove output that is the same as input
        # response = [res.replace(inputs, '') for res in response]
        # response = [res.replace('### ASSISTANT:\n', '').strip() for res in response]

        response = [{'generated_text': res} for res in response]
        return response