import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from .Model import Model


class Llama(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]

        api_pos = int(config["api_key_info"]["api_key_use"])
        hf_token = config["api_key_info"]["api_keys"][api_pos]

        # self.tokenizer = LlamaTokenizer.from_pretrained(self.name, token=hf_token)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, token=hf_token, trust_remote_code=True)
        # self.model = LlamaForCausalLM.from_pretrained(self.name, torch_dtype=torch.float16, use_auth_token=hf_token).to(
        #     self.device
        # )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name, torch_dtype=torch.float16, token=hf_token, trust_remote_code=True
        ).to(self.device)

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            model_kwargs={"torch_dtype": torch.float16},
        )

    def query(self, msg, return_json=False, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = self.max_output_tokens
        inputs = self.tokenizer(msg, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=self.temperature,
                max_new_tokens=max_new_tokens,
                # early_stopping=True
            )

        out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = out[len(msg) :]
        return result

    def pipeline_query(self, msg, return_json=False, max_new_tokens=None):
        messages = [
            {"role": "user", "content": msg},
        ]
        if max_new_tokens is None:
            max_new_tokens = self.max_output_tokens
        outputs = self.pipeline(
            messages,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.95,
            top_k=50,
        )
        out = outputs[0]["generated_text"]
        result = out[1]["content"]
        return result
