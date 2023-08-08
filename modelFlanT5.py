import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig


class SummaryModel:
    def __init__(self):
        peft_model_id = "config/T5LSum"
        config = PeftConfig.from_pretrained("config/T5LSum")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.base_model_name_or_path, torch_dtype="auto", device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        self.model = PeftModel.from_pretrained(model, peft_model_id)

    def generate(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = inputs.to("cuda")
        outputs = self.model.generate(input_ids=inputs["input_ids"])
        return self.tokenizer.batch_decode(
            outputs.detach().cpu().numpy(), skip_special_tokens=True
        )[0]
