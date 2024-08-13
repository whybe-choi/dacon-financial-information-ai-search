import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from accelerate import Accelerator

from langchain_huggingface.llms import HuggingFacePipeline


def load_pipeline(model_id):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.use_default_system_prompt = False

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=None,
        top_p=None,
        return_full_text=False,
        max_new_tokens=384,
        do_sample=False,
    )

    hf_pipeline = HuggingFacePipeline(pipeline=text_generation_pipeline)

    return hf_pipeline
