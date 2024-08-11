import torch
import transformers
from tqdm import tqdm
from .promptor_interface import PromptorInterface

class Promptor(PromptorInterface):
    def __init__(self, 
                 file_system: PromptorInterface):
        self.file_system = file_system()
    
    def do_llm(self, instruction):
        return self.file_system.do_llm(instruction)

class Gemma2Promptor(PromptorInterface):
    def __init__(self):
        model_id = "rtzr/ko-gemma-2-9b-it"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        self.pipeline.model.eval()

    def do_llm(self, instruction):
        messages = [
            {"role": "user", "content": f"{instruction}"}
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        return outputs, outputs[0]["generated_text"][len(prompt):]