from src.models.QwenModel import QwenModel
from transformers import AutoTokenizer
from peft import PeftModel


class FineTunedQwenModel(QwenModel):
    def __init__(self, name, finetune_path, quantized=False, enable_thinking=False):
        super().__init__(name, quantized=quantized, enable_thinking=enable_thinking)

        self.finetune_path = finetune_path
        self.max_workers = 1
        self.tokenizer = AutoTokenizer.from_pretrained(self.finetune_path)
        self.model = PeftModel.from_pretrained(self.model, self.finetune_path)

    def base_model_forward(self, *args, **kwargs):
        self.model.disable_adapter_layers()
        r = self.forward(*args, **kwargs)
        self.model.enable_adapter_layers()
        return r
