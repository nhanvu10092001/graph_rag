import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

qwen_variant = "Qwen3-8B"
graph_name = "prime"
base_model = f"Qwen/{qwen_variant}"
fine_tuned_model_name = "graphagent-600-noreject"
lora_dir = f"../data/finetuning/{graph_name}/{qwen_variant}/{fine_tuned_model_name}"
output_dir = f"../data/finetuning/{graph_name}/{qwen_variant}/{fine_tuned_model_name}/merged"

os.makedirs(output_dir, exist_ok=True)

print(f"Loading base model: {base_model}")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Loading LoRA adapter from: {lora_dir}")
model = PeftModel.from_pretrained(
    model,
    lora_dir,
)

print("Merging LoRA weights into base model...")
model = model.merge_and_unload()

print(f"Saving merged model to: {output_dir}")
model.save_pretrained(output_dir)

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True,
)
tokenizer.save_pretrained(output_dir)

print("\nDone ✅")
print("You can now run vLLM like:")
print(
    f"  python -m vllm.entrypoints.openai.api_server "
    f"--model {output_dir} --host 0.0.0.0 --port 8000"
)
