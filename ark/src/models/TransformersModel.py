import torch
from typing import Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class Function:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = arguments


class ToolCall:
    def __init__(self, id: str, type: str, function: Function):
        self.id = id
        self.type = type
        self.function = function


class TransformersMessage:
    def __init__(self, role: str, content: str, tool_calls: list[ToolCall], reasoning_content: str):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class TransformersModel:
    def __init__(
        self,
        name,
        quantized: bool = False,
        enable_thinking: bool = False,
    ) -> None:
        self.name = name
        self.quantized = quantized
        self.tokenizer, self.model = self.load_base_model(quantized=quantized)
        self.enable_thinking = enable_thinking

    def load_base_model(self, quantized: bool = False):
        if quantized:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.name, device_map="auto", quantization_config=bnb_config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.name, device_map="auto", dtype=torch.float16
            )

        tokenizer = AutoTokenizer.from_pretrained(self.name)
        return tokenizer, model

    def parse_tool_calls(self, content: str) -> list:
        raise NotImplementedError("Not implemented")

    def parse_thinking(self, content: str) -> tuple[Optional[str], str]:
        raise NotImplementedError("Not implemented")

    def forward(self, messages, tools=[], tool_choice="auto", enable_thinking=None):
        if enable_thinking is None:
            enable_thinking = self.enable_thinking

        formatted_input = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )

        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.7,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        reasoning_content, cleaned_response_text = self.parse_thinking(response_text)

        tool_calls = self.parse_tool_calls(cleaned_response_text)

        message = TransformersMessage(
            role="assistant",
            content=cleaned_response_text,
            tool_calls=[
                ToolCall(
                    id=tool_call["id"],
                    type=tool_call["type"],
                    function=Function(
                        name=tool_call["function"]["name"],
                        arguments=tool_call["function"]["arguments"],
                    ),
                )
                for tool_call in tool_calls
            ],
            reasoning_content=reasoning_content,
        )

        return message
