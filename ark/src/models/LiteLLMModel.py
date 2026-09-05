from litellm import completion


class LiteLLMModel:
    def __init__(self, name, enable_thinking=False, max_workers=3):
        self.name = name
        self.enable_thinking = enable_thinking
        self.max_workers = max_workers

    def forward(self, messages, tools=[], tool_choice="auto", enable_thinking=None):
        if enable_thinking is None:
            enable_thinking = self.enable_thinking

        gpt_args = {}
        if enable_thinking:
            gpt_args["reasoning_effort"] = "medium"

        response = completion(
            model=self.name,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            timeout=30,
            **gpt_args,
        )
        message = response["choices"][0]["message"]

        if not hasattr(message, "reasoning_content"):
            message.reasoning_content = None

        return message
