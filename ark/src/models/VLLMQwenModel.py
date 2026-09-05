from litellm import completion


class VLLMQwenModel:
    def __init__(
        self,
        server_base_url: str,
        name: str,
        enable_thinking: bool = False,
        max_workers: int = 3,
    ) -> None:
        self.name = name
        self.server_base_url = server_base_url
        self.enable_thinking = enable_thinking
        self.max_workers = max_workers

    def forward(self, messages, tools=[], tool_choice="auto", enable_thinking=None):

        if enable_thinking is None:
            enable_thinking = self.enable_thinking

        response = completion(
            model=f"openai/{self.name}",
            api_base=self.server_base_url,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            api_key="_",
            timeout=120,
        )
        message = response["choices"][0]["message"]

        if not hasattr(message, "reasoning_content"):
            message.reasoning_content = None

        return message
