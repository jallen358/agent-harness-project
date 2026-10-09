import os

from anthropic import Anthropic

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

response = client.messages.create(
    model=os.environ["HARNESS_MODEL"],
    max_tokens=256,
    tools=[
        {
            "name": "echo",
            "description": "Echo a string back to the caller.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to echo."},
                },
                "required": ["text"],
            },
        }
    ],
    messages=[{"role": "user", "content": "Call echo with text hello."}],
)
print(response)
