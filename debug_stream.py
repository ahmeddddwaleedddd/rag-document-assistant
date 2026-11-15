# debug_stream.py
from dotenv import load_dotenv
load_dotenv()

import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    model="qwen3:4b",
    max_tokens=3000,  # raised from 500 so thinking has room to finish
    system="Always respond by calling the tool. Do not think for long; respond quickly.",
    tools=[{
        "name": "answer_from_document",
        "description": "test",
        "input_schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    }],
    tool_choice={"type": "tool", "name": "answer_from_document"},  # force it, skip the ambiguity
    messages=[{"role": "user", "content": "Call the tool with answer='This is a longer test sentence to see if it streams in pieces or all at once.'"}],
) as stream:
    i = 0
    for event in stream:
        if event.type == "thinking":
            pass  # swallow thinking text, we don't care about it here
        elif event.type == "input_json":
            i += 1
            print(f"[chunk {i}] len={len(event.partial_json)} -> {event.partial_json!r}")
    print("TOTAL CHUNKS:", i)
    final = stream.get_final_message()
    print("\nstop_reason:", final.stop_reason)
    print("output_tokens:", final.usage.output_tokens)