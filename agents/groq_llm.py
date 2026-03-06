import os
import json
from typing import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from groq import AsyncGroq


class GroqLlm(BaseLlm):
    model: str = "openai/gpt-oss-20b"
    generation_kwargs: dict = {}

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:

        self._maybe_append_user_content(llm_request)

        client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

        messages = []
        if llm_request.config and llm_request.config.system_instruction:
            messages.append(
                {"role": "system", "content": llm_request.config.system_instruction}
            )

        for content in llm_request.contents:
            role = content.role
            if role == "model":
                role = "assistant"

            text_parts = []
            for part in content.parts:
                if part.text:
                    text_parts.append(part.text)
                # Note: Currently we don't handle complex multi-modal or function-call mappings for Groq in this wrapper.

            text = "\n".join(text_parts)
            if not text:
                text = " "  # Groq requires non-empty content

            messages.append({"role": role, "content": text})

        # Optional: mapping ADK tools to Groq tools schema if needed.
        tools = []
        if llm_request.config and llm_request.config.tools:
            for tool_obj in llm_request.config.tools:
                if getattr(tool_obj, "function_declarations", None):
                    for fn_dec in tool_obj.function_declarations:
                        # Extract the required parameters from ADK Schema object if it exists.
                        parameters_dict = {"type": "object", "properties": {}}
                        if getattr(fn_dec, "parameters", None) and hasattr(
                            fn_dec.parameters, "properties"
                        ):
                            props = fn_dec.parameters.properties
                            required_fields = getattr(fn_dec.parameters, "required", [])
                            for prop_name, prop_val in props.items():
                                parameters_dict["properties"][prop_name] = {
                                    "type": (
                                        getattr(prop_val, "type", "string").lower()
                                        if getattr(prop_val, "type", None)
                                        else "string"
                                    ),
                                    "description": getattr(prop_val, "description", ""),
                                }
                            if required_fields:
                                parameters_dict["required"] = required_fields

                        tools.append(
                            {
                                "type": "function",
                                "function": {
                                    "name": fn_dec.name,
                                    "description": fn_dec.description,
                                    "parameters": parameters_dict,
                                },
                            }
                        )

        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }

        if self.generation_kwargs:
            if "temperature" in self.generation_kwargs:
                kwargs["temperature"] = self.generation_kwargs["temperature"]
            if self.generation_kwargs.get("response_mime_type") == "application/json":
                kwargs["response_format"] = {"type": "json_object"}

        if tools:
            kwargs["tools"] = tools

        try:
            if stream:
                stream_resp = await client.chat.completions.create(**kwargs)
                full_text = ""
                async for chunk in stream_resp:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and getattr(delta, "content", None):
                        full_text += delta.content
                        res_part = types.Part.from_text(text=delta.content)
                        yield LlmResponse(
                            content=types.Content(role="model", parts=[res_part]),
                            partial=True,
                        )
                # Final chunk
                final_part = types.Part.from_text(text=full_text)
                yield LlmResponse(
                    content=types.Content(role="model", parts=[final_part]),
                    partial=False,
                    turn_complete=True,
                    finish_reason=types.FinishReason.STOP,
                )
            else:
                resp = await client.chat.completions.create(**kwargs)
                content = resp.choices[0].message

                parts: list[types.Part] = []

                # Check for function calls
                if getattr(content, "tool_calls", None):
                    for tc in content.tool_calls:
                        if tc.type == "function":
                            fn_call = types.FunctionCall(
                                name=tc.function.name,
                                args=(
                                    json.loads(tc.function.arguments)
                                    if tc.function.arguments
                                    else {}
                                ),
                            )
                            # Add an id to match expected ADK schema if possible
                            if hasattr(fn_call, "id"):
                                fn_call.id = tc.id
                            parts.append(types.Part(function_call=fn_call))

                if getattr(content, "content", None):
                    parts.insert(0, types.Part.from_text(text=content.content))

                final_res = LlmResponse(
                    content=types.Content(role="model", parts=parts),
                    partial=False,
                    turn_complete=True,
                    finish_reason=types.FinishReason.STOP,
                )
                yield final_res

        except Exception as e:
            yield LlmResponse(
                error_code="GROQ_ERROR", error_message=str(e), turn_complete=True
            )
