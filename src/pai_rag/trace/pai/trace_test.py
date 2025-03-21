"""Test llama-index instrumentation."""
## refer to:
##  https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent/

# this depends:
# pip install llama-index-agent-openai llama-index-llms-openai
import unittest
import json
from typing import Sequence, List

from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from openai.types.chat import ChatCompletionMessageToolCall

from pai_rag.trace.pai.pai_instrumentor import set_custom_attributes
from pai_rag.trace.pai.pai_instrumentor import init_opentelemetry, get_tracer
from llama_index.llms.openai_like import OpenAILike

from pai_rag.trace.trace_config import TraceConfig

llm = OpenAILike(
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-499ae44046ee48dfbcb38b6a47045b05",
    model="qwen-max",
    is_function_calling_model=True,
    is_chat_model=True,
)


def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer."""
    custom_attr = {
        "input.value": f"THIS IS CUSTOMIZED ATTRIBUTE: {a} * {b}",
    }
    set_custom_attributes(custom_attr)

    with get_tracer().start_as_current_span("customize_span") as child_span:
        child_span.set_attributes(custom_attr)

    return a * b


multiply_tool = FunctionTool.from_defaults(
    fn=multiply, name="tool_for_multiply", description="用于计算两个整数相乘结果"
)


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer."""
    custom_attr = {
        "input.value": f"THIS IS CUSTOMIZED ATTRIBUTE: {a} + {b}",
    }
    set_custom_attributes(custom_attr)

    with get_tracer().start_as_current_span("customize_span") as child_span:
        child_span.set_attributes(custom_attr)

    return a + b


add_tool = FunctionTool.from_defaults(
    fn=add, name="tool_for_add", description="用于计算两个整数相加结果"
)


class LlamaIndexInstrumentationTest(unittest.TestCase):
    """Test open ai instrument."""

    def setUp(self):
        # pylint: disable=line-too-long
        self.app_name = "feiyue_trace_2025"
        config = TraceConfig(
            service_name=self.app_name,
            endpoint="http://tracing-analysis-dc-hz.aliyuncs.com:8090",
            token="xx",
        )
        init_opentelemetry(
            config=config,
            service_version="0.0.1",
            service_id="",
            deployment_environment="",
            service_owner_id="",
            service_owner_sub_id="",
        )

    def test_llama_index_agent_instrument(self):
        """Test llama index agent instrument."""

        class YourOpenAIAgent:
            def __init__(
                self,
                tools: Sequence[BaseTool] = [],
                llm: OpenAI = OpenAI(temperature=0, model="gpt-4o-mini"),
                chat_history: List[ChatMessage] = [],
            ) -> None:
                self._llm = llm
                self._tools = {tool.metadata.name: tool for tool in tools}
                self._chat_history = chat_history

            def reset(self) -> None:
                self._chat_history = []

            def chat(self, message: str) -> str:
                chat_history = self._chat_history
                chat_history.append(ChatMessage(role="user", content=message))
                tools = [
                    tool.metadata.to_openai_tool() for _, tool in self._tools.items()
                ]

                ai_message = self._llm.chat(chat_history, tools=tools).message
                additional_kwargs = ai_message.additional_kwargs
                chat_history.append(ai_message)

                tool_calls = additional_kwargs.get("tool_calls", None)
                # parallel function calling is now supported
                if tool_calls is not None:
                    for tool_call in tool_calls:
                        function_message = self._call_function(tool_call)
                        chat_history.append(function_message)
                        ai_message = self._llm.chat(chat_history).message
                        chat_history.append(ai_message)

                return ai_message.content

            def _call_function(
                self, tool_call: ChatCompletionMessageToolCall
            ) -> ChatMessage:
                id_ = tool_call.id
                function_call = tool_call.function
                tool = self._tools[function_call.name]
                output = tool(**json.loads(function_call.arguments))
                return ChatMessage(
                    name=function_call.name,
                    content=str(output),
                    role="tool",
                    additional_kwargs={
                        "tool_call_id": id_,
                        "name": function_call.name,
                    },
                )

        agent = YourOpenAIAgent(tools=[multiply_tool, add_tool], llm=llm)
        trace_id = None
        custom_attr = {
            "service.app.user_id": "123456789",
            "service.app.user_name": "123456789@阿里巴巴",
        }

        with get_tracer().start_as_current_span(
            "llama_index_agent_multiply_test"
        ) as current_span:
            trace_id = current_span.get_span_context().trace_id
            self.assertNotEqual(str(trace_id), "0")
            set_custom_attributes(custom_attr)

            resp = agent.chat("Q1: 123 * 456 等于多少")
            print(resp)
            self.assertTrue("56,088" in resp or "56088" in resp)

        with get_tracer().start_as_current_span("llama_index_agent_add_test"):
            self.assertNotEqual(str(trace_id), "0")
            set_custom_attributes(custom_attr)

            agent.reset()
            resp = agent.chat("Q2: What is 123 + 456 等于多少")
            print(resp)
            self.assertTrue("579" in resp)

        print(f"trace_id: {trace_id}, hex: {hex(trace_id)[2:]}")

    def test_agent_simple(self):
        """Test llama index agent instrument."""
        agent = OpenAIAgent.from_tools(
            [multiply_tool, add_tool],
            llm=llm,
        )
        with get_tracer().start_as_current_span("llama_index_agent_multiply_add_test"):
            response = agent.query("What is (121 * 3) + 42?")
            print(response)
            self.assertTrue("405" in response.response)
