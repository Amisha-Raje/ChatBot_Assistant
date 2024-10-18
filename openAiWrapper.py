import json
import requests
from typing import (
    Any, Dict, List, Optional, Sequence, Type, Union, Callable
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.pydantic_v1 import BaseModel, Field

# import curlify
import uuid
import tiktoken
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class ChatResultWithCost(ChatResult):
    cost: float
class ChatIntuition(BaseChatModel):
    url: str
    model: str
    provider: str
    input_token_cost: Optional[float]
    output_token_cost: Optional[float]
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    tools: Optional[List[BaseTool]] = []
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResultWithCost:
        prompt_messages = self._convert_messages(messages)
        # print(prompt_messages)
        input_tokens = self.num_tokens_from_messages(prompt_messages, self.model)
        # print(input_tokens)
        response_data = self._get_gpt_response(prompt_messages)
        # print(response_data)
        return self._parse_response(response_data, input_tokens)
    
    def _count_tokens(
        self,
        messages: List[BaseMessage]
    ) -> int:
        prompt_messages = self._convert_messages(messages)
        return self.num_tokens_from_messages(prompt_messages, self.model)
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_token_cost)/1000000 + (output_tokens * self.output_token_cost)/1000000
    
    @property
    def _llm_type(self) -> str:
        return "intuition-chat-model"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"url": self.url}
    
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        for tool in tools:
            if not isinstance(tool, BaseTool):
                raise ValueError(f"Tool {tool.name} is not an instance of BaseTool") 
        self.tools.extend(tools)
    
    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        role_map = {SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant"}
        return [{"role": role_map[type(msg)], "content": msg.content} for msg in messages if type(msg) in role_map]
    
    def _get_gpt_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        payload = self._make_gpt_payload(messages)
        response = requests.post(self.url,cookies = {
            'user.env.type': 'ENTERPRISE',
            'AWSALB': 'xVo0K7eIKmgk5YAHETX9djeOIf2tlZJau6SUm1MGOXy0%2Bz9dlNUme9vwNJO5AhOu2kxFKZnan8VgJTleDk2kzNlWdzCC6ZG3V%2Ff1Gc1T%2Fn0OAUw5DcggiplgNF7H',
            'AWSALBCORS': 'xVo0K7eIKmgk5YAHETX9djeOIf2tlZJau6SUm1MGOXy0%2Bz9dlNUme9vwNJO5AhOu2kxFKZnan8VgJTleDk2kzNlWdzCC6ZG3V%2Ff1Gc1T%2Fn0OAUw5DcggiplgNF7H',
            }, headers={'Content-Type': 'application/json'}, data=json.dumps(payload), timeout=300
        )
        #print(curlify.to_curl(response.request))
        output = response.json()
        return output
    
    def _make_gpt_payload(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "provider": self.provider,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "client_identifier": "backend-insights-dev",
            "messages": messages
        }
        if len(self.tools) > 0:
            payload['functions'] = [convert_to_openai_tool(tool)['function'] for tool in self.tools]
        return payload
    
    def _parse_response(self, response_data: Dict[str, Any], input_tokens: int) -> ChatResultWithCost:
        if 'choices' not in response_data:
            raise ValueError(f"Invalid response from API: {response_data}")
        
        message = response_data['choices'][0].get('message', {})
        # function_call = message.get('function_call')
        # print(message)
        # print(function_call)
        
        # output_tokens = self.num_tokens_from_messages([message], self.model)
        # cost = self._calculate_cost(input_tokens, output_tokens)
        
        # if function_call:
        #     function_args = json.loads(function_call['arguments'])
        #     tool_call_data = {
        #         "id": str(uuid.uuid4()),  # Generate a unique ID
        #         "name": function_call['name'],
        #         "args": function_args
        #     }
        #     ai_message = AIMessage(content="", tool_calls=[tool_call_data])
        # else:
        # print("helo")
        ai_message = AIMessage(content=message.get('content', ""))
        # print("hello")
        
        return ChatResultWithCost(generations=[ChatGeneration(message=ai_message)],cost = 0.0)
    
    def num_tokens_from_messages(self, messages, model):
        """Return the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gemini-1.5-flash-001",
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            return self.num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
    

# # Example Usage
# if __name__ == "__main__":
#     # Create an instance of ChatIntuition
#     chat_intuition = ChatIntuition(
#         # url='https://prod0-intuitionx-llm-router.sprinklr.com/chat-completion',
#         model="gpt-3.5-turbo",
#         provider= "AZURE_OPEN_AI",        
#     )
#     # chat_intuition.bind_tools(GradeHallucinations)
#     content = """Note: Existing Shingles algorithm that we use compares how “same” two sentences are while KNN search considers how “similar” two sentences are. In short, we can say that the Shingles algorithm performs lexical search while KNN search performs semantic search. k-Nearest Neighbours (k-NN) is a supervised machine learning algorithm used for classification and regression tasks. For a given data point, it finds the k-nearest data points in the training dataset based on a distance metric and makes predictions based on the majority class (for classification) or average value (for regression) of these neighbours. Elasticsearch provides 4 methods of calculating the similarity of nearest neighbours. The methods are as follows: l2_norm – uses Euclidean distance to calculate similarity. Dot product – uses dot product between vectors to calculate similarity. Can be used only with unit vectors. Cosine – same as dot product but does not allow vectors with zero magnitude.

# Index – the unique index value of this data. String text – the string value of the data. Vector embedding – the embedding value of the string. List - stores the list of parents of the current index. Each Parent consists of a pair of , where index is the index of the parent and similarity is the score of the similarity between the current text and the parent’s text. When adding a document, we perform KNN search on the embedding of that document, and a list of the ‘k’ nearest neighbours are returned. Then out of these neighbours, we add to the list all the neighbours which have a similarity score >= THRESHOLD (which is a parameter that can be defined). We found that the best results are achieved when using the cosine similarity function, with a THRESHOLD of 0.9. In the cosine similarity, the similarity score is a number between 0 and 1. It is not a percentage, however the larger it gets, the more likely that documents A and B are similar, and vice versa. Source: Blog

# Exclude vector fields from _source Time Consumption: The runtime search complexity is O(N * logk), where N is the number of vectors and k is the number of nearest neighbours. KNN Benchmarks The new _knn_search endpoint uses HNSW graphs to efficiently retrieve similar vectors. Unlike exact KNN, which performs a full scan of the data, it scales well to large datasets. Here is an example that compares _knn_search to the exact approach based on script_score queries on a dataset of 1 million image vectors with 128 dimensions, averaging over 10,000 different queries: Approach         Queries Per Second    Recall (k=10) script_score           5.257                     1.000 _knn_search          849.286                 0.945 Source: Blog Memory consumption: Although HNSW provides very good approximate nearest neighbour search at low latencies, it can consume a large amount of memory. Each HNSW graph uses 1.1 * (4 * d + 8 * m) * num_vectors bytes of memory:"""

#     generation = "k-Nearest Neighbours (k-NN) is a supervised machine learning algorithm used for classification and regression tasks. It finds the k-nearest data points in the training dataset based on a distance metric and makes predictions based on the majority class (for classification) or average value (for regression) of these neighbors. Thanks for asking!"
#     # generation = " knn is bad."

#     # Create some messages
#     messages = [
#         SystemMessage(content="""You are a grader assessing whether an LLM generation is supported by a set of retrieved facts. \n 
#             Give a binary score 'yes' or 'no'. 'Yes' means that the answer is supported by the set of facts."""),
#         HumanMessage(content=f"Set of facts: \n {content} \n\n LLM generation: {generation}"),
#     ]

#     # Generate a response
#     try:
#         chat_result_with_cost = chat_intuition._generate(messages)
#         print(chat_result_with_cost.generations[0].message.content)
#         # print(f"Cost: {chat_result_with_cost.cost}")
#     except Exception as e:
#         print(f"Error: {e}")        