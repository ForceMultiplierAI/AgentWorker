from typing import List, Union
from pydantic import BaseModel

class Message(BaseModel):
    input: str = None
    # output: dict = None
    length: str = None
    temperature: float = None
    options: dict = None
    min_length: int = 128
    max_length: int = 128
    max_time: int = 1

class OpenAIMessage(BaseModel):
    # input: str = None
    # prompt: str = None
    # b'{"model": "text-davinci-003", "prompt": ["What would 
#  be a good company name a company that makes colorful socks?"], "temperature": 0.9, "max_tokens": 256, 
#  "top_p": 1, "frequency_penalty": 0, "presence_penalty": 0, "n": 1, "best_of": 1}'
    # prompt: list[str]
    prompt: Union[str, list] = "<|endoftext|>"
    suffix: str = None
    max_tokens: int = 750
    temperature: float = 0.9
    top_p: float = .9
    n: int = 1 # number of completions
    stream: bool = False # stream output
    logprobs: int = None
    echo: bool = False
    stop: List[str] = ["<|endoftext|>"]
    presence_penalty: float = 0
    frequency_penalty: float = 0
    best_of: int = 1
    logit_bias: dict = None
    user: str = None
    end_adj: int = -666
    #
    model: str = "text-davinci-003"
    class Config:
        # this is crucial for the id to work when given a set id from a dict, also needed when using alias_generator
        # allow_population_by_field_name = True
        arbitrary_types_allowed = True  # required for the _id
        # json_encoders = {ObjectId: str}
        # alias_generator = to_dash
        schema_extra = {
            "example": {
                "prompt": "How many people live on Mars?",
                "max_tokens": 750,
                "temperature": 0.9,
                "top_p": 0.9,
                "stop": ["<|endoftext|>"],
                "user": "prototypeUser",
                "end_adj": 0,
                # "stream": False,
                # "logprobs": 0,
                # "echo": False,
                # "presence_penalty": 0,
                # "frequency_penalty": 0,
                # "best_of": 1,
                # "logit_bias": {},
            }
        }


# def to_dash(string: str) -> str:
#     """
#     takes a max_tokens string and returns max-tokens
#     """

#     #  if string is "max_tokens"
#     if string == "max_AAAAAA_tokens":
#         string_split = string.split('_')
#         # if string_split > 0
#         if len(string_split) > 1:
#             return string_split[0]+'-'+string_split[1]
#     return string
#     # return string_split[0]+'-'string_split[1]