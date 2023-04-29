from datetime import datetime
import re
import json
import sys
import openai
from abc import ABC
from typing import Any, List, Mapping

import gc
model = {}

class ChatGPTPassThroughBackend(ABC):
    def __init__(self, name, **config):
        self.name = name
        self.model = {}
        self.modelStates = {}
        self.generate_config = config.get('GENERATE_CONFIG', {})
        self.tokenizer_config = config.get('TOKENIZER_CONFIG', {})
        self.model_config = config.get('MODEL_CONFIG', {})

        print(f'Loaded model: {self.name}')
        print(f'generate_config: {self.generate_config}')
        print(f'tokenizer_config: {self.tokenizer_config}')

        self.stopStrings = self.generate_config.get("stop", ["<|endoftext|>"])
        self.stopTokens = self.generate_config.get("stopTokens", [0])
        self.temp = self.generate_config.get("temperature", .9)
        self.top_p = self.generate_config.get("top_p", .9)
        self.end_adj = self.generate_config.get("end_adj", -666)
        self.number = self.generate_config.get("number", 100)

        
    def start(self):
        print("Starting ChatGPTPassThroughBackend model")

    def chat_complete(self, messages : List, **kwargs):

        # pop temperature
        temperature = kwargs.pop("temperature", self.temp)
        max_tokens = kwargs.pop("max_tokens", self.number)
        model = kwargs.pop("model", None)

        print(f"ChatGPTPassThroughBackend: chat_complete: temperature: {temperature} max_tokens: {max_tokens} model: {model}")

        # messages_json = json.dumps(messages)
        # print(f"Messages: {messages_json}")
            # Add a default name if 'name' is missing or None
        for message in messages:
            if 'name' not in message or message['name'] is None:
                message['name'] = 'default_name'
        

        print(f"Messages: {messages}")

        response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
        return response
    
    def get_embeddings(self, input : List[str], **kwargs):
        return openai.Embedding.create(
            input=input, model="text-embedding-ada-002"
            ) #["data"][0]["embedding"]

    def completion(self, prompt : str, **kwargs):
        # pop temperature
        temperature = kwargs.pop("temperature", self.temp)
        max_tokens = kwargs.pop("max_tokens", self.number)
        model = kwargs.pop("model", None)

        print(f"ChatGPTPassThroughBackend: completion: temperature: {temperature} max_tokens: {max_tokens} model: {model}")

        response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
        return response

    def streaming_completion(self, prompt : str, **kwargs):
        print(f"streaming_completion kwargs: {kwargs}")
        stopStrings = kwargs.pop("stop", self.stopStrings)
        stopTokens = kwargs.pop("stopTokens", self.stopTokens)
        temp = float(kwargs.pop("temperature", self.temp))
        top_p = float(kwargs.pop("top_p", self.top_p))
        # end_adj = max(-4000, min(kwargs.pop("end_adj", self.end_adj), 0))
        number = kwargs.pop("max_tokens", self.number)

        max_new_tokens = number         # borrowing variable name from transformers "max_new_tokens" for more descriptive code
        print(f"All of the variables: stopStrings:{stopStrings}, stopTokens:{stopTokens}, temp:{temp}, top_p:{top_p}, max_new_tokens:{number}")
        print(f"Remaining kwargs: {kwargs} (should be empty)")
        
        ################################################
        generated_text = ""
        done = False
        completion = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temp,
            stream=True            
        )

        result = ""

        for resp in completion:
            result += resp.choices[0].text
            yield resp.choices[0].text

    def reset(self):
        pass