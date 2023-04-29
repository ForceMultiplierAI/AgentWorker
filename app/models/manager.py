import json
from .base import BaseModel
from .rwkv import RWKVBackend
from .llama import LLaMaCausalLMBackend
from .neox import GPTNeoXForCausalLMBackend
from .openai import ChatGPTPassThroughBackend

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.models = []
            cls._instance.model_names = []  # Define model_names here
            cls._instance.load_models()
        return cls._instance

    def load_models(self):
        # Read the config file
        with open('./config.json') as f:
            config = json.load(f)

        for name, model_config in config['MODELS'].items():
            if model_config['ENABLED']:
                model_class = self.get_model_class(model_config['TYPE'])
                model = model_class(name, **model_config)
                self.models.append(model)
                self.model_names.append(name)

    def get_model_class(self, model_type):
        if model_type == 'RWKV':
            return RWKVBackend
        if model_type == 'LLaMaCausalLM':
            return LLaMaCausalLMBackend
        if model_type == 'GPTNeoX': 
            return GPTNeoXForCausalLMBackend
        if model_type == 'ChatGPTPassThrough':
            return ChatGPTPassThroughBackend
        else:
            raise ValueError(f'Invalid model type: {model_type}')

    def get_named_model(self, name):
        try:
            index = self.model_names.index(name)
        except ValueError:
            raise ValueError(f'No model named {name} found')
        return self.models[index]

    def get_default_model(self):
        if self.models:
            return self.models[0]
        else:
            raise ValueError('No models loaded')


# In this specific case, the ModelManager is implemented as a singleton pattern. The singleton pattern ensures that only one instance of the class can be created throughout the program. The __new__ method is overridden to enforce this constraint. When you try to create a new instance of the ModelManager class, the __new__ method checks whether an instance already exists (cls._instance is None). If an instance exists, it returns the existing instance, effectively preventing the creation of multiple instances of the class. If no instance exists, it creates a new one using the superclass's __new__ method, initializes it, and stores it in the _instance class attribute.

