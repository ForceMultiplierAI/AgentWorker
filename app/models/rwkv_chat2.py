import json
import torch
import codecs
from abc import ABC
from typing import Any, List, Mapping
from rwkvstic.load import RWKV
from rwkvstic.agnostic.backends import TORCH, TORCH_QUANT, TORCH_STREAM
# from rwkvstic.tokenizer import Tokenizer

import gc

model = {}

interface = ":"
user = "cryscan"
bot = "Eliose"

init_prompt = f'''
The following is a verbose detailed conversation between {user} and a young girl chat robot {bot}. {bot} is intelligent, friendly and cute. {bot} is unlikely to disagree with {user}.

{user}{interface} Hello {bot}, how are you doing?

{bot}{interface} Hi {user}! Thanks, I'm fine. What about you?

{user}{interface} I am very good! It's nice to see you. Would you mind me chatting with you for a while?

{bot}{interface} Not at all! I'm listening.

{user}{interface} Tell me about Philosophy

{bot}{interface} Certainly!, Philosophy as a discipline and way of thinking has been practiced by humans for thousands of years, and its origins can be traced back to ancient civilizations such as Greece and China. However, it is difficult to attribute the invention of philosophy to a single person or civilization. Instead, philosophy has evolved over time through the contributions of many thinkers, each building upon the ideas of their predecessors. Some of the earliest and most influential philosophers include Socrates, Plato, and Aristotle in ancient Greece, and Confucius and Laozi in ancient China.

{user}{interface} What is the meaning of life?

{bot}{interface} The meaning of life is a philosophical question that has been debated by scholars and thinkers for centuries, and there is no single answer that is universally accepted. Different people and cultures have offered a wide range of perspectives and interpretations on the meaning of life, ranging from religious and spiritual beliefs to scientific and existential perspectives.

Some people believe that the meaning of life is to seek happiness, fulfillment, and personal growth. Others believe that the meaning of life is to serve a higher power or purpose, such as a god or gods, or to contribute to the greater good of society.

Still others believe that life has no inherent meaning and that it is up to each individual to create their own purpose and meaning through their experiences and actions.

Ultimately, the meaning of life is a deeply personal and subjective question, and each person must determine their own answer based on their own beliefs, values, and experiences.

{user}{interface} How many people are there in the world?

{bot}{interface} As of my training data cut-off in 2021, the estimated world population was around 7.9 billion people. However, the exact number of people in the world changes constantly due to births, deaths, and migration, so the current figure may be higher or lower. It's also worth noting that population estimates can vary depending on the methodology used and the sources of data, so different sources may provide slightly different numbers.

'''



class RWKVBackend(ABC):
    def __init__(self, name, **config):
        self.name = name
        self.model = {}
        self.modelStates = {}
        self.generate_config = config.get('GENERATE_CONFIG', {})
        self.tokenizer_config = config.get('TOKENIZER_CONFIG', {})
        self.model_config = config.get('MODEL_CONFIG', {})
        # print loaded
        print(f'Loaded model: {self.name}')
        # print all variables
        print(f'generate_config: {self.generate_config}')
        # tokenizer_config: {self.tokenizer_config}')
        print(f'tokenizer_config: {self.tokenizer_config}')
        print(f"tokenizer_config: {self.tokenizer_config}")
        #  print config
        # print(f'config: {config}')

        runtimedtype = torch.float32 # torch.float64, torch.bfloat16
            # this is the dtype used for matrix-vector operations, and is the dtype that will determine the performance and memory usage of the model
        dtype = torch.bfloat16 # torch.float32, torch.float64, torch.bfloat16
        useGPU = True # False

        # self.model = RWKV("/home/shazam/RWKV-4-Pile-7B-20230109-ctx4096.pth", mode=TORCH, useGPU=True, runtimedtype=runtimedtype, dtype=dtype)

    def start(self):
        print("Starting RWKV model")
        runtimedtype = torch.bfloat16 #float32 # torch.float64, torch.bfloat16
            # this is the dtype used for matrix-vector operations, and is the dtype that will determine the performance and memory usage of the model
        dtype = torch.bfloat16 # torch.float32, torch.float64, torch.bfloat16
        useGPU = True # False
        # if model_config has  "MODEL_CONFIG" : {
                # "FILE" : "/home/shazam/RWKV-4-Pile-14B-20230228-ctx4096-test663"
            # },
            #  then use that model

        # print tokenizer_config
        if self.model_config:
            print(f"Loading model {self.model_config}")

            dtype = torch.bfloat16 if self.model_config.get('dtype') == 'torch.bfloat16' else torch.float32
            # runtimedtype = self.model_config.get("runtimedtype", runtimedtype)
            runtimedtype = torch.bfloat16 if self.model_config.get('runtimedtype') == 'torch.bfloat16' else torch.float32

            mode = TORCH_QUANT if self.model_config.get("mode") == "TORCH_QUANT" else TORCH
            target = self.model_config.get("target", 22)
            maxQuantTarget = self.model_config.get("maxQuantTarget", 22)
            self.model = RWKV(self.model_config["FILE"], mode=mode, useGPU=True, target=target, dtype=dtype, runtimedtype=runtimedtype, maxQuantTarget=maxQuantTarget)
        else:
            # error message model FILE not found
            print(f"Error: model {self.model_config.get('FILE')} not found")
            exit()


        # Agent variables
        # number = 750 # the number of tokens to generate
        # stopStrings = ["<|endoftext|>", "Best,", "Response:", "Summary:", "Experts", "Please", "http", "https", "Regards", "Hope", "Thank", "Thanks", "Disclaimer", "©", "Copyright"] # When read, the model will stop generating output
        # stopTokens = [0] # advanced, when the model has generated any of these tokens, it will stop generating output
        # temp = .9  #.87 #1 # the temperature of the model, higher values will result in more random output, lower values will result in more predictable output
        # top_p = .9 # the top_p of the model, higher values will result in more random output, lower values will result in more predictable output
        # end_adj = -666

        #print loading config
        print(f"Loading config {self.generate_config}")

        self.stopStrings = self.generate_config.get("stop", ["<|endoftext|>"])
        self.stopTokens = self.generate_config.get("stopTokens", [0])
        self.temp = self.generate_config.get("temperature", .9)
        self.top_p = self.generate_config.get("top_p", .9)
        self.end_adj = self.generate_config.get("end_adj", -666)
        self.number = self.generate_config.get("number", 100)

        # load context

        # intro = open("ELDR-agent_lite.txt", "r").read()
        
        self.model.resetState()
        self.model.loadContext(newctx=init_prompt)

        output = self.model.forward(number=1, stopStrings=self.stopStrings, stopTokens=self.stopTokens, temp=self.temp, end_adj=self.end_adj, top_p_usual=self.top_p)

        self.chatState = self.model.getState()

    def completions(self, prompt : str, **kwargs):
        print (prompt)

        choices = []
        prompt_tokens_count = 0
        completion_tokens_count = 0

        # model.loadContext(newctx=f"Q: who is Jim Butcher?\n\nA:")
            # if prompt is equal to "" then set prompt to " "
        if prompt == "":
            prompt = " "
    
        print(f"prompt: `{prompt}`")

        self.model.loadContext(newctx=prompt)
        # def loadContext(model, ctx, newctx, statex, progressCallBack=lambda x: x):

        print(f"kwargs: {kwargs}")
        stopStrings = kwargs.get("stop", self.stopStrings)
        stopTokens = kwargs.get("stopTokens", self.stopTokens)
        temp = kwargs.get("temperature", self.temp)
        top_p = kwargs.get("top_p", self.top_p)
        end_adj = kwargs.get("end_adj", self.end_adj)
        number = kwargs.get("max_tokens", self.number)
        
        output = self.model.forward(
            number=1,
            stopStrings=stopStrings,
            stopTokens=stopTokens,
            temp=temp,
            top_p_usual=top_p,
            end_adj=end_adj
         )["output"]
        
        choices.append({
            'text': output,
            'index': 0,
        })

        response = {
            'choices': choices,
            'usage': {
                'prompt_tokens': prompt_tokens_count,
                'completion_tokens': completion_tokens_count,
                'total_tokens': prompt_tokens_count + completion_tokens_count
            },
            "model": self.name,
            "kwargs": kwargs,
        }
        print(f"response: `{response}`")
        return response

    def streaming_completion(self, prompt : str, **kwargs):
        # print (prompt)
        # softPrompt = f"{prompt}\n\nExpert Long Detailed Response:\n\nHi, thanks for reaching out, we would be happy to answer your question."
        # print(f"ELDRprompt: `{softPrompt}`")
        softprompt = f"\n{bot}{interface} {prompt}\n\n"
        print(f"softprompt: `{softprompt}`")
        self.model.loadContext(newctx=softprompt)

        print(f"kwargs: {kwargs}")
        stopStrings = kwargs.get("stop", self.stopStrings)
        stopTokens = kwargs.get("stopTokens", self.stopTokens)
        temp = float(kwargs.get("temperature", self.temp))
        top_p = float(kwargs.get("top_p", self.top_p))
        end_adj = max(-4000, min(kwargs.get("end_adj", self.end_adj), 0))
        number = kwargs.get("max_tokens", self.number)

        
        # stopStrings = ["<|endoftext|>"] #"<|endoftext|>", "Best,", "Response:", "Summary:", "Experts", "Please", "http", "https", "Regards", "Hope", "Thank", "Thanks", "Disclaimer", "©", "Copyright"] # When read, the model will stop generating output
        # stopTokens = [] # advanced, when the model has generated any of these tokens, it will stop generating output
        # temp = .9  #.87 #1 # the temperature of the model, higher values will result in more random output, lower values will result in more predictable output
        # top_p = .9 # the top_p of the model, higher values will result in more random output, lower values will result in more predictable output
        # end_adj = 0 #-666

        # add "bot" to stopStrings
        # stopStrings.append(bot)
        # stopStrings.append(interface)
        # stopStrings.append(user)
        stopStrings.append("\n\n")

        max_new_tokens = number # 
        print(f"max_new_tokens: {max_new_tokens}")

        print(f"All of the variables: stopStrings:{stopStrings}, stopTokens:{stopTokens}, temp:{temp}, top_p:{top_p}, end_adj:{end_adj}, number:{number}")
        
        ################################################
        generated_text = ""
        done = False
        with torch.no_grad():
            for _ in range(max_new_tokens):
                char = self.model.forward(stopStrings=stopStrings, 
                stopTokens=stopTokens, temp=temp, top_p_usual=top_p, end_adj=end_adj)[
                    "output"]
                # instead, print char but no new line
                # print(char, end="")

                generated_text += char
                # generated_text = generated_text.lstrip("\n ")

                for stop_word in stopStrings:
                    stop_word = codecs.getdecoder("unicode_escape")(stop_word)[0]
                    if stop_word != '' and stop_word in generated_text:
                        done = True
                        print("line 189 <stopped>\n")
                        break
                yield char
                if done:
                    print("<line 193 stopped>\n")
                    break

            for stop_word in stopStrings:
                stop_word = codecs.getdecoder("unicode_escape")(stop_word)[0]
                if stop_word != '' and stop_word in generated_text:
                    generated_text = generated_text[:generated_text.find(stop_word)]

            gc.collect()
            print("gc collect, reset, load state")
            # yield generated_text
            self.model.resetState()
            self.model.setState(self.chatState)
            
    def reset(self):
        self.model.resetState()
    
    def loadStateIdemponent(self, state_name):
        if state_name in self.modelStates:
            self.model.setState(self.modelStates[state_name])
        else:
            self.modelStates[state_name] = self.model.getState()

    def saveState(self, state_name):
        self.modelStates[state_name] = self.model.getState()

# list all self.modelStates by name return json
    def listStates(self):
        # return json.dumps(self.modelStates)
        #  dont dump the model state itself, drop the modelStates[state_name] or list of state_names
        return json.dumps(list(self.modelStates.keys()))
        