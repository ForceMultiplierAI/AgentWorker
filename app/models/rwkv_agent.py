import torch
import codecs
from abc import ABC
from typing import Any, List, Mapping
from rwkvstic.load import RWKV
from rwkvstic.agnostic.backends import TORCH
# from rwkvstic.tokenizer import Tokenizer

import gc

model = {}

class RWKVAgentBackend(ABC):
    def __init__(self, name, **config):
        self.name = name
        self.model = {}
        self.modelStates = {}
        self.generate_config = config.get('GENERATE_CONFIG', {})
        self.tokenizer_config = config.get('TOKENIZER_CONFIG', {})
        # print loaded
        print(f'Loaded model: {self.name}')
        # print all variables
        print(f'generate_config: {self.generate_config}')
        # tokenizer_config: {self.tokenizer_config}')
        print(f'tokenizer_config: {self.tokenizer_config}')

        runtimedtype = torch.float32 # torch.float64, torch.bfloat16
            # this is the dtype used for matrix-vector operations, and is the dtype that will determine the performance and memory usage of the model
        dtype = torch.bfloat16 # torch.float32, torch.float64, torch.bfloat16
        useGPU = True # False

        # self.model = RWKV("/home/shazam/RWKV-4-Pile-7B-20230109-ctx4096.pth", mode=TORCH, useGPU=True, runtimedtype=runtimedtype, dtype=dtype)

    def start(self):
        print("Starting RWKV model")
        runtimedtype = torch.float32 # torch.float64, torch.bfloat16
            # this is the dtype used for matrix-vector operations, and is the dtype that will determine the performance and memory usage of the model
        dtype = torch.bfloat16 # torch.float32, torch.float64, torch.bfloat16
        useGPU = True # False
        self.model = RWKV("/home/shazam/RWKV-4-Pile-7B-20230109-ctx4096.pth", mode=TORCH, useGPU=True, runtimedtype=runtimedtype, dtype=dtype)


        # Agent variables
        number = 750 # the number of tokens to generate
        stopStrings = ["<|endoftext|>", "Best,", "Response:", "Summary:", "Experts", "Please", "http", "https", "Regards", "Hope", "Thank", "Thanks", "Disclaimer", "©", "Copyright"] # When read, the model will stop generating output
        stopTokens = [0] # advanced, when the model has generated any of these tokens, it will stop generating output
        temp = .9  #.87 #1 # the temperature of the model, higher values will result in more random output, lower values will result in more predictable output
        top_p = .9 # the top_p of the model, higher values will result in more random output, lower values will result in more predictable output
        end_adj = -666

        # load context

        intro = open("ELDR-agent_lite.txt", "r").read()
        self.model.resetState()
        self.model.loadContext(newctx=intro)
        output = self.model.forward(number=1, stopStrings=stopStrings, stopTokens=stopTokens, temp=temp, end_adj=end_adj, top_p_usual=top_p)
        self.chatState = self.model.getState()


    def streaming_completion(self, prompt : str, **kwargs):
        print (prompt)
        ELDRprompt = f"{prompt}\n\nExpert Long Detailed Response:\n\nHi, thanks for reaching out, we would be happy to answer your question. Have a look:\""
        # model.loadContext(newctx=ELDRprompt)

        choices = []
        prompt_tokens_count = 0
        completion_tokens_count = 0

        # model.loadContext(newctx=f"Q: who is Jim Butcher?\n\nA:")
            # if prompt is equal to "" then set prompt to " "
        # if prompt == "":
        #     prompt = " "

        # stop = kwargs.get("stop", ["\n"])
        # max_new_tokens = kwargs.get("max_new_tokens", 100)
        # temperature = kwargs.get("temperature", 0.9)
        # top_p = kwargs.get("top_p", 0.9)
        # end_adj = kwargs.get("end_adj", 0.9)

                # Agent variables
        number = 250 # the number of tokens to generate
        stop = ["<|endoftext|>", "Best,", "Response:", "Summary:", "Experts", "Please", "http", "https", "Regards", "Hope", "Thank", "Thanks", "Disclaimer", "©", "Copyright"] # When read, the model will stop generating output
        stopTokens = [0] # advanced, when the model has generated any of these tokens, it will stop generating output
        temperature = .9  #.87 #1 # the temperature of the model, higher values will result in more random output, lower values will result in more predictable output
        top_p = .9 # the top_p of the model, higher values will result in more random output, lower values will result in more predictable output
        end_adj = -666

        max_new_tokens = number # 

        print(f"ELDRprompt: `{ELDRprompt}`")

        self.model.loadContext(newctx=ELDRprompt)
        ################################################
        generated_text = ""
        done = False
        with torch.no_grad():
            for _ in range(max_new_tokens):
                char = self.model.forward(stopStrings=stop, temp=temperature, top_p_usual=top_p, end_adj=end_adj)[
                    "output"]
                print(char, end='', flush=True)
                generated_text += char
                generated_text = generated_text.lstrip("\n ")

                for stop_word in stop:
                    stop_word = codecs.getdecoder("unicode_escape")(stop_word)[0]
                    if stop_word != '' and stop_word in generated_text:
                        done = True
                        break
                yield char
                if done:
                    print("<stopped>\n")
                    break

            # print(f"{generated_text}")

            for stop_word in stop:
                stop_word = codecs.getdecoder("unicode_escape")(stop_word)[0]
                if stop_word != '' and stop_word in generated_text:
                    generated_text = generated_text[:generated_text.find(stop_word)]

            gc.collect()
            # yield generated_text
            self.model.resetState()
            self.model.setState(self.chatState)
        ################################################
        # # def loadContext(model, ctx, newctx, statex, progressCallBack=lambda x: x):

        # output = self.model.forward(**kwargs)["output"]
        # choices.append({
        #     'text': output,
        #     'index': 0,
        # })

        # response = {
        #     'choices': choices,
        #     'usage': {
        #         'prompt_tokens': prompt_tokens_count,
        #         'completion_tokens': completion_tokens_count,
        #         'total_tokens': prompt_tokens_count + completion_tokens_count
        #     },
        #     "model": self.name,
        #     "kwargs": kwargs,
        # }
        # print(f"response: `{response}`")

        # return response
    def reset(self, **kwargs):
        self.model.resetState()


    #     # self.chatState = self.model.getState()
    # def loadStateIdemponent(self, state_name, **kwargs):
    #     # load a state called 'name'
    #     self.model.setState(self.modelStates[state_name])
    #     #  but if it doesn't exist, create it
    #     # self.modelStates[state_name] = self.model.getState()
    
    def loadStateIdemponent(self, state_name, **kwargs):
        if state_name in self.modelStates:
            self.model.setState(self.modelStates[state_name])
        else:
            self.modelStates[state_name] = self.model.getState()

    def saveState(self, state_name):
        self.modelStates[state_name] = self.model.getState()
    # def make_embedding(text):
    #     parts = []
    #     ctx = tokenizer.tokenizer.encode(text)
    #     ctx.insert(0, 0)
    #     print(ctx)
    #     src_len = len(ctx)
    #     src_ctx = ctx.copy()

    #     init_out = None
    #     init_state = None
    #     for i in range(1):
    #         x = ctx[: src_len]
    #         init_state = model.forward(x, init_state, preprocess_only=True)
    #         parts.append(init_state[-1].detach().cpu().numpy())
    #         gc.collect()
    #         torch.cuda.empty_cache()

    #     # return init_state[-5:0]
    #     return np.concatenate(init_state[-5:])


# fullstack — Today at 2:52 PM
# Thanks. Should that function return parts? I guess I need to experiment with Jupiter and see this data structure.
# undeleted — Today at 3:03 PM
# no, that's something I poorly hacked together while I was still figuring it out
# basic flow is to tokenize the text, prefix the tokens with 0 (<|endoftext|>), then feed the text to the model normally
# init_state[0] contains the embedding
# it seems to anyway, since cosine similarity tests return sane values
# someone correct me if I'm wrong
# fullstack — Today at 3:04 PM
# I missed the 0/<|endoftext|>, see it now
# Mateon1 — Today at 3:14 PM
# Uh, doesn't model.forward() return a tuple of logits and state? init_state[0] in this case is just the final logits. init_state[1] should be the state, which would be useful for an embedding
# undeleted — Today at 3:35 PM
# normally, but preprocess_only makes it only return the state
# Mateon1 — Today at 3:35 PM
# Ah, right, I missed that
# Still, init_state[0] is a very low level embedding, and it's only a small part of the first layer's state
# Concat-ing the last 5 vectors in the state should provide the best embedding, I think
# undeleted — Today at 3:37 PM
# np.concatenate(init_state[0:5]) ?
# Mateon1 — Today at 3:38 PM
# init_state[-5:]
# But yes, I think that should work
# hazardous1222 — Today at 4:10 PM
# init_state.flatten()