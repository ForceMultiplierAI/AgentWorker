import torch
from abc import ABC
from typing import Any, List, Mapping
from rwkvstic.load import RWKV
from rwkvstic.agnostic.backends import TORCH
# from rwkvstic.tokenizer import Tokenizer

model = {}

class RWKVBackend(ABC):
    def __init__(self, name, **config):
        self.name = name
        self.model = {}
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

        output = self.model.forward(**kwargs)["output"]
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
    def reset(self, **kwargs):
        self.model.resetState()

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