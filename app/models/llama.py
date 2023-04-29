# https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaModel.forward.use_cache
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

import json
import time
import torch
import codecs
from abc import ABC
from typing import Any, List, Mapping


import gc
import transformers
from transformers.generation.logits_process import LogitsProcessorList

from transformers.generation.stopping_criteria import MaxLengthCriteria, StoppingCriteria, StoppingCriteriaList
from transformers.utils.generic import find_labels

from transformers.generation.configuration_utils import GenerationConfig

model = {}

from transformers import AutoTokenizer, LlamaForCausalLM
from .callbacks import (Iteratorize, Stream)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Inference device: {device}")

# self.model_config["FILE"] = "/home/shazam/llama-converted-weights"
# model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
# tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

# prompt = "Hey, are you consciours? Can you talk to me?"
# inputs = tokenizer(prompt, return_tensors="pt")

# Generate
# generate_ids = model.generate(inputs.input_ids, max_length=30)
# tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


from transformers import LogitsWarper
class CallbackLogitsWarper(LogitsWarper):
    def __init__(self, tokenizer, callback):
        self.tokenizer = tokenizer
        self.callback = callback
        self.res_tokens = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        self.res_tokens.append(input_ids[0][-1])
        result = self.tokenizer.decode(self.res_tokens).lstrip()
        self.callback(result) # return current generation already in words format back
        return scores

class LLaMaCausalLMBackend(ABC):
    def __init__(self, name, **config):
        self.name = name
        self.model = {}
        self.modelStates = {}
        self.generate_config = config.get('GENERATE_CONFIG', {})
        self.tokenizer_config = config.get('TOKENIZER_CONFIG', {})
        self.model_config = config.get('MODEL_CONFIG', {})
        
        self.load_8bit: bool = self.model_config.get('LOAD_8BIT', True)
        # self.base_model: str = self.model_config.get('BASE_MODEL', "gpt2")
        self.lora_weights: str = self.model_config.get('LORA_WEIGHTS', False)

        self.stopStrings = self.generate_config.get('stop', [])
        self.number = self.generate_config.get("number", 100)

        print(f'Loaded model: {self.name}')
        print(f'generate_config: {self.generate_config}')
        print(f'tokenizer_config: {self.tokenizer_config}')
        print(f'model_config: {self.model_config}')

        print(f'LOAD_8BIT: {self.load_8bit}')
        # print(f'BASE_MODEL: {self.base_model}')
        print(f'LORA_WEIGHTS: {self.lora_weights}')
# # Quantized model
# elif shared.args.wbits > 0:
#     from modules.GPTQ_loader import load_quantized

#     model = load_quantized(model_name)
    # def load_quant(model, checkpoint, wbits, groupsize):
    #     from transformers import LlamaConfig, LlamaForCausalLM 
    #     config = LlamaConfig.from_pretrained(model)
    #     def noop(*args, **kwargs):
    #         pass
    #     torch.nn.init.kaiming_uniform_ = noop 
    #     torch.nn.init.uniform_ = noop 
    #     torch.nn.init.normal_ = noop 

    #     torch.set_default_dtype(torch.half)
    #     transformers.modeling_utils._init_weights = False
    #     torch.set_default_dtype(torch.half)
    #     model = LlamaForCausalLM(config)
    #     torch.set_default_dtype(torch.float)
    #     model = model.eval()
    #     layers = find_labels(model)
    #     for name in ['lm_head']:
    #         if name in layers:
    #             del layers[name]
    #     make_quant(model, layers, wbits, groupsize)

    #     print('Loading model ...')
    #     if checkpoint.endswith('.safetensors'):
    #         from safetensors.torch import load_file as safe_load
    #         model.load_state_dict(safe_load(checkpoint))
    #     else:
    #         model.load_state_dict(torch.load(checkpoint))
    #     model.seqlen = 2048
    #     print('Done.')

        # return model
        
    def start(self):
        print("Starting LLaMa model")
        # // https://github.com/search?q=LlamaForCausalLM+8bit&type=code
        # start time
        time_start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config["FILE"])
        # how long?
        print(f"Loading tokenizer took {time.time() - time_start} seconds")
        # reset time
        time_start = time.time()
        self.model = LlamaForCausalLM.from_pretrained(self.model_config["FILE"],  
        torch_dtype=torch.float16,
        # low_cpu_mem_usage=True,
        # use_cache=False,
        # gradient_checkpointing=True,
        device_map='auto',
        load_in_8bit=self.load_8bit)
        # how long?
        print(f"Loading model took {time.time() - time_start} seconds")
        self.model.sqllen = 2048

        # save pretrained to get one time token file
        # https://github.com/huggingface/transformers/issues/22669
        self.tokenizer.save_pretrained("/tmp/llama_tokenizer")

        # test model
        output_stream = self.model.generate(input_ids=self.tokenizer("Hello, my name is", return_tensors="pt").input_ids, max_length=30)
        print(self.tokenizer.batch_decode(output_stream, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

        self.model.eval()


    def streaming_completion(self, prompt : str, **kwargs):
        if prompt == "":
            prompt = " "
    
        print(f"prompt: `{prompt}`")
        print(f"kwargs: {kwargs}")

        stopStrings = self.generate_config.get('stop', [])
        temp = self.generate_config.get('temperature', 0.9)
        top_p = self.generate_config.get('top_p', 0.9)
        number = kwargs.get("max_tokens", self.number)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to('cuda')



#### Broken: https://github.com/huggingface/transformers/issues/22436
        # Extremely annoying undocumented hugging face api..
        class StopKeyStoppingCriteria(transformers.StoppingCriteria):
            def __init__(self, stop_ids: list, starting_idx: int = 0):
                super().__init__()
                self.stop_ids = stop_ids
                self.starting_idx = starting_idx

            def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor) -> bool:
                for sample in input_ids:
                    trimmed_sample = sample[self.starting_idx:]

                    for i in range(len(self.stop_ids)):
                        # Can't unfold, output is still too tiny. Skip.
                        if trimmed_sample.shape[-1] < self.stop_ids[i].shape[-1]:
                            continue
                        for window in trimmed_sample.unfold(0, self.stop_ids[i].shape[-1], 1):
                            if torch.all(torch.eq(self.stop_ids[i][0], window)):
                                print(f"!!!!!Stop reached!!!!!")
                                return True
                return False 

        def inside_loop_encode(prompt):
            # is truncation even a word?
            _input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, add_special_tokens=False).input_ids
            print(f"input_ids {_input_ids}")
            return _input_ids.cuda()

        stopping_criteria_list = StoppingCriteriaList()

        if type(stopStrings) is list and len(stopStrings) > 0:
            print(f"Using stopStrings {stopStrings}")
            stop_token_input_ids = [inside_loop_encode(w) for w in stopStrings]
            stopping_criteria_list.append(StopKeyStoppingCriteria(stop_ids=stop_token_input_ids, starting_idx=len(input_ids[0])))


        # append max_tokens as stopping criteria
        # max_length = 
        print(f"Length of number + len(input_ids[0]) = {number+len(input_ids[0])}")
        print(f"Number {number}")
        # stopping_criteria_list.append(MaxLengthCriteria(max_length=number+len(input_ids[0])))


        # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig

        config = {
            "temperature": temp,
            "top_p": top_p,
        }

        generate_config = GenerationConfig(**config)

        generate_params = {
            # "input_ids": input_ids,
            "generation_config" : generate_config,
            "stopping_criteria": stopping_criteria_list,
            # "logits_processor": logits_processor,
            # "output_scores": True,
            # "return_dict_in_generate": True,
            "max_new_tokens": number,
        }
# https://github.com/tloen/alpaca-lora/blob/e2ed209d3bf83ad8963d185dc97d6cc81bb51bf8/generate.py
        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator,
        # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

        def generate_with_callback(callback=None, **kwargs):
            kwargs.setdefault(
                "stopping_criteria", transformers.StoppingCriteriaList()
            )
            kwargs["stopping_criteria"].append(
                Stream(callback_func=callback)
            )
            print(f"kwargs: {kwargs}")

            with torch.no_grad():
                output_stream = self.model.generate(input_ids=
                    input_ids,
                # self.tokenizer("Hello, my name is", return_tensors="pt").input_ids,
                    # max_length=30,
                    **kwargs)
                # print(self.tokenizer.batch_decode(output_stream, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

        def generate_with_streaming(**kwargs):
            # print(f"Generate with streaming....")
            return Iteratorize(
                generate_with_callback, kwargs, callback=None
            )

        total_generated_so_far = 0
        total_actually_text = ''

        with generate_with_streaming(**generate_params) as generator:
            count = 1
            stop_token_input_ids = [inside_loop_encode(w) for w in stopStrings]
            for output in generator:

                if output[-1] in [stop_token_input_ids]:
                    print(f"STOPPED STOP TOKEN: output[-1] {output[-1]} in stop_token_input_ids")
                    _flush_output = output[-count:]
                    decoded = self.tokenizer.decode(_flush_output, skip_special_tokens=True)
                    yield decoded
                    break

                if output[-1] in [self.tokenizer.eos_token_id]:
                    print(f"STOPPED EOS: output[-1] in [self.tokenizer.eos_token_id]")
                    _flush_output = output[-count:]
                    decoded = self.tokenizer.decode(_flush_output, skip_special_tokens=True)
                    yield decoded
                    break

                if( count == 8):
                    # get the last 8 tokens
                    _flush_output = output[-8:]
                    decoded = self.tokenizer.decode(_flush_output, skip_special_tokens=True)
                    print(f"decoded: `{decoded}`")
                    count = 1
                    yield f" {decoded}"
                else:
                    count += 1
                    continue

                # # decode last token
                # last_token = output[-1]
                # decoded_last_token = self.tokenizer.decode(last_token, skip_special_tokens=False,
                #     clean_up_tokenization_spaces=False)
                
                # add space to decoded_last_token
                # decoded_last_token = f" {decoded_last_token}"

                total_generated_so_far += 1
                if total_generated_so_far > number:
                    print(f"STOPPED NUMBER: total_generated_so_far {total_generated_so_far} > number {number}")
                    # decode reaminder at count
                    # get the last count tokens
                    _flush_output = output[-count:]
                    decoded = self.tokenizer.decode(_flush_output, skip_special_tokens=True)
                    yield decoded
                    break

                # yield decoded
                # yield f" {decoded}"


        # with generate_with_streaming(**generate_params) as generator:
        #     for output in generator:
        #         new_tokens = len(output) - len(input_ids[0])

        #         last_token = output[-1]
        #         decoded_last_token = self.tokenizer.decode(last_token, skip_special_tokens=True, 
        #             # clean_up_tokenization_spaces=True
        #         )

        #         # take the second of the last decoded last token
        #         second_last_token = output[-3:]
        #         print(f"Length of second_last_token: {len(second_last_token)}")
        #         decoded_second_last_token = self.tokenizer.decode(second_last_token, skip_special_tokens=True,
        #             # clean_up_tokenization_spaces=True
        #         )

        #         print(f"Second last token: `{decoded_second_last_token}`")

        #         # count spaces, if there is one less space from the previous one,  trim decoded_last_token
        #         if decoded_second_last_token.count(' ') > prev_count:
        #         # if decoded_second_last_token.count(' ') <  0:
        #             print(f"Failsafe reached, stripping decoded_last_token: decoded_second_last_token.count(' ') {decoded_second_last_token.count(' ')} > count {prev_count}  `{decoded_last_token}` to `{decoded_last_token.strip()}`")
        #             decoded_last_token = decoded_last_token.strip()

        #         prev_count = decoded_second_last_token.count(' ')

        #         if output[-1] in [self.tokenizer.eos_token_id]:
        #             print(f"STOPPED EOS: output[-1] in [self.tokenizer.eos_token_id]")
        #             break

        #         # add new_tokens to a counter and if it is more than number, break
        #         total_generated_so_far += 1
        #         if total_generated_so_far > number:
        #             print(f"STOPPED NUMBER: total_generated_so_far {total_generated_so_far} > number {number}")
        #             break


        #         # decode the last 8 tokens
        #         decoded_new_tokens = self.tokenizer.decode(output[-8:])
        #         yield decoded_last_token

        # with generate_with_streaming(**generate_params) as generator:
        #     for output in generator:
        #         # print(f"output: `{output}`")
        #         new_tokens = len(output) - len(input_ids[0])
        #         reply = self.tokenizer.decode(output[-new_tokens:])

        #         last_token = output[-1]
        #         decoded_last_token = self.tokenizer.decode(last_token, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        #         # compare reply and total_actually_text, count the number of spaces
        #         # if the difference is greater than 1, then we we trim reply of all white spaces left and right
        #         reply_spaces = reply.count(' ')
        #         total_actually_text_spaces = total_actually_text.count(' ')
        #         if abs(reply_spaces - total_actually_text_spaces) > 1:
        #         # if (total_actually_text_spaces > reply_spaces):
        #             decoded_last_token = decoded_last_token.strip()

        #         total_actually_text = reply


        #         if output[-1] in [self.tokenizer.eos_token_id]:
        #             print(f"STOPPED EOS: output[-1] in [self.tokenizer.eos_token_id]")
        #             break

        #         # add new_tokens to a counter and if it is more than number, break
        #         total_generated_so_far += 1
        #         if total_generated_so_far > number:
        #             print(f"STOPPED NUMBER: total_generated_so_far {total_generated_so_far} > number {number}")
        #             break


        #         # decode the last 8 tokens
        #         decoded_new_tokens = self.tokenizer.decode(output[-8:])
        #         yield decoded_last_token

                # print(f"Yielding decoded_output: `{decoded_output}`")
                # yield decoded_output
                
                # no plan, yield only new_tokens of decoded_output 
                # yield decoded_output[-new_tokens:]
        # return

                # reply = self.tokenizer.decode(output)
                # yield decoded_output
                # yield prompter.get_response(decoded_output)
        # return  # early return for stream_output

        # def generate_with_callback(callback=None, **kwargs):
        #     kwargs['stopping_criteria'].append(Stream(callback_func=callback))
        #     # print(f"BING GENERATE WITH CALLBACK {kwargs}")
        #     # clear_torch_cache()
        #     with torch.no_grad():
        #         print(f"BING MADE IT HEREself.model.generate")
        #         return self.model.generate(**kwargs)
        #         # output = 
        #         # print(f"BING GENERATE WITH OUTPUT {output}")
        #         # yield output

        # def generate_with_streaming(**kwargs):
        #     print(f"BING BOT IS GOOD {kwargs}")
        #     return Iteratorize(generate_with_callback, kwargs, callback=None)

        # def formatted_outputs(reply, model_name):
        #     return reply

        # with generate_with_streaming(**generate_params) as generator:
        #     print(f"BING!")
        #     for output in generator:
        #         # print(f"HELLO WORLD BING bot is good!")
        #         # print(output)
        #         new_tokens = len(output) - len(input_ids[0])
        #         reply = self.tokenizer.decode(new_tokens)
        #         print(f"BING REPLY {reply}")
        #         # if output[-1] == self.tokenizer.eos_token_id:
        #             # break
        #         yield "word" #reply

        # while not stopping_criteria(input_ids, scores):
        #     outputs = self.model.forward(input_ids, past_key_values)
        #     past_key_values = outputs.past_key_values
        #     logits = outputs.logits.softmax(dim=-1)
        #     output = self.tokenizer.decode(outputs.sequences[0])
        #     scores = logits_processor(logits)
        #     input_ids = logits.argmax(dim=-1) # 
        #     yield output


        # generate_ids = self.model.generate(
            # input_ids=self.tokenizer(prompt, return_tensors="pt").input_ids,
            # max_length=number,
            # logits_warper=CallbackLogitsWarper(self.tokenizer, callback)
            # )

        # self tokenizer
        # with no grad
        
        # output_str = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.


# stopping_criteria (StoppingCriteriaList, optional) — An instance of StoppingCriteriaList. List of instances of class derived from StoppingCriteria used to tell if the generation loop should stop.
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria

# class KeywordsStoppingCriteria(StoppingCriteria):
#     def __init__(self, keywords_ids:list):
#         self.keywords = keywords_ids

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         if input_ids[0][-1] in self.keywords:
#             return True
#         return False

#     # if 'stop' in generate_args:
#     if 'stop' in generate_args:
#         stop_words = generate_args['stop']
#         generate_args.pop('stop', None)
#         # stop_words = ['}', ' }', '\n']
#         stop_ids = [tokenizer.encode(w)[0] for w in stop_words]
#         stop_criteria = KeywordsStoppingCriteria(stop_ids)
#         generate_args['stopping_criteria'] = StoppingCriteriaList([stop_criteria])


        

    def completions(self, prompt : str, **kwargs):
        choices = []
        prompt_tokens_count = 0
        completion_tokens_count = 0

        if prompt == "":
            prompt = " "
    
        print(f"prompt: `{prompt}`")
        print(f"kwargs: {kwargs}")

        stopStrings = self.generate_config.get('stop', [])
        # stopTokens = kwargs.get("stopTokens", self.stopTokens)

        # temp = kwargs.get("temperature", self.temp)
        temp = self.generate_config.get('temperature', 0.9)
        # top_p = kwargs.get("top_p", self.top_p)
        top_p = self.generate_config.get('top_p', 0.9)

        # end_adj = kwargs.get("end_adj", self.end_adj)
        # number = kwargs.get("max_tokens", self.number)

        generate_ids = self.model.generate(
            input_ids=self.tokenizer(prompt, return_tensors="pt").input_ids,
            max_length=number)

        # self tokenizer
        # with no grad
        
        output_str = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
