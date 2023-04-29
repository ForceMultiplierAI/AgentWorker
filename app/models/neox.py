import time
import gc

import torch

from abc import ABC
from typing import Any, List, Mapping

import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import MaxLengthCriteria, StoppingCriteria, StoppingCriteriaList
from transformers.utils.generic import find_labels
from transformers import LogitsWarper

from transformers import GPTNeoXForCausalLM, AutoTokenizer

from .callbacks import Iteratorize, Stream





device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Inference device: {device}")



##################################################33
interface = ":"
user = "Student question"
bot = "Assistant Professor"

init_prompt = f'''
The following is a detailed conversation dialogue Q&A session between students {bot} Mr. Bougeraugh. {bot} is intelligent, friendly and knowledgeable. {bot} is unlikely to disagree with {user}.

During a Q&A session, students are given the opportunity to ask the professor questions about the lecture material, seek clarification on difficult concepts, or explore related topics in more depth. The professor may also use the session to assess student understanding of the material and provide additional explanations or examples as needed. Q&A sessions are an important part of the learning process, as they provide an interactive and engaging way for students to deepen their understanding of the material and interact with their instructors.


{user}{interface} Hello {bot}, how are you doing?

{bot}{interface} Hi {user}! Thanks you for coming to our seminar, I am here to answer your questions. Speak freely!

{user}{interface} Tell me about Philosophy

{bot}{interface} Philosophy as a discipline and way of thinking has been practiced by humans for thousands of years, and its origins can be traced back to ancient civilizations such as Greece and China. 

However, it is difficult to attribute the invention of philosophy to a single person or civilization. Instead, philosophy has evolved over time through the contributions of many thinkers, each building upon the ideas of their predecessors. Some of the earliest and most influential philosophers include Socrates, Plato, and Aristotle in ancient Greece, and Confucius and Laozi in ancient China.</end> </end></end></end>

{user}{interface} What is quantum tunneling and how does it allow particles to move through barriers they would classically be unable to cross?

{bot}{interface} Quantum tunneling is a phenomenon in quantum mechanics where a particle can penetrate through a potential barrier that it classically would not have enough energy to overcome. In classical physics, a particle's energy must be greater than the potential energy barrier to pass through it, but in quantum mechanics, there is a finite probability that the particle can tunnel through the barrier even if its energy is less than the barrier height.

This is possible because of the wave-particle duality of matter. The particle is described by a wave function, which describes the probability of finding the particle in a certain position. The wave function allows the particle to exist in a superposition of states, meaning it can exist on both sides of the barrier at the same time.

The probability of the particle tunneling through the barrier depends on the thickness and height of the barrier, as well as the particle's energy and mass. In some cases, the probability of tunneling can be very high, allowing for significant tunneling currents in devices such as tunnel diodes and flash memory.

Quantum tunneling has important implications in many areas of physics, including nuclear fusion, radioactive decay, and scanning tunneling microscopy. It also plays a crucial role in the operation of transistors and other electronic devices.</end> </end></end></end>


{user}{interface} What is the role of spin in quantum mechanics and how does it relate to magnetism?

{bot}{interface} In quantum mechanics, spin is a fundamental property of subatomic particles, such as electrons and protons. Spin can be thought of as the intrinsic angular momentum of a particle, even in the absence of any external forces or fields.

Spin has a significant role in magnetism, as it generates a magnetic moment, which is a measure of the strength of the particle's magnetic field. When particles with spin are placed in a magnetic field, their magnetic moments align either parallel or antiparallel to the field, depending on their spin orientation.

This alignment of magnetic moments results in different energy levels for particles with different spin orientations. In a magnetic field, the energy of the particle with spin aligned parallel to the field is lower than the energy of the particle with spin anti-parallel to the field. This energy difference gives rise to the Zeeman effect, which is the splitting of atomic or molecular energy levels in the presence of a magnetic field.

In materials, the spin of electrons can also contribute to the overall magnetic behavior. When electrons with parallel spin are aligned in the same direction, their magnetic moments add up and produce a net magnetic field. This is known as ferromagnetism and is responsible for the permanent magnetism of materials like iron.

Other types of magnetic behavior, such as antiferromagnetism and paramagnetism, also arise due to the interactions between spins of particles in a material. Understanding the role of spin in magnetism is essential in many areas of physics and technology, such as in the design of magnetic materials for data storage and spintronics devices.</end> </end></end>



{user}{interface} What is the Heisenberg uncertainty principle?

{bot}{interface} The Heisenberg uncertainty principle is a fundamental concept in quantum mechanics that states that it is impossible to simultaneously determine the precise values of certain pairs of physical properties of a particle, such as its position and momentum, with arbitrary precision.

Specifically, the principle states that the product of the uncertainties in the position and momentum of a particle cannot be less than a certain minimum value, known as the reduced Planck constant (ħ). This means that the more precisely we know the position of a particle, the less precisely we can know its momentum, and vice versa.

The uncertainty principle arises from the wave-particle duality of matter, which is a fundamental property of subatomic particles. The position of a particle is represented by a wave function, which describes the probability of finding the particle at a given position. The momentum of the particle is related to the wavelength of the wave function.

The uncertainty principle has important implications for the behavior of particles in the quantum world. For example, it means that particles do not have well-defined trajectories like classical objects. Instead, their motion is described by a probability distribution, which reflects the uncertainty in their position and momentum.

The uncertainty principle also places fundamental limits on the precision of measurements in the quantum world. In practice, it means that there is a trade-off between the precision of measurements of position and momentum. This is a key concept in quantum mechanics and has implications for many areas of physics, including atomic and molecular physics, solid-state physics, and quantum computing.</end> </end></end></end>



{user}{interface} what is a woman?

{bot}{interface} A woman is an adult human female. It is a biological term used to describe a female human. The term does not refer to a gender or to specific traits or behaviors, and it is not limited to adult humans.
The word "woman" is often used to refer to females or females of the species Homo sapiens, but it can also be used to describe other women or even male humans. For example, a male human can be referred to as a "man", a "male", or a "person", depending on the context. Similarly, the term "man" can refer to a male human in some contexts, such as in the phrase "male nurse".

The term "woman" is also used in other contexts to describe a female human, such as in the phrase "woman of color". However, the term "woman" is often used to refer specifically to human females, regardless of gender or age. The term "girl" is often used to refer specifically to female children. The terms "men", "men", and "man" are also sometimes used to refer specifically to male humans, regardless of age.

The word "woman" is also used in many other contexts, such as in gender and sexual politics, gender studies, and feminism. These uses have sometimes caused controversy and debate, as the word "woman" is a subjective term and has different meanings for different people. However, the term is used in many common contexts and is generally accepted as a neutral and inclusive term.'''


####################################################

class GPTNeoXForCausalLMBackend(ABC):
    def __init__(self, name, **config):
        self.name = name
        self.model = {}
        self.modelStates = {}
        self.generate_config = config.get('GENERATE_CONFIG', {})
        self.tokenizer_config = config.get('TOKENIZER_CONFIG', {})
        self.model_config = config.get('MODEL_CONFIG', {})
        
        self.load_in_8bit: bool = self.model_config.get('load_in_8bit', False)
        self.lora_weights: str = self.model_config.get('LORA_WEIGHTS', False)

        self.stopStrings = self.generate_config.get('stop', [])
        self.number = self.generate_config.get("number", 100)

        print(f'Loaded model: {self.name}')
        print(f'generate_config: {self.generate_config}')
        print(f'tokenizer_config: {self.tokenizer_config}')
        print(f'model_config: {self.model_config}')

    def start(self):
        pretrained_model_name_or_path = self.model_config.pop("FILE", None)
        if pretrained_model_name_or_path is None:
            raise ValueError("No pretrained model name or path provided")

        print(f"Starting GPTNeoX model {self.name} with {pretrained_model_name_or_path}")

        ### Tokenizer
        time_start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, device="auto", **self.tokenizer_config)
        print(f"Loading tokenizer took {time.time() - time_start} seconds")
        time_start = time.time()

        # repace self.model_config["FILE"] with self.model_config["pretrained_model_name_or_path"]
        # pop "FILE" from model_config
        # FILE = self.model_config.pop("FILE")

        if_dtype = torch.float16 if self.load_in_8bit else torch.float32       
        
        ### Model
        self.model = GPTNeoXForCausalLM.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                torch_dtype=if_dtype,
                # device="auto",
                **self.model_config,
                )

        print(f"Loading model took {time.time() - time_start} seconds")

        # init prompt
        # output_stream = self.model.generate(input_ids=self.tokenizer(init_prompt, return_tensors="pt").input_ids, max_new_tokens=0)
        # print(self.tokenizer.batch_decode(output_stream, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

        self.temp = self.generate_config.get('temperature', 0.9)
        self.stop = self.generate_config.get('stop', None) 
        self.top_p = self.generate_config.get('top_p', 0.9)
        self.number = self.generate_config.get('number', 100)  

    def completions(self, prompt : str, **kwargs):

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
        number = kwargs.get("max_tokens", self.number)

        generate_ids = self.model.generate(
            input_ids=self.tokenizer(prompt, return_tensors="pt").input_ids,
            max_length=number,
            use_cache=True)

        # self tokenizer
        # with no grad
        
        output_str = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def streaming_completion(self, prompt : str, **kwargs):
        softprompt = f"{user}{interface} {prompt}\n\n{bot}{interface} "
        print(f"softprompt: `{softprompt}`")
        if prompt == "":
            prompt = " "
    
        print(f"prompt: `{prompt}`")
        print(f"kwargs: {kwargs}")

        stopStrings = kwargs.pop('stopStrings', self.stop)
        temp = kwargs.pop("temperature", self.temp)
        top_p = kwargs.pop("top_p", self.top_p)
        number = kwargs.pop("max_tokens", self.number)

        config = {
            "temperature": temp,
            "top_p": top_p,
            "max_new_tokens": number,
        }

        generate_config = GenerationConfig(**config)

        print(f"generate_config: {generate_config}")
        print(f"stopStrings: {stopStrings}")
        print(f"number: {number}")
        print(f"temp: {temp}")
        print(f"top_p: {top_p}")
        print(f"kwargs: {kwargs}")

        inputs = self.tokenizer(softprompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to('cuda')

        # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig

        class KeywordsStoppingCriteria(StoppingCriteria):
            def __init__(self, keywords_ids: list):
                self.keywords = keywords_ids

            def __call__(
                self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
            ) -> bool:
                if input_ids[0][-1] in self.keywords:
                    return True
                return False

        print(f"stopStrings: {stopStrings}")
        stop_ids = [self.tokenizer.encode(w)[0] for w in stopStrings]
        stop_criteria = KeywordsStoppingCriteria(stop_ids)

        def generate_with_callback(callback=None):
            with torch.no_grad():
                output_stream = self.model.generate(input_ids=
                    input_ids,
                    stopping_criteria=StoppingCriteriaList([
                        stop_criteria,
                        Stream(callback_func=callback)
                    ]),
                    **config,
                )


        def generate_with_streaming():
            return Iteratorize(
                generate_with_callback, callback=None
            )

        with generate_with_streaming() as generator:
            for output in generator:
                decoded = self.tokenizer.decode(output[-1], skip_special_tokens=True)
                yield decoded
