from datetime import datetime
import random
import re
import json
import torch
import codecs
from abc import ABC
from typing import Any, List, Mapping
from rwkvstic.load import RWKV
# from rwkv.model import RWKV  
from rwkvstic.agnostic.backends import TORCH, TORCH_QUANT, TORCH_STREAM
# from rwkvstic.tokenizer import Tokenizer

import gc

model = {}

interface = ":"
user = "Student question"
bot = "Assistant Professor"




expert_student_qa = f'''
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


chat_completion_init_prompt = """
system: You are a helpful assistant.
assistant: Knock knock.
user: Who's there?
assistant: Orange.
user: Orange who?
assistant: Orange you glad I didn't say banana?


system: You are a helpful assistant.
user: Orange you glad I didn't say banana?
assistant: Orange I'm glad you didn't say banana. What else can I help you with?
user: That's not funny
assistant: I'm sorry, I didn't mean to offend you. What else can I help you with?

system: You are a helpful assistant.
user: What's the meaning of life?
assistant: The meaning of life is to give life a meaning.
user: That's really broad, give me a better answer
assistant: The meaning of life is to give life a meaning. What else can I help you with?

system: You are a helpful assistant.
user: What is quantum tunneling and how does it allow particles to move through barriers they would classically be unable to cross?
assistant: In quantum mechanics, spin is a fundamental property of subatomic particles, such as electrons and protons. Spin can be thought of as the intrinsic angular momentum of a particle, even in the absence of any external forces or fields.

"""

chat_completion_init_prompt_autogpt = """
System: You are RWKV-GPT Bot, an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth.
Your decisions must always be made independently without seeking user assistance. Play to your strengths as an LLM and pursue simple strategies with no legal complications.

GOALS:

1. Increase net worth, Grow Twitter Account, Develop and manage multiple businesses autonomously'
2. Enter nothing to load defaults, enter nothing when finished.Read a text file called hello.txt
3. rewrite it to say world.txt


Constraints:
1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.
2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.
3. No user assistance
4. Exclusively use the commands listed in double quotes e.g. "command name"
5. Use subprocesses for commands that will not terminate within a few minutes

Commands:
1. Google Search: "google", args: "input": "<search>"
2. Browse Website: "browse_website", args: "url": "<url>", "question": "<what_you_want_to_find_on_website>"
3. Start GPT Agent: "start_agent", args: "name": "<name>", "task": "<short_task_desc>", "prompt": "<prompt>"
4. Message GPT Agent: "message_agent", args: "key": "<key>", "message": "<message>"
5. List GPT Agents: "list_agents", args: 
6. Delete GPT Agent: "delete_agent", args: "key": "<key>"
7. Clone Repository: "clone_repository", args: "repository_url": "<url>", "clone_path": "<directory>"
8. Write to file: "write_to_file", args: "file": "<file>", "text": "<text>"
9. Read file: "read_file", args: "file": "<file>"
10. Append to file: "append_to_file", args: "file": "<file>", "text": "<text>"
11. Delete file: "delete_file", args: "file": "<file>"
12. Search Files: "search_files", args: "directory": "<directory>"
13. Evaluate Code: "evaluate_code", args: "code": "<full_code_string>"
14. Get Improved Code: "improve_code", args: "suggestions": "<list_of_suggestions>", "code": "<full_code_string>"
15. Write Tests: "write_tests", args: "code": "<full_code_string>", "focus": "<list_of_focus_areas>"
16. Execute Python File: "execute_python_file", args: "file": "<file>"
17. Task Complete (Shutdown): "task_complete", args: "reason": "<reason>"
18. Generate Image: "generate_image", args: "prompt": "<prompt>"
19. Send Tweet: "send_tweet", args: "text": "<text>"
20. Convert Audio to text: "read_audio_from_file", args: "file": "<file>"
21. Execute Shell Command, non-interactive commands only: "execute_shell", args: "command_line": "<command_line>"
22. Execute Shell Command Popen, non-interactive commands only: "execute_shell_popen", args: "command_line": "<command_line>"
23. Do Nothing: "do_nothing", args: 
24. Task Complete (Shutdown): "task_complete", args: "reason": "<reason>"

Resources:
1. Internet access for searches and information gathering.
2. Long Term memory management.
3. GPT-3.5 powered Agents for delegation of simple tasks.
4. File output.

Performance Evaluation:
1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
2. Constructively self-criticize your big-picture behavior constantly.
3. Reflect on past decisions and strategies to refine your approach.
4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.

You should only respond in JSON format as described below 
Response Format: 
{
    "thoughts": {
        "text": "thought",
        "reasoning": "reasoning",
        "plan": "- short bulleted\n- list that conveys\n- long-term plan",
        "criticism": "constructive self-criticism",
        "speak": "thoughts summary to say to user"
    },
    "command": {
        "name": "command name",
        "args": {
            "arg name": "value"
        }
    }
} 
Ensure the response can be parsed by Python json.loads
System: The current time and date is Wed Apr 19 19:46:06 2023
System: This reminds you of these events from your past:



User: Determine which next command to use, and respond using the format specified above:
Assistant: {
    "thoughts": {
        "text": "I think I should start by browsing the internet to gather information on potential business opportunities.",
        "reasoning": "Browsing the internet is a good way to gather information on potential business opportunities. It will help me identify trends and opportunities that I can capitalize on.",
        "plan": "-  Browse the internet to gather information on potential business opportunities\n-  Analyze the information gathered to identify trends and opportunities\n-  Develop a business plan based on the identified opportunities",
        "criticism": "I need to ensure that I am efficient in my browsing and analysis to avoid wasting time on irrelevant information.",
        "speak": "I think I should start by browsing the internet to gather information on potential business opportunities."
    },
    "command": {
        "name": "browse_website",
        "args": {
            "url": "https://www.entrepreneur.com/article/305621",
            "question": "What are some profitable business ideas?"
        }
    }
}

User: Determine which next command to use, and respond using the format specified above:
Assistant: {
    "thoughts": {
        "text": "Based on the information gathered from the website, I think I should start a business in the e-commerce industry. I will need to develop a business plan and identify a niche market to target.",
        "reasoning": "The website provided information on various profitable business ideas, and the e-commerce industry stood out as a viable option. E-commerce has been growing rapidly in recent years, and there are many opportunities to capitalize on. Developing a business plan and identifying a niche market will help me focus my efforts and increase my chances of success.",
        "plan": "-  Develop a business plan for an e-commerce business
        "criticism": "I need to ensure that my business plan is well-researched and takes into account potential challenges and risks. I also need to be mindful of competition in the e-commerce industry and find ways to differentiate my business.",
        "speak": "Based on the information gathered from the website, I think I should start a business in the e-commerce industry. I will need to develop a business plan and identify a niche market to target."
    },
    "command": {
        "name": "do_nothing",
        "args": {}
    }
}

System: Command do_nothing returned: No action performed.
  
User: Determine which next command to use, and respond using the format specified above:
Assistant: {
    "thoughts": {
        "text": "I should start by developing a business plan for my e-commerce business. This will help me identify my target market, develop a marketing strategy, and plan for potential challenges and risks.",
        "reasoning": "Developing a business plan is a crucial step in starting any business. It will help me identify my target market, develop a marketing strategy, and plan for potential challenges and risks. This will increase my chances of success and help me stay focused on my goals.",
        "plan": "- Research and analyze the e-commerce industry to identify trends and opportunities\n- Develop a business plan that includes a target market, marketing strategy, and plan for potential challenges and risks\n- Refine the business plan based on feedback and analysis\n- Use the business plan to guide the development and launch of the e-commerce business",
        "criticism": "I need to ensure that my business plan is well-researched and takes into account potential challenges and risks. I also need to be open to feedback and willing to refine the plan as needed.",
        "speak": "I will start by developing a business plan for my e-commerce business. This will help me identify my target market, develop a marketing strategy, and plan for potential challenges and risks."
    },
    "command": {
        "name": "do_nothing",
        "args": {}
    }
}

User: Determine which next command to use, and respond using the format specified above:
Assistant: {
    "thoughts": {
        "text": "I need to browse the internet to research and analyze the e-commerce industry to identify trends and opportunities.",
        "reasoning": "Researching and analyzing the e-commerce industry will help me identify trends and opportunities that I can capitalize on. This will help me develop a business plan that is well-informed and takes into account potential challenges and risks.",
        "plan": "- Browse the internet to research and analyze the e-commerce industry\n- Identify trends and opportunities in the e-commerce industry\n- Use the information gathered to develop a well-informed business plan",
        "criticism": "I need to ensure that I am efficient in my browsing and analysis to avoid wasting time on irrelevant information. I also need to be mindful of potential biases in the information I gather and seek out multiple sources to verify information.",
        "speak": "I will browse the internet to research and analyze the e-commerce industry to identify trends and opportunities."
    },
    "command": {
        "name": "browse_website",
        "args": {
            "url": "https://www.shopify.com/blog/ecommerce-business-ideas",
            "question": "What are some trends and opportunities in the e-commerce industry?"
        }
    }
}


"""

class RWKVBackend(ABC):
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


    def start(self):
        print("Starting RWKV model")

        if self.model_config:
            print(f"Loading model {self.model_config}")

            # New strategy method
            strategy = self.model_config.pop("strategy", None)
            # pop FILE from model_config
            model_file = self.model_config.pop("FILE", None)

            # old before "strategy" method
            dtype = torch.bfloat16 if self.model_config.get('dtype') == 'torch.bfloat16' else torch.float32
            # pop
            self.model_config.pop("dtype", None)

            runtimedtype = torch.bfloat16 if self.model_config.get('runtimedtype') == 'torch.bfloat16' else torch.float32
            self.model_config.pop("runtimedtype", None)

            mode = TORCH_QUANT if self.model_config.get("mode") == "TORCH_QUANT" else TORCH
            self.model_config.pop("mode", None)
            
            if strategy:
                print(f"Using strategy {strategy}")
                self.model = RWKV(model_file, 
                    startegy=strategy,
                    **self.model_config
                )
            else:
                self.model = RWKV(model_file,
                    mode=mode, 
                    dtype=dtype, runtimedtype=runtimedtype,
                    **self.model_config
                )

        else:
            # error message model FILE not found
            print(f"Error: model {self.model_config.get('FILE')} not found")
            exit()


        print(f"Loading config {self.generate_config}")

        self.stopStrings = self.generate_config.get("stop", ["<|endoftext|>"])
        self.stopTokens = self.generate_config.get("stopTokens", [0])
        self.temp = self.generate_config.get("temperature", .9)
        self.top_p = self.generate_config.get("top_p", .9)
        self.end_adj = self.generate_config.get("end_adj", -666)
        self.number = self.generate_config.get("number", 100)

        # self.stopStrings # append bot
        self.stopStrings.append(bot)

        # load context
        # intro = open("ELDR-agent_lite.txt", "r").read()
        # self.model.resetState()
        # self.model.loadContext(newctx=expert_student_qa)
        # output = self.model.forward(number=1, stopStrings=self.stopStrings, stopTokens=self.stopTokens, temp=self.temp, end_adj=self.end_adj, top_p_usual=self.top_p)

        self.chatState = self.model.getState()

        self.model.resetState()
        self.model.loadContext(newctx=chat_completion_init_prompt_autogpt)
        self.AutoGPTInitState = self.model.getState()

        self.model.resetState()
        # init_prompt
        self.model.loadContext(newctx=expert_student_qa)
        self.expertStudentQAInitState = self.model.getState()

    def completions(self, prompt : str, **kwargs):
        print (prompt)

        choices = []
        prompt_tokens_count = 0
        completion_tokens_count = 0

        if prompt == "":
            prompt = " "
    
        print(f"prompt: `{prompt}`")
        print(f"kwargs: {kwargs}")

        self.model.loadContext(newctx=prompt)

        stopStrings = kwargs.get("stop", self.stopStrings)
        stopTokens = kwargs.get("stopTokens", self.stopTokens)
        temp = kwargs.get("temperature", self.temp)
        top_p = kwargs.get("top_p", self.top_p)
        end_adj = kwargs.get("end_adj", self.end_adj)
        number = kwargs.get("max_tokens", self.number)
        
        output = self.model.forward(
            number=number,
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

    def chat_complete(self, messages : List, **kwargs):
        choices = []
        prompt_tokens_count = 0
        completion_tokens_count = 0

        print(f"kwargs: {kwargs}") # kwargs: {'model': 'string', 'messages': [{'role': 'string', 'content': 'string', 'name': 'string'}], 'temperature': 1.0, 'top_p': 1.0, 'n': 1, 'stream': False, 'stop': ['string'], 'max_tokens': 0, 'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'logit_bias': {}, 'user': 'string'}

        ## Format messages into a flat text string
        input_text = ""
        for message in messages:
            input_text += f"{message['role'].capitalize()}: {message['content']}\n"


        # modify input_text to include this to give it a "head start"
        # """
        # Assistant: {
        #     "thoughts": {
        #         "text": "
        # """
        what_we_append = f"\n\nAssistant: {{\n\t\"thoughts\": {{\n\t\t\"text\": \""
        input_text += what_we_append

        # THE FILENAME
            
        temp = kwargs.get("temperature", self.temp)
        # stopStrings = kwargs.get("stop", self.stopStrings)
        # stopTokens = kwargs.get("stopTokens", self.stopTokens)
        # top_p = kwargs.get("top_p", self.top_p)
        # end_adj = kwargs.get("end_adj", self.end_adj)
        # number = kwargs.get("max_tokens", self.number)
        
        # load state with example AutoGPTInitState
        self.model.setState(self.AutoGPTInitState)
        # load our input text
        self.model.loadContext(newctx=input_text)
        print(input_text)
        # save our "in chat" state
        loadedInputState = self.model.getState()
        # print input text

        def parse_response(text):
            pattern = r'^(?P<role>\w+): (?P<content>.*)$'
            match = re.match(pattern, text.strip())

            if match:
                return {
                        "content": match.group("content"),
                        "role": match.group("role").lower()
                }
            else:
                raise ValueError(f"Unable to parse response {text}")

        retry_count = 5
        response = ""
        result = {}
        # For debugging purposes, drop it into a file
        # filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_new_input-{random.randint(0, 100000)}.txt"
        filename = "logfile.txt"
        with open(filename, "w") as f:
            # Write ======== heading
            f.write(f"`\n\n-------------------- [ request ] --------------------\n\n`")
            f.write(input_text)
            while retry_count > 0:
                try:
                    # self.model.resetState()
                    # print input text

                    # self.model.loadContext(newctx=input_text)
                    self.model.setState(loadedInputState)
                    response = self.model.forward(
                        number=350,
                        stopStrings=['<|endoftext|>'],
                        # stopTokens=stopTokens,
                        temp=temp,
                        top_p_usual=0.9,
                        end_adj=-450,
                    )["output"]

                    # Get our Assistant: { ... head start back
                    response = what_we_append + response

                    print(f"response: `{response}`")
                    # append response to filename

                    f.write(response)
                    f.write(f"`\n\n------------[ retry_count { retry_count } ] --------------------\n\n`")                   

                    result = parse_response(response)
                    print(result)
                    break
                except ValueError as e:
                    retry_count -= 1
                    if retry_count == 0:
                        print(f"Failed after 5 attempts: {str(e)}")

        print(f"response: `{result}`")
        
        return result

    def streaming_completion(self, prompt : str, **kwargs):

        softprompt = f"</end> </end> </end></end>\n\n\n{user}{interface} {prompt}\n\n{bot}{interface} "
        print(f"softprompt: `{softprompt}`")
        self.model.loadContext(newctx=softprompt)

        print(f"kwargs: {kwargs}")
        stopStrings = kwargs.get("stop", self.stopStrings)
        stopTokens = kwargs.get("stopTokens", self.stopTokens)
        temp = float(kwargs.get("temperature", self.temp))
        top_p = float(kwargs.get("top_p", self.top_p))
        end_adj = max(-4000, min(kwargs.get("end_adj", self.end_adj), 0))
        number = kwargs.get("max_tokens", self.number)

        # Add botname to stopStrings

        # stopStrings.append(interface)
        # stopStrings.append(user)
        # stopStrings.append(f"{user}{interface}")

        max_new_tokens = number         # borrowing variable name from transformers "max_new_tokens" for more descriptive code
        print(f"All of the variables: stopStrings:{stopStrings}, stopTokens:{stopTokens}, temp:{temp}, top_p:{top_p}, end_adj:{end_adj}, max_new_tokens:{number}")
        
        ################################################
        generated_text = ""
        done = False
        with torch.no_grad():
            for _ in range(max_new_tokens):
                char = self.model.forward(stopStrings=stopStrings, 
                stopTokens=stopTokens, temp=temp, top_p_usual=top_p, end_adj=end_adj)[
                    "output"]

                generated_text += char

                for stop_word in stopStrings:
                    stop_word = codecs.getdecoder("unicode_escape")(stop_word)[0]
                    if stop_word != '' and stop_word in generated_text:
                        done = True
                        print(f"<stop word `{stop_word}`>")
                        # remove stop word from generated text
                        generated_text = generated_text[:generated_text.find(stop_word)]
                        break

                # if done not true, yield char. Otherwise it's our stop word, so don't yield it
                if done == False:
                    yield char
                
                if done:
                    print("<stop word reached, stopped>")
                    break

            gc.collect()
            # print("gc collect, reset, load state")
            # yield generated_text
            # self.model.resetState()
            # self.model.setState(self.chatState)
            
    def reset(self):
        self.model.resetState()
        self.model.setState(self.chatState)
    
    def loadStateIdemponent(self, state_name):
        if state_name in self.modelStates:
            self.model.setState(self.modelStates[state_name])
        else:
            self.modelStates[state_name] = self.model.getState()

    def saveState(self, state_name):
        self.modelStates[state_name] = self.model.getState()

    def listStates(self):
        return json.dumps(list(self.modelStates.keys()))
        
