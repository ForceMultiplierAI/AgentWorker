# Agent


Autonomous OpenAI compatible LLM Asynchronous Worker Queue Agent 

ForceMultiplierAI

An Autonomous OpenAI compatible LLM Asynchronous Worker Queue Agent is a system that manages and processes tasks related to large language models like  OpenAI's API series, but asynchronously. This agent leverages the capabilities of these language models to perform various tasks while maintaining an efficient workflow.

Key components of such an agent include:

Task Queue: A centralized queue that holds the incoming tasks and processes them asynchronously. It ensures that the tasks are completed in an orderly fashion, allowing for scalability and efficient resource management.

Worker Agents: A pool of worker agents that carry out the tasks assigned to them by the queue agent. These workers utilize the LLM (Large Language Model) to perform tasks such as generating responses, providing summaries, or extracting information from text.

Task Prioritization: The agent is capable of prioritizing tasks based on factors such as urgency or importance. This ensures that high-priority tasks are completed first, improving overall efficiency.

Load Balancing: The agent can dynamically allocate tasks to the worker agents based on their current workload and capacity. This ensures that tasks are distributed evenly and prevents any worker from being overwhelmed with tasks.

Fault Tolerance: The system is designed to handle failures and errors. If a worker agent encounters an issue, the task is rescheduled to another worker or placed back in the queue for processing.

Monitoring and Analytics: The agent includes monitoring and analytics capabilities, providing insights into the performance and health of the system. This helps in identifying bottlenecks, optimizing performance, and ensuring the overall effectiveness of the agent.

Such an agent can be deployed in various scenarios, such as customer support chatbots, content generation, or information extraction systems, where the tasks need to be processed asynchronously and efficiently.

## Supported models

OpenAI Passthrough
RWKV-4-Raven-1B5-V10
RWKV-4-Raven-14B-V9
Dolly-V2-12B
Pythia-12B
Vicuna-13B
RWKV-4-Pile-7B-Instruct-Test4
RWKV-4-7B-Alpaca-Finetuned-8Bit-Test2
RWKV-4-7B-Alpaca-Finetuned-8Bit
RWKV-4-7B-Alpaca-Finetuned
RWKV-4-Pile-14B
RWKV-4-Pile-7B

hundreds more see config.json

## How to use

```
$ ./update_runtime.sh
```

```
$ ./run.sh
```

## Configuration

See config.json