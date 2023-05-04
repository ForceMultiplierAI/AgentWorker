import json
import os
import traceback
import uuid
import time
import requests
from sseclient import SSEClient
from models import manager

GRAPHQL_URL = os.environ.get('GRAPHQL_ENDPOINT', 'https://chat.forcemultiplier.ai/graphql')
FMX_API_KEY = os.environ.get('FMX_API_KEY', 'fmx_CtJ3gbuH+MIIQtOkQtbxA0FO04QSk/JPMOn7GFhG')

MUTATE_UPDATE_CHAT_RESPONSE = '''
mutation MyMutation($pId: Int!, $pResponse: String!) {
  updateChatResponse(input: {pId: $pId, pResponse: $pResponse}) {
    clientMutationId
  }
}
'''


MUTATE_UPDATE_TASK = '''
mutation MyMutation2($id: Int!, $status: TaskStatus!) {
  updateTask(input: {patch: {status: $status}, id: $id}) {
    clientMutationId
  }
}
'''


### HTTP Keep Alive
session = requests.Session()
session.headers.update({'API_KEY': FMX_API_KEY})

keep_alive_printed = False

token = None
taskId = None

def with_requests(url, headers, last_task_id=None):
    """Get a streaming response for the given event feed using requests."""
    if last_task_id is not None:
        headers['LAST_TASK_ID'] = str(last_task_id)
    return requests.get(url, stream=True, headers=headers)

def get_system_information():
    try:
        os_name = os.name
        cpu_name = os.uname().machine
        load_avg = os.getloadavg()
        vram_usage = os.popen('nvidia-smi --query-gpu=memory.used --format=csv,noheader').read().strip()
        vram_total = os.popen('nvidia-smi --query-gpu=memory.total --format=csv,noheader').read().strip()
        vram_usage_percentage = os.popen('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits').read().strip()
        getnode = str(uuid.getnode())

    except Exception as e:
        os_name, cpu_name, load_avg, vram_usage, vram_total, vram_usage_percentage, getnode = None, None, None, None, None, None, None
        print(f"Error retrieving system information: {e}")

    return {
        key: value
        for key, value in {
            'fmx_os_name': os_name,
            'fmx_cpu_name': cpu_name,
            'fmx_load_avg': str(load_avg),
            'fmx_vram_usage': vram_usage,
            'fmx_vram_total': vram_total,
            'fmx_vram_usage_percentage': vram_usage_percentage,
            'fmx_getnode': getnode,
            
        }.items()
        if value is not None
    }

def handle_event(event, manager):
    global token
    global taskId

    print(f"Handle event: {event}")
    if event.event == "stream":
        data = json.loads(event.data)
        # print(f"Event received: {event.event} event, {len(event.data)} bytes {event.data}")
        payload = data.get("payload")
        message = payload.get("message")
        chatId = int(payload.get("chatId"))
        taskId = int(data.get("id"))
        token = payload.get("token")
        print(f"Accepting job: {taskId}")
        update_task_status(taskId, 'PROCESSING')
        model = manager.get_default_model()
        completions = model.streaming_completion(prompt=message)

        for word in completions:
            print(word, end='')
            # in python how do I check if word is ''
            if word != '':
                update_chat_response(chatId, word)

        update_task_status(taskId, 'DONE')
    if event.event == "request":
        data = json.loads(event.data)
        # get from event.data
        print(f"Event received: {event.event} event, {len(event.data)} bytes {event.data}\n {data}")

        # get all the variables
        id, taskName, status, userId, payload, createdAt, chatId, event = (data.pop(x) for x in ["id", "taskName", "status", "userId", "payload", "createdAt", "chatId", "event"])
        token, method, headers, url_path = (payload.get(x) for x in ["token", "method", "headers", "url_path"])
        print(f"Accepting job: {id} (ignoring params {data}")
        update_task_status(id, 'PROCESSING')
        # print all variables
        print(f"id: {id} taskName: {taskName} status: {status} userId: {userId} payload: {payload} createdAt: {createdAt} chatId: {chatId} event: {event}")

        # sending method to url_path with headers
        URL = "http://localhost:8000" + url_path
        print(f"Sending {method} to {URL} with headers: {headers}....")
        response = requests.request(method, URL, headers=headers)
        
        print(f"[{URL} {method}] Response: {response}")
        print(response.text)

        update_task_status(id, 'DONE')

def update_task_status(taskId, status):
    global keep_alive_printed 
    variables = {'id': taskId, 'status': status}
    print(f"FMX_API_KEY: {FMX_API_KEY}")
    headers={'Authorization': f'Bearer {token}'}
    # headers={'API_KEY': f'{FMX_API_KEY}'}
    response = requests.post(GRAPHQL_URL, headers=headers, json={'query': MUTATE_UPDATE_TASK, 'variables': variables})
    keep_alive_printed = False
    if not keep_alive_printed and response.headers.get('Connection', '').lower() == 'keep-alive':
        print("Keep-Alive enabled")
        keep_alive_printed = True
        # if response wasn't ok
    if response.status_code != 200:
        print("Error updating chat response")
    print(response.text)


def update_chat_response(chatId, word):
    global keep_alive_printed 
    variables = {'pId': chatId, 'pResponse': word}
    # print(f"Updating chat response: {word} for chatId: {chatId}")
    headers={'Authorization': f'Bearer {token}'}
    headers={'API_KEY': f'{FMX_API_KEY}'}
    response = requests.post(GRAPHQL_URL, headers=headers, json={'query': MUTATE_UPDATE_CHAT_RESPONSE, 'variables': variables})
    if not keep_alive_printed and response.headers.get('Connection', '').lower() == 'keep-alive':
        print("Keep-Alive enabled")
        keep_alive_printed = True
    # if response wasn't ok
    if response.status_code != 200:
        print("Error updating chat response")
    # print(response.text)


async def run_event_listener(manager):
    url = 'https://chat.forcemultiplier.ai/sse/worker'
    # headers = get_system_information()
    headers = {}

    headers['API_KEY'] = FMX_API_KEY
    headers['Accept'] = 'text/event-stream'
    headers['fmx_worker_name'] = f"worker#123" #{headers['fmx_getnode']}"

    last_task_id = None

    print(f"Starting event listener {url}")
    while True:
        try:
            response = with_requests(url, headers, last_task_id)
            client = SSEClient(response)       

            for event in client.events():
                print("Event received:", event, event.data)
                if event.event == 'init':
                    # for model in manager.models:
                    #     model.start()
                    print("\U0001F493", end='\r', flush=True)
                    continue

                # if reload
                if event.event == 'reload':
                    print("Reloading models")
                    manager.reload_models()
                    continue

                handle_event(event, manager)

        except Exception as e:
            traceback.print_exc(limit=25)
            print(f"Error outerloop: {e}")
            print("Handle disconnection")
            time.sleep(1) # 

