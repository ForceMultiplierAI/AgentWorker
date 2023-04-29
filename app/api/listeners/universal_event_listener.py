# import ppint, model, requests, time, sseclient
import json
import os
import traceback
import uuid
from sseclient import SSEClient
import time
import requests

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

async def run_super_event_listener(model):
    url = 'https://chat.forcemultiplier.ai/sse/worker'
    print(f"Using FMX_API_KEY: {FMX_API_KEY}")
    headers = {'Accept': 'text/event-stream', 'API_KEY': FMX_API_KEY}

    # TODO replace and test 
    # headers = {}
    # headers.update(get_system_information())
    os_name = os.name
    cpu_name = os.uname().machine
    load_avg = os.getloadavg()
    vram_usage = os.popen('nvidia-smi --query-gpu=memory.used --format=csv,noheader').read().strip()
    vram_total = os.popen('nvidia-smi --query-gpu=memory.total --format=csv,noheader').read().strip()
    vram_usage_percentage = os.popen('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits').read().strip()
    headers['fmx_os_name'] = os_name
    headers['fmx_cpu_name'] = cpu_name
    headers['fmx_load_avg'] = str(load_avg)
    headers['fmx_vram_usage'] = vram_usage
    headers['fmx_vram_total'] = vram_total
    headers['fmx_vram_usage_percentage'] = vram_usage_percentage
    headers['fmx_getnode'] = str(uuid.getnode())

    # using the above variables, generate a unique name like "worker#getnode"
    headers['fmx_worker_name'] = f"worker#{headers['fmx_getnode']}"

    # queue = []
    last_task_id = None
    processing = False
    print("Starting event listener")
    while True:
        try:
            response = with_requests(url, headers, last_task_id)
            print("Reconnecting SSEClient...")  # Add this line
            client = SSEClient(response)       

            for event in client.events():
                print("Event received:", event, event.data)
                if event.event == 'init':
                    print("\U0001F493", end='\r', flush=True)
                    continue

                # if event.event == 'command':
                    # extract the variables from data

                userId, taskId, taskName, status, message, chatId, type = (json.loads(event.data).get("data").get("listen").get("relatedNode").get(x) for x in ["userId", "id", "taskName", "status", "payload.text", "payload.chatId", "payload.type"])

                related_node = json.loads(event.data).get("data").get("listen").get("relatedNode")
                userId, taskId, taskName, status, payload, type = (related_node.get(x) for x in ["userId", "id", "taskName", "status", "payload", "payload.type"])
                message, chatId = payload.get("text"), payload.get("chatId")

                # print all variables
                print(f"All variables: userId: {userId}, taskId: {taskId}, taskName: {taskName}, status: {status}, type: {type}, chatId: {chatId}, message: {message}")

                if event.event != "listen":
                    print(f"Event not listen: {event.event}")
                    continue

                if status != "CREATED":
                    print(f"Status not CREATED: {status}")
                    continue

                # print processing status
                print(f"Processing: {processing}")
                if not processing:
                    # accept job
                    print(f"Accepting job: {taskId}")
                    variables = { 'id' : taskId, 'status': 'PROCESSING' }
                    _response = requests.post(GRAPHQL_URL, headers={'API_KEY':FMX_API_KEY}, json={'query': MUTATE_UPDATE_TASK, 'variables': variables})
                    print(f"Response: {_response}")
                    processing = True

                    completions = model.streaming_completion(
                        prompt=message,
                    )

                    for word in completions:
                        print(word, end='')
                        variables = { 'pId' : chatId, 'pResponse': word }
                        # print(f"Variables: {variables}")
                        _response = requests.post(GRAPHQL_URL, headers={'API_KEY':FMX_API_KEY}, json={'query': MUTATE_UPDATE_CHAT_RESPONSE, 'variables': variables})
                        # response = _response.json()

                    # complete job
                    variables = { 'id' : taskId, 'status': 'DONE' }
                    _response = requests.post(GRAPHQL_URL, headers={'API_KEY':FMX_API_KEY}, json={'query': MUTATE_UPDATE_TASK, 'variables': variables})
                    print(f"Response: {_response}")
                    processing = False
                    last_task_id = taskId

        except Exception as e:
            traceback.print_exc(limit=25)
            print(f"Error outerloop: {e}")
            # traceback.print_exc() 
            # if e = Incorrect password
            print("Handle disconnection")
            time.sleep(1) # 

# def with_requests(url, headers, cookies_file='cookie.txt'):
#     session = requests.Session()
#     cookies = {}
#     #  print with_request args
#     print(f"with_requests args: {url}, {headers}, {cookies_file}")

#     # Load cookies from file if it exists
#     if os.path.exists(cookies_file):
#         with open(cookies_file, 'r') as f:
#             cookies = f.read()
#     else:
#         # Try to connect to the URL and get a cookie
#         print(f"Trying to connect to the URL and get a cookie: {url}, {headers}")
#         response = session.get(url, headers=headers)
#         print(f"response: {response}")
#         if 'set-cookie' in response.headers:
#             cookies = response.headers['set-cookie']
#             print(f"cookies: {cookies}")
#         else:
#             raise Exception('Unable to get a cookie')

#     # Save the cookies to the file
#     with open(cookies_file, 'w') as f:
#         f.write(cookies)

#     print(f"cookies: {cookies}")

#     # Set the cookies in the session
#     session.cookies.update(requests.utils.cookiejar_from_dict(requests.utils.cookiejar_from_cookie_string(cookies)))

#     # Try to reconnect to the URL with the cookie
#     print(f"Trying to reconnect to the URL with the cookie: {url}, {headers}")
#     response = session.get(url, headers=headers)
#     if response.status_code == 302:
#         print(f"code 302 response.status_code: {response.status_code}")
#         # Follow the redirect and get a new cookie
#         print(f"Follow the redirect and get a new cookie: {response.headers['location']}, {headers}")
#         response = session.get(response.headers['location'], headers=headers)
#         if 'set-cookie' in response.headers:
#             cookies = response.headers['set-cookie']
#         else:
#             raise Exception('Unable to get a new cookie')
#         # Save the new cookie to the file
#         with open(cookies_file, 'w') as f:
#             f.write(cookies)
#     else:
#         print(f"response.status_code: {response.status_code}")
#         raise Exception('Unable to connect to the URL')
# return response
# # def with_requests(url, headers):
# #     """Get a streaming response for the given event feed using requests."""
# #     import requests
# #     return requests.get(url, stream=True, headers=headers)
# import requests

# def with_requests(url, headers):
#     """Get a streaming response for the given event feed using requests."""
#     session = requests.Session()
#     # session.cookies.update(cookie)
#     response = session.get(url, stream=True, headers=headers)
#     # if response is redirect, follow the redirect and get a new cookie
#     if response.status_code == 302:
#         print(f"code 302 response.status_code: {response.status_code}")
#         response = session.get(response.headers['location'], headers=headers)
#         if 'set-cookie' in response.headers:
#             cookies = response.headers['set-cookie']
#             # update cookies
#             session.cookies.update(requests.utils.cookiejar_from_dict(requests.utils.cookiejar_from_cookie_string(cookies)))
#             response = session.get(url, stream=True, headers=headers)
#         else:
#             raise Exception('Unable to get a new cookie')
#     return response
# API_KEY = fmx_ruG5drQmR4ZHIJNrcYrZRbPPuW9RnByaYJIvctKa
# def with_requests(url, api_key):
#     """Get a streaming response for the given event feed using requests."""
#     import requests
#     headers = {'API_KEY': api_key}
#     return requests.get(url, stream=True, headers=headers)


            ##################
            # Add these lines to print the raw response content
            # for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            #     if chunk:
            #         print(chunk.strip())
            ##################