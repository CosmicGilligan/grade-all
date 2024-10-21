import openai

def get_key():
    with open("/home/drkeithcox/openai.key", 'r') as file:
        line = file.read()

    api_key = line.strip()
    return(api_key)

def get_client():
    
    api_key = get_key()

    client=openai
    client.api_key = api_key
    return(client)
