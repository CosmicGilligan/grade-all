


from openai import OpenAI

def get_key():
  with open("/home/drkeithcox/llama.key", 'r') as file:
    line = file.read()

  llama_key = line.strip()
  return(llama_key)

def get_client():

  llama_key=get_key()

  client = OpenAI(base_url = "https://integrate.api.nvidia.com/v1", api_key=llama_key)
  return(client)

#completion = client.chat.completions.create(
#  model="nvidia/llama-3.1-nemotron-70b-instruct",
#  messages=[{"role":"user","content":"give me a general overview of the concept of the common good"}],
#  temperature=0.5,
#  top_p=1,
#  max_tokens=1024,
#  stream=True
#)

#for chunk in completion:
#  if chunk.choices[0].delta.content is not None:
#    print(chunk.choices[0].delta.content, end="")

