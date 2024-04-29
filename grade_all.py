from operator import itemgetter
#from wa_html import queryBot
import openai
#import gradio as gr
import os
from operator import itemgetter
import pandas as pd
import numpy as np
import tiktoken
from ast import literal_eval
from scipy.spatial.distance import cosine
import os
from bs4 import BeautifulSoup
#import sys

client=openai

domain = "text/Transcripts/"
subdomain109 = "text/Transcripts/109/"
subdomain110 = "text/Transcripts/110/"
full_url = "text/Transcripts/"
suburl109 = "text/Transcripts/109/"
suburl110 = "text/Transcripts/110/"
max_tokens = 500
lecStrList = []
shortened = []
df = []
df_embeddings = []
df_similarities = []
prmtTitleList=[]
qFileList=[]
US1List=[]
US2List=[]
gradeMode=True
examMode=False
directory = './submissions/'
entryList=[]

def extract_content(html_content):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract name enclosed in <h1>
    name = soup.find('h1').get_text(strip=True)
    
    # Initialize a variable to hold the extracted text
    extracted_text = ''
    
    # Find the <h1> tag, then start searching for subsequent text containers
    for sibling in soup.find('h1').next_siblings:
        if sibling.name in ['p', 'ol', 'div']:  # Add any other tags you expect to contain the text
            extracted_text += sibling.get_text(" ", strip=True) + ' '
        # Optional: Break out of the loop if a certain tag is encountered indicating the end of the relevant text
    
    # Clean up the extracted text
    extracted_text = extracted_text.strip()
    
    return name, extracted_text

def makeEntryList():
    for filename in os.listdir(directory):
        if filename.endswith('.html'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                thisEntry=[]
                html_content = file.read()
                name, text = extract_content(html_content)
                i=name.find(":")
                strEnd=len(name)
                shortName=name[i+2:strEnd]
                thisEntry.append(shortName)
                thisEntry.append(text)
                thisEntry.append("\n")
                entryList.append(thisEntry)
    return entryList

# search embedded docs based on cosine similarity

def calculate_similarity(embedding1, embedding2):
    # Using 1 - cosine distance to get cosine similarity
    result = 1 - cosine(embedding1, embedding2)
    return result

def get_embedding(text, model="text-embedding-ada-002"):
   return client.embeddings.create(input = [text], model='text-embedding-ada-002').data[0].embedding
              
def search_docs(df, user_query, top_n=3, to_print=True):
    embedding = get_embedding(
        user_query,
        model="text-embedding-ada-002"
    )
    
    df_similarities=df.copy()
    df_similarities["similarities"] = df.embeddings.apply(lambda x: calculate_similarity(embedding, x))

    res = (
        df_similarities.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    if to_print:
        print(res)
    return res

def askDB(question):
    # get the standard system prompt    
    filename=full_url+"query.prmt"
    f=open(filename)
    promptList=f.readlines()
    f.close()
    qStr=promptList[0]+" "+question+" "
    initial_prompt = qStr
    result = ''
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    df_tok=df.copy()
    df_tok['n_tokens'] = df['text'].apply(lambda x: len(tokenizer.encode(x)))

    df_embeddings = df_tok.copy()
    df_embeddings['ada_v2_embedding'] = df_tok.text.apply(lambda x: openai.embeddings.create(input=[x], model='text-embedding-ada-002').data[0].embedding)
    df_similarities = df_embeddings.copy()

    # search the embeddings for the most appropriate source

    res = search_docs(df_embeddings, question, top_n=1)
    ai_question = question
    context= res.text.values    

#    initial_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly."

    combined_prompt = initial_prompt + str(context) + "Q: " + ai_question
    completion = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
             {"role": "user", "content": combined_prompt}
        ]
    )
    answer = completion.choices[0].message 

def split_into_many(tokenizer, text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1
        
    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

# Create a dataframe from the list of texts
def createDataframe():
    global shortened
    global df
    global df_embeddings
    global df_similarities

    df = pd.DataFrame(lecStrList, columns = ['fname', 'text'])
    df.to_csv('embeddings.csv')
    df.head()

    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    df = pd.read_csv('embeddings.csv', index_col=0)
    df.columns = ['title', 'text']

    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    # Visualize the distribution of the number of tokens per row using a histogram
    df.n_tokens.hist()

    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1].text is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1].n_tokens > max_tokens:
            shortened += split_into_many(tokenizer,row[1].text)
        
        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append(row[1].text)

    df = pd.DataFrame(shortened, columns = ['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    df.n_tokens.hist()

    # Note that you may run into rate limit issues depending on how many files you try to embed
    # Please check out our rate limit guide to learn more on how to handle this: https://platform.openai.com/docs/guides/rate-limits
#    df['embeddings'] = df.text.apply(lambda x: client.embeddings.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    df['embeddings'] = df.text.apply(lambda x: client.embeddings.create(input = x, model='text-embedding-ada-002').data[0].embedding)
                                               
    df.to_csv('tokens.csv')
    df.head()

    df=pd.read_csv('tokens.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)

    df.head()

def sort_by_number(string_list):
  """Sorts a list of strings in the format 'sss: xxx' by the numeric part (xxx) in ascending order.

  Args:
      string_list: A list of strings in the format 'sss: xxx'.

  Returns:
      A new list with the strings sorted by the numeric part (xxx) in ascending order.
  """

  # Function to extract the numeric part (xxx) from a string
  def get_number(text):
    return int(text.split(':')[-1])

  # Sort the list using the extracted numeric part as the key
  return sorted(string_list, key=get_number)

def crawl():
    ospath=os.fspath(full_url)
    osdir=os.listdir(full_url)
    dirTree=[]
    max_tokens = 500
    max_len = 1800
    shortened = []
    local_domain = domain
    queue=[]
    seen=set([])
    global qFileList
    global lecStrList
    global US1List
    global US2List

    for (root,dirs,files) in os.walk(domain, topdown=True): 
        dirTree.append(dirs)      
        while files:
            foundQ=False
            fileList=files.pop(0)
            x=fileList.find("q.txt")
            if(x>=0):
                newFile="./"+root+"/"+fileList
                f=open(newFile)
                qinFileList=f.readlines()
                f.close()
                qFileList.append(qinFileList)
                foundQ=True
            x=fileList.find(".txt")
            if(( x>=0) and (foundQ==False)):
                newFile="./"+root+"/"+fileList
                titleList=[newFile]
                f=open(newFile, "r")
                inputList=(f.readlines())
                f.close
                lStr=inputList[2]
                l1Str=inputList[1].replace("\n","")
                lnStr=l1Str+lStr[:3]
                x=inputList[1].find("US1")
                if(x>=0):                    
                    US1List.append(lnStr)
                else:
                    US2List.append(lnStr)

                removeStr=(inputList[1].replace("\n","")+inputList[2].replace("\n",""))
                idStr=inputList[2].replace("\n","")
                idStr1=idStr.replace("-","")
                idStr2=idStr1.replace(" ","")
                id=int(idStr2)
                titleList.append(id)
                titleStr=removeStr+inputList[3]
                lecStr=inputList[4].replace("\n","")
                tlecStrList=[]
                tlecStrList.append(titleStr)
                tlecStrList.append(lecStr)
                lecStrList.append(tlecStrList)

    sorted_list=sorted(lecStrList,key=itemgetter(1))
    lecStrList=sorted_list
    sorted_list=sorted(qFileList,key=itemgetter(0))
    qFileList=sorted_list
    sorted_list=sort_by_number(US1List)
    US1List=sorted_list
    sorted_list=sort_by_number(US2List)
    US2List=sorted_list
    createDataframe()

model="US1"
sStr=""

"""
def rs1_change(c):
    global sStr
    sStr=c
    print(sStr)
    return(sStr)
#    return gr.Dropdown(choices=choices[c], interactive=True) # Make it interactive as it is not by default

def rs2_change(c):
    global sStr
    sStr=c
    print(sStr)
    return(sStr)
#    return gr.Dropdown(choices=choices[c], interactive=True) # Make it interactive as it is not by default

def model_change(value):
    global model
    if(value=="109"):
        model="US1"
        return
    model="US2"
    return

def progress_bar(progress, total):
    length=40
    
    Displays a simple text-based progress bar.

    Args:
        progress (int): Current progress.
        total (int): Total number of steps.
        length (int): Length of the progress bar in characters.

    percent = progress / float(total) * 100
    filled_length = int(length * percent // 100)
    bar = '█' * filled_length + '-' * (length - filled_length)
    pctStr=round(percent, 1)
    progressStr="\rProgress: |"+bar+"| "+str(pctStr)+"%"
    sys.stdout.write(progressStr)
    sys.stdout.flush()  # Ensure the text is displayed 
    return progressStr
"""

def getPromptStr(course,base):
    print("Begin grading completions...\n")
    filename=full_url+"query.prmt"
    f=open(filename)
    promptList=f.readlines()
    f.close()
    basetitleStr=domain+str(course)+"/"+str(base)
    titleStr=basetitleStr+".lec"
    qtitleStr=basetitleStr+"q.txt"
    filename=titleStr
    f=open(filename)
    lecStr=f.readline()
    f.close()

    baseStr=str(base)
    if course == 109:
        rootStr=suburl109
    else:
        rootStr=suburl110
    for x in qFileList:
        y=x[0].find(baseStr)
        if y>=0:
            filename=rootStr+x[1].replace("\n","")
            f=open(filename,"r")
            qStrList=f.readlines()
            f.close()
            qLen=len(qStrList)
            qStr=""
            x=2
            while x<qLen:
                qStr=qStr+qStrList[x]
                x+=1
            prompt1=promptList[0]
            prompt2=promptList[1]

    promptStr=prompt1+" "+lecStr+" "+prompt2+" "+qStr+"\n"
    return(promptStr)
"""
def submitPrompt(course,base):
    global domain
    global full_url
    promptList=[]
    global progressTxt

    promptStr=getPromptStr(course)


def create_gradio_interface():
    global grProgressBar
    with gr.Blocks(title="Professor Cosmic's Magic Lecture Grader") as demo:    
        gr.Markdown("# **Professor Cosmic's Magic Lecture Grader**")
        with gr.Row():
            mBtn=gr.Radio(choices=["109", "110"], label="Model")
            mBtn.change(model_change, inputs=mBtn)
            s1Box=gr.Dropdown(
                choices=US1List, label="109 Module Section"
            )
            s1Box.select(fn=rs1_change, inputs=s1Box)
            s2Box=gr.Dropdown(
                choices=US2List, label="110 Module Section"
            )
            s2Box.select(fn=rs2_change, inputs=s2Box)

        global progressTxt 
        progressTxt = gr.Textbox(label="Progress", value="Select Course and Module then press button to grade.",interactive=False)  # This component will be updated

        

        def respond():
            global sStr
            global model
            x = sStr.find("US1")
            crsStr = "US1" if x >= 0 else "US2"

            if crsStr != model:
                gr.Warning("You have selected a lesson from a different course.")
                return

            msg = f"Begin Grading Completions: {sStr}"
            print(msg)
            lecStr = sStr[-3:]
            print(lecStr)
            crs = 110 if model == "US2" else 109
            submitPrompt(crs, int(lecStr))  # Function to process submissions

        button = gr.Button("Grade Submissions")
        button.click(respond, inputs=[], outputs=progressTxt)  # Respond updates the Textbox based on its return


#        progressTxt=gr.Textbox(label="Progress")
#        button.click(respond, inputs=progressTxt, outputs=progressTxt)
#        button.click(respond, inputs=None, outputs=progressTxt, _js={"inputs": ["progressTxt"], "outputs": ["progressTxt"]})
    return demo
"""
def main():
    import sys    
    crawl()
    makeEntryList()
#    interface=create_gradio_interface()
#    interface.launch()
    print("done")

if __name__ == "__main__":
    main()
