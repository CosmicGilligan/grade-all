import streamlit as st
import grade_all
import openai
import csv
import os
import time
from create_xlsx import create_xlsx

client=openai
client.api_key = "sk-xxxx" # Replace with your key

# Get various user information
uid = os.getuid()  # Get the effective user ID 
username = os.getlogin()  # Get the login name of the current user

# Title for your app
st.markdown("## Professor Cosmic's Magic Lecture Grader")

# Conditional to run crawl only if needed
if 'US1List' not in st.session_state or 'US2List' not in st.session_state:
    with st.spinner("Initializing data..."):
        grade_all.crawl()        
        st.session_state.US1List = grade_all.US1List 
        st.session_state.US2List = grade_all.US2List

if 'entryList' not in st.session_state:
    st.session_state.entryList=grade_all.makeEntryList()

entryList=st.session_state.entryList
list1 = st.session_state.US1List
list2 = st.session_state.US2List

col1, col2 = st.columns(2)

if 'selectbox_options' not in st.session_state:
    st.session_state.selectbox_options = list1
if 'radio_choice' not in st.session_state:
    st.session_state.radio_choice = '109'  # Default 

#my_selectbox = st.selectbox("Select an option", st.session_state.selectbox_options)

def update_options(inList):
    st.session_state.selectbox_options = inList

    # Conditional actions based on the selected radio option
if st.session_state.radio_choice == '109':
    st.markdown("Current Selected Course: HIST109")
    update_options(list1)  
else:
    st.markdown("Current Selected Course: HIST110")
    update_options(list2)

with col1:
    selected_option = st.radio("Course:", ("109", "110"), key='radio_choice')
with col2:
    moduleList=st.selectbox("Select an option", st.session_state.selectbox_options)

progress_bar = st.progress(0)

# A simple function to simulate some work
def do_some_work():

    promptStr=grade_all.getPromptStr(int(st.session_state.radio_choice), int(moduleList[-3:]))
    i=0
    total_steps=len(entryList)
    iPctStep=1/total_steps

    for x in entryList:
        with st.spinner("Grading Completions..."):
            progress_bar.progress(i * iPctStep)
            i+=1
            responseStr=x[1]

            completion = client.chat.completions.create(
                model="gpt-4-0125-preview",
                messages=[
                    {"role": "system", "content": promptStr},
                    {"role": "user", "content": responseStr}
                ]
            )

            result=""
            for choice in completion.choices:
                cleanresult=choice.message.content.replace("\n", " ")
                result += cleanresult
            result += "\n"
            x[2]=result

    filename='./submissions/completions.csv'
    with open(filename, 'w+', newline='') as f:  # 'newline='' prevents extra blank lines
        writer = csv.writer(f)
        writer.writerows(entryList)
        f.close()
    # Brief pause (if it helps)
    time.sleep(0.5)  
    create_xlsx(filename)

    os.system("wps ./submissions/completions.xlsx &")

# Button to start the work simulation
if st.button("Start"):
    do_some_work()
    st.balloons()  # Optional celebratory touch