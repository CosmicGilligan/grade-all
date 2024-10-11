import streamlit as st
import grade_all
import openai
import csv
import os
import time
from create_xlsx import create_xlsx

CRS109=0
CRS110=1
CRSRVW=2

HIST109_1CRN='2479709'
HIST109_2CRN='2473754'
HIST110CRN='2473937'

list3 = []

with open("~/openai.key", 'r') as file:
    line = file.read()

api_key = line.strip()

client=openai
client.api_key = api_key

# Title for your app
st.markdown("## Professor Cosmic's Magic Grading Machine")

# Conditional to run crawl only if needed
if 'US1List' not in st.session_state or 'US2List' not in st.session_state:
    with st.spinner("Initializing data..."):
        grade_all.crawl()        
        st.session_state.US1List = grade_all.US1List 
        st.session_state.US2List = grade_all.US2List

if 'entryList' not in st.session_state:
    st.session_state.entryList=grade_all.makeEntryList(HIST109_2CRN)

entryList=st.session_state.entryList
list1 = st.session_state.US1List
list2 = st.session_state.US2List
list3 = ["109","110"]

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
elif st.session_state.radio_choice == '110':
    st.markdown("Current Selected Course: HIST110")
    update_options(list2)
else:
    st.markdown("Current Selected Course: REVIEWS")
    update_options(list3)

with col1:
    selected_option = st.radio("Course:", ("109", "110","Reviews"), key='radio_choice')
with col2:
    moduleList=st.selectbox("Select an option", st.session_state.selectbox_options)

progress_bar = st.progress(0)



def do_some_work():

    rvw=False
    if st.session_state.radio_choice == 'Reviews':
        promptStr=grade_all.getReviewPromptStr()
        rvw=True
    else:
        promptStr=grade_all.getPromptStr(int(st.session_state.radio_choice), int(moduleList[-3:]))
    i=0
    total_steps=len(entryList)
    if total_steps<=0:
        st.error("No student submissions found")
        return
    iPctStep=1/total_steps

    for x in entryList:
        with st.spinner("Grading Completions..."):
            progress_bar.progress(i * iPctStep)
            i+=1
            responseStr=x[1]

            completion = client.chat.completions.create(
                model="gpt-4o",
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
#    if rvw==True:
#        os.system("wps ./submissions/completions.csv &")
#        return
    create_xlsx(filename)
    os.system("wps ./submissions/completions.xlsx &")

# Button to start the work simulation
if st.button("Grade Submissions"):
    do_some_work()
    st.balloons()  # Optional celebratory touch

if st.button("New Submissions"):
    st.session_state.entryList=[]
    entryList=[]

    if st.session_state.radio_choice == '109':
        st.markdown("Current Selected Course: HIST109")
        crn=HIST109_1CRN
        update_options(list1)  
    elif st.session_state.radio_choice == '110':
        st.markdown("Current Selected Course: HIST110")
        crn=HIST110CRN
        update_options(list2)
    else:
        st.markdown("Current Selected Course: REVIEWS")
        update_options(list3)

    st.session_state.entryList=grade_all.makeEntryList(crn)
    entryList=st.session_state.entryList
    st.markdown("New Submissions Loaded")