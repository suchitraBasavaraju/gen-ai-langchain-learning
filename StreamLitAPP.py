import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file,get_table_data
import streamlit as st
from langchain.callbacks import get_openai_calback
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain
from src.mcqgenerator.logger import logging

#loading json file
with open('/config/workspace/Response.json','r') as file:
    RESPONSE_JSON=json.load(file)

#creating a title for the app
st.title("MCQ Creator applicaiton with langcahin")

#create a form using st.form
with st.form("user_inputs"):
    #File upload
    uploaded_file=st.file_uploader("upload a pdf or text file")

    #input  fields
    mcq_count=st.number_input("No of MCQ's",min_value=3,max_value=50)

    #Subject
    subject=st.text_input("insert Subject",max_chars=20)

    #Quiz Tone
    tone=st.text_input("Complexity level of Questions",max_chars=20,placeholder="simple")

    #Add Buton
    button=st.form_submit_button("create MCQ's")

    #Check if button is clicked and all fields have input
    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading.."):
            try: 
                text=read_file(uploaded_file)
                #count tokens and the cost of API call
                with get_openai_calback() as cb: 
                    response=generate_evaluate_chain(
                    {
                        "text": text,
                        "number":mcq_count,
                        "subject": subject,
                        "tone": tone,
                        "response_json":json.dumps(RESPONSE_JSON)
                    
                    }
                    )
                #st.write(response)
            except Exception as e: 
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error("Error")

            else: 
                print(f"Total Tokens: {cb.total_tokens}")
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Cost: {cb.total_cost}")
                if isinstance(response,dict):
                    #Extract the quiz data from the response
                    quiz=response.get("quiz",None)
                    if quiz is not None:
                        table_data=get_table_data(quiz)
                        if table_data is not None:
                            df=pd.DataFrame(table_data)
                            df.index=df.index+1
                            st.table(df)
                            #display the review in atext box as well
                            st.text_area(label="Review",value=response["review"])
                        else:
                            st.error("Error in the table data")

                else:
                    st.write(response)
                            


