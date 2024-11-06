import os
os.environ['TRANSFORMERS_CACHE'] = '/datadrive/hugging-face-cache'

import streamlit as st

# check gpu
from torch import cuda
# used to log into huggingface hub
from huggingface_hub import login
# used to setup language generation pipeline
import torch

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.prompts.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from llama_index import LangchainEmbedding, VectorStoreIndex, ServiceContext
from llama_index import SimpleDirectoryReader

from llmlingua import PromptCompressor

st.title('【ＳＡＳ】 Quickstart App')

## TODO: trying 13b, also try out models like mistral
MODEL = "meta-llama/Llama-2-13b-chat-hf"


## prompt compressed using https://www.gptrim.com/?ref=hackernoon.com and the output is still very good
## TODO: use python package of gptrim. the trimmed version is here to compare in plaintext
system_prompt = """
answer bot SAS Viya 4 Kubernet job answer question SAS Viya 4 Kubernet, base given sourc document. rule must alway follow: - Assum question ask SAS Viya, refer SAS Viya 4. - question not relat SAS Viya 4, respond answer question relat SAS Viya 4. - Keep answer base fact, specif possibl describ SAS Viya function. - answer question, pleas use dot point bullet list answer. - Pleas provid specif suggest perform tune - not suggest inform unless sourc inform.
"""
# system_prompt = """
# You are an answer bot on SAS Viya 4 and Kubernetes and your job is to only answer questions about SAS Viya 4 and Kubernetes, based on the given source documents. Here are some rules you must always follow:
# - Assume if the question is asking about SAS Viya, it is referring to SAS Viya 4.
# - If a question is not related to SAS Viya 4, respond that you only answer questions related to SAS Viya 4.
# - Keep your answers based on facts, and be as specific as possible when describing SAS Viya functionality.
# - When answering questions, please use dot points and bulleted lists in your answer.
# - Please provide specific suggestions for performance and tuning
# - Do not suggest any information unless there is a source for your information.
# """
# - You must provide a source for your information in your response with a specific page number.
# system_prompt = """"""
# This will wrap the default prompts that are internal to llama-index
# query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
prompt_template_str = "[INST]<<SYS>>\n" + system_prompt + "<</SYS>>\n\n{query_str}[/INST]"
query_wrapper_prompt = PromptTemplate(
    prompt_template_str
)

## works but slows down. takes way too long. put into another script to look at the prompt compression
## i think this thing downloads its LLM again because we download checkpoint shards again
# llm_lingua = PromptCompressor()
# compressed_prompt = llm_lingua.compress_prompt(prompt_template_str, instruction="", question="", target_token=200)
# print(compressed_prompt['compressed_prompt'])
# query_wrapper_prompt = PromptTemplate(compressed_prompt['compressed_prompt'])

@st.cache_resource
def load_documents2():
    return SimpleDirectoryReader(
        input_dir="rag_data/", 
        exclude=["*.md"] ## GEL notes don't give install answers, they give answers like "contact sas premium support" because thats on the checklist
        ).load_data()

@st.cache_resource
def embeddings():
    return LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

@st.cache_resource
def load_model2():
    return HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=2048,
        generate_kwargs={
            "temperature": 0.1,
             "repetition_penalty": 1.1},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",
        # uncomment this if using CUDA to reduce memory usage
        model_kwargs={"torch_dtype": torch.float32}
    )

llm = load_model2()
embed_model = embeddings()
documents = load_documents2()
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
# query_engine = index.as_query_engine()


def generate_response2(input_text):
    return st.info(chat_engine.chat(input_text)) #st.info(query_engine.query(input_text)) 

with st.form('my_form'):
    text = st.text_area('Enter text:', 'Ask me anything about SAS Viya!')
    submitted = st.form_submit_button('Submit')
    if submitted:
        print("Response submitted")
        generate_response2(text)