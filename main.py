import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY ="*******************************" #openAI key

#upload PDF FILE
st.header ("LingoAI")

with st.sidebar:
      st.title("Your Documents")
      file = st.file_uploader("upload your PDF file and start asking your questions", type="pdf")

# extract the text
if file is not None:
      pdf_reader= PdfReader(file)
      text=""
      for page in pdf_reader.pages:
          text+=page.extract_text()
          #st.write(text)


#break it into tokens
      text_splitter = RecursiveCharacterTextSplitter(
            separators="\n",
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len)
      chunks= text_splitter.split_text(text)
      #st.write(chunks)



      #generating embeddings
      embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


     #creating vector store ----FAISS
      vector_store = FAISS.from_texts(chunks,embeddings)    #creating database

     #get user question
      user_question= st.text_input("Ask your Question here ")

     # do similarity Search
      if user_question:
         match= vector_store.similarity_search(user_question)
         st.write(match)
         # define the LLM
         llm= ChatOpenAI(
               openai_api_key=OPENAI_API_KEY,
               temperature=0,
               max_tokens=1000,                       #fine tunning
               model_name="gpt-4"
         )

         # output result
         # chain ->take the question,get relevant document,pass it to the llm,generate the output
         chain= load_qa_chain(llm, chain_type="stuff")
         response=chain.run(input_documents=match,question=user_question)
         st.write(response)







