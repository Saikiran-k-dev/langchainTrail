#for setting up the envirnament
from dotenv import load_dotenv

#to read the pdf file and saved it into the variable{{pdf}}
from PyPDF2 import PdfReader
# from sentence_transformers import SentenceTransformers
"""Below are 
All the Langchain Imports
that i am using"""
from langchain.chat_models import ChatOpenAI
#to split the text using the text splitter and create the chunk using the data present in the pdf
from langchain.text_splitter import CharacterTextSplitter
#to embed the given chunk into vectorstore and to create custom knowledge base vectorstore
from langchain.embeddings import HuggingFaceInstructEmbeddings,OpenAIEmbeddings,LlamaCppEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
#to read the .pdf files present in the current directory 
import os

#gets a PDF and returns the RAW text present in the PDF file
def getPdfText():
    text = ""
    pdf = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.pdf')]
    pdf_reader = PdfReader(pdf[0])
    for page in pdf_reader.pages:
            text += page.extract_text()
    return text

model_path = "./modelForEmbedding/ggml-model-q4_0.bin"
def initialize_embeddings() -> OpenAIEmbeddings:
      return OpenAIEmbeddings()
# modelForEmbedding
# def initialize_embeddings() -> LlamaCppEmbeddings:
#      return LlamaCppEmbeddings(model_path=model_path)

# def initialize_embeddings() -> HuggingFaceInstructEmbeddings:
#     return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

#retruns the chunks for the raw text using langchain text splitter
def getTextChunks(text):
    textSplitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = textSplitter.split_text(text)
    return chunks
# model = SentenceTransformers('all-MiniLM-L6-v2')
# embeddings = model.encode(sentences)
#returns the vectorstore for the chunk data using instructor embedding model
def getVectorstore(chunkData):
     embeddings = OpenAIEmbeddings()
    #  embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
     vectorstore = FAISS.from_texts(texts=chunkData, embedding=embeddings)
     return vectorstore

# rawText = getPdfText()
# print(rawText)
# textChunks = getTextChunks(rawText)
# print(textChunks)
# vectorstore = getVectorstore(textChunks)
# print(vectorstore)
# vectorstore.save_local("samplepdf")
embeddings = initialize_embeddings()
def getConversationChain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
llm = ChatOpenAI()
# llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1, "max_length":1000})
index = FAISS.load_local("./supportHandbookComplete", embeddings)
qa = ConversationalRetrievalChain.from_llm(llm, index.as_retriever(), max_tokens_limit=10000)
chat_history = []
print("Welcome to support team chatbot ask your queries")
#some copied code works as expected
while True:
    query = input("Please enter your question: ")
    
    if query.lower() == 'exit':
        break
    result = qa({"question": query, "chat_history": chat_history})

    print("Answer:", result['answer'])