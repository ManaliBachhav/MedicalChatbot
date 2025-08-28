# from flask import Flask, render_template, jsonify, request
# from src.helper import download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import ChatOpenAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from src.prompt import *
# import os
# from langchain_google_genai import ChatGoogleGenerativeAI


# app = Flask(__name__)


# load_dotenv()

# PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
# OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# embeddings = download_hugging_face_embeddings()

# index_name = "medical-chatbot" 
# # Embed each chunk and upsert the embeddings into your Pinecone index.
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )


from flask import Flask, render_template, request
import os
from dotenv import load_dotenv

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore


# Importatnt line hai ye langchain ki manu 


# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)
load_dotenv()

# ENV
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
# PINECONE_API_KEY = os.environ.get("pcsk_22LMkR_NCV3DH9GgCfVLBECweV6jrdaZYo16w1ZSLyY1WZXNahcbxpzBeSTaxj2DiuPhFi")
GOOGLE_API_KEY  = os.environ.get("GOOGLE_API_KEY")


if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY missing in .env")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY missing in .env")

# Vector store (existing index must be ready)
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Gemini LLM
chatModel = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",           # fast & cheap; pro also works
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    convert_system_message_to_human=True  # keeps system_prompt behavior sane
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)

qa_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    result = rag_chain.invoke({"input": msg})
    return str(result["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
