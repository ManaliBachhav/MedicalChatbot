# from dotenv import load_dotenv
# import os
# from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
# from pinecone import Pinecone
# from pinecone import ServerlessSpec 
# from langchain_pinecone import PineconeVectorStore

# load_dotenv()


# PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
# OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# extracted_data=load_pdf_file(data='data/')
# filter_data = filter_to_minimal_docs(extracted_data)
# text_chunks=text_split(filter_data)

# embeddings = download_hugging_face_embeddings()

# pinecone_api_key = PINECONE_API_KEY
# pc = Pinecone(api_key=pinecone_api_key)



# index_name = "medical-chatbot"  # change if desired

# if not pc.has_index(index_name):
#     pc.create_index(
#         name=index_name,
#         dimension=384,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )

# index = pc.Index(index_name)


# docsearch = PineconeVectorStore.from_documents(
#     documents=text_chunks,
#     index_name=index_name,
#     embedding=embeddings, 
# )


from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")


PINECONE_API_KEY = os.environ.get(" pcsk_22LMkR_NCV3DH9GgCfVLBECweV6jrdaZYo16w1ZSLyY1WZXNahcbxpzBeSTaxj2DiuPhFi ")
if not PINECONE_API_KEY:
    # raise RuntimeError("PINECONE_API_KEY missing in .env")

    raise RuntimeError(" pcsk_22LMkR_NCV3DH9GgCfVLBECweV6jrdaZYo16w1ZSLyY1WZXNahcbxpzBeSTaxj2DiuPhFi ")

# 1) Load & split docs
extracted = load_pdf_file(data="data/")
filtered = filter_to_minimal_docs(extracted)
chunks = text_split(filtered)

# 2) Embeddings (384 dims)
embeddings = download_hugging_face_embeddings()

# 3) Pinecone index
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# 4) Upsert
docsearch = PineconeVectorStore.from_documents(
    documents=chunks,
    index_name=index_name,
    embedding=embeddings,
)
print("✅ Pinecone upsert done.")
