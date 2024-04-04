from uuid import UUID
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(page_icon="🏠", page_title="GPT")
memory = ConversationBufferMemory(return_return_messages= True)

api_key = st.sidebar.text_input("Your API KEY", type="password")
os.environ['OPENAI_API_KEY'] = api_key


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    # file_path = f"./.cache/files/{file.name}"

    file_dir = f"./.cache/files"
    file_path = os.path.join(file_dir, file.name)
    os.makedirs(file_dir, exist_ok=True)



    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)

    docs = loader.load_and_split(text_splitter=splitter)

    try:

      embeddings = OpenAIEmbeddings()
      cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

      vectorstore = FAISS.from_documents(docs, cached_embeddings)
  
    except Exception as e:
      error_type = type(e).__name__  # 예외 객체의 클래스 이름을 가져옴

      st.write(f"❌ Retriever 생성 실패 - 에러 종류: {error_type}") 
      st.write(f"에러 메세지: {e}")
      return None

    # cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    # vectorstore = FAISS.from_documents(docs, cached_embeddings)

    retriever = vectorstore.as_retriever()
    return retriever


def save_messages(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_messages(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_messages(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

def invoke_chain(question):
    try:
      result=chain.invoke(question)
      memory.save_context({"input":question}, {"output":result.content})
    except Exception as e:
      error_type = type(e).__name__  # 예외 객체의 클래스 이름을 가져옴

      st.write(f"❌ Retriever 생성 실패 - 에러 종류: {error_type}") 
      st.write(f"에러 메세지: {e}")
      result=None
    return result

if(api_key):
  llm = ChatOpenAI(
      api_key=api_key,
      temperature=0.1,
      streaming=True,
      callbacks=[
          ChatCallbackHandler(),
      ],
  )
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    Answer the question using Only the following context. If you don't know the answer just say you don't know. DON'T make anything up.
     context: {context}
     """,
        ),
        ("human", "{question}"),
    ]
)
st.title("Chat-Bot")

st.markdown(
    """
Welcome!

Use this chatbot to ask questions to a AI about your files!

first, Input your OPEN_API_KEY.

second, Upload your files on the sidebar
            
"""
)
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .socx file",
        type=["pdf", "txt", "docx"],
    )

# 업로드한 파일을 캐쉬디렉토리에 저장한다.
if file:
    retriever = embed_file(file)
    if retriever is None:
      st.session_state["messages"] = []
    else :
      send_message("I'm ready! Ask away!", "ai", save=False)
      paint_history()
      message = st.chat_input("Ask anything about your file...")

      if message:
          send_message(message, "human")
          chain = (
              {
                  "context": retriever | RunnableLambda(format_docs),
                  "question": RunnablePassthrough(),
              }
              | prompt
              | llm
          )
          with st.chat_message("ai"):
              response = invoke_chain(message)
else:
    st.session_state["messages"] = []
