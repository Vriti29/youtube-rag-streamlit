import streamlit as st
import os

from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Chat with YouTube Videos (RAG)")
st.title("🤖 Chat with YouTube Videos (RAG)")

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ---------------- SIDEBAR ----------------
st.sidebar.header("Configuration")

groq_api_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password"
)

youtube_url = st.sidebar.text_input(
    "Enter YouTube Video URL"
)

process_btn = st.sidebar.button("Process Video")

st.sidebar.markdown(
    """
**Note:**  
- Model: `llama-3.3-70b-versatile`  
- Embeddings: `all-MiniLM-L6-v2`
"""
)

# ---------------- FUNCTIONS ----------------
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

def get_transcript(url):
    try:
        video_id = url.split("v=")[1].split("&")[0]

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            transcript = transcript_list.find_manually_created_transcript(["en", "hi"])
        except:
            transcript = transcript_list.find_generated_transcript(["en", "hi"])

        fetched = transcript.fetch()
        return " ".join([item["text"] for item in fetched])

    except TranscriptsDisabled:
        return None
    except NoTranscriptFound:
        return None
    except Exception as e:
        return None


def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(docs, embeddings)

def answer_question(question):
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    context = "\n\n".join(d.page_content for d in docs)

    prompt = PromptTemplate(
        template="""
Answer the question using ONLY the context below.
If not present, say "I don't know".

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"],
    )

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

# ---------------- PROCESS VIDEO ----------------
if process_btn:
    if not groq_api_key or not youtube_url:
        st.sidebar.error("Please enter API key and YouTube URL")
    else:
        with st.spinner("Splitting text and creating embeddings..."):
            transcript = get_transcript(youtube_url)

            if not transcript:
                st.error("❌ Transcript not available for this video.")
                st.stop()

            st.session_state.vectorstore = build_vectorstore(transcript)


        st.success("Video processed successfully!")

# ---------------- DISPLAY CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- CHAT INPUT ----------------
user_input = st.chat_input("Ask something from the video...")

if user_input and st.session_state.vectorstore:
    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = answer_question(user_input)
            st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
