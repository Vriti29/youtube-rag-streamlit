import streamlit as st
import os

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# ---------------- CONFIG ----------------
st.set_page_config(page_title="YouTube RAG Assistant", layout="centered")
st.title("🎥 YouTube Video Q&A + Summary (RAG)")

# ---------------- API KEY ----------------
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found. Please set it in Streamlit secrets.")
    st.stop()

# ---------------- FUNCTIONS ----------------

def get_transcript(youtube_url):
    try:
        video_id = youtube_url.split("v=")[1].split("&")[0]
        transcript_data = YouTubeTranscriptApi.fetch(
            video_id, languages=["en", "hi"]
        )
        return " ".join([i.text for i in transcript_data])
    except TranscriptsDisabled:
        return None
    except Exception as e:
        return None


@st.cache_resource
def create_vector_store(transcript):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    docs = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(docs, embeddings)


def run_rag(question, vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = PromptTemplate(
        template="""
You are an assistant.
Answer the question using ONLY the information given in the transcript.
If the answer is not present, say "I don't know".

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


# ---------------- UI ----------------

youtube_url = st.text_input("🔗 Enter YouTube Video URL")

if youtube_url:
    with st.spinner("Fetching transcript..."):
        transcript = get_transcript(youtube_url)

    if not transcript:
        st.error("❌ Transcript not available for this video.")
        st.stop()

    st.success("✅ Transcript loaded")

    with st.spinner("Creating embeddings & vector store..."):
        vector_store = create_vector_store(transcript)

    question = st.text_input("💬 Ask a question from the video")

    if st.button("Get Answer") and question:
        with st.spinner("Thinking..."):
            answer = run_rag(question, vector_store)
        st.subheader("📌 Answer")
        st.write(answer)

    if st.button("Generate Video Summary"):
        with st.spinner("Generating summary..."):
            summary = run_rag("Write summary of this video", vector_store)
        st.subheader("📝 Summary")
        st.write(summary)
