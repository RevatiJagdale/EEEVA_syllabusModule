import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Paths
CHROMA_PATH = "chroma"

# Prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

@st.cache_resource
def load_vectorstore():
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    return db

# Initialize Groq LLM
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama3-8b-8192",  # or "llama3-70b-8192"
        api_key=os.environ["GROQ_API_KEY"]
    )

# Streamlit UI
st.set_page_config(page_title="📚 Syllabus Q&A", layout="wide")
st.title("📖 Ask your Syllabus")

query_text = st.text_input("Enter your question:", "")

if query_text:
    db = load_vectorstore()
    llm = load_llm()

    # Search Chroma DB
    results = db.similarity_search_with_relevance_scores(query_text, k=5)

    # show raw results for debugging
st.write("Raw results:", results)

    if len(results) == 0 or results[0][1] < 0.5:
        st.warning("⚠️ No good match found in syllabus.")
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        with st.spinner("Thinking..."):
            response = llm.invoke(prompt).content

        st.subheader("💡 Answer")
        st.write(response)

        # Show sources
        with st.expander("📂 Sources"):
            for doc, score in results:
                st.markdown(f"- **{doc.metadata.get('source','Unknown')}** (score: {score:.2f})")
