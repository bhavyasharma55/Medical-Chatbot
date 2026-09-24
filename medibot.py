import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq


# ============================================================
# Configuration
# ============================================================

DB_FAISS_PATH = "vectorstore/db_faiss"
load_dotenv(find_dotenv(), override=False)


# ============================================================
# Load FAISS Vector Store
# ============================================================

@st.cache_resource
def get_vectorstore():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db


# ============================================================
# Custom Prompt
# ============================================================

def set_custom_prompt():

    CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.

Rules:
1. Answer only using the information provided in the context.
2. If the answer is not available in the context, say "I don't know."
3. Do not make up information.
4. Do not use outside knowledge.
5. Keep the answer clear and concise.
6. Start the answer directly without small talk.

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = ChatPromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)

    return prompt


# ============================================================
# Main Application
# ============================================================

def main():

    st.set_page_config(
        page_title="MediBot",
        page_icon="🩺",
        layout="centered"
    )

    st.title("🩺 MediBot")
    st.write("Ask questions based on the medical documents in the knowledge base.")

    # --------------------------------------------------------
    # Check GROQ API Key
    # --------------------------------------------------------

    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key:

        st.error(
            "GROQ_API_KEY is not set. "
            "Please add it to your environment or .env file as: GROQ_API_KEY=your_key"
        )

        st.stop()

    # --------------------------------------------------------
    # Initialize Chat History
    # --------------------------------------------------------

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --------------------------------------------------------
    # Display Previous Messages
    # --------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --------------------------------------------------------
    # Chat Input
    # --------------------------------------------------------

    prompt = st.chat_input("Ask your medical question...")

    if prompt:

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Save user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        try:

            # ------------------------------------------------
            # Load Vector Store
            # ------------------------------------------------

            with st.spinner("Searching medical documents..."):

                vectorstore = get_vectorstore()

            if vectorstore is None:

                st.error("Failed to load the FAISS vector store.")
                st.stop()

            # ------------------------------------------------
            # Create Retriever
            # ------------------------------------------------

            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": 3
                }
            )

            # ------------------------------------------------
            # Create Groq LLM
            # ------------------------------------------------

            llm = ChatGroq(
                model="openai/gpt-oss-20b",
                temperature=0,
                groq_api_key=groq_api_key
            )

            # ------------------------------------------------
            # Create Prompt
            # ------------------------------------------------

            prompt_template = set_custom_prompt()

            # ------------------------------------------------
            # Build Retrieval-Augmented Generation Chain
            # ------------------------------------------------

            retrieval_chain = (
                {
                    "context": lambda x: "\n\n".join(
                        doc.page_content for doc in retriever.invoke(x)
                    ),
                    "question": RunnablePassthrough(),
                }
                | prompt_template
                | llm
                | StrOutputParser()
            )

            # ------------------------------------------------
            # Get Response
            # ------------------------------------------------

            with st.spinner("Generating answer..."):

                source_documents = retriever.invoke(prompt)
                result = retrieval_chain.invoke(prompt)

            # ------------------------------------------------
            # Extract Answer
            # ------------------------------------------------

            response = {
                "answer": result,
                "context": source_documents
            }

            # ------------------------------------------------
            # Display Assistant Response
            # ------------------------------------------------

            with st.chat_message("assistant"):

                st.markdown(result)

                # --------------------------------------------
                # Display Source Documents
                # --------------------------------------------

                source_documents = response.get(
                    "context",
                    []
                )

                if source_documents:

                    with st.expander("📚 Source Documents"):

                        for i, doc in enumerate(
                            source_documents,
                            start=1
                        ):

                            st.markdown(
                                f"### Source {i}"
                            )

                            if hasattr(
                                doc,
                                "metadata"
                            ):

                                metadata = doc.metadata

                                if metadata:

                                    st.write(
                                        "Metadata:",
                                        metadata
                                    )

                            st.write(
                                doc.page_content
                            )

            # ------------------------------------------------
            # Save Assistant Response
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result
                }
            )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )


# ============================================================
# Run Application
# ============================================================

if __name__ == "__main__":
    main()
