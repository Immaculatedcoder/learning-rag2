import os
from pathlib import Path
import bs4


from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# A) CONFIG (you can change these)
# -----------------------------
URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
OLLAMA_MODEL = "llama3.1"
PERSIST_DIR = str(Path(__file__).parent / "chroma_db")  # saves DB to disk
COLLECTION_NAME = "agents_article"
USER_AGENT = "rag-local-tutorial"


# -----------------------------
# B) HELPERS
# -----------------------------
def format_docs(docs):
    """Turn a list of Document objects into one big string for the prompt."""
    return "\n\n".join(d.page_content for d in docs)

# -----------------------------
# C) INGESTION (Load → Split → Embed → Store)
# -----------------------------
def build_or_load_vectorstore():
    # 1) Local embeddings model 

    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2) Create/load a Chroma DB on disk
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )

    # 3) If DB is empty, we ingest the webpage and store embddings
    if vectorstore._collection.count() == 0:
        print("No index found. Ingesting webpage...")

        # Load HTML/Document from the Web
        loader = WebBaseLoader(
            web_path=(URL,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(class_=("post-content","post-title", "post-header"))
            ),
            header_template={"User-Agent":USER_AGENT}
        )
        docs = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = splitter.split_documents(docs)

        # Add chunks to vector DB using Chroma
        vectorstore.add_documents(splits)

        # Save DB to disk
        vectorstore.persist()
        print(f"✅ Ingested {len(splits)} chunks and saved to {PERSIST_DIR}")
    else:
        print(f"✅ Loaded existing index from {PERSIST_DIR}")
    return vectorstore


# -----------------------------
# D) RAG PIPELINE (Retrieve → Prompt → Generate)
# -----------------------------
def build_rag_chain(vectorstore):
    # 1) Retriever: returns top-k relevant chunks for a question
    retriever = vectorstore.as_retriever(
        search_kwargs={"k":4}
    )

    # 2) Prompt: tells the LLM to use retrieved context
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Use the provided context to answer the question. "
         "If the answer is not in the context, say you don't know. Do not make things up."),
        ("human", "Context: \n{context} \n\nQuestion: \n{question}")
    ])

    # 3) Initiate your local LLM via Ollama
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

    # 4) Chain: question -> retrieve context -> prompt -> LLM -> string output
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
         | prompt
         | llm
         | StrOutputParser()         
    )

    return rag_chain

# -----------------------------
# E) MAIN (run the app)
# -----------------------------
def main():
    # Build/Load vector DB
    vectorstore = build_or_load_vectorstore()

    # Build the rag chain
    rag_chain = build_rag_chain(vectorstore)
    print("\nLocal RAG is ready. Ask questions (type 'exit' to quit). \n")
    while True:
        question = input("You> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        answer = rag_chain.invoke(question)
        print("\nAssistant>\n" + answer + "\n")

if __name__ == "__main__":
    main()
