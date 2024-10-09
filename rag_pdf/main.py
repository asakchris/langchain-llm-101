from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    # Load the file
    pdf_path = "../data/react.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    # Split documents
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    # Embed the document and store it in vector database
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save index into local
    vectorstore.save_local("../tmp/faiss_index_react")

    # Load index from local
    new_vectorstore = FAISS.load_local(
        "../tmp/faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    # Create chain
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(), retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )

    # Search document
    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(res["answer"])
