import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()
# Global chain variable
rag_chain = None

def process_pdf(file):
    global rag_chain
    if file is None:
        return "Please upload a valid PDF file."

    # Gradio 6 returns a file path string for single file uploads
    loader = PyPDFLoader(file)
    docs = loader.load()

    # Chunking: 1000 characters, 200 overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embeddings & Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # Top-3 similarity search retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # System Prompt Specification
    system_prompt = (
        "Answer the question based only on the following context:\n"
        "{context}\n\n"
        "If the answer is not in the context, say 'I don't know based on the provided document.'"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # LLM & Chain Assembly
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return f"Successfully processed {len(docs)} page(s) into {len(splits)} chunks! You can now ask questions."

def answer_question(user_query, history):
    global rag_chain
    if rag_chain is None:
        return "Please upload and process a PDF first."
    
    response = rag_chain.invoke({"input": user_query})
    return response["answer"]

# Build Gradio Interface
with gr.Blocks(title="DocuChat Basics — Single Document Q&A") as demo:
    gr.Markdown("# DocuChat Basics — Single Document Q&A")
    
    with gr.Row():
        file_input = gr.File(label="Upload PDF Document", file_types=[".pdf"])
        status_output = gr.Textbox(label="Status", interactive=False)
    
    upload_button = gr.Button("Process Document")
    
    chatbot = gr.ChatInterface(
        fn=answer_question,
        title="Document Q&A Chat"
    )

    upload_button.click(
        fn=process_pdf,
        inputs=[file_input],
        outputs=[status_output]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)