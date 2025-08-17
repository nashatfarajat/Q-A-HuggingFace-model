from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
import torch

load_dotenv()

def get_wikipedia_data(query):
    try:
        loader = WikipediaLoader(query=query, load_max_docs=1)
        docs = loader.load()
        if docs:
            docs[0].metadata["source"] = "Wikipedia"
            return docs[0]
        return None
    except:
        return None

def get_web_search_results(query):
    search = DuckDuckGoSearchRun()
    try:
        results = search.run(query)
        return Document(
            page_content=results,
            metadata={"source": "Web Search"}
        )
    except:
        return Document(
            page_content="No web results found",
            metadata={"source": "None"}
        )

def create_qa_chain():
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    
    # Initialize GPT-2 Medium
    model_name = "gpt2-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.3,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.1,
        device=0 if torch.cuda.is_available() else -1
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=FAISS.from_texts(
            ["dummy content"],
            embedding=embeddings,
            metadatas=[{"source": "system"}]
        ).as_retriever(),
        return_source_documents=True
    )
    
    return qa_chain, embeddings, text_splitter

def answer_question(qa_chain, embeddings, text_splitter, question):
    # Format question for GPT-2
    formatted_question = f"Q: {question}\nA:"
    
    # Get documents
    wiki_doc = get_wikipedia_data(question)
    web_doc = get_web_search_results(question)
    
    # Prepare documents
    docs = []
    if wiki_doc:
        docs.append(wiki_doc)
    if web_doc:
        docs.append(web_doc)
    
    if not docs:
        docs = [Document(
            page_content="No information found",
            metadata={"source": "System"}
        )]
    
    # Process documents
    split_docs = text_splitter.split_documents(docs)
    db = FAISS.from_documents(split_docs, embeddings)
    
    # Get answer
    try:
        result = qa_chain({"query": formatted_question})
        answer = result.get("result", "").split("A:")[-1].strip()
        
        # Extract sources
        sources = []
        if "source_documents" in result:
            sources = list(set([
                doc.metadata.get("source", "Unknown") 
                for doc in result["source_documents"]
            ]))
        
        return {
            "answer": answer if answer else "No answer generated",
            "sources": ", ".join(sources) if sources else "Various sources"
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": ""
        }