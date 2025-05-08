from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def get_qa_chain(llm, vectorstore):
    """Creates a RetrievalQA chain."""
    if not vectorstore:
        return None
        
    retriever = vectorstore.as_retriever()
    
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible.
    Context: {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )
    return qa_chain