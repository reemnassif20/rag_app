import logging
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_qa_chain(llm, vectorstore, use_history=False, k=4):
    """
    Creates an enhanced RetrievalQA chain with better prompting and retrieval.

    Args:
        llm: Language model to use
        vectorstore: Vector store for document retrieval
        use_history: Whether to incorporate conversation history
        k: Number of documents to retrieve (default: 4)

    Returns:
        RetrievalQA chain object or None if vectorstore is empty
    """
    if not vectorstore:
        logger.warning("Cannot create QA chain: vectorstore is None")
        return None

    try:
        # Create a retriever with customized search parameters
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

        # Enhanced prompt template
        prompt_template = """
        You are an assistant that provides accurate and helpful answers based on the given context.

        Use the following pieces of context to answer the question at the end. The context comes from documents 
        that the user has uploaded, so you should prioritize this information when formulating your answer.

        Context:
        {context}

        Question: {question}

        Instructions for answering:
        1. Answer directly based on the context provided
        2. If the context doesn't contain the answer, just say "I don't have enough information to answer that question based on the documents you've provided."
        3. Don't make up information that isn't in the context
        4. Keep your answer concise but informative
        5. If relevant, mention which document(s) your answer comes from

        Answer:
        """

        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

        # Configure chain with memory if requested
        chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}

        if use_history:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            chain_type_kwargs["memory"] = memory

        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            chain_type="stuff",  # "stuff" method combines all docs into one context
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True,
            verbose=True
        )

        logger.info(f"QA chain created successfully with retriever k={k}")
        return qa_chain

    except Exception as e:
        logger.error(f"Error creating QA chain: {str(e)}", exc_info=True)
        return None


def get_advanced_qa_chain(llm, vectorstore, use_history=False):
    """
    Creates an advanced QA chain with multi-retrieval approach for higher quality answers.

    Args:
        llm: Language model to use
        vectorstore: Vector store for document retrieval
        use_history: Whether to incorporate conversation history

    Returns:
        RetrievalQA chain object or None if vectorstore is empty
    """
    if not vectorstore:
        logger.warning("Cannot create advanced QA chain: vectorstore is None")
        return None

    try:
        # Create a hybrid retriever combining similarity and MMR search methods
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6, "fetch_k": 10}
        )

        # Enhanced prompt template with better reasoning
        prompt_template = """
        You are an assistant that helps users find information in their documents. You're analyzing documents that have been processed and stored for retrieval.

        CONTEXT INFORMATION:
        ---------------------
        {context}
        ---------------------

        QUESTION: {question}

        INSTRUCTIONS:
        - Provide a comprehensive, accurate answer based ONLY on the information in the CONTEXT section above
        - If the context doesn't contain enough information, acknowledge this limitation politely
        - Never make up facts that aren't supported by the context
        - Include specific details and examples from the context to support your answer
        - Organize information in a logical structure when appropriate
        - Citation is not needed - the user knows the answer comes from their documents

        ANSWER:
        """

        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

        # Configure chain with memory if requested
        chain_type_kwargs = {
            "prompt": QA_CHAIN_PROMPT,
            "verbose": True
        }

        if use_history:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            chain_type_kwargs["memory"] = memory

        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True,
        )

        logger.info("Advanced QA chain created successfully")
        return qa_chain

    except Exception as e:
        logger.error(f"Error creating advanced QA chain: {str(e)}", exc_info=True)
        return None