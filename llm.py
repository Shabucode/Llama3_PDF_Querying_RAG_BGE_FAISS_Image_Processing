# llm.py
#requirements.txt or pip install langchain langchain-community chains Ollama llama3

#Hugging face Bge Embeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
embeddings = hf

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
import re

# Initialize the LLM model
llm = Ollama(model="llama3")

# Mapping of marketplaces to corresponding PDF files
document_indexer = {
    "steamworks documentation": "E:\LLM_SC_integration\docs\Steamworks Documentation.pdf",
    "epic game store": "E:\LLM_SC_integration\docs\epic game store documentation.pdf"
}

# Function to process a single PDF file
def process_pdf(file):
    loader = PyPDFLoader(file)
    documents = loader.load()
    return documents


#Initialize embeddings and create or load the vectorstore
def create_or_load_vectorstore(documents, vectorstorefaissbge_dir):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = hf
    
    # Check if the vectorstore directory exists
    if not os.path.exists(vectorstorefaissbge_dir):
        os.makedirs(vectorstorefaissbge_dir)
        # Initialize FAISS
        vectorstorefaissbge = FAISS.from_documents(documents=texts, embedding=embeddings)
        vectorstorefaissbge.save_local(vectorstorefaissbge_dir)
    else:
        vectorstorefaissbge = FAISS.load_local(vectorstorefaissbge_dir, embeddings, allow_dangerous_deserialization=True)

    return vectorstorefaissbge
def get_dimensions(marketplace, context):
    if marketplace in document_indexer:
        # Process the relevant PDF document
        pdf_file = document_indexer[marketplace]
        vectorstorefaissbge_dir = f'vectorstoresfaissbge/{marketplace.replace(" ", "_")}'
        
        # Check if the vectorstore already exists
        if os.path.exists(vectorstorefaissbge_dir):
            vectorstorefaissbge = FAISS.load_local(vectorstorefaissbge_dir, embeddings, allow_dangerous_deserialization=True)
        else:
            documents = process_pdf(pdf_file)
            vectorstorefaissbge = create_or_load_vectorstore(documents, vectorstorefaissbge_dir)
        
        retriever = vectorstorefaissbge.as_retriever()
        query = f"What is the size of the {context}Required or Optional in the {marketplace}"
        docs1 = vectorstorefaissbge.similarity_search(query)
        #print(docs1[0].page_content)
        retrieved_docs = retriever.invoke(query)
        context1 = "\n".join([doc.page_content for doc in retrieved_docs])
        template1="""
                            You will provide the size in pixel for a main at a provided game store name that is {marketplace} in the {context1}. 
                            When you find the Required pixel sizes after {context}Required or Optional separated by x the first value is the width and the second value is the height for the {context}. 
                            Lookup documentation for: {marketplace}
                            Lookup the following asset: {context}
                            Lookup in the retrieved document : {context1}
                            Provide the width and height in a JSON object with key width for width and key height for height and no preamble or explanation.
                            """
        prompt = PromptTemplate(input_variables=["marketplace", "context"], template=template1)
        chain = prompt | llm
        result = chain.invoke({"marketplace": marketplace, "context":context, "context1":context1})
        print(type(result))
        print(result)
        # Remove whitespace and newlines from the string
        x_cleaned = result.strip()

        # Parse the JSON-like string into a dictionary
        parsed_dict = json.loads(x_cleaned)

        print(parsed_dict)
        print(type(parsed_dict))
        b=parsed_dict["width"]
        print(b)
        c=parsed_dict["height"]
        print(c)
         # Write the data to the JSON file

        # Example text returned by LL
        # text = str({
        #             "width": 616, 
        #             "height": 353})
        # print(type(text))
        # pattern = r'\'(\w+)\': (\d+)'
        # matches = re.findall(pattern, result)
        # # Create a dictionary from the matches and convert values to integers
        # data = {key: int(value) for key, value in matches}
        # print(type(data))
        # result_json = json.dumps(result)
        # print(type(result_json))
        # try:
        #      result_dict = json.loads(result_json)
        # except json.JSONDecodeError as e:
        #      print(f"JSON decode error: {e}")
        #      result_dict = {}
        # result_dict = json.loads(result)
        return parsed_dict

if __name__ == "__main__":
    marketplace = input("Enter the marketplace: ")
    context = input("Enter the context: ")
    widthheight = get_dimensions(marketplace, context)
    print(type(widthheight))
    print(widthheight)
    #add the dimensions in the output json file whenever the code has executed along with marketplace and context and their corresponding 
    #height and width
    #if marketplace and context already exists in the json file which will contain, marketplace, context, width and height, it will return the 
    #values directly after going into get_dimensions, else it will get the width and height from get_dimensions() function from llm and from pdf.
    #to minimize the execution time each time
    #also specify the time it is taking to execute this whole code
