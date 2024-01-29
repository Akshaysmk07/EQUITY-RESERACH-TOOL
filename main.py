import os
import streamlit as st
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load OpenAI API key
os.environ['OPENAI_API_KEY'] = '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'
# Initialise LLM with required params
llm = OpenAI(temperature=0.9, max_tokens=1000)

st.title("EQUITY NEWS RESEARCH TOOL 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

vectorstore = None

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    vectorstore = vectorstore_openai
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")

    st.success("Processing Completed!")


query = main_placeholder.text_input("Question: ")
if query:
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

        result = chain({"question": query}, return_only_outputs=True)

        # result will be a dictionary of this format --> {"answer": "", "sources": [] }

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)
else:
    st.warning("Please process URLs to build vectorstore_openai.")
