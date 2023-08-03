import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from IPython.display import display, Markdown
from langchain.embeddings import OpenAIEmbeddings


openai_api_key = os.environ['OPENAI_API_KEY']


# Get File
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)

index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch).from_loaders([loader])
# query = "Please list all your shirts in with sun protection in a table in markdown and summarize each one"
# response = index.query(query)
# display(Markdown(response))
docs = loader.load()
# print(docs[0])

embeddings = OpenAIEmbeddings()
embed = embeddings.embed_query("Hi my name is Lois")
# print(len(embed))
# print(embed[:5])

db = DocArrayInMemorySearch.from_documents(
    docs,
    embeddings
)
# query = "Please suggest a shirt with sunblocking"
# docs = db.similarity_search(query)
# print(len(docs))

retriever = db.as_retriever()
llm = ChatOpenAI(temperature=0.0)

# without llm we would do this code commented out below qdocs = "".join([docs[i].page_content for i in range(len(
# docs))]) response = llm.call_as_llm(f"{qdocs} Question: Please list all your shirts with sun protection in a table
# in markdown " f"and summarise") display(Markdown(response))

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)
query = "Please list all your shirts with sun protection in a table in markdown and summarize each one"
response = qa_stuff.run(query)
display(Markdown(response))
response = index.query(query, llm=llm)
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])

