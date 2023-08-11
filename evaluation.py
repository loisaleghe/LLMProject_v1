from dotenv import load_dotenv, find_dotenv
_= load_dotenv(find_dotenv())

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

llm = ChatOpenAI(temperature = 0.0)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)

# print(data[10])

#HARD-CODED EXAMPLES
examples = [{
                "query": "Do the Cozy Comfort Pullover Set have side pockets?",
                "answer": "Yes"
            },
            {
                "query": "What collection is the Ultra-Lofty 850 Stretch Down Hooded Jacket from?",
                "answer": "The DownTek collection"
            }]

#LLM GENERATED-EXAMPLES
from langchain.evaluation.qa import QAGenerateChain
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI())
new_examples = example_gen_chain.apply_and_parse(
    [{"doc":t} for t in data[:5]]
)

# print(new_examples[0])
# print(new_examples[1])
# print(new_examples[2])

#COMBINE EXAMPLES
examples += new_examples
# print(qa.run(examples[5]["query"]))

#IF YOU WANT TO SEE THE MANUAL EVALUATION
# import langchain
# langchain.debug = True
# print(qa.run(examples[0]["query"]))

# TO TURN OFF DEBUGGING TOOL
# langchain.debug = False

# LLM ASSISTED EVALUATION
predictions = qa.apply(examples) #This calls the chain on all the inputs in the list

from langchain.evaluation.qa import QAEvalChain

llm = ChatOpenAI(temperature=0)
eval_chain = QAEvalChain.from_llm(llm)
graded_output = eval_chain.evaluate(examples, predictions)

for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_output[i]['results'])
    print()