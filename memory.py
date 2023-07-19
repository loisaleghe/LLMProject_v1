import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationTokenBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain


llm = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0.0)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

print(conversation.predict(input="Hi, my name is Lois"))
print(conversation.predict(input="do you know my name"))
print(conversation.predict(input="what is 1+1"))
print(conversation.predict(input="what is my name"))
print(memory.buffer)
print(memory.load_memory_variables({}))

ConversationBufferWindowMemory
memory1 = ConversationBufferWindowMemory(k=1)
# k=1 means one input, one output is stored in the memory & it only stores the last input and output
conversation1 = ConversationChain(
    llm=llm,
    memory=memory1,
    verbose=False
)
# memory1.save_context({"input": "hey"}, {"output": "what's up"})
# memory1.save_context({"input": "not much, just chilling, you?"}, {"output": "I'm cool"})
# print(memory1)
print(conversation1.predict(input="Hi, my name is Lois"))
print(conversation1.predict(input="what is 1+1"))
print(conversation1.predict(input="what is my name"))  # the reason why the model couldn't answer this question is
# because the model only has the most recent chat stored in memory which is 1+1

# print(memory1.buffer)
# print(memory1.load_memory_variables({}))  # because I set k=1, I only get the last statement printed from memory

memory2 = ConversationTokenBufferMemory(llm=llm, max_token_limit=30)
conversation2 = ConversationChain(
    llm=llm,
    memory=memory2,
    verbose=True
)
memory2.save_context({"input": "AI is what"}, {"output": "Amazing!"})
memory2.save_context({"input": "Backpropagation is what"}, {"output": "Beautiful"})
memory2.save_context({"input": "Chatbots are what?"}, {"output": "Charming"})
print(memory2.load_memory_variables({}))

# ConversationSummaryBufferMemory
# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian restaurant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory3 = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory3.save_context({"input": "Hello"}, {"output": "What's up?"})
memory3.save_context({"input": "Nothing much, just chilling"}, {"output": "cool"})
memory3.save_context({"input": "what's on the schedule today?"}, {"output": f"{schedule}"})

# print(memory3.load_memory_variables({}))
conversation3 = ConversationChain(
    llm=llm,
    memory=memory3,
    verbose=True
)
print(conversation3.predict(input="what would be a good demo to show?"))
print(memory3.load_memory_variables({}))
