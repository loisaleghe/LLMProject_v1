import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import warnings
warnings.filterwarnings("ignore")

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.chat_models import ChatOpenAI

llm=ChatOpenAI(temperature=0) # I am setting temperature to 0 to get the most accurate reasoning
tools = load_tools(["llm-math", "wikipedia"], llm=llm) # llm_math is actually a chain that uses language model in
# conjunction with a calculator to do math problems, while wikipedia is an API that connects to wikipedia allowing you
# to run search queries against wikipedia to get back results

# set up agent
agent = initialize_agent(
    tools,
    llm,
    agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors = True,
    verbose = True
)
# important things to know from syntax above are:
# CHAT: This is an agent that has to work with chat models
# REACT: This is a prompting technique designed to get the best reasoning performance out of language models

# print(agent("What is the 25% of 700"))

# question = "If I'm eating a pink lady, which fruit am I eating?"
# result = agent(question)


agent1 = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True
)

customer_list = [["Harrison", "Chase"],
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"],
                 ["Geoff","Fusion"],
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]
# print(agent1(f"""Sort these customers by \
# last name and then first name \
# and print the output: {customer_list}"""))

# The commented code above works but another way to write this would be:
# agent1.run(f"""Sort these customers by \
# last name and then first name \
# and print the output: {customer_list}""")

# TO TEST FOR DATE BUT CREATING OUR OWN TOOL
from langchain.agents import tool
from datetime import date

@tool
def time(text:str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())

agent = initialize_agent(
    tools + [time],
    llm,
    agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handling_pasrsing_errors = True,
    verbose=True
)

agent.run("What is the date today?")

# The agent will sometimes come to the wrong conclusion (agents are a work in progress!).
# if it does, please try running it again
# try:
#     result = agent("whats the date today?")
# except:
#     print("exception on external access")