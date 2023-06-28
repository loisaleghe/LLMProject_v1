# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

chat = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'])
print(chat)

# template = """\
# For the following text, extract the following information:
# columns: what are the data columns listed  or referenced in the text \
# Answer as a json array representing the columns.
#
# relationships: Extract the hierarchical relationships from the given text while excluding any date or time hierarchies. \
# Answer in json format like  "items" : [hierarchies] where each hierarchy looks like this \
# "label" : "hierarchy label" , "members" : [grand parent, parent , child ]. use " instead of ' round strings \
# filter out  temporal, date and time hierarchies, do  not include any hierarchy that has only one column
#
# Format the output as JSON with the following keys:
# columns
# relationships
#
# text: {text}
# """
#
# prompt_template = ChatPromptTemplate.from_template(template)
# messages = prompt_template.format_messages(text="تحتوي بياناتي على فئة المنتج، فرع فئة المنتج، المنتج، المنطقة، البلد، المدينة، السنة، الشهر، اليوم، القسم، الموظف، والحجم.")
# response=chat(messages)
# print(response.content)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

template2 = """\
From the given schema, generate a SQL for the city with the highest number of sales.
The schema should include number of units sold, name of products, price per unit, and city of sale
"""
prompt_template1 = ChatPromptTemplate.from_template(template2)
messages = prompt_template1.format_messages()
response1 = chat(messages)
print(response1.content)
