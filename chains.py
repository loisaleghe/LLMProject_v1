import os
import pandas as pd
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser

df = pd.read_csv('product_template.csv', on_bad_lines='skip', skipinitialspace=True)
# print(df.head())
# llm = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0.9)
llm = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0)

prompt = ChatPromptTemplate.from_template("what is one company that makes the best {product}")
product = "Queen Size Sheet Set"
chain = LLMChain(llm=llm, prompt=prompt)
# print(chain.run(product))

# SIMPLESEQUENTIALCHAIN - only one input/output

prompt1 = ChatPromptTemplate.from_template("write a 20 words description for the following company: {company_name}")
chain1 = LLMChain(llm=llm, prompt=prompt1)
overall_simple_chain = SimpleSequentialChain(chains=[chain, chain1], verbose=True)
# print(overall_simple_chain.run(product))

# SEQUENTIALCHAIN - same as above but multiple input/output

# first prompt, translate review to pidgin english
first_prompt = ChatPromptTemplate.from_template("Translate the following review to pidgin english:"
                                                "\n \n {Review}")
chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="pidgin_review")

# second prompt, write summary of review above
second_prompt = ChatPromptTemplate.from_template("Summarize the following review in 1 sentence:"
                                                 "{pidgin_review}")
chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")

# third prompt, get language of original review
third_prompt = ChatPromptTemplate.from_template("What language is the following review:"
                                                "\n\n{Review}")
chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="language")

# fourth prompt, write a follow-up message
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary} \n\n Language: {language}"
)
chain_four = LLMChain(llm=llm, prompt=fourth_prompt, output_key="response")

sequential_response = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["pidgin_review", "summary", "language", "response"],
    verbose=True
)
review = df.Review[4]
# print(sequential_response(review))

# ROUTER_CHAIN

physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here is a question:
{input}"""

math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts, 
answer the component parts, and then put them together\
to answer the broader question.

Here is a question:
{input}"""

history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""

computerscience_template = """ You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity. 

Here is a question:
{input}"""

english_template = """You are a very good english professor. \
You are great at speaking and teaching english in a way that \
everyone understands. You know a lot of words, more than the \
average English native speaker. You scored the highest points \
in the IGCSE English exam.

Here is a question:
{input}"""

prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template
    },
    {
        "name": "math",
        "description": "Good for answering questions about math",
        "prompt_template": math_template
    },
    {
        "name": "history",
        "description": "Good for answering history questions",
        "prompt_template": history_template
    },
    {
        "name": "computer science",
        "description": "Good for answering computer science questions",
        "prompt_template": computerscience_template
    },
    {
        "name": "english",
        "description": "Good for answering english questions",
        "prompt_template": english_template
    }
]

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

# in the line below, this is creating a string and putting it in a string literal like this: "<name>:<description>"
# using the f-string literal
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

# this is the result that will be given if the input doesn't match anything from the prompt_infos above
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser()  # this is important as it will help the chain decide which subchains to
    # route between
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(router_chain=router_chain,
                         destination_chains=destination_chains,
                         default_chain=default_chain,
                         verbose=True)

# print(chain.run("What is black body radiation"))
print(chain.run("What is the hardest question in the world"))
