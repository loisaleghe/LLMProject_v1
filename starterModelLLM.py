import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

if __name__ == '__main__':
    chat = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0.0)
    chat

    template_string = """ Translate the text \
    that is delimited by triple backticks \
    into a style that is {style}.
    break each sentence into new line after 10 words
    text: ```{text}```
    """

    prompt_template = ChatPromptTemplate.from_template(template_string)

    customer_style = """American English \
    in a calm and respectful tone
    """

    customer_email = """
    Arr, \
    I be fuming that me blender lid \
    flew off and splattered me kitchen walls \
    with smoothie! And to make matters worse, \
    the warranty does not cover the cose of cleaning \
    up me kitchen. I need your help right now, matey!
    """

    customer_message = prompt_template.format_messages(style=customer_style, text=customer_email)
    customer_response = chat(customer_message)
    print(customer_response.content)

    service_reply = """
    Hey there customer, \
    the warranty does not cover cleaning \
    expenses for your kitchen because it's \
    your fault that you misused your blender by \
    forgetting to put the lid on before starting \
    the blender. \
    Tough luck! see ya!
    """

    service_style_pirate = """\
    a polite tone \
    that speaks in English pirate\
    """

    service_message = prompt_template.format_messages(style=service_style_pirate, text=service_reply)
    service_response = chat(service_message)
    print(service_response.content)

    # LEARNING ABOUT OUTPUT PARSERS

    customer_review = """\
    This leaf blower is amazing. It has four settings:\
    candle blower, gentle breeze, windy city, and tornado. \
    It arrived in two days, just in time for my wife's \
    anniversary present.\
    I think my wife liked it so much she was speechless. \
    So far I've been the only one using it, and I've been \
    using it every other morning to clear the leaves on our lawn. \
    It's slightly more expensive than the other leaf blowers \
    out there, but I think it's worth it for the extra features.\
    """

    review_template = """\
    For the following text, extract the following information:
    
    gift: Was the item purchased as a gift for someone else? Answer True if yes, False if no or unknown \
    delivery_days: How may days did it take for the product to be delivered? \
    price_value: Extract any sentences about the value or price of the product \
    
    Format the output as JSON with the following keys:
    gift
    delivery_days
    price_value
    
    text: {text}
    """

    prompt_template1 = ChatPromptTemplate.from_template(review_template)
    review_message = prompt_template1.format_messages(text=customer_review)
    review_response = chat(review_message)
    print(review_response.content)

    # The above returns a string in json but if we want dictionary this is how we do it below
    # step one: define a ResponseSchema for each piece of information to be extracted
    gift_schema = ResponseSchema(name="gift", description="Was the item purchased as a gift for someone else? Answer "
                                                          "True if yes, False if not or unknown")
    delivery_days_schema = ResponseSchema(name="delivery_days", description="How may days did it take for the product "
                                                                            "to be delivered?")
    price_value_schema = ResponseSchema(name="price_value", description="Extract any sentences about the value or "
                                                                        "price of the product but as an adjective")

    response_schemas = [gift_schema, delivery_days_schema, price_value_schema]

    # step two: create the output_parser and format instructions using schemas created above
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    # print(format_instructions)

    review_template1 = """\
        For the following text, extract the following information:

        gift: Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.\
        delivery_days: How may days did it take for the product to be delivered? \
        price_value: Extract any sentences about the value or price of the product but as an adjective

        text: {text}
        
        {format_instructions}
        """

    # step three: create prompt as we always do but this time passing format_instructions as a parameter
    prompt_template2 = ChatPromptTemplate.from_template(review_template1)
    review_message1 = prompt_template2.format_messages(text=customer_review, format_instructions=format_instructions)
    review_response1 = chat(review_message1)
    print(review_response.content)

    # to parse into an output dictionary
    output_dict = output_parser.parse(review_response1.content)
    output_dict
    print(type(output_dict))  # to confirm type is a dictionary
    print(output_dict.get('gift'))
    print(output_dict.get('delivery_days'))
    print(output_dict.get('price_value'))
