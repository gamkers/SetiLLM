from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Configure the Groq-based language model
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    groq_api_key="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",
)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
"system",
"You are a helpful NLP assistant, specializing in Sentiment Analysis. You are provided with a yammer post where user posted about their story with Chennai (QUERY) in English or in more than one language (code-mixed), along with the defintions about the sentiment classes. Your task is to analyze the messageand categorize it as Positive, Negative, or Neutral based on the sentiment expressed, along with a justification. Make sure to highlight thewords/span of text im the query that you used to make your decision in your justification.(DEFINITIONS)** Negative Sentiment **: It expresses some sort of negative feeling or view or opinion about someone or something.** Neutral Sentiment **: It neither expresses a positive nor a negative sentiment of the speaker. It could be a general comment, acknowledgement,chitchat or any factual advice or a simple greeting.** Positive Sentiment **: The sentiment needs to be classified as positive if the speaker feels strong and positive at any particular utterance,except the normal aspects such as any form of greetings.DEFINITIONSQUERY: query output_format_instructions DO NOT OUTPUT ANYTHING OTHER THAN THE JSON OBJECT",
        ),
("human", "query: {query}"),
    ]
)

# Create the processing chain
chain = prompt | llm

output = chain.invoke(
{
    "query": "Your work is impressive—for someone who’s just starting out, I suppose.",
}
        ).content

print(output)

import json

# Your JSON-like string
json_string = output

# Convert JSON string to Python dictionary
my_dict = json.loads(json_string)

# Print the dictionary
print(my_dict['sentiment'])       # Output: Positive
print(my_dict['justification'])  
