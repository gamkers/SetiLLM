from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import json

def senti(query):
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
    "You are a helpful NLP assistant, specializing in Sentiment Analysis. You are provided with a yammer post where user posted about their story with Chennai (QUERY) in English or in more than one language (code-mixed), along with the defintions about the sentiment classes. Your task is to analyze the message and categorize it as Positive, Negative, or Neutral based on the sentiment expressed, along with a justification and score between 1 to 10 in seprate key value pares key named score and make sure catogries the post based on the content with the keywords in seprate key value pares, key is keywords and values should be in list maximum of 3 and its should have the places and food. Make sure to highlight the words/span of text in the query that you used to make your decision in your justification. (DEFINITIONS)** Negative Sentiment **: It expresses some sort of negative feeling or view or opinion about someone or something.** Neutral Sentiment **: It neither expresses a positive nor a negative sentiment of the speaker. It could be a general comment, acknowledgement,chitchat or any factual advice or a simple greeting.** Positive Sentiment **: The sentiment needs to be classified as positive if the speaker feels strong and positive at any particular utterance, except the normal aspects such as any form of greetings. DEFINITIONS QUERY: query output_format_instructions DO NOT OUTPUT ANYTHING OTHER THAN THE JSON OBJECT",
            ),
    ("human", "query: {query}"),
        ]
    )

    # Create the processing chain
    chain = prompt | llm

    # Invoke the LLM with the query and get the output
    output = chain.invoke({"query": query}).content

    try:
        # Convert JSON string to Python dictionary
        result_dict = json.loads(output)
        return result_dict
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

# Replace 'your_file.json' with the actual path to your JSON file
with open('raw_extract.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# List to store all results
all_results = []

# Extract the 'body' part from each message
for message in data['messages']:
    body = message.get('body', {}).get('parsed') or message.get('content_excerpt', {}).get('parsed')

    if body and len(body) > 200:
        print("---------------------------------------------------------------------------------------")
        print(body)
        result = senti(body)
        if result:  # Check if the result is not None
            all_results.append(result)

# Save all results to a JSON file
with open('senti_analysis_results.json', 'w', encoding='utf-8') as result_file:
    json.dump(all_results, result_file, ensure_ascii=False, indent=4)

print(f"Sentiment analysis results have been saved to 'senti_analysis_results.json'")
