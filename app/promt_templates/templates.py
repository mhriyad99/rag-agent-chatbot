from typing import List

from langchain.prompts import PromptTemplate
from app.core.settings import COLLECTION_SCOPE

template = """
Answer the question based the on the context below. If you
don't know the answer just say "I don't know".

Context: {context}

Queston: {question}
"""


temp_llama_question_router = PromptTemplate(
    template=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
      user question to a vectorstore or web search. Use the vectorstore for questions on 
      {COLLECTION_SCOPE}. You do not need to be too strict with the keywords 
      in the question related to these topics. Otherwise, use web-search. Give a binary choice 
      'web_search' or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
      no preamble or explanation. Question to route: {{question}} 
      <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
      input_variables=["question"],
)

temp_llama_retrival_grader = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
       of a retrieved document to a user question. If the document contains keywords related to the user question, 
       grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
       Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
       Provide the binary score as a JSON with a single key 'score' and no preamble or 
       explanation.
       <|eot_id|><|start_header_id|>user<|end_header_id|>
       Here is the retrieved document: \n\n {document} \n\n
       Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
       """,
    input_variables=["question", "document"],
)

temp_llama_hallucination_grader = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
        an answer is grounded in / supported by the information in the documents. Give a binary 
        'yes' or 'no' score to indicate whether the answer is grounded in / supported by the 
        documents. Provide the binary score as a JSON with a single key 'score' and no 
        preamble or explanation. Strictly give the answer in json 
        with just one key named 'score'. 
         
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )

temp_llama_answer_grader = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
       answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
       useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
       <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
       \n ------- \n
       {generation} 
       \n ------- \n
       Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)

temp_llama_generation = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)

temp_llama_question_rephrase = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an 
    assistant who's responsibility is to rephrase users question based on the chat history, 
    if the question is relevant to the previous conversation. Your task is to identify and replace
    anaphoric expressions (pronouns like "it", "they", etc.) in the follow-up question with the
    entities they refer to from the chat history if the question is relevant to the chat 
    history. Just rephrase the question and give a json output with key 'output' and rephrased 
    question as the value and no preamble or explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>
    \n
    Chat History: {chat_history}
    Question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["chat_history", "question"]
)