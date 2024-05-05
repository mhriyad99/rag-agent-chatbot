from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from resources import llm
from store import retriever

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Be as verbose and educational in your response as possible. 

    context: {context}
    Question: "{question}"
    Answer:
    """
)

chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
)


async def chat(msg, func):
    await func(chain.invoke(msg))


# rag_chain_with_source = RunnableParallel(
#     # {"context": retriever, "question": RunnablePassthrough()}
# ).assign(answer=chain)
#
#
# async def chat(msg, func=None):
#     print(msg)
#     output = {}
#     curr_key = None
#     for chunk in rag_chain_with_source.stream(msg):
#         for key in chunk:
#             print(key)
#             if key not in output:
#                 output[key] = chunk[key]
#             else:
#                 output[key] += chunk[key]
#             if  key == 'answer':
#                 if key != curr_key:
#                     await func(chunk[key])
#                 else:
#                     await func(chunk[key])
#             curr_key = key
#     await func("", active=False)
#
