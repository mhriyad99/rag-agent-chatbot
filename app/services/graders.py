from app.promt_templates.templates import temp_llama_retrival_grader, \
    temp_llama_hallucination_grader, temp_llama_answer_grader
from langchain_core.output_parsers import JsonOutputParser

from app.schemas.schemas import GraphState
from app.services.llm import llm


def get_retrieval_grader():
    prompt = temp_llama_retrival_grader
    retrieval_grader = prompt | llm | JsonOutputParser()

    return retrieval_grader


def get_hallucination_grader():
    prompt = temp_llama_hallucination_grader
    hallucination_grader = prompt | llm | JsonOutputParser()

    return hallucination_grader


def get_answer_grader():
    prompt = temp_llama_answer_grader
    answer_grader = prompt | llm | JsonOutputParser()

    return answer_grader


def grade_documents(state: GraphState):
    question = state["question"]
    documents = state["documents"]

    retrieval_grader = get_retrieval_grader()

    filtered_docs = []
    web_search = "No"

    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]

        if grade.lower() == "yes":
            filtered_docs.append(d)

    if len(filtered_docs) == 0:
        web_search = "Yes"

    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def grade_generation_v_documents_and_question(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    iteration = state["iteration"]

    if iteration >= 2:
        return "limit exceeded"

    hallucination_grader = get_hallucination_grader()
    answer_grader = get_answer_grader()

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        # Check question-answering
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            return "useful"
        else:
            return "not useful"
    else:
        return "not supported"


