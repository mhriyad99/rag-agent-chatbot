from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from app.schemas.schemas import GraphState
from app.services.tools import web_search, decide_to_generate
from app.services.tools import retrieve
from app.services.tools import generate
from app.services.graders import grade_documents
from app.services.tools import route_question
from app.services.graders import grade_generation_v_documents_and_question

workflow = StateGraph(GraphState)

workflow.add_node("websearch", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)


app = workflow.compile()
inputs = {"question": "What is result of bayern munich vs real madrid"}
for output in app.stream(inputs):
    print("output", output)
    for key, value in output.items():
        print(f"Finished running: {key}:")
print(value["generation"])
