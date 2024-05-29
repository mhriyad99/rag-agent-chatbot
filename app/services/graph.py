from langgraph.graph import END, StateGraph

from app.schemas.schemas import GraphState
from app.services.tools import web_search, decide_to_generate
from app.services.tools import retrieve, question_rephraser
from app.services.tools import generate
from app.services.graders import grade_documents
from app.services.tools import route_question
from app.services.graders import grade_generation_v_documents_and_question
from app.services.tools import store_history
from app.services.global_state import history


def get_workflow():
    workflow = StateGraph(GraphState)

    workflow.add_node("question_rephraser", question_rephraser)
    workflow.add_node("websearch", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("store_history", store_history)

    workflow.set_entry_point("question_rephraser")
    workflow.add_conditional_edges(
        "question_rephraser",
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
            "useful": "store_history",
            "not useful": "websearch",
            "limit exceeded": "store_history",
        },
    )

    workflow.add_edge("store_history", END)

    app = workflow.compile()
    return app


# app = get_workflow()
# inputs = {"question": "score card of the match played between bangladesh and USA"}
# for output in app.stream(inputs):
#     for key, value in output.items():
#         print(f"Finished running: {key}:")
# print(value["generation"])

async def chat_graph(msg, func):
    app = get_workflow()
    output = app.invoke(msg)
    history.extend(output["chat_history"][-2:])
    # print("history", history)
    await func(output["generation"])
