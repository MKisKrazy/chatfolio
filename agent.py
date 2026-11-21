from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from prompts import RAG_WORKFLOW_INSTRUCTIONS,CHUNK_ANALYST_INSTRUCTIONS,SUBAGENT_DELEGATION_INSTRUCTIONS
from tools import search_documentation,backend
from langchain.messages import HumanMessage

max_concurrent_analysts=3

INSTRUCTIONS=(
    RAG_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "="*80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_analysts=max_concurrent_analysts
    )
)

model= init_chat_model(model="openai:gpt-5.5")

chunk_analyst_subagent={
    "name":"chunk_analyst",
    "description":(
        "Analyse one retrieved document chunk file"
        "Pass the user question and a single path file under /retrieved/."
    ),
    "system_prompt":CHUNK_ANALYST_INSTRUCTIONS
}

agent = create_deep_agent(
    model=model,
    tools=[search_documentation],
    subagents=[chunk_analyst_subagent],
    system_prompt=INSTRUCTIONS,
    backend=backend
)

example_query = "What are the reasons for climate change?"

result= agent.invoke(
    {"messages":[HumanMessage(content=example_query)]
    })

print(result["messages"][-1].content)