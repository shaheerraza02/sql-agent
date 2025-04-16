from fastapi import FastAPI, Request
from pydantic import BaseModel
import os

from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import init_chat_model
from langchain import hub
from langgraph.prebuilt import create_react_agent

# ENV setup (use dotenv or secrets manager in prod)
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
os.environ["LANGSMITH_API_KEY"] = "your_langsmith_api_key"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "your_project_name"

# DB config
db_uri = "mysql+pymysql://orko_ai:NewRandomPassword456@cvp-pilot-mysql.mysql.database.azure.com:3306/orko_ai_pilot"
db = SQLDatabase.from_uri(db_uri)

# LangChain setup
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
system_message = prompt_template.format(dialect="MySQL", top_k=5)

agent_executor = create_react_agent(llm, tools, prompt=system_message)

# FastAPI app
app = FastAPI()

# Input schema
class QueryRequest(BaseModel):
    question: str

# Route
@app.post("/ask")
async def ask_sql_agent(query: QueryRequest):
    user_message = {"messages": [{"role": "user", "content": query.question}]}
    response = agent_executor.invoke(user_message)
    return {"answer": response["messages"][-1].content}

