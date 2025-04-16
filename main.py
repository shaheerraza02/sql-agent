from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import init_chat_model
from langchain import hub
from langgraph.prebuilt import create_react_agent

# Load from .env locally (optional, for dev)
load_dotenv()

db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT", "3306")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")

openai_api_key = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

# DB connection URI for LangChain
from langchain_community.utilities import SQLDatabase
db_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
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
allow_origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,              # or ["*"] for all origins (not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class QueryRequest(BaseModel):
    question: str

# Route
@app.post("/ask")
async def ask_sql_agent(query: QueryRequest):
    user_message = {"messages": [{"role": "user", "content": query.question}]}
    response = agent_executor.invoke(user_message)
    return {"answer": response["messages"][-1].content}

