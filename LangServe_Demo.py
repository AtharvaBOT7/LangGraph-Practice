from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI
import uvicorn
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

system_template = """You are a helpful assistant that translates English to {language}."""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")  
])

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", convert_system_message_to_user_message=True)

parser = StrOutputParser()

chain = prompt_template | model | parser

app = FastAPI(
    title = "MY LLM API",
    description = "This is a demo LLM API using LangServe",
    version = "0.0.1"
)

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)