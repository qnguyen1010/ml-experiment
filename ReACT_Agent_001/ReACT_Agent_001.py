# Importing the Packages
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from googlesearch import search
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuring the Gemini LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
llm = Gemini()
Settings.llm = llm

# Definition of the tool 
def searchingTool(query:str) -> str:
  """
  This tool searches the internet for the query that is being passed.
  This tool can be used for gathering the latest information about the topic.
  This tool uses Google's Search, and returns the context based on the top results obtained.

  Args:
      query: prompt from the agent
  Returns:
      context(str): a complete combined context 
  """
  time.sleep(1)

  response = search(query, num_results=20, advanced=True)
  context = ""
  for result in response:
    context += result.description
  return context

# Converting the function to the ReACT tool
search_tool = FunctionTool.from_defaults(fn=searchingTool)

# Configuring the ReACT Agent
agent = ReActAgent.from_tools([search_tool],
                               llm=llm,
                               verbose=True,
                               allow_parallel_tool_calls=True,
                             )

if __name__ == "__main__":
    while True:
        template = input("Enter your question: ")
        response = agent.chat(template)
        print(response)

