import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Initialize Tavily search tool with API key
tavily_search_tool = TavilySearchResults(max_results=3)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

@tool
def divide(a: int, b: int) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers."""
    return a % b

@tool
def wiki_search(query: str) -> str:
    """
    Search Wikipedia for factual information and encyclopedic content.
    
    This tool searches Wikipedia's vast database of articles for information on a wide range of topics
    including history, science, biographies, geography, and general knowledge. It's ideal for finding
    well-sourced, encyclopedic information about established facts and concepts.
    
    Args:
        query (str): The search term or topic to look up on Wikipedia. Can be a person's name,
                    place, concept, historical event, scientific term, etc.
                    Examples: "Albert Einstein", "climate change", "Roman Empire", "photosynthesis"
    
    Returns:
        str: Formatted search results containing up to 2 Wikipedia articles with their content.
            Each result includes the article source URL and page content. Returns an error
            message if the search fails or no results are found.
    
    Note:
        - Limited to 2 articles to keep response manageable
        - Best for established, factual information rather than current events
        - Content may be lengthy, so results are provided in full for comprehensive information
    """
    try:
        search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                for doc in search_docs
            ])
        return formatted_search_docs
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

@tool
def web_search(query: str) -> str:
    """
    Search the web for current information, news, and real-time data using Tavily.
    
    This tool performs web searches to find the most current and up-to-date information available
    online. It's particularly useful for recent news, current events, trending topics, real-time
    data, and information that may not be available in encyclopedic sources. Tavily provides
    high-quality, relevant results from across the web.
    
    Args:
        query (str): The search query to look up on the web. Can be current events, news topics,
                    recent developments, product information, or any topic requiring current data.
                    Examples: "latest AI developments 2024", "current stock market trends", 
                    "recent climate change news", "new iPhone features"
    
    Returns:
        str: Formatted search results containing up to 3 web pages with their content.
            Each result includes the source URL and up to 1000 characters of content.
            Returns an error message if the search fails or API key is invalid.
    
    Note:
        - Requires TAVILY_API_KEY environment variable to be set
        - Limited to 3 results and 1000 characters per result for efficiency
        - Best for current, real-time information and recent developments
        - Content is truncated to provide concise, relevant information
    """
    try:
        search_docs = tavily_search_tool.invoke(query)
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.get("url", "")}"/>\n{doc.get("content", "")[:1000]}\n</Document>'
                for doc in search_docs
            ])
        return formatted_search_docs
    except Exception as e:
        return f"Error searching web: {str(e)}"

@tool
def arvix_search(query: str) -> str:
    """
    Search arXiv for academic papers, research articles, and scholarly publications.
    
    This tool searches arXiv, a repository of electronic preprints and published papers approved
    for publication after moderation. It covers physics, mathematics, computer science, quantitative
    biology, quantitative finance, statistics, electrical engineering, systems science, and economics.
    Ideal for finding cutting-edge research, academic papers, and scholarly work.
    
    Args:
        query (str): The academic search query. Can include author names, paper titles, research topics,
                    mathematical concepts, or scientific terms. Use specific academic terminology for
                    best results.
                    Examples: "machine learning transformers", "quantum computing algorithms",
                    "neural networks optimization", "climate modeling", "John Smith author:Smith"
    
    Returns:
        str: Formatted search results containing up to 3 academic papers with their abstracts and
            metadata. Each result includes the paper source URL and up to 1000 characters of content
            (typically the abstract and key information). Returns an error message if the search fails.
    
    Note:
        - Limited to 3 papers and 1000 characters per paper for readability
        - Best for academic research, scientific papers, and scholarly articles
        - Content includes abstracts, which provide concise summaries of research
        - Papers may be preprints (not yet peer-reviewed) or published works
        - Use specific academic terminology for more relevant results
    """
    try:
        search_docs = ArxivLoader(query=query, load_max_docs=3).load()
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
                for doc in search_docs
            ])
        return formatted_search_docs
    except Exception as e:
        return f"Error searching Arxiv: {str(e)}"

tools = [
    multiply, add, subtract, divide, modulus,
    wiki_search, web_search, arvix_search,
]

# Load system prompt
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
except FileNotFoundError:
    system_prompt = "You are a helpful AI assistant. Use tools when needed to provide accurate and helpful responses."

sys_msg = SystemMessage(content=system_prompt)

# Setup HuggingFace endpoint (fallback model)
endpoint = None
try:
    endpoint = HuggingFaceEndpoint(
        repo_id="microsoft/DialoGPT-medium",
        temperature=0.1,
        max_new_tokens=256,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    _ = endpoint.invoke("Hello")  # quick check
except Exception as e:
    print(f"Primary model failed, using fallback: {e}")
    endpoint = HuggingFaceEndpoint(
        repo_id="gpt2",
        temperature=0.1,
        max_new_tokens=256,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

# Gemini (primary model)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    model_kwargs={"max_new_tokens": 512},
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Bind tools if supported
try:
    llm_with_tools = llm.bind_tools(tools)
except Exception:
    llm_with_tools = llm

def assistant(state: MessagesState):
    """Main LLM node that receives messages and decides to respond or call a tool."""
    try:
        messages = state["messages"]
        if not messages:
            return {"messages": [AIMessage(content="I'm ready to help! What would you like to know?")]}
        
        last_user_message = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        if not last_user_message:
            return {"messages": [AIMessage(content="I didn't receive a clear question. Please ask me something!")]}

        # CHANGED: Call the LLM with tools and check for tool calls
        response = llm_with_tools.invoke(messages)
        
        # If the response is a tool call, let ToolNode handle it
        if hasattr(response, 'tool_calls') and response.tool_calls:
            return {"messages": [response]}  # ToolNode will process tool call

        # Otherwise, return the final answer
        answer = response.strip() if isinstance(response, str) else str(response).strip()
        if answer.startswith("Question:"):
            answer = answer.split("Answer:", 1)[-1].strip()
        if not answer or len(answer) < 3:
            answer = f"I understand you're asking about: {last_user_message}. Let me help you with that."
        return {"messages": [AIMessage(content=answer)]}
    except Exception as model_error:
        print(f"Model invocation error: {model_error}")
        return {"messages": [AIMessage(content=f"I received your question about: {last_user_message}. While I'm experiencing some technical difficulties, I understand your question and would normally provide a detailed response.")]}
    except Exception as e:
        print(f"Error in assistant node: {e}")
        return {"messages": [AIMessage(content="I encountered a technical error. Please try again.")]}

def should_continue(state: MessagesState):
    """Determine if we should continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    # CHANGED: use hasattr to check for tool_calls
    return "tools" if hasattr(last_message, 'tool_calls') and last_message.tool_calls else END

# Build LangGraph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.set_entry_point("assistant")
builder.add_conditional_edges("assistant", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "assistant")
compiled_graph = builder.compile()

from langchain_core.messages import AIMessage, HumanMessage

class HFCompatAgent:
    """Hugging Face-compatible agent wrapper for LangGraph."""
    def __init__(self, graph):
        self.graph = graph

    def run(self, question: str, *args, **kwargs) -> str:
        """
        Run the LangGraph agent with a single string input and return only the final answer string.
        This is required for correct grading in the evaluation system.
        """
        try:
            # Wrap input in a HumanMessage
            input_msg = HumanMessage(content=question)
            result = self.graph.invoke({"messages": [input_msg]})
            
            # Extract the final AI message and return its content only
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage):
                    return msg.content.strip()
            
            return "Unable to generate a valid response."
        except Exception as e:
            print(f"Error in agent run: {e}")
            return f"Error: {e}"

my_agent = HFCompatAgent(compiled_graph)
