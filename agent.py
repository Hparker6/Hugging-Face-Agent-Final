import os
import re
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Initialize Tavily search tool with API key
tavily_search_tool = TavilySearchResults(max_results=5)

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
    """Search Wikipedia for factual information and encyclopedic content."""
    try:
        search_docs = WikipediaLoader(query=query, load_max_docs=3).load()
        if not search_docs:
            return f"No Wikipedia results found for: {query}"
        
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
    """Search the web for current information, news, real-time data, and YouTube videos using Tavily."""
    try:
        search_docs = tavily_search_tool.invoke(query)
        if not search_docs:
            return f"No web search results found for: {query}"
        
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.get("url", "")}"/>\n{doc.get("content", "")}\n</Document>'
                for doc in search_docs
            ])
        return formatted_search_docs
    except Exception as e:
        return f"Error searching web: {str(e)}"

@tool
def arvix_search(query: str) -> str:
    """Search arXiv for academic papers, research articles, and scholarly publications."""
    try:
        search_docs = ArxivLoader(query=query, load_max_docs=3).load()
        if not search_docs:
            return f"No arXiv results found for: {query}"
        
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1500]}\n</Document>'
                for doc in search_docs
            ])
        return formatted_search_docs
    except Exception as e:
        return f"Error searching Arxiv: {str(e)}"

tools = [
    multiply, add, subtract, divide, modulus,
    wiki_search, web_search, arvix_search,
]

def extract_clean_answer(text: str) -> str:
    """Extract the cleanest, most concise answer from AI response."""
    if not text or not text.strip():
        return "No answer found"
    
    text = text.strip()
    
    # Remove LaTeX formatting
    text = re.sub(r'\$+[^$]*\$+', '', text)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    
    # Look for explicit answer markers
    answer_patterns = [
        r'(?:final answer|answer|result):\s*(.+?)(?:\n|$)',
        r'(?:the answer is|answer is):\s*(.+?)(?:\n|$)',
        r'(?:therefore|thus|so),?\s*(.+?)(?:\n|$)',
        r'(?:in conclusion|to conclude),?\s*(.+?)(?:\n|$)',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # Clean up the extracted answer
            answer = re.sub(r'[*_`]', '', answer)  # Remove markdown
            answer = re.sub(r'\s+', ' ', answer)   # Normalize whitespace
            return answer
    
    # Handle string reversal specifically
    if "reverse" in text.lower() and len(text) < 100:
        # Look for quoted strings or obvious reversals
        reversed_match = re.search(r'["\'`]([^"\'`]+)["\'`]', text)
        if reversed_match:
            return reversed_match.group(1)
    
    # Handle numerical answers
    number_match = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
    if number_match and len(text.split()) < 15:
        return number_match.group(1)
    
    # For short responses, return the whole thing cleaned up
    if len(text) < 100:
        # Remove common prefixes
        text = re.sub(r'^(?:based on|according to|the information shows that|i found that)\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[*_`]', '', text)  # Remove markdown
        text = re.sub(r'\s+', ' ', text)   # Normalize whitespace
        return text
    
    # For longer responses, try to extract the most relevant sentence
    sentences = re.split(r'[.!?]+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) < 200:
            # Skip introductory sentences
            if not re.match(r'^(?:based on|according to|the information|i found)', sentence, re.IGNORECASE):
                sentence = re.sub(r'[*_`]', '', sentence)  # Remove markdown
                return sentence
    
    # Fallback: return first 150 characters
    text = re.sub(r'[*_`]', '', text)
    return text[:150].strip()

def detect_direct_answer_questions(question: str) -> str:
    """Handle questions that don't need tools (like string reversal)."""
    question_lower = question.lower()
    
    # String reversal detection
    if "reverse" in question_lower:
        # Extract the string to reverse
        string_match = re.search(r'["\'`]([^"\'`]+)["\'`]', question)
        if string_match:
            string_to_reverse = string_match.group(1)
            return string_to_reverse[::-1]
        
        # Look for words that might need reversing
        words = question.split()
        for word in words:
            if len(word) > 2 and word.replace('.', '').replace(',', '').isalpha():
                # This might be the word to reverse
                clean_word = re.sub(r'[^\w]', '', word)
                if len(clean_word) > 2:
                    return clean_word[::-1]
    
    return None

# Enhanced system prompt focused on concise answers
system_prompt = """You are a helpful AI assistant. Your job is to provide CONCISE, DIRECT answers to questions.

CRITICAL RULES:
1. Always provide SHORT, DIRECT answers - no explanations, reasoning, or verbose text
2. For factual questions, use tools (wiki_search, web_search, arvix_search) to get accurate information
3. After using tools, extract ONLY the specific fact or number requested
4. For calculations, use math tools and return only the numerical result
5. For YouTube videos or current events, use web_search
6. For academic topics, use arvix_search
7. For general facts/biographies, use wiki_search

ANSWER FORMAT:
- Return only the final answer, nothing else
- No "Based on the search results..." or "According to..."
- No markdown, LaTeX, or special formatting
- Just the direct answer (number, name, fact, etc.)

Examples:
- Question: "What is 15 + 27?" → Answer: "42"
- Question: "Who won the 2020 Olympics?" → Answer: "USA" (after searching)
- Question: "How many albums did X release?" → Answer: "5" (after searching)

Be precise, factual, and concise."""

sys_msg = SystemMessage(content=system_prompt)

# Initialize LLM with fallback models
def initialize_llm():
    """Initialize LLM with fallback to reliable models."""
    models_to_try = [
        "gemini-2.0-flash",
        "gemini-1.5-flash", 
        "gemini-1.5-pro",
        "gemini-pro"
    ]
    
    for model in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=0,  # Set to 0 for more deterministic responses
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            print(f"Successfully initialized {model}")
            return llm
        except Exception as e:
            print(f"Failed to initialize {model}: {e}")
            continue
    
    raise Exception("Failed to initialize any Gemini model")

llm = initialize_llm()
llm_with_tools = llm.bind_tools(tools)

def assistant(state: MessagesState):
    """Enhanced assistant node with better answer extraction."""
    messages = state["messages"]
    
    # Get the original question
    human_message = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            human_message = msg.content
            break
    
    # Check if this is a direct answer question (like string reversal)
    if human_message:
        direct_answer = detect_direct_answer_questions(human_message)
        if direct_answer:
            return {"messages": [AIMessage(content=direct_answer)]}
    
    try:
        response = llm_with_tools.invoke(messages)
        
        # If no tool calls, clean up the response
        if not (hasattr(response, 'tool_calls') and response.tool_calls):
            clean_answer = extract_clean_answer(response.content)
            response.content = clean_answer
        
        return {"messages": [response]}
    except Exception as e:
        print(f"Error in assistant: {e}")
        return {"messages": [AIMessage(content=f"Error: {str(e)}")]}

def should_continue(state: MessagesState):
    """Check if we should continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are tool calls, continue to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return END

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("assistant")
builder.add_conditional_edges(
    "assistant",
    should_continue,
    {
        "tools": "tools",
        END: END,
    }
)
builder.add_edge("tools", "assistant")

compiled_graph = builder.compile()

class HFCompatAgent:
    """Hugging Face-compatible agent wrapper for LangGraph."""
    def __init__(self, graph):
        self.graph = graph

    def run(self, question, *args, **kwargs) -> str:
        """Run the agent and return the final clean answer."""
        try:
            # Handle both string and HumanMessage inputs
            if isinstance(question, str):
                question_content = question
            elif hasattr(question, 'content'):
                question_content = question.content
            else:
                question_content = str(question)
            
            print(f"\n=== Processing: {question_content[:100]}... ===")
            
            # Create initial state
            initial_state = {
                "messages": [sys_msg, HumanMessage(content=question_content)]
            }
            
            # Run the graph with higher recursion limit
            config = {"recursion_limit": 25}
            final_state = self.graph.invoke(initial_state, config=config)
            messages = final_state.get("messages", [])
            
            # Extract the final answer
            final_answer = None
            for message in reversed(messages):
                if isinstance(message, AIMessage) and message.content:
                    # Skip messages that only contain tool calls
                    if hasattr(message, 'tool_calls') and message.tool_calls and not message.content.strip():
                        continue
                    
                    # Clean and extract the answer
                    final_answer = extract_clean_answer(message.content)
                    break
            
            if not final_answer:
                final_answer = "Unable to determine answer"
            
            print(f"Final answer: {final_answer}")
            return final_answer
            
        except Exception as e:
            print(f"Error in agent run: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return f"Error: {str(e)}"

# Initialize the agent
my_agent = HFCompatAgent(compiled_graph)