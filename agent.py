#agent.py

import os
from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.constants import Send
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator
from langgraph.graph import StateGraph, START, END
from IPython.display import display, Image
from typing import Annotated, List, Optional, Literal, Union, Dict, Any
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from langgraph.constants import Send
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import asyncio
from dataclasses import asdict, dataclass
import tiktoken


load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# defines structure for each section in the report
class Section(BaseModel):
    name: str = Field(
        description="Name for a particular section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web search for this section of the report."
    )
    content: str = Field(
        description="The content for this section."
    )

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="All the Sections of the overall report.",
    )

# defines structure for queries generated for deep research
class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of web search queries.",
    )

# consists of input topic and output report generated
class ReportStateInput(TypedDict):
    topic: str # Report topic

class ReportStateOutput(TypedDict):
    final_report: str # Final report

# overall agent state which will be passed and updated in nodes in the graph
class ReportState(TypedDict):
    topic: str # Report topic
    clarifications: List[str]           # We'll store clarifications here
    awaiting_clarification: bool
    clarifying_question: str
    clarification_attempts: int
    sections: list[Section] # List of report sections
    completed_sections: Annotated[list, operator.add] # Send() API
    report_sections_from_research: str # completed sections to write final sections
    final_report: str # Final report
    filename: str # Filename of the report

# defines the key structure for sections written using the agent 
class SectionState(TypedDict):
    section: Section # Report section
    search_queries: list[SearchQuery] # List of search queries
    source_str: str # String of formatted source content from web search
    report_sections_from_research: str # completed sections to write final sections
    completed_sections: list[Section] # Final key in outer state for Send() API

class SectionOutputState(TypedDict):
    completed_sections: list[Section] # Final key in outer state for Send() API

# just to handle objects created from LLM reponses
@dataclass
class SearchQuery:
    search_query: str
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

tavily_search = TavilySearchAPIWrapper()


def generate_prompt(placeholders: dict, prompt_file: str) -> str:
    """
    Replaces placeholders in the prompt template with their respective values.

    Args:
        placeholders (dict): keys are placeholders, values are replacements
        prompt_file (str): path to the prompt file

    Returns:
        str: The modified prompt with placeholders replaced.
    """
    with open(prompt_file, 'r', encoding='utf-8') as file:
        prompt = file.read()
    
    for key, value in placeholders.items():
        prompt = prompt.replace(f'{{{key}}}', str(value))

    return prompt



async def run_search_queries(
    search_queries: List[Union[str, SearchQuery]],
    num_results: int = 5,
    include_raw_content: bool = False
) -> List[Dict]:
    search_tasks = []
    for query in search_queries:
        # Handle both string and SearchQuery objects
        # Just in case LLM fails to generate queries as:
        # class SearchQuery(BaseModel):
        #     search_query: str
        query_str = query.search_query if isinstance(query, SearchQuery)else str(query) # text query
        try:
            # get results from tavily async (in parallel) for each search query
            search_tasks.append(
                tavily_search.raw_results_async(
                    query=query_str,
                    max_results=num_results,
                    search_depth='advanced',
                    include_answer=False,
                    include_raw_content=include_raw_content
                )
            )
        except Exception as e:
            print(f"Error creating search task for query '{query_str}': {e}")
            continue
    # Execute all searches concurrently and await results
    try:
        if not search_tasks:
            return []
        search_docs = await asyncio.gather(*search_tasks, return_exceptions=True)
        # Filter out any exceptions from the results
        valid_results = [
            doc for doc in search_docs
            if not isinstance(doc, Exception)
        ]
        return valid_results
    except Exception as e:
        print(f"Error during search queries: {e}")
        return []
    
def format_search_query_results(
    search_response: Union[Dict[str, Any], List[Any]],
    max_tokens: int = 2000,
    include_raw_content: bool = False
) -> str:
    encoding = tiktoken.encoding_for_model("gpt-4")
    sources_list = []

    # Handle different response formats if search results is a dict
    if isinstance(search_response, dict):
        if 'results' in search_response:
            sources_list.extend(search_response['results'])
        else:
            sources_list.append(search_response)
    # if search results is a list
    elif isinstance(search_response, list):
        for response in search_response:
            if isinstance(response, dict):
                if 'results' in response:
                    sources_list.extend(response['results'])
                else:
                    sources_list.append(response)
            elif isinstance(response, list):
                sources_list.extend(response)

    if not sources_list:
        return "No search results found."

    # Deduplicate by URL and keep unique sources (website urls)
    unique_sources = {}
    for source in sources_list:
        if isinstance(source, dict) and 'url' in source:
            if source['url'] not in unique_sources:
                unique_sources[source['url']] = source

    # Format output
    formatted_text = "Content from web search:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source.get('title', 'Untitled')}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source.get('content', 'No content available')}\n===\n"

        if include_raw_content:
            # truncate raw webpage content to a certain number of tokens to prevent exceeding LLM max token window
            raw_content = source.get("raw_content", "")
            if raw_content:
                tokens = encoding.encode(raw_content)
                truncated_tokens = tokens[:max_tokens]
                truncated_content = encoding.decode(truncated_tokens)
                formatted_text += f"Raw Content: {truncated_content}\n\n"


    return formatted_text.strip()

# Structure Guideline
DEFAULT_REPORT_STRUCTURE = """The report structure should focus on breaking-down the user-provided topic
                              and building a comprehensive report in markdown using the following format:


                              1. Introduction (no web search needed)
                                    - Brief overview of the topic area


                              2. Main Body Sections:
                                    - Each section should focus on a sub-topic of the user-provided topic
                                    - Include any key concepts and definitions
                                    - Provide real-world examples or case studies where applicable


                              3. Conclusion (no web search needed)
                                    - Aim for 1 structural element (either a list of table) that distills the main body sections
                                    - Provide a concise summary of the report


                              When generating the final response in markdown, if there are special characters in the text,
                              such as the dollar symbol, ensure they are escaped properly for correct rendering e.g $25.5 should become \$25.5
                          """

llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=OPENAI_KEY)

async def generate_report_plan(state: ReportState):
    """Generate the overall plan for building the report"""
    topic = state["topic"]
    print('--- Generating Report Plan ---')

    report_structure = DEFAULT_REPORT_STRUCTURE
    number_of_queries = 8

    structured_llm = llm.with_structured_output(Queries)

    placeholders = {
        "topic": topic,
        "report_organization": report_structure,
        "number_of_queries": str(number_of_queries)
    }

    #Load the prompt from external file
    prompt_file = os.path.join("prompts", "REPORT_PLAN_QUERY_GENERATOR_PROMPT.txt")
    system_instructions_query = generate_prompt(placeholders, prompt_file)

    try:
        # Generate queries
        results = structured_llm.invoke([
            SystemMessage(content=system_instructions_query),
            HumanMessage(content='Generate search queries that will help with planning the sections of the report.')
        ])
        # Convert SearchQuery objects to strings
        query_list = [
            query.search_query if isinstance(query, SearchQuery) else str(query)
            for query in results.queries
        ]
        # Search web and ensure we wait for results
        search_docs = await run_search_queries(
            query_list,
            num_results=5,
            include_raw_content=False
        )
        if not search_docs:
            print("Warning: No search results returned")
            search_context = "No search results available."
        else:
            search_context = format_search_query_results(
                search_docs,
                include_raw_content=False
            )

        # Generate sections
        placeholders = {
            "topic": topic,
            "report_organization": report_structure,
            "search_context": search_context
        }

        #Load the prompt from external file
        prompt_file = os.path.join("prompts", "REPORT_PLAN_SECTION_GENERATOR_PROMPT.txt")

        system_instructions_sections = generate_prompt(placeholders, prompt_file)

        structured_llm = llm.with_structured_output(Sections)
        report_sections = structured_llm.invoke([
            SystemMessage(content=system_instructions_sections),
            HumanMessage(content="Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. Each section must have: name, description, plan, research, and content fields.")
        ])

        print('--- Generating Report Plan Completed ---')
        return {"sections": report_sections.sections}

    except Exception as e:
        print(f"Error in generate_report_plan: {e}")
        return {"sections": []}

def generate_queries(state: SectionState):
    """ Generate search queries for a specific report section """

    # Get state
    section = state["section"]
    print('--- Generating Search Queries for Section: '+ section.name +' ---')
    # Get configuration
    number_of_queries = 5
    # Generate queries
    structured_llm = llm.with_structured_output(Queries)
    # Format system instructions


    # Generate sections
    placeholders = {
        "section_topic":section.description,
        "number_of_queries":number_of_queries
    }

    #Load the prompt from external file
    prompt_file = os.path.join("prompts", "REPORT_SECTION_QUERY_GENERATOR_PROMPT.txt")

    system_instructions = generate_prompt(placeholders, prompt_file)  
    
    # Generate queries
    user_instruction = "Generate search queries on the provided topic."
    search_queries = structured_llm.invoke([SystemMessage(content=system_instructions),
                                     HumanMessage(content=user_instruction)])

    print('--- Generating Search Queries for Section: '+ section.name +' Completed ---')
    return {"search_queries": search_queries.queries}

async def search_web(state: SectionState):
    """ Search the web for each query, then return a list of raw sources and a formatted string of sources."""

    # Get state
    search_queries = state["search_queries"]
    print('--- Searching Web for Queries ---')
    # Web search
    query_list = [query.search_query for query in search_queries]
    search_docs = await run_search_queries(search_queries, num_results=6, include_raw_content=True)
    # Deduplicate and format sources
    search_context = format_search_query_results(search_docs, max_tokens=4000, include_raw_content=True)

    print('--- Searching Web for Queries Completed ---')
    return {"source_str": search_context}

def write_section(state: SectionState):
    """ Write a section of the report """

    # Get state
    section = state["section"]
    source_str = state["source_str"]
    print('--- Writing Section : '+ section.name +' ---')

    # Format system instructions
    placeholders = {
        "section_title":section.name,
        "section_topic":section.description,
        "context":source_str
    }

    #Load the prompt from external file
    prompt_file = os.path.join("prompts", "SECTION_WRITER_PROMPT.txt")

    system_instructions = generate_prompt(placeholders, prompt_file)

    # Generate section
    user_instruction = "Generate a report section based on the provided sources."
    section_content = llm.invoke([SystemMessage(content=system_instructions),
                                  HumanMessage(content=user_instruction)])
    # Write content to the section object
    section.content = section_content.content

    print('--- Writing Section : '+ section.name +' Completed ---')
    # Write the updated section to completed sections
    return {"completed_sections": [section]}


# Add nodes and edges
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")
section_builder.add_edge("write_section", END)
section_builder_subagent = section_builder.compile()

# Display the graph

Image(section_builder_subagent.get_graph().draw_mermaid_png())

def parallelize_section_writing(state: ReportState):
    """ This is the "map" step when we kick off web research for some sections of the report in parallel and then write the section"""


    # Kick off section writing in parallel via Send() API for any sections that require research
    return [
        Send("section_builder_with_web_search", # name of the subagent node
             {"section": s})
            for s in state["sections"]
              if s.research
    ]

def format_sections(sections: list[Section]) -> str:
    """ Format a list of report sections into a single text string """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
        {'='*60}
        Section {idx}: {section.name}
        {'='*60}
        Description:
        {section.description}
        Requires Research:
        {section.research}

        Content:
        {section.content if section.content else '[Not yet written]'}

        """
    return formatted_str

def format_completed_sections(state: ReportState):
    """ Gather completed sections from research and format them as context for writing the final sections """

    print('--- Formatting Completed Sections ---')
    # List of completed sections
    completed_sections = state["completed_sections"]
    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(completed_sections)

    print('--- Formatting Completed Sections is Done ---')
    return {"report_sections_from_research": completed_report_sections}

def write_final_sections(state: SectionState):
    """ Write the final sections of the report, which do not require web search and use the completed sections as context"""

    # Get state
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]

    print('--- Writing Final Section: '+ section.name + ' ---')
    # Format system instructions

    placeholders = {
        "section_title": section.name,
        "section_topic": section.description,
        "context": completed_report_sections
    }

    #Load the prompt from external file
    prompt_file = os.path.join("prompts", "FINAL_SECTION_WRITER_PROMPT.txt")
    system_instructions = generate_prompt(placeholders, prompt_file)

    # Generate section
    user_instruction = "Craft a report section based on the provided sources."
    section_content = llm.invoke([SystemMessage(content=system_instructions),
                                  HumanMessage(content=user_instruction)])

    # Write content to section
    section.content = section_content.content

    print('--- Writing Final Section: '+ section.name + ' Completed ---')
    # Write the updated section to completed sections
    return {"completed_sections": [section]}


def parallelize_final_section_writing(state: ReportState):
    """ Write any final sections using the Send API to parallelize the process """

    # Kick off section writing in parallel via Send() API for any sections that do not require research
    return [
        Send("write_final_sections",
             {"section": s, "report_sections_from_research": state["report_sections_from_research"]})
                 for s in state["sections"]
                    if not s.research
    ]



def generate_report_filename(report_content: str) -> str:
    """Generate a filename based on report content using LLM"""
    title_prompt = """Analyze this report content and generate a concise filename in snake_case format. 
    Follow these rules:
    1. Use only lowercase letters, numbers, and underscores
    2. Maximum 5 words
    3. Reflect main topic from first section
    4. No special characters or spaces
    
    Content: {content}"""
    
    response = llm.invoke([
        SystemMessage(content=title_prompt.format(content=report_content[:2000])),  # Truncate to save tokens
        HumanMessage(content="Generate filename following the rules:")
    ])
    
    # Clean up any extra quotes or spaces
    return response.content.strip().replace('"', '').replace("'", "").replace(" ", "_") + ".md"

def generate_clarification_question(topic: str) -> str:
    """
    Use the LLM to generate a clarifying question tailored to the given topic.
    For instance, if the topic is 'Business', the LLM might ask:
    "What type of business report do you want? Which industry should be the focus?"
    """
    prompt = (
        f"Given the topic '{topic}', generate a clarifying question that asks the user for additional context. "
        "The question should be specific to the topic and help refine the report. "
        "For example, if the topic is 'Business', you might ask, 'What type of business report do you want? "
        "Which industry should be the focus?'"
    )
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="")
    ])
    return response.content.strip()

# This node now does not block but sets a flag and returns the clarifying question in the state.
async def ask_for_clarification(state: ReportState) -> ReportState:
    """
    1) We generate a new clarifying question based on the current state['topic'].
    2) We set state['awaiting_clarification'] to True.
    """
    max_attempts = 5
    attempts = state.get("clarification_attempts", 0)
    # Generate a clarifying question dynamically based on the current topic.
    clarifying_question = generate_clarification_question(state["topic"])
    # Store the clarifying question and flag in the state.
    state["clarifying_question"] = clarifying_question
    state["awaiting_clarification"] = True
    state["clarification_attempts"] = attempts
    return state

# This function updates the state with the user's clarification.
async def update_clarification(state: ReportState, user_response: str) -> ReportState:
    """
    1) We append the user's new clarification to a list of clarifications in state['clarifications'].
    2) We check if we have enough detail to finalize the topic.
       - If yes, we unify (summarize) everything with the LLM into a final topic.
       - If no, we generate a new clarifying question (unless we exceed attempts).
    """

    # Ensure we have a clarifications list
    if "clarifications" not in state:
        state["clarifications"] = []

    # 1) Append this new response to the clarifications list
    state["clarifications"].append(user_response)

    # 2) Remove 'awaiting_clarification' so we can see if we still need it
    state.pop("awaiting_clarification", None)
    state.pop("clarifying_question", None)

    # 3) The LLM decides if the topic is now sufficiently detailed
    evaluation_prompt = (
        "We have a topic plus multiple clarifications. "
        "Do we have enough detail for a focused, in-depth report? Answer 'yes' or 'no'.\n\n"
        f"Base topic: {state['topic']}\n"
        f"Clarifications so far: {state['clarifications']}"
    )
    evaluation_response = llm.invoke([
        SystemMessage(content=evaluation_prompt),
        HumanMessage(content="")
    ])
    evaluation = evaluation_response.content.strip().lower()

    if evaluation.startswith("yes"):
        # 4) If we have enough detail, unify everything into a single final 'topic'
        combine_prompt = (
            "Combine the base topic and all clarifications into a single refined, concise topic. "
            "Focus on capturing every important detail. No extraneous text.\n\n"
            f"Base topic: {state['topic']}\n"
            f"Clarifications: {state['clarifications']}"
        )
        combine_response = llm.invoke([
            SystemMessage(content=combine_prompt),
            HumanMessage(content="")
        ])
        refined_topic = combine_response.content.strip()
        state["topic"] = refined_topic

        # We do NOT set awaiting_clarification => We are done clarifying
    else:
        # 5) We do not have enough detail, ask for more clarifications
        attempts = state.get("clarification_attempts", 0) + 1
        state["clarification_attempts"] = attempts
        if attempts < 5:
            # Generate a new clarifying question
            question = generate_clarification_question(state["topic"])
            state["clarifying_question"] = question
            state["awaiting_clarification"] = True

    return state

def compile_final_report(state: ReportState):
    """ Compile the final report """

    # Get sections
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}

    print('--- Compiling Final Report ---')
    # Update sections with completed content while maintaining original order
    for section in sections:
        section.content = completed_sections[section.name]

    # Compile final report
    all_sections = "\n\n".join([s.content for s in sections])
    # Escape unescaped $ symbols to display properly in Markdown
    formatted_sections = all_sections.replace("\\$", "TEMP_PLACEHOLDER")  # Temporarily mark already escaped $
    formatted_sections = formatted_sections.replace("$", "\\$")  # Escape all $
    formatted_sections = formatted_sections.replace("TEMP_PLACEHOLDER", "\\$")  # Restore originally escaped $

    # Generate filename
    filename = generate_report_filename(formatted_sections)
    file_path = os.path.join(os.getcwd(), filename)
    # Save to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(formatted_sections)
        print(f"\n--- Report saved as {filename} ---")

    # Add filename to the state before returning
    state["final_report"] = formatted_sections
    state["filename"] = filename
    return state

# Update the ResearchChatbot to incorporate clarification before proceeding
class ResearchChatbot:
    def __init__(self, agent):
        """
        agent: A compiled StateGraph or some LLM-based pipeline that
               expects {'topic': <string>, ...} as input and
               produces events with 'final_report' etc.
        """
        self.agent = agent
        self.console = Console()
        
    async def handle_input(self, state: ReportState):
        """
        1) If the state says we need clarifications, we do that.
        2) Once we have a final 'topic', we call the main agent pipeline.
        """
         # Check if we still want clarifications
        if state.get("awaiting_clarification"):
            self.console.print("[bold yellow]Still awaiting clarification...[/bold yellow]")
            return {"reply": state.get("clarifying_question", "No clarifying question found.")}

        # If we have no clarifications needed, let's run the main plan
        self.console.print(f"[bold green]Refined Topic: {state['topic']}[/bold green]")

        # Next, you call your sub-agent (the planning + report pipeline)
        final_report_response = await call_planner_agent(self.agent, state["topic"])

        # The final agent pipeline should produce: {'final_report': "...", 'filename': "..."}
        return final_report_response



async def call_planner_agent(agent, prompt, config={"recursion_limit": 50}):
    console = Console()
    events = agent.astream(
        {'topic' : prompt},
        config,
        stream_mode="values",
    )

    async for event in events:
        if 'final_report' in event:
            md = RichMarkdown(event['final_report'])
            console.print(md)
            return event  # Return for API integration

async def initialize_agent() -> StateGraph:
    builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput)
    builder.add_node("ask_for_clarification", ask_for_clarification)
    builder.add_node("generate_report_plan", generate_report_plan)
    builder.add_node("section_builder_with_web_search", section_builder_subagent)
    builder.add_node("format_completed_sections", format_completed_sections)
    builder.add_node("write_final_sections", write_final_sections)
    builder.add_node("compile_final_report", compile_final_report)
    builder.add_edge(START, "ask_for_clarification")
    builder.add_edge("ask_for_clarification", "generate_report_plan")
    builder.add_conditional_edges("generate_report_plan", parallelize_section_writing, ["section_builder_with_web_search"])
    builder.add_edge("section_builder_with_web_search", "format_completed_sections")
    builder.add_conditional_edges("format_completed_sections", parallelize_final_section_writing, ["write_final_sections"])
    builder.add_edge("write_final_sections", "compile_final_report")
    builder.add_edge("compile_final_report", END)

    return builder.compile()


async def main():
    """
    docs = await run_search_queries(['langgraph'], include_raw_content=True)
    output = format_search_query_results(docs, max_tokens=500, 
    include_raw_content=True)
    """


    builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput)

    # Add a new node for asking clarifying questions
    builder.add_node("ask_for_clarification", ask_for_clarification)

    builder.add_node("generate_report_plan", generate_report_plan)
    builder.add_node("section_builder_with_web_search", section_builder_subagent)
    builder.add_node("format_completed_sections", format_completed_sections)
    builder.add_node("write_final_sections", write_final_sections)
    builder.add_node("compile_final_report", compile_final_report)

    builder.add_edge(START, "ask_for_clarification")
    builder.add_edge("ask_for_clarification", "generate_report_plan")

    builder.add_conditional_edges("generate_report_plan",
                                parallelize_section_writing,
                                ["section_builder_with_web_search"])
    builder.add_edge("section_builder_with_web_search", "format_completed_sections")
    builder.add_conditional_edges("format_completed_sections",
                                parallelize_final_section_writing,
                                ["write_final_sections"])
    builder.add_edge("write_final_sections", "compile_final_report")
    builder.add_edge("compile_final_report", END)

    reporter_agent = builder.compile()
    try:
        display(Image(reporter_agent.get_graph(xray=True).draw_mermaid_png()))
    except Exception:
        pass

    # Create a chatbot interface and prompt the user for a topic
    chatbot = ResearchChatbot(reporter_agent)
    topic = input("Enter your research topic: ")
    await chatbot.handle_input(topic)


if __name__ == "__main__":
    asyncio.run(main())