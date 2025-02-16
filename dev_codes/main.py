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
    sections: list[Section] # List of report sections
    completed_sections: Annotated[list, operator.add] # Send() API
    report_sections_from_research: str # completed sections to write final sections
    final_report: str # Final report

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

REPORT_PLAN_QUERY_GENERATOR_PROMPT = """You are an expert technical report writer, helping to plan a report.

The report will be focused on the following topic:
{topic}

The report structure will follow these guidelines:
{report_organization}

Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information for planning the report sections.

The query should:
1. Be related to the topic
2. Help satisfy the requirements specified in the report organization

Make the query specific enough to find high-quality, relevant sources while covering the depth and breadth needed for the report structure.
"""
REPORT_PLAN_SECTION_GENERATOR_PROMPT = """You are an expert technical report writer, helping to plan a report.

Your goal is to generate the outline of the sections of the report.

The overall topic of the report is:
{topic}

The report should follow this organizational structure:
{report_organization}

You should reflect on this additional context information from web searches to plan the main sections of the report:
{search_context}

Now, generate the sections of the report. Each section should have the following fields:
- Name - Name for this section of the report.
- Description - Brief overview of the main topics and concepts to be covered in this section.
- Research - Whether to perform web search for this section of the report or not.
- Content - The content of the section, which you will leave blank for now.

Consider which sections require web search.
For example, introduction and conclusion will not require research because they will distill information from other parts of the report.
"""

llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=OPENAI_KEY)

async def generate_report_plan(state: ReportState):
    """Generate the overall plan for building the report"""
    topic = state["topic"]
    print('--- Generating Report Plan ---')

    report_structure = DEFAULT_REPORT_STRUCTURE
    number_of_queries = 8

    structured_llm = llm.with_structured_output(Queries)

    system_instructions_query = REPORT_PLAN_QUERY_GENERATOR_PROMPT.format(
        topic=topic,
        report_organization=report_structure,
        number_of_queries=number_of_queries
    )

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
        system_instructions_sections = REPORT_PLAN_SECTION_GENERATOR_PROMPT.format(
            topic=topic,
            report_organization=report_structure,
            search_context=search_context
        )
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


REPORT_SECTION_QUERY_GENERATOR_PROMPT = """Your goal is to generate targeted web search queries that will gather comprehensive information for writing a technical report section.

Topic for this section:
{section_topic}

When generating {number_of_queries} search queries, ensure that they:
1. Cover different aspects of the topic (e.g., core features, real-world applications, technical architecture)
2. Include specific technical terms related to the topic
3. Target recent information by including year markers where relevant (e.g., "2024")
4. Look for comparisons or differentiators from similar technologies/approaches
5. Search for both official documentation and practical implementation examples

Your queries should be:
- Specific enough to avoid generic results
- Technical enough to capture detailed implementation information
- Diverse enough to cover all aspects of the section plan
- Focused on authoritative sources (documentation, technical blogs, academic papers)"""

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
    system_instructions = REPORT_SECTION_QUERY_GENERATOR_PROMPT.format(section_topic=section.description,                                                                       number_of_queries=number_of_queries)
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

SECTION_WRITER_PROMPT = """You are an expert technical writer crafting one specific section of a technical report.

Title for the section:
{section_title}

Topic for this section:
{section_topic}

Guidelines for writing:

1. Technical Accuracy:
- Include specific version numbers
- Reference concrete metrics/benchmarks
- Cite official documentation
- Use technical terminology precisely

2. Length and Style:
- Strict 150-200 word limit
- No marketing language
- Technical focus
- Write in simple, clear language do not use complex words unnecessarily
- Start with your most important insight in **bold**
- Use short paragraphs (2-3 sentences max)

3. Structure:
- Use ## for section title (Markdown format)
- Only use ONE structural element IF it helps clarify your point:
  * Either a focused table comparing 2-3 key items (using Markdown table syntax)
  * Or a short list (3-5 items) using proper Markdown list syntax:
    - Use `*` or `-` for unordered lists
    - Use `1.` for ordered lists
    - Ensure proper indentation and spacing
- End with ### Sources that references the below source material formatted as:
  * List each source with title, date, and URL
  * Format: `- Title : URL`

3. Writing Approach:
- Include at least one specific example or case study if available
- Use concrete details over general statements
- Make every word count
- No preamble prior to creating the section content
- Focus on your single most important point

4. Use this source material obtained from web searches to help write the section:
{context}

5. Quality Checks:
- Format should be Markdown
- Exactly 150-200 words (excluding title and sources)
- Careful use of only ONE structural element (table or bullet list) and only if it helps clarify your point
- One specific example / case study if available
- Starts with bold insight
- No preamble prior to creating the section content
- Sources cited at end
- If there are special characters in the text, such as the dollar symbol,
  ensure they are escaped properly for correct rendering e.g $25.5 should become \$25.5
- Maintain 3rd person academic tone
- Include 1-2 inline citations per paragraph
- Highlight conflicting research findings
"""

def write_section(state: SectionState):
    """ Write a section of the report """

    # Get state
    section = state["section"]
    source_str = state["source_str"]
    print('--- Writing Section : '+ section.name +' ---')
    # Format system instructions
    system_instructions = SECTION_WRITER_PROMPT.format(section_title=section.name, 
                                                       section_topic=section.description,                                                       
                                                       context=source_str)
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

FINAL_SECTION_WRITER_PROMPT = """You are an expert technical writer crafting a section that synthesizes information from the rest of the report.

Title for the section:
{section_title}

Topic for this section:
{section_topic}

Available report content of already completed sections:
{context}

1. Section-Specific Approach:

For Introduction:
- Use # for report title (Markdown format)
- 50-100 word limit
- Write in simple and clear language
- Focus on the core motivation for the report in 1-2 paragraphs
- Use a clear narrative arc to introduce the report
- Include NO structural elements (no lists or tables)
- No sources section needed

For Conclusion/Summary:
- Use ## for section title (Markdown format)
- 100-150 word limit
- For comparative reports:
    * Must include a focused comparison table using Markdown table syntax
    * Table should distill insights from the report
    * Keep table entries clear and concise
- For non-comparative reports:
    * Only use ONE structural element IF it helps distill the points made in the report:
    * Either a focused table comparing items present in the report (using Markdown table syntax)
    * Or a short list using proper Markdown list syntax:
      - Use `*` or `-` for unordered lists
      - Use `1.` for ordered lists
      - Ensure proper indentation and spacing
- End with specific next steps or implications
- No sources section needed

3. Writing Approach:
- Use concrete details over general statements
- Make every word count
- Focus on your single most important point

4. Quality Checks:
- For introduction: 50-100 word limit, # for report title, no structural elements, no sources section
- For conclusion: 100-150 word limit, ## for section title, only ONE structural element at most, no sources section
- Markdown format
- Do not include word count or any preamble in your response
- If there are special characters in the text, such as the dollar symbol,
  ensure they are escaped properly for correct rendering e.g $25.5 should become \$25.5"""


def write_final_sections(state: SectionState):
    """ Write the final sections of the report, which do not require web search and use the completed sections as context"""

    # Get state
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]

    print('--- Writing Final Section: '+ section.name + ' ---')
    # Format system instructions
    system_instructions = FINAL_SECTION_WRITER_PROMPT.format(section_title=section.name,
                                                             section_topic=section.description,
                                                             context=completed_report_sections)

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
    clarifying_question = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="")  # No human input is provided here
    ]).content.strip()
    return clarifying_question

def ask_for_clarification(state: ReportState) -> ReportState:
    max_attempts = 5
    attempts = state.get("clarification_attempts", 0)
    
    while attempts < max_attempts:
        # Dynamically generate a clarifying question based on the current topic
        clarifying_question = generate_clarification_question(state["topic"])
        user_response = input(clarifying_question + "\n")
        # Append the user's response to the topic
        state["topic"] += " " + user_response
        
        # Evaluate if sufficient context is now provided
        evaluation_prompt = (
            f"Based on the combined topic '{state['topic']}', does this provide enough context for a focused report? "
            "Answer 'yes' if sufficient or 'no' if further clarification is needed."
        )
        evaluation = llm.invoke([
            SystemMessage(content=evaluation_prompt),
            HumanMessage(content="")
        ]).content.strip().lower()
        
        attempts += 1
        state["clarification_attempts"] = attempts
        
        if evaluation.startswith("yes"):
            # Combine all clarifications into a final, refined topic.
            combine_prompt = (
                f"Combine the original topic and all the clarifications provided to create a focused, refined topic. "
                f"Current topic: '{state['topic']}'"
            )
            refined_topic = llm.invoke([
                SystemMessage(content=combine_prompt),
                HumanMessage(content="")
            ]).content.strip()
            print(f"Refined Topic: {refined_topic}")
            state["topic"] = refined_topic
            break

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
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(formatted_sections)
        print(f"\n--- Report saved as {filename} ---")

    return {"final_report": formatted_sections}


# Update the ResearchChatbot to incorporate clarification before proceeding
class ResearchChatbot:
    def __init__(self, agent):
        self.agent = agent
        self.console = Console()
        
    async def handle_input(self, topic: str):
        """Process user research request with clarification step."""
        self.console.print(f"\n[bold green]Initial Topic: {topic}[/bold green]")
        # First, run the clarification using the LLM-assisted node
        # Here we simulate a synchronous call for clarification
        clarified_state = ask_for_clarification({"topic": topic})
        refined_topic = clarified_state["topic"]
        self.console.print(f"\n[bold blue]Refined Topic: {refined_topic}[/bold blue]")
        # Then pass the refined topic to the planner agent
        await call_planner_agent(self.agent, refined_topic)



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
            return event['final_report']  # Return for API integration

VALID_TOPIC_REGEX = r'^[a-zA-Z0-9\s\-,.:;()]{10,200}$'


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