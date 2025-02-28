#agent.py
import os
import re
import asyncio
import tiktoken
from dataclasses import asdict, dataclass
from dotenv import load_dotenv
from typing import Annotated, List, Optional, Literal, Union, Dict, Any, get_type_hints, get_origin, get_args
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator

# LangChain imports
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

# Visualization imports
from IPython.display import display, Image
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown

# Load environment variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

#############################################
# DATA MODELS
#############################################

# Bottom-up hierarchy models
class EvidencePoint(BaseModel):
    fact: str = Field(description="A specific fact, statistic, or example")
    source: str = Field(description="Source of this evidence")
    relevance: str = Field(description="How this evidence relates to the section topic")
    subsection: str = Field(description="Suggested subsection where this evidence belongs")

class EvidencePoints(BaseModel):
    evidence_points: List[EvidencePoint] = Field(
        description="All the Evidence Points extracted from search results.",
    )

class SubPoint(BaseModel):
    content: str = Field(description="Content for this specific point")
    sources: List[str] = Field(description="Sources supporting this point")

class Paragraph(BaseModel):
    main_idea: str = Field(description="The central idea of this paragraph")
    points: List[SubPoint] = Field(description="Supporting points for this paragraph")
    synthesized_content: str = Field(description="Final paragraph text synthesized from points")

class SubSection(BaseModel):
    title: str = Field(description="Title of this subsection")
    paragraphs: List[Paragraph] = Field(description="Paragraphs in this subsection")
    synthesized_content: str = Field(description="Final subsection text synthesized from paragraphs")

# Define subsection structure
class OrganizedSubsection(BaseModel):
    title: str = Field(description="Title for this subsection")
    evidence: List[str] = Field(description="Evidence points that belong in this subsection")
    main_points: List[str] = Field(description="Main points to cover in this subsection")

class OrganizedSubsections(BaseModel):
    subsections: List[OrganizedSubsection] = Field(
        description="All the organized subsections.",
    )

# Section model for report structure
class Section(BaseModel):
    name: str = Field(description="Name for a particular section of the report")
    description: str = Field(description="Brief overview of the main topics and concepts")
    research: bool = Field(description="Whether to perform web search for this section")
    subsections: List[SubSection] = Field(description="Subsections within this section")
    content: str = Field(description="The final content for this section")

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="All the Sections of the overall report.",
    )

# Search query models
class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of web search queries.",
    )

# State models for graph
class ReportStateInput(TypedDict):
    topic: str  # Report topic
    config: dict

class ReportStateOutput(TypedDict):
    final_report: str  # Final report

class ReportState(TypedDict):
    topic: str  # Report topic
    clarifications: List[str]  # We'll store clarifications here
    awaiting_clarification: bool
    clarifying_question: str
    clarification_attempts: int
    sections: list[Section]  # List of report sections
    completed_sections: Annotated[list, operator.add]  # Send() API
    report_sections_from_research: str  # completed sections to write final sections
    final_report: str  # Final report
    filename: str  # Filename of the report
    config: dict  # Configuration for the report
    last_feedback: str
    confirmation_summary: str 
    awaiting_confirmation: bool
    awaiting_feedback_response: bool

class SectionState(TypedDict):
    section: Section  # Report section
    search_queries: list[SearchQuery]  # List of search queries
    source_str: str  # String of formatted source content from web search
    report_sections_from_research: str  # completed sections to write final sections
    completed_sections: list[Section]  # Final key in outer state for Send() API
    evidence_points: list[EvidencePoint]  # Evidence points for this section

class SectionOutputState(TypedDict):
    completed_sections: list[Section]  # Final key in outer state for Send() API

# Helper dataclass for SearchQuery
@dataclass
class SearchQuery:
    search_query: str
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

#############################################
# UTILITIES
#############################################

# Initialize API clients
tavily_search = TavilySearchAPIWrapper()
llm = ChatOpenAI(model_name="o3-mini", temperature=None, openai_api_key=OPENAI_KEY, reasoning_effort="high")
chat = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=OPENAI_KEY)

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

#############################################
# WEB SEARCH FUNCTIONS
#############################################

async def run_search_queries(
    search_queries: List[Union[str, SearchQuery]],
    num_results: int = 5,
    include_raw_content: bool = False
) -> List[Dict]:
    search_tasks = []
    for query in search_queries:
        # Handle both string and SearchQuery objects
        query_str = query.search_query if isinstance(query, SearchQuery)else str(query)  # text query
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

    print(formatted_text.strip())
    return formatted_text.strip()

#############################################
# CLARIFICATION FUNCTIONS
#############################################

def generate_clarification_question(topic: str) -> str:
    """
    Use the LLM to generate a clarifying question tailored to the given topic.
    For instance, if the topic is 'Business', the LLM might ask:
    "What type of business report do you want? Which industry should be the focus?"
    """
    prompt = (
        f"Given the topic '{topic}', generate a clarifying question that asks the user for additional context. "
        "The question should be specific to the topic and help refine the report. "
        "Use the WWWWWH format (Who, What, When, Where, Why, How) to guide your question."
        "Begin by restating or summarizing the user's question or topic to ensure you have understood it correctly."
        "Ask the user a series of questions to gather more context about their research needs. For example:"
        f"-What specific aspect/subtopics/ angles of this topic are you most interested in?(List out possible angles/subtopics that relates to {topic})"
        "-Could you explain what you mean by [key term or concept] in your question?"
        "-Is there a particular time period, geographical region, or cultural context you want to focus on?"
        "-Are you looking for a general overview or a more detailed analysis?"
        "-Do you have any sources or examples that have influenced your perspective on this topic?"
        "-What is the goal of your research—are you preparing for an assignment, writing a paper, or just exploring the topic?"
    )
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="")
    ])
    return response.content.strip()

async def ask_for_clarification(state: ReportState) -> ReportState:
    """
    1) We generate a new clarifying question based on the current state['topic'].
    2) We set state['awaiting_clarification'] to True.
    """
    attempts = state.get("clarification_attempts", 0)
    # Generate a clarifying question dynamically based on the current topic.
    clarifying_question = generate_clarification_question(state["topic"])
    # Store the clarifying question and flag in the state.
    state["clarifying_question"] = clarifying_question
    state["awaiting_clarification"] = True
    state["clarification_attempts"] = attempts
    return state

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

    combine_prompt = (
        "Combine the base topic and all clarifications into a single refined, concise topic. "
        "Focus on capturing every important detail. No extraneous text.\n\n"
        f"Base topic: {state['topic']}\n"
        f"Clarifications: {state['clarifications']}")
    combine_response = llm.invoke(
        [SystemMessage(content=combine_prompt),
            HumanMessage(content="")])
    refined_topic = combine_response.content.strip()
    state["topic"] = refined_topic
    evaluation_prompt = f"""
    ### LLM Topic Evaluation Prompt ###
    You are evaluating whether the given topic and clarifications have enough detail to create a reasonably well-structured report.
    **Instructions:**
    1. Review the provided topic, clarifications, and user instructions.
    2. Refer to the checklist below to see if the user has provided the essential details from the rubric. 
    3. If the user has addressed most of the major points in a general or specific way, proceed with APPROVED.
    4. Only request more information if truly necessary for the user's intended scope (e.g., the user wants a highly specialized or advanced analysis).
    **Base Topic:** {state['topic']}
    **Clarifications Provided:** {state['clarifications']}
    **User Report Instructions:** {state['config']}
    ### Evaluation Criteria (Rubric) ###
    1. **Type of Research** – Has the user specified the type of research (e.g., academic paper, case study, white paper)?
    2. **Style of Report** – Has the user indicated a writing style or tone (formal, conversational, technical, etc.)?
    3. **Target Audience** – Do we know who the report is intended for?
    4. **Structural Expectations** – Has the user mentioned how they want the report organized (even if just general sections)?
    5. **Word Limit** – Is there a defined word or page count (if relevant)?
    6. **Writing Style** – Do we have at least some detail on style requirements (again, even if minimal)?
    7. **Overall Goal** – Is the main purpose or outcome of this research clear?
    8. **Topics or Angles to Cover** – Has the user identified an aspect or angle, even if broad, that they want to explore?
    If the user has largely answered these questions at a basic level, then provide approval (APPROVED). If any critical element is missing and the user's instructions cannot be followed without that info, request it (MORE_INFO_NEEDED).
    Using the base topic, clarification, and user instructions, provide feedback in this EXACT format:
    **Feedback**:
    - Missing: [list any crucial criteria that must be clarified for the research to proceed; if none, say "None"]
    - Strong: [summarize what the user has done well or provided clearly]
    - Clarification: [ask only for info that is truly necessary to proceed; if everything is sufficient, keep it minimal or say "No further details required."]
    - Suggestion: [optional ways to improve or refine the scope if relevant; otherwise keep it light]
    **Status**: [APPROVED or MORE_INFO_NEEDED]
    """
    evaluation_response = llm.invoke([
        SystemMessage(content=evaluation_prompt),
        HumanMessage(content="")
    ])
    feedback_text = evaluation_response.content
    
    # Extract status using regex
    status_match = re.search(r"(?:\*\*)?Status(?:\*\*)?:\s*(\w+)", feedback_text, re.IGNORECASE)
    status = status_match.group(1) if status_match else "MORE_INFO_NEEDED"
    print(feedback_text)
    print(status)
    
    if status.lower() == "approved":
        # Generate summary for confirmation
        summary_prompt = f"""Summarize this research request for user confirmation:
        Topic: {state['topic']}
        Config: {state['config']}
        Clarifications: {state['clarifications']}"""
        
        summary = llm.invoke([SystemMessage(content=summary_prompt)]).content
        state["awaiting_confirmation"] = True
        state["confirmation_summary"] = f"""**Please confirm**:\n{summary}\n\nRespond with 'yes' to proceed or provide additional clarifications."""
        return state
    
    # If needing more info, store feedback and await response
    state["awaiting_feedback_response"] = True
    state["last_feedback"] = feedback_text
    print(state)
    return state

async def handle_confirmation(state: ReportState, user_response: str) -> ReportState:
    """Process user confirmation response"""
    if user_response.lower().strip() in ["yes", "y", "continue", "proceed"]:
        # Finalize topic and clear flags
        state.pop("awaiting_clarification", None)
        state.pop("awaiting_confirmation", None)
        state.pop("confirmation_summary", None)
        return state
    
    # Treat non-confirmation as additional clarification
    state["clarifications"].append(user_response)
    state.pop("awaiting_confirmation", None)
    return await update_clarification(state, user_response)

#############################################
# REPORT PLANNING FUNCTIONS
#############################################

def generate_report_brief(state: ReportState):
    """Generate a brief overview of the report using LLM based on user configuration."""
    print('--- Generating Report Brief ---')
    # Get the topic and configuration from the state
    topic = state["topic"]
    config = state["config"]
    
    # Build system instructions by including the configuration details
    system_instructions = f"""
    You are a report writing assistant. Your task is to create a brief overview template for a research report.
    The report is of type: {config.get("research_type", "General")}.
    Target Audience: {config.get("target_audience", "General Audience")}.
    Expected Structure: {config.get("structure", "Introduction, Main Body, Conclusion")}.
    Section Word Limit: {config.get("section_word_limit", "No limit specified")} words.
    Writing Style: {config.get("writing_style", "Neutral")}.
    Based on the topic "{topic}", generate a concise report brief template that outlines the overall structure for the report.
    Keep your response clear and aligned with the above configuration. also add in the brief that the structure of the report should be in a markdown format.
    also When generating the final response in markdown, if there are special characters in the text,such as the dollar symbol, ensure they are escaped properly for correct rendering e.g $25.5 should become \$25.5
    """
    # Use the LLM to generate the report brief by passing the instructions and a human prompt
    report_brief = llm.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Generate a brief overview of the report.")
    ])
    
    print('--- Generating Report Brief Completed ---')
    return {"report_brief": report_brief.content}

async def generate_report_plan(state: ReportState):
    """Generate the overall plan for building the report"""
    topic = state["topic"]
    print('--- Generating Report Plan ---')
    report_structure = generate_report_brief(state)
    number_of_queries = 8
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
        structured_llm = llm.with_structured_output(Queries)
        results = structured_llm.invoke([
            SystemMessage(content=system_instructions_query),
            HumanMessage(content='Generate search queries that will help with planning the sections of the report.')
        ])
        # Convert SearchQuery objects to strings
        query_list = [
            query.search_query if isinstance(query, SearchQuery) else str(query)
            for query in results.queries
        ]
        print(query_list)
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

#############################################
# BOTTOM-UP SECTION BUILDING FUNCTIONS
#############################################

def generate_queries(state: SectionState):
    """ Generate search queries for a specific report section """
    # Get state
    section = state["section"]
    print('--- Generating Search Queries for Section: '+ section.name +' ---')
    # Get configuration
    number_of_queries = 5
    # Generate queries
    structured_llm = llm.with_structured_output(Queries)
    
    # Generate sections
    placeholders = {
        "section_topic": section.description,
        "number_of_queries": number_of_queries
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

async def collect_evidence(state: SectionState):
    """Collect specific evidence points from search results"""
    # Get state
    search_queries = state["search_queries"]
    section = state["section"]
    print(f'--- Collecting Evidence for Section: {section.name} ---')
    
    # Web search
    query_list = [query.search_query for query in search_queries]
    search_docs = await run_search_queries(search_queries, num_results=3, include_raw_content=True)
    
    # Format search results
    search_context = format_search_query_results(search_docs, max_tokens=6000, include_raw_content=True)
    
    # Extract evidence points from search results
    placeholders = {
        "section_title": section.name,
        "section_topic": section.description,
        "search_results": search_context
    }
    
    prompt_file = os.path.join("prompts", "EVIDENCE_EXTRACTION_PROMPT.txt")
    system_instructions = generate_prompt(placeholders, prompt_file)
    
    # Use structured output to get evidence points
    structured_llm = llm.with_structured_output(EvidencePoints)
    evidence_points_container = structured_llm.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Extract specific evidence points from these search results. Your response must include an 'evidence_points' field containing a list of evidence points.")
    ])
    
    print(f'--- Collected {len(evidence_points_container.evidence_points)} Evidence Points for Section: {section.name} ---')
    return {"evidence_points": evidence_points_container.evidence_points, "source_str": search_context}

def organize_subsections(state: SectionState):
    """Organize evidence into subsections"""
    # Get state
    section = state["section"]
    evidence_points = state["evidence_points"]
    
    print(f'--- Organizing Subsections for Section: {section.name} ---')
    
    # Organize evidence into subsections
    placeholders = {
        "section_title": section.name,
        "section_topic": section.description,
        "evidence_points": "\n".join([
            f"- {e.fact} (Source: {e.source})" for e in evidence_points
        ])
    }
    
    prompt_file = os.path.join("prompts", "SUBSECTION_ORGANIZATION_PROMPT.txt")
    system_instructions = generate_prompt(placeholders, prompt_file)
    
    # Use structured output to get organized subsections
    structured_llm = llm.with_structured_output(OrganizedSubsections)
    organized_subsections_container = structured_llm.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Organize these evidence points into logical subsections. Your response must include a 'subsections' field containing a list of organized subsections.")
    ])
    
    # Create subsection objects
    subsections = []
    for subsec in organized_subsections_container.subsections:
        paragraphs = []
        for point in subsec.main_points:
            # Create a paragraph for each main point
            relevant_evidence = [
                e for e in evidence_points 
                if any(keyword in e.fact.lower() for keyword in point.lower().split())
            ]
            
            sub_points = [
                SubPoint(
                    content=e.fact,
                    sources=[e.source]
                ) for e in relevant_evidence[:3]  # Limit to 3 evidence points per paragraph
            ]
            
            paragraphs.append(
                Paragraph(
                    main_idea=point,
                    points=sub_points,
                    synthesized_content=""  # Will be filled in later
                )
            )
        
        subsections.append(
            SubSection(
                title=subsec.title,
                paragraphs=paragraphs,
                synthesized_content=""  # Will be filled in later
            )
        )
    
    # Update section with subsections
    section.subsections = subsections
    
    print(f'--- Organized {len(subsections)} Subsections for Section: {section.name} ---')
    return {"section": section}

def write_paragraphs(state: SectionState):
    """Write individual paragraphs based on evidence points"""
    # Get state
    section = state["section"]
    
    print(f'--- Writing Paragraphs for Section: {section.name} ---')
    
    # Process each subsection
    for subsection in section.subsections:
        for paragraph in subsection.paragraphs:
            # Format evidence points
            evidence_text = "\n".join([
                f"- {point.content} (Source: {', '.join(point.sources)})"
                for point in paragraph.points
            ])
            
            # Generate paragraph
            placeholders = {
                "main_idea": paragraph.main_idea,
                "evidence": evidence_text,
                "section_topic": section.description,
                "subsection_title": subsection.title,
                "target_audience": state.get("config", {}).get("target_audience", "General audience"),
                "writing_style": state.get("config", {}).get("writing_style", "Academic")
            }
            
            prompt_file = os.path.join("prompts", "PARAGRAPH_WRITER_PROMPT.txt")
            system_instructions = generate_prompt(placeholders, prompt_file)
            
            # Generate paragraph
            paragraph_content = llm.invoke([
                SystemMessage(content=system_instructions),
                HumanMessage(content="Write a paragraph based on this evidence.")
            ])
            
            # Update paragraph
            paragraph.synthesized_content = paragraph_content.content
    
    print(f'--- Wrote Paragraphs for Section: {section.name} ---')
    return {"section": section}

def synthesize_subsections(state: SectionState):
    """Synthesize paragraphs into subsections"""
    # Get state
    section = state["section"]
    
    print(f'--- Synthesizing Subsections for Section: {section.name} ---')
    
    #Process each subsection
    for subsection in section.subsections:
        # Combine paragraphs
        paragraphs_text = "\n\n".join([
            paragraph.synthesized_content for paragraph in subsection.paragraphs
        ])
        
        # Generate subsection
        placeholders = {
            "subsection_title": subsection.title,
            "paragraphs": paragraphs_text,
            "section_topic": section.description,
            "target_audience": state.get("config", {}).get("target_audience", "General audience"),
            "writing_style": state.get("config", {}).get("writing_style", "Academic")
        }
        
        prompt_file = os.path.join("prompts", "SUBSECTION_SYNTHESIS_PROMPT.txt")
        system_instructions = generate_prompt(placeholders, prompt_file)
        
        # Generate subsection
        subsection_content = llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content="Synthesize these paragraphs into a cohesive subsection.")
        ])
        
        # Update subsection
        subsection.synthesized_content = subsection_content.content
    
    print(f'--- Synthesized Subsections for Section: {section.name} ---')
    return {"section": section}

def synthesize_section(state: SectionState):
    """Synthesize subsections into a complete section"""
    # Get state
    section = state["section"]
    
    print(f'--- Synthesizing Section: {section.name} ---')
    
    # Combine subsections
    subsections_text = "\n\n".join([
        subsection.synthesized_content for subsection in section.subsections
    ])
    
    # Generate section
    placeholders = {
        "section_title": section.name,
        "section_topic": section.description,
        "subsections": subsections_text,
        "target_audience": state.get("config", {}).get("target_audience", "General audience"),
        "writing_style": state.get("config", {}).get("writing_style", "Academic"),
        "section_word_limit": state.get("config", {}).get("section_word_limit", 500)
    }
    
    prompt_file = os.path.join("prompts", "SECTION_SYNTHESIS_PROMPT.txt")
    system_instructions = generate_prompt(placeholders, prompt_file)
    
    # Generate section
    section_content = llm.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Synthesize these subsections into a complete section.")
    ])
    
    # Update section
    section.content = section_content.content
    
    print(f'--- Synthesized Section: {section.name} ---')
    return {"completed_sections": [section]}


# Create section builder subgraph
section_builder = StateGraph(SectionState, output=SectionOutputState)

# Add nodes for bottom-up section building
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("collect_evidence", collect_evidence)
section_builder.add_node("organize_subsections", organize_subsections)
section_builder.add_node("write_paragraphs", write_paragraphs)
section_builder.add_node("synthesize_subsections", synthesize_subsections)
section_builder.add_node("synthesize_section", synthesize_section)

# Connect nodes in bottom-up sequence
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "collect_evidence")
section_builder.add_edge("collect_evidence", "organize_subsections")
section_builder.add_edge("organize_subsections", "write_paragraphs")
section_builder.add_edge("write_paragraphs", "synthesize_subsections")
section_builder.add_edge("synthesize_subsections", "synthesize_section")
section_builder.add_edge("synthesize_section", END)

section_builder_subagent = section_builder.compile()

#############################################
# FINAL SECTION WRITING FUNCTIONS
#############################################

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

#############################################
# FINAL REPORT COMPILATION FUNCTIONS
#############################################

def compile_final_report(state: ReportState):
    """Compile the final report with a bottom-up approach"""
    # Get sections
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}
    print('--- Compiling Final Report ---')
    
    # Update sections with completed content while maintaining original order
    for section in sections:
        section.content = completed_sections.get(section.name, "")
    
    # Extract key insights from each section
    section_insights = []
    for section in sections:
        if not section.content:
            continue
            
        insight_prompt = f"""
        Extract 3-5 key insights from this section:
        
        {section.name}:
        {section.content}
        
        Format each insight as a bullet point.
        """
        
        insights = llm.invoke([
            SystemMessage(content=insight_prompt),
            HumanMessage(content="Extract key insights")
        ]).content
        
        section_insights.append({
            "section": section.name,
            "insights": insights
        })
    
    # Generate executive summary based on insights
    insights_text = "\n\n".join([
        f"### {item['section']} Insights:\n{item['insights']}"
        for item in section_insights
    ])
    
    summary_prompt = f"""
    Create an executive summary for a report on {state['topic']}.
    
    Use these key insights from each section:
    
    {insights_text}
    
    The executive summary should:
    1. Synthesize the most important findings across all sections
    2. Highlight connections between different sections
    3. Present a coherent overview of the entire report
    4. Be approximately 250-350 words
    5. Follow the {state['config'].get('writing_style', 'Academic')} style
    6. Be appropriate for {state['config'].get('target_audience', 'General audience')}
    """
    
    executive_summary = llm.invoke([
        SystemMessage(content=summary_prompt),
        HumanMessage(content="Generate executive summary")
    ]).content
    
    # Compile final report with executive summary
    report_content = f"# {state['topic']}\n\n## Executive Summary\n\n{executive_summary}\n\n"
    
    # Add table of contents
    toc = "## Table of Contents\n\n"
    for i, section in enumerate(sections, 1):
        if section.content:
            toc += f"{i}. [{section.name}](#section-{i})\n"
    
    report_content += f"{toc}\n\n"
    
    # Add sections with anchors
    for i, section in enumerate(sections, 1):
        if section.content:
            report_content += f"<a id='section-{i}'></a>\n\n{section.content}\n\n"
    
    # Escape unescaped $ symbols to display properly in Markdown
    formatted_report = report_content.replace("\\$", "TEMP_PLACEHOLDER")
    formatted_report = formatted_report.replace("$", "\\$")
    formatted_report = formatted_report.replace("TEMP_PLACEHOLDER", "\\$")
    
    # Generate filename
    filename = generate_report_filename(formatted_report)
    file_path = os.path.join(os.getcwd(), filename)
    
    # Save to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(formatted_report)
        print(f"\n--- Report saved as {filename} ---")
    
    # Add filename to the state before returning
    state["final_report"] = formatted_report
    state["filename"] = filename
    return state

#############################################
# PARALLELIZATION FUNCTIONS
#############################################

def parallelize_section_writing(state: ReportState):
    """ This is the "map" step when we kick off web research for some sections of the report in parallel and then write the section"""
    # Kick off section writing in parallel via Send() API for any sections that require research
    return [
        Send("section_builder_with_web_search", # name of the subagent node
             {"section": s})
            for s in state["sections"]
              if s.research
    ]

def parallelize_final_section_writing(state: ReportState):
    """ Write any final sections using the Send API to parallelize the process """
    # Kick off section writing in parallel via Send() API for any sections that do not require research
    return [
        Send("write_final_sections",
             {"section": s, "report_sections_from_research": state["report_sections_from_research"]})
                 for s in state["sections"]
                    if not s.research
    ]

#############################################
# CHATBOT AND AGENT SETUP
#############################################

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
        final_report_response = await call_planner_agent(self.agent, state)
        # The final agent pipeline should produce: {'final_report': "...", 'filename': "..."}
        return final_report_response

async def call_planner_agent(agent, full_state, config={"recursion_limit": 50}):
    print(full_state)
    console = Console()
    events = agent.astream(
        full_state,
        config,
        stream_mode="values",
    )
    async for event in events:
        if 'final_report' in event:
            md = RichMarkdown(event['final_report'])
            console.print(md)
            return event  # Return for API integration

async def initialize_agent() -> StateGraph:
    
    # Create main report builder graph
    builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput)
    
    # Add nodes
    builder.add_node("ask_for_clarification", ask_for_clarification)
    builder.add_node("generate_report_plan", generate_report_plan)
    builder.add_node("section_builder_with_web_search", section_builder_subagent)
    builder.add_node("format_completed_sections", format_completed_sections)
    builder.add_node("write_final_sections", write_final_sections)
    builder.add_node("compile_final_report", compile_final_report)

    # Connect nodes
    builder.add_edge(START, "ask_for_clarification")
    builder.add_edge("ask_for_clarification", "generate_report_plan")
    builder.add_conditional_edges("generate_report_plan", parallelize_section_writing, ["section_builder_with_web_search"])
    builder.add_edge("section_builder_with_web_search", "format_completed_sections")
    builder.add_conditional_edges("format_completed_sections", parallelize_final_section_writing, ["write_final_sections"])
    builder.add_edge("write_final_sections", "compile_final_report")
    builder.add_edge("compile_final_report", END)
    
    return builder.compile()

#############################################
# MAIN EXECUTION
#############################################

async def main():
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