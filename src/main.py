import os
import sys
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, FileWriterTool
from dotenv import load_dotenv

# Load API keys from .env (For local testing only. Charm injects these automatically.)
load_dotenv()

def run_lite_crew(inputs=None):
    """
    Charm Cloud Entry Point.
    
    Args:
        inputs (dict, optional): Input data from the frontend. 
                                 Example: {'task': 'I want to build a scraping bot...'}
    """
    print("=== Initializing SaaS Launchpad Crew (Lite Version) ===")
    
    # 2. Initialize Core Tools
    search_tool = SerperDevTool()
    file_writer = FileWriterTool()

    # 3. Define Agents
    
    # Agent A: The Analyst
    analyst = Agent(
        role='Lead Product Analyst',
        goal='Analyze the user idea and produce a clear Product Requirements Document (PRD).',
        backstory=(
            "You are an expert at translating vague ideas into logical development "
            "documentation and identifying potential market risks."
        ),
        tools=[search_tool, file_writer],
        verbose=True,
        memory=True
    )

    # Agent B: The Resource Hunter
    resource_hunter = Agent(
        role='Tech Resource Scout',
        goal='Find the most suitable Python libraries, APIs, and open-source projects.',
        backstory=(
            "You are deeply familiar with the Python ecosystem and GitHub. "
            "You always find existing tools and libraries to accelerate development."
        ),
        tools=[search_tool, file_writer],
        verbose=True,
        memory=True
    )

    # Agent C: The Architect
    architect = Agent(
        role='Senior Technical Architect',
        goal='Produce the core MVP code structure based on analysis and resources.',
        backstory=(
            "You excel at rapidly building functional MVPs. "
            "You focus on code simplicity, modularity, and best practices."
        ),
        tools=[file_writer],
        verbose=True,
        memory=True
    )

    # 4. Define Tasks
    
    # [Critical Change] 
    # We use '{task}' placeholder here instead of f-string.
    # CrewAI will automatically replace {task} with the value from inputs['task'] 
    # when the platform calls crew.kickoff(inputs=...).
    
    # Task 1: Specification & Analysis
    task_analysis = Task(
        description=(
            "Analyze the user's idea: '{task}'.\n"  # <--- Matches charm.yaml input property
            "1. Define 3-5 core features (MVP scope).\n"
            "2. Identify potential technical challenges.\n"
            "3. Use search_tool to find similar products and list 2 competitors.\n"
            "Write the result to 'lite_output/1_spec.md'."
        ),
        expected_output="A Markdown document containing the feature list, technical challenges, and competitor analysis.",
        agent=analyst
    )

    # Task 2: Tech Stack Selection
    task_resources = Task(
        description=(
            "Based on the analyst's spec, recommend the technology stack.\n"
            "1. Search and list the best 3 Python libraries (include 'pip install' commands).\n"
            "2. If external APIs (e.g., OpenAI, Line, Weather) are required, list recommended providers.\n"
            "Write the result to 'lite_output/2_tech_stack.md'."
        ),
        expected_output="A Markdown document containing the Python package list and API recommendations.",
        agent=resource_hunter
    )

    # Task 3: Skeleton Code Generation
    task_coding = Task(
        description=(
            "Based on the spec and tech stack, write the core 'main.py'.\n"
            "1. This is a 'Skeleton', not the full product.\n"
            "2. Include necessary imports, class definitions, and function placeholders (pass).\n"
            "3. Add detailed comments explaining the purpose of each block.\n"
            "Write the complete Python code to 'lite_output/3_mvp_skeleton.py'."
        ),
        expected_output="An executable Python file containing the complete architectural skeleton.",
        agent=architect
    )

    # 5. Assemble the Crew
    crew = Crew(
        agents=[analyst, resource_hunter, architect],
        tasks=[task_analysis, task_resources, task_coding],
        process=Process.sequential
    )

    # [Critical Change] 
    # Do NOT call crew.kickoff() here.
    # Just return the object. The Charm Platform handles the execution.
    return crew


# --- Local Testing Logic ---
# This block runs ONLY when you execute `python src/main.py` on your machine.
# It simulates what the Charm Platform does.
if __name__ == "__main__":
    print("## Local Testing Mode ##")
    
    # 1. Simulate Frontend Input
    user_idea = input("\nPlease enter your project idea: ")
    if not user_idea:
        print("No idea entered. Exiting...")
        sys.exit()
    
    # 2. Get the Crew Object
    my_crew = run_lite_crew()
    
    # 3. Simulate Platform Execution
    # We construct the inputs dictionary just like the Frontend would.
    test_inputs = {"task": user_idea}
    
    print(f"\n[System] Kicking off crew with inputs: {test_inputs}\n")
    result = my_crew.kickoff(inputs=test_inputs)
    
    print("\n\n########################")
    print("## Workflow Complete! ##")
    print("########################\n")
    print(result)