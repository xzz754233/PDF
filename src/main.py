import os
import sys
import glob
import PyPDF2
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from crewai_tools import FileWriterTool, SerperDevTool, GithubSearchTool, LinkupSearchTool, EXASearchTool
from dotenv import load_dotenv

# Load environment variables
_ = load_dotenv()

# --- Dynamic Configuration Functions ---

def select_pdf_file():
    """Scans the directory for PDF files and prompts the user to select one."""
    pdf_files = glob.glob("*.pdf")
    
    if not pdf_files:
        print("Error: No PDF files found in the current directory.")
        print("Please place your project PDF here and run the script again.")
        sys.exit(1)
        
    print("\n[Detected PDF Files]")
    for idx, file in enumerate(pdf_files):
        print(f"  [{idx + 1}] {file}")
    
    while True:
        try:
            choice = input("\n> Select file number (e.g., 1): ").strip()
            file_index = int(choice) - 1
            if 0 <= file_index < len(pdf_files):
                selected_pdf = pdf_files[file_index]
                print(f"Selected: {selected_pdf}")
                return selected_pdf
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def configure_llm():
    """Prompts the user to select an LLM configuration."""
    print("\n[LLM Configuration]")
    print("  [1] OpenAI GPT-4o (Standard)")
    print("  [2] OpenAI GPT-4-Turbo (Standard)")
    print("  [3] Custom / Other (e.g., AIML API, Local LLM)")

    choice = input("\n> Select option (Default 1): ").strip() or "1"
    
    # Reset OPENAI_API_BASE to ensure standard OpenAI works if previously modified
    if "OPENAI_API_BASE" in os.environ:
        del os.environ["OPENAI_API_BASE"]

    if choice == "1":
        return "gpt-4o"
    elif choice == "2":
        return "gpt-4-turbo"
    elif choice == "3":
        custom_model = input("Enter model name (e.g., openai/gpt-5-chat-latest): ").strip()
        use_custom_base = input("Do you need a custom API Base URL? (y/n): ").lower()
        if use_custom_base == 'y':
            custom_base = input("Enter API Base URL: ").strip()
            os.environ["OPENAI_API_BASE"] = custom_base
        return custom_model
    else:
        print("Invalid choice, defaulting to GPT-4o.")
        return "gpt-4o"

# --- Initialization Phase ---

# 1. Select File and Model
print("=== CrewAI System Initialization ===")
target_pdf_file = select_pdf_file()
llm_config = configure_llm()

# 2. Check Critical Keys
if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY not found in .env file.")

if not os.getenv("SERPER_API_KEY"):
    print("WARNING: SERPER_API_KEY not found. Search functionality will fail.")

# 3. Create Output Folders
output_folder = Path("project_analysis_output")
output_folder.mkdir(exist_ok=True)

resource_folder = Path("resource_output")
resource_folder.mkdir(exist_ok=True)

code_folder = Path("code_output")
code_folder.mkdir(exist_ok=True)

# 4. Initialize Tools (Conditionally)
print("\nInitializing Tools...")
file_writer = FileWriterTool()
serper_tool = SerperDevTool()

# Base tools for agents
analyst_tools = [file_writer, serper_tool]
resource_tools = [file_writer, serper_tool]

# Initialize GitHub Tool (Optional)
if os.getenv("GITHUB_TOKEN"):
    print("- Loading GithubSearchTool")
    github_search_tool = GithubSearchTool(
        gh_token=os.getenv("GITHUB_TOKEN"),
        content_types=['code', 'issue'],
        max_results=500
    )
    resource_tools.append(github_search_tool)
else:
    print("- Skipping GithubSearchTool (GITHUB_TOKEN missing)")

# Initialize Linkup Tool (Optional)
if os.getenv("LINKUP_API_KEY"):
    print("- Loading LinkupSearchTool")
    linkup_tool = LinkupSearchTool(api_key=os.getenv("LINKUP_API_KEY"))
    resource_tools.append(linkup_tool)

# Initialize EXA Tool (Optional)
if os.getenv("EXA_API_KEY"):
    print("- Loading EXASearchTool")
    exascience_tool = EXASearchTool(api_key=os.getenv("EXA_API_KEY"))
    resource_tools.append(exascience_tool)


# --- Core Logic ---

def read_pdf_content(pdf_path: str) -> str:
    """Read and extract text content from PDF"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text() + "\n"
            return text_content
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# Read the content of the selected PDF
print(f"\nReading PDF: {target_pdf_file}...")
pdf_content = read_pdf_content(target_pdf_file)

# --- Agent Definitions ---

project_analyst = Agent(
    role='Project Analyst',
    goal='Analyze project documents to identify risks, strengths, and opportunities.',
    backstory=(
        "You are a skilled project analyst who can read project documents and "
        "extract meaningful insights, risks, and opportunities. "
        "You have access to the full project document content and can research "
        "similar projects on the web to provide comprehensive market analysis."
    ),
    tools=analyst_tools,
    verbose=True,
    memory=True,
    llm=llm_config,
    allow_code_execution=False
)

resource_search_agent = Agent(
    role='Resource Search Specialist',
    goal='Efficiently locate and curate the most relevant, high-quality resources from multiple platforms to accelerate development and research.',
    backstory=(
        "A specialized research librarian with deep knowledge of developer communities, "
        "academic databases, and open-source ecosystems. Expert at evaluating resource quality, "
        "licensing compatibility, and relevance scoring."
    ),
    allow_code_execution=False,
    tools=resource_tools, # Dynamically assigned tools
    verbose=True,
    memory=True,
    llm=llm_config
)

coding_agent = Agent(
    role='Senior Full-Stack Developer & Code Architect',
    goal='Generate production-ready, modular code that accelerates development while maintaining best practices and extensibility.',
    backstory=(
        "A senior full-stack developer and architect with expertise across multiple programming languages, "
        "frameworks, and design patterns. Specializes in rapid prototyping while maintaining code quality and scalability. "
        "Expert in Python, JavaScript/TypeScript, React, Node.js, and modern development practices. "
        "Can write, execute, and debug code to solve complex problems. "
    ),
    tools=[file_writer],
    verbose=True,
    memory=True,
    allow_code_execution=False,
    llm=llm_config
)

# --- Task Definitions ---

# Task 1: Project Context Analysis
project_context_task = Task(
    description=(
        f"CRITICAL: You MUST read the PDF content from the input context and extract REAL information.\n\n"
        f"SPECIFIC INSTRUCTIONS:\n"
        f"1. Look at the 'full_content' field in the input\n"
        f"2. Extract the actual project name from the 'project_name' field\n"
        f"3. Find real requirements mentioned in the PDF content\n"
        f"4. Identify actual technology stack mentioned in the 'technology_stack' field\n"
        f"5. Extract real project objectives and scope from the PDF text\n"
        f"6. Find actual constraints and limitations mentioned\n"
        f"7. Identify real dependencies and relationships\n"
        f"8. Extract actual risks and mitigation strategies\n"
        f"9. Find real timeline and milestones from the 'timeline_info' field\n"
        f"10. Extract actual team information and roles from the 'team_info' field\n"
        f"11. Find real budget and resource information from the 'budget_info' field\n\n"
        f"WEB RESEARCH: Use SerperDevTool to search for similar AI-powered platforms and analyze:\n"
        f"- Current market leaders\n"
        f"- Their key features and pricing\n"
        f"- Market gaps and opportunities\n\n"
        f"USE FileWriterTool to write this to '{output_folder}/project_analysis.md'\n"
        f"DO NOT use placeholders like [Detail] or [Project Name] - use REAL data from the PDF!\n\n"
        f"Project Document Content:\n{pdf_content}"
    ),
    expected_output=(
        f"A REAL project analysis report saved as '{output_folder}/project_analysis.md' containing extracted data and market research."
    ),
    agent=project_analyst
)

# Task 2: Objective Clarification
objective_task = Task(
    description=(
        f"CRITICAL: Based on the ACTUAL PDF content, break down real project goals into specific objectives.\n\n"
        f"SPECIFIC INSTRUCTIONS:\n"
        f"1. Extract the actual project goals mentioned in the PDF\n"
        f"2. Identify real secondary goals from the PDF content\n"
        f"3. Find specific acceptance criteria mentioned in the PDF\n"
        f"4. Determine recommended project phases based on the real scope\n\n"
        f"WEB RESEARCH: Use SerperDevTool to research market validation strategies and industry benchmarks.\n\n"
        f"USE FileWriterTool to write this to '{output_folder}/project_objectives.md'\n"
        f"DO NOT use placeholders - use REAL data from the PDF!\n\n"
        f"Project Document Content:\n{pdf_content}"
    ),
    expected_output=(
        f"A REAL objectives document saved as '{output_folder}/project_objectives.md' containing goals, phases, and benchmarks."
    ),
    agent=project_analyst
)

# Task 3: Technical Feasibility
technical_task = Task(
    description=(
        f"CRITICAL: Evaluate the technical complexity of the ACTUAL project described in the PDF content.\n\n"
        f"SPECIFIC INSTRUCTIONS:\n"
        f"1. Analyze the real technology stack\n"
        f"2. Assess the actual project complexity based on real requirements\n"
        f"3. Identify real prerequisite skills needed\n"
        f"4. Recommend project-specific fallback solutions\n\n"
        f"WEB RESEARCH: Research latest tech stacks, API limitations, and technical challenges for this specific domain.\n\n"
        f"USE FileWriterTool to write this to '{output_folder}/technical_assessment.md'\n"
        f"Project Document Content:\n{pdf_content}"
    ),
    expected_output=(
        f"A REAL technical assessment saved as '{output_folder}/technical_assessment.md' containing complexity ratings and technical analysis."
    ),
    agent=project_analyst
)

# Task 4: Resource Planning
resource_task = Task(
    description=(
        f"CRITICAL: Based on the ACTUAL project requirements from the PDF, determine what real resources are needed.\n\n"
        f"SPECIFIC INSTRUCTIONS:\n"
        f"1. Identify real datasets needed for the specific project\n"
        f"2. Determine actual documentation requirements\n"
        f"3. Find real resources and tools needed for the specific technologies\n"
        f"4. Identify actual APIs and services required\n\n"
        f"WEB RESEARCH: Research current costs for development services, API pricing, and hosting costs.\n\n"
        f"USE FileWriterTool to write this to '{output_folder}/resource_planning.md'\n"
        f"Project Document Content:\n{pdf_content}"
    ),
    expected_output=(
        f"A REAL resource plan saved as '{output_folder}/resource_planning.md' containing dataset needs, tool requirements, and cost estimates."
    ),
    agent=project_analyst
)

# Task 5: Multi-platform resource discovery
multi_platform_discovery_task = Task(
    description=(
        f"Based on the project analysis files created by the first agent, search across GitHub, Kaggle, ArXiv, StackOverflow, and documentation sites.\n\n"
        f"SPECIFIC INSTRUCTIONS:\n"
        f"1. Read the analysis files from '{output_folder}/' to understand the project requirements\n"
        f"2. Use available search tools to find relevant resources based on the technology stack\n"
        f"3. CRITICAL: You MUST find and document AT LEAST 10 different resources across all platforms\n"
        f"4. Focus on: AI marketing tools, social media automation, content generation APIs\n\n"
        f"USE FileWriterTool to write this to '{resource_folder}/multi_platform_resources.md'\n"
    ),
    expected_output=(
        f"A comprehensive resource discovery report saved as '{resource_folder}/multi_platform_resources.md' containing at least 10 resources with links and descriptions."
    ),
    agent=resource_search_agent
)

# Task 6: Code repository analysis
code_repository_analysis_task = Task(
    description=(
        f"Based on the project analysis, analyze GitHub repositories for code quality and compatibility.\n\n"
        f"SPECIFIC INSTRUCTIONS:\n"
        f"1. Read the analysis files from '{output_folder}/'\n"
        f"2. Find relevant GitHub repositories using search tools\n"
        f"3. Evaluate repositories for quality, maintenance, and licensing\n"
        f"4. CRITICAL: You MUST find and analyze AT LEAST 10 different GitHub repositories\n\n"
        f"USE FileWriterTool to write this to '{resource_folder}/code_repositories.md'\n"
    ),
    expected_output=(
        f"A curated repository analysis saved as '{resource_folder}/code_repositories.md' containing at least 10 repositories with quality metrics."
    ),
    agent=resource_search_agent
)

# Task 7: Dataset discovery
dataset_discovery_task = Task(
    description=(
        f"Based on the project requirements, find relevant datasets on Kaggle and other portals.\n\n"
        f"SPECIFIC INSTRUCTIONS:\n"
        f"1. Read the analysis files from '{output_folder}/'\n"
        f"2. Search for relevant datasets (social media data, AI training data)\n"
        f"3. Evaluate data quality and formatting\n"
        f"4. CRITICAL: You MUST find and document AT LEAST 10 different datasets\n\n"
        f"USE FileWriterTool to write this to '{resource_folder}/datasets.md'\n"
    ),
    expected_output=(
        f"A dataset catalog saved as '{resource_folder}/datasets.md' containing at least 10 datasets with metadata and download links."
    ),
    agent=resource_search_agent
)

# Task 8: Academic paper retrieval
academic_paper_task = Task(
    description=(
        f"Based on the project scope, search for relevant papers, tutorials, and implementation guides.\n\n"
        f"SPECIFIC INSTRUCTIONS:\n"
        f"1. Read the analysis files from '{output_folder}/'\n"
        f"2. Search academic sources for AI/ML research and marketing automation papers\n"
        f"3. CRITICAL: You MUST find and document AT LEAST 10 different academic resources\n\n"
        f"USE FileWriterTool to write this to '{resource_folder}/academic_resources.md'\n"
    ),
    expected_output=(
        f"An annotated bibliography saved as '{resource_folder}/academic_resources.md' containing at least 10 resources with summaries."
    ),
    agent=resource_search_agent
)

# Task 9: Real-time monitoring
realtime_monitoring_task = Task(
    description=(
        f"Monitor for new releases, updates, or trending resources related to the project domain.\n\n"
        f"SPECIFIC INSTRUCTIONS:\n"
        f"1. Read the analysis files from '{output_folder}/'\n"
        f"2. Search for recent developments (last 6 months) in AI tools and social media APIs\n"
        f"3. CRITICAL: You MUST find and document AT LEAST 10 different recent/trending resources\n\n"
        f"USE FileWriterTool to write this to '{resource_folder}/realtime_updates.md'\n"
    ),
    expected_output=(
        f"A live resource update feed saved as '{resource_folder}/realtime_updates.md' containing at least 10 recent trends/resources."
    ),
    agent=resource_search_agent
)

# Task 10: Project Architecture Design
architecture_design_task = Task(
    description=(
        f"Design the overall system architecture based on the analysis files.\n\n"
        f"SPECIFIC INSTRUCTIONS:\n"
        f"1. Read the analysis files from '{output_folder}/'\n"
        f"2. Design a scalable, modular architecture\n"
        f"3. Define module boundaries, data flow, and technology stack\n"
        f"4. Address security and scalability\n\n"
        f"USE FileWriterTool to write this to '{code_folder}/architecture_design.md'\n"
        f"Also generate a Python script to create the folder structure."
    ),
    expected_output=(
        f"Complete architecture design saved as '{code_folder}/architecture_design.md' including diagrams and structure scripts."
    ),
    agent=coding_agent
)

# Task 11: Starter Template Generation
starter_template_task = Task(
    description=(
        f"Generate a complete project scaffolding with boilerplate code.\n\n"
        f"SPECIFIC INSTRUCTIONS:\n"
        f"1. Create a fully functional project template structure\n"
        f"2. Include dependency files (requirements.txt), config files (.env.example), and basic API endpoints\n"
        f"3. Create comprehensive setup instructions\n\n"
        f"USE FileWriterTool to write setup instructions to '{code_folder}/setup_instructions.md'\n"
        f"Generate all necessary Python files in the '{code_folder}/project_template/' directory."
    ),
    expected_output=(
        f"Complete project template saved in '{code_folder}/project_template/' with setup instructions."
    ),
    agent=coding_agent
)

# Task 12: Custom Components
custom_components_task = Task(
    description=(
        f"Generate specific functions, classes, and components for the project.\n\n"
        f"SPECIFIC INSTRUCTIONS:\n"
        f"1. Create core business logic functions\n"
        f"2. Generate AI content generation components and user management systems\n"
        f"3. Include error handling and unit tests\n\n"
        f"USE FileWriterTool to write documentation to '{code_folder}/custom_components.md'\n"
        f"Generate all Python files in the '{code_folder}/components/' directory."
    ),
    expected_output=(
        f"Custom components and functions saved in '{code_folder}/components/' with documentation."
    ),
    agent=coding_agent
)

# Task 13: API Integration
api_integration_task = Task(
    description=(
        f"Create wrapper functions and integration code for external APIs and databases.\n\n"
        f"SPECIFIC INSTRUCTIONS:\n"
        f"1. Create API clients for social media platforms and database connections\n"
        f"2. Generate authentication and rate limiting handlers\n"
        f"3. Ensure security best practices\n\n"
        f"USE FileWriterTool to write documentation to '{code_folder}/api_integrations.md'\n"
        f"Generate all Python files in the '{code_folder}/integrations/' directory."
    ),
    expected_output=(
        f"API integration code saved in '{code_folder}/integrations/' with documentation."
    ),
    agent=coding_agent
)

# Task 14: Testing Framework
testing_validation_task = Task(
    description=(
        f"Generate comprehensive unit tests and validation scripts.\n\n"
        f"SPECIFIC INSTRUCTIONS:\n"
        f"1. Create unit tests for critical functions\n"
        f"2. Generate integration tests for APIs\n"
        f"3. Set up automated testing configurations\n\n"
        f"USE FileWriterTool to write documentation to '{code_folder}/testing_framework.md'\n"
        f"Generate all test files in the '{code_folder}/tests/' directory."
    ),
    expected_output=(
        f"Testing framework saved in '{code_folder}/tests/' with comprehensive test suites."
    ),
    agent=coding_agent
)

# --- Crew Execution ---

crew = Crew(
    agents=[project_analyst, resource_search_agent, coding_agent],
    tasks=[
        # Phase 1: Analysis
        project_context_task, 
        objective_task, 
        technical_task, 
        resource_task,
        # Phase 2: Resource Discovery
        multi_platform_discovery_task,
        code_repository_analysis_task,
        dataset_discovery_task,
        academic_paper_task,
        realtime_monitoring_task,
        # Phase 3: Development
        architecture_design_task,
        starter_template_task,
        custom_components_task,
        api_integration_task,
        testing_validation_task
    ],
    process=Process.sequential
)

# Run the crew
print("\n" + "="*50)
print("STARTING WORKFLOW")
print("="*50)
print(f"Project analysis output: {output_folder}")
print(f"Resource discovery output: {resource_folder}")
print(f"Code generation output:    {code_folder}")

try:
    result = crew.kickoff()
    print("\n" + "="*50)
    print("WORKFLOW COMPLETE!")
    print("="*50)
    print("Files have been generated in the output directories.")
    print(result)
except Exception as e:
    print(f"\nCRITICAL ERROR during execution: {e}")
