import os
import sys
import markdown
from xhtml2pdf import pisa
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool, FileWriterTool
from pydantic import BaseModel, Field
from typing import Type
from dotenv import load_dotenv

load_dotenv()

# --- Custom Tool for PDF Generation ---
# 我們將 PDF 生成邏輯封裝成一個 CrewAI 工具，這樣 Agent 就可以在任務中呼叫它
class PDFGeneratorInput(BaseModel):
    """Input schema for PDFGeneratorTool."""
    content: str = Field(..., description="The full markdown content to convert to PDF.")
    filename: str = Field(..., description="The output filename (e.g., 'lite_output/final_report.pdf').")

class PDFGeneratorTool(BaseTool):
    name: str = "Generate PDF File"
    description: str = "Converts Markdown text into a PDF file and saves it locally."
    args_schema: Type[BaseModel] = PDFGeneratorInput

    def _run(self, content: str, filename: str) -> str:
        try:
            # 確保目錄存在
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # 定義樣式 (解決中文亂碼通常需要字型，這裡使用通用英文樣式)
            css_style = """
            <style>
                @page { size: A4; margin: 2cm; }
                body { font-family: Helvetica, sans-serif; font-size: 10pt; line-height: 1.5; }
                h1 { color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }
                h2 { color: #e67e22; margin-top: 20px; }
                pre { background-color: #f4f4f4; padding: 10px; border: 1px solid #ddd; white-space: pre-wrap; }
                code { font-family: Courier; color: #c7254e; background-color: #f9f2f4; }
            </style>
            """
            
            # 轉換 Markdown -> HTML -> PDF
            html_content = markdown.markdown(content, extensions=['fenced_code', 'codehilite'])
            full_html = f"<html><head>{css_style}</head><body>{html_content}</body></html>"

            with open(filename, "wb") as pdf_file:
                pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)

            if pisa_status.err:
                return f"Error creating PDF: {pisa_status.err}"
            return f"Successfully created PDF at: {filename}"
            
        except Exception as e:
            return f"Exception during PDF creation: {str(e)}"

# --- Main Crew Logic ---

def run_lite_crew(inputs=None):
    """
    Charm Cloud Entry Point.
    """
    print("=== Initializing SaaS Launchpad Crew (Lite Version) ===")
    
    # 1. Initialize Tools
    search_tool = SerperDevTool()
    file_writer = FileWriterTool()
    pdf_tool = PDFGeneratorTool() # 實例化我們的新工具

    # 2. Define Agents
    # Note: verbose=False turns off the console event logging
    
    # Agent A: The Analyst
    analyst = Agent(
        role='Lead Product Analyst',
        goal='Analyze the user idea and produce a clear Product Requirements Document (PRD).',
        backstory=(
            "You are an expert at translating vague ideas into logical development "
            "documentation and identifying potential market risks."
        ),
        tools=[search_tool, file_writer],
        verbose=False, # [關閉 Log]
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
        verbose=False, # [關閉 Log]
        memory=True
    )

    # Agent C: The Architect (Now also responsible for the PDF Report)
    architect = Agent(
        role='Senior Technical Architect',
        goal='Produce the MVP code structure and compile the final PDF report.',
        backstory=(
            "You excel at rapidly building functional MVPs. "
            "You allow focus on code simplicity, modularity, and best practices."
        ),
        tools=[file_writer, pdf_tool], # [新增] 給架構師 PDF 工具
        verbose=False, # [關閉 Log]
        memory=True
    )

    # 3. Define Tasks
    
    # Task 1: Specification & Analysis
    task_analysis = Task(
        description=(
            "Analyze the user's idea: '{task}'.\n"
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

    # [New Task] Task 4: Final PDF Report
    # 因為這個 task 是最後一個，CrewAI 會自動把前面 task 的 context 傳進來
    task_report = Task(
        description=(
            "Compile a comprehensive final report.\n"
            "1. Combine the content from the Analysis, Tech Stack, and Skeleton Code tasks.\n"
            "2. Organize it with clear Markdown headers (Phase 1, Phase 2, Phase 3).\n"
            "3. Use the 'Generate PDF File' tool to save this combined content to 'lite_output/final_report.pdf'.\n"
            "IMPORTANT: Do not just write a file, you MUST use the PDF tool."
        ),
        expected_output="A confirmation that the PDF has been generated at 'lite_output/final_report.pdf'.",
        agent=architect
    )

    # 4. Assemble the Crew
    crew = Crew(
        agents=[analyst, resource_hunter, architect],
        tasks=[task_analysis, task_resources, task_coding, task_report], # [新增 Task]
        process=Process.sequential,
        verbose=False # [關閉 Crew 層級的 Log]
    )

    return crew


# --- Local Testing Logic ---
if __name__ == "__main__":
    print("## Local Testing Mode ##")
    
    user_idea = input("\nPlease enter your project idea: ")
    if not user_idea:
        print("No idea entered. Exiting...")
        sys.exit()
    
    my_crew = run_lite_crew()
    test_inputs = {"task": user_idea}
    
    print(f"\n[System] Kicking off crew... (Logs are silenced)\n")
    result = my_crew.kickoff(inputs=test_inputs)
    
    print("\n\n########################")
    print("## Workflow Complete! ##")
    print("########################\n")
    print(result)