import os
import sys
import html
import re
import markdown
from xhtml2pdf import pisa
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field
from typing import Type
from dotenv import load_dotenv

load_dotenv()


class PDFGeneratorInput(BaseModel):
    output_filename: str = Field(..., description="The final PDF filename.")


class PDFGeneratorTool(BaseTool):
    name: str = "Generate PDF Report"
    description: str = "Compiles generated files into a PDF report."
    args_schema: Type[BaseModel] = PDFGeneratorInput

    # ---------------------------
    # Helpers to stabilize layout
    # ---------------------------
    def _strip_page_markers(self, s: str) -> str:
        """Remove lines like 'Page 2' that sometimes appear in copied/merged text."""
        return re.sub(r"(?m)^\s*Page\s+\d+\s*$", "", s or "")

    def _unwrap_hardwrap_paragraphs(self, s: str) -> str:
        """
        Join hard-wrapped lines inside paragraphs, while preserving:
        - blank lines (paragraph breaks)
        - headings (# ...)
        - list items (-, *, +, •, 1., 1))
        This prevents random mid-sentence line breaks from becoming headings.
        """
        s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = s.split("\n")

        out = []
        buf = ""

        def is_heading(line: str) -> bool:
            return bool(re.match(r"^\s*#{1,6}\s+\S", line))

        def is_list(line: str) -> bool:
            return bool(re.match(r"^\s*([-*+•]|\d+[.)])\s+\S", line))

        for raw in lines:
            line = raw.rstrip()

            # paragraph break
            if not line.strip():
                if buf:
                    out.append(buf.strip())
                    buf = ""
                out.append("")
                continue

            # structural line
            if is_heading(line) or is_list(line):
                if buf:
                    out.append(buf.strip())
                    buf = ""
                out.append(line.strip())
                continue

            # normal text line -> join
            if not buf:
                buf = line.strip()
            else:
                buf += " " + line.strip()

        if buf:
            out.append(buf.strip())

        joined = "\n".join(out)
        joined = re.sub(r"\n{3,}", "\n\n", joined).strip()
        return joined

    def _standardize_tool_list_section(self, md: str) -> str:
        """
        Standardize 'Libraries and Tools' section into consistent bullets:
        - **Tool**: description

        Handles cases where:
          - tool name is on its own line (no bullet)
          - description line accidentally starts with '•'
          - bullets are inconsistent
        """
        md = (md or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = md.split("\n")

        out = []
        i = 0

        def is_heading(l: str) -> bool:
            return bool(re.match(r"^\s*#{1,6}\s+\S", l.strip()))

        def is_list_item(l: str) -> bool:
            return bool(re.match(r"^\s*([-*+]|\d+\.)\s+\S", l.strip()))

        def clean_bullet_prefix(l: str) -> str:
            return re.sub(r"^\s*[•\-*+]\s+", "", (l or "").strip())

        def looks_like_tool_name(l: str) -> bool:
            t = (l or "").strip()
            if not t:
                return False
            if is_heading(t) or is_list_item(t):
                return False
            if len(t) > 60:
                return False
            if re.search(r"[。.!?]$", t):
                return False
            if re.match(r"^(A|An|The|This|It|Used|Provides|Part|A part)\b", t, flags=re.IGNORECASE):
                return False
            return True

        def looks_like_description(l: str) -> bool:
            t = clean_bullet_prefix(l)
            if not t:
                return False
            if is_heading(t) or is_list_item(t):
                return False
            return (
                len(t) > 30
                or bool(re.match(r"^(A|An|The|Used|Provides|Part|A part)\b", t, flags=re.IGNORECASE))
            )

        in_lib_section = False

        while i < len(lines):
            line = lines[i]
            stripped = (line or "").strip()

            # Enter Libraries and Tools section (supports both "## ..." and plain)
            if re.match(r"(?i)^\s*##\s*Libraries and Tools\s*$", stripped) or re.match(
                r"(?i)^\s*Libraries and Tools\s*$", stripped
            ):
                in_lib_section = True
                out.append("## Libraries and Tools")
                i += 1
                continue

            # Exit on next heading (excluding the Libraries heading itself)
            if in_lib_section and is_heading(line) and not re.search(r"(?i)Libraries and Tools", stripped):
                in_lib_section = False

            if not in_lib_section:
                out.append(line)
                i += 1
                continue

            # Inside Libraries and Tools:
            if not stripped:
                out.append("")
                i += 1
                continue

            candidate = clean_bullet_prefix(line)

            # If already a list item, keep it as-is (agent followed template)
            if is_list_item(line):
                out.append(line)
                i += 1
                continue

            # Try: ToolName line + next description line
            if looks_like_tool_name(candidate):
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                if (next_line or "").strip() and looks_like_description(next_line):
                    desc = clean_bullet_prefix(next_line)
                    out.append(f"- **{candidate}**: {desc}")
                    i += 2
                    continue
                else:
                    out.append(f"- **{candidate}**")
                    i += 1
                    continue

            # Otherwise keep as paragraph line (but remove weird bullet prefix)
            out.append(candidate)
            i += 1

        return "\n".join(out)

    # ---------------------------
    # Normalize messy LLM markdown into renderable markdown
    # ---------------------------
    def _normalize_markdown(self, s: str) -> str:
        """
        Make LLM outputs renderable AND stable for PDF:
        - Strip broken fences like ``markdown ... `` and ```markdown ... ```
        - Remove stray 'Page N' artifacts
        - Unwrap hard-wrapped paragraphs so sentences don't split into fake headings
        - Standardize Libraries and Tools bullet list (Phase 2)
        - Promote ONLY standalone section lines (Overview/Features/...) into headings
        - Normalize bullets and numbered lists (line-start only)
        - Add blank lines for consistent markdown parsing
        """
        if not s:
            return ""

        s = s.replace("\r\n", "\n").replace("\r", "\n")

        # 0) remove "Page N" artifacts
        s = self._strip_page_markers(s)

        # 1) Remove code fences wrappers (keep inner content)
        s = re.sub(r"```[a-zA-Z0-9_-]*\n?", "", s)
        s = s.replace("```", "")
        s = re.sub(r"``\s*markdown\s*", "", s, flags=re.IGNORECASE)
        s = s.replace("``", "")

        # 2) unwrap hard-wrapped paragraphs BEFORE any heading/list heuristics
        s = self._unwrap_hardwrap_paragraphs(s)

        # 2.5) standardize Phase-2 tool list if present (safe no-op if not)
        s = self._standardize_tool_list_section(s)

        # 3) promote ONLY standalone section lines to headings
        section_words = ["Overview", "Features", "Risks", "Competitors", "Conclusion"]
        for w in section_words:
            s = re.sub(rf"(?mi)^\s*{w}\s*$", f"## {w}", s)

        # 4) Normalize bullets and ordered list markers at line start
        s = re.sub(r"(?m)^\s*•\s+", "- ", s)
        s = re.sub(r"(?m)^\s*(\d+)\)\s+", r"\1. ", s)

        # 5) Ensure blank line after headings (helps markdown render into <p>/<ul>/<ol>)
        s = re.sub(r"(?m)^(#{1,6}\s+.+)$", r"\1\n", s)

        # cleanup
        s = re.sub(r"\n{3,}", "\n\n", s).strip()
        return s

    def _render_markdown_to_html(self, md_text: str) -> str:
        md_text = self._normalize_markdown(md_text)
        return markdown.markdown(
            md_text,
            extensions=["extra", "sane_lists"],
            output_format="xhtml",
        )

    def _run(self, output_filename: str) -> str:
        try:
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)

            css_style = """
            <style>
                @page {
                    size: A4;
                    margin: 2cm;
                    @frame footer_frame {
                        -pdf-frame-content: footerContent;
                        bottom: 1cm;
                        margin-left: 2cm;
                        margin-right: 2cm;
                        height: 1cm;
                    }
                }

                body {
                    font-family: Helvetica, sans-serif;
                    font-size: 10pt;
                    line-height: 1.45;
                    color: #333;
                }

                h1 {
                    color: #2c3e50;
                    border-bottom: 2px solid #2c3e50;
                    padding-bottom: 5px;
                    margin-top: 10px;
                    margin-bottom: 12px;
                    font-size: 18pt;
                    font-weight: bold;
                }

                h2 {
                    color: #e67e22;
                    margin-top: 12px;
                    margin-bottom: 8px;
                    border-bottom: 1px solid #eee;
                    font-size: 14pt;
                    font-weight: bold;
                }

                h3 {
                    color: #2c3e50;
                    margin-top: 10px;
                    margin-bottom: 6px;
                    font-size: 12pt;
                    font-weight: bold;
                }

                strong, b { font-weight: bold; color: #000; }

                p { margin: 0 0 8px 0; }

                ul, ol { margin: 0 0 10px 0; padding-left: 18px; }
                li { margin: 0 0 3px 0; }

                code {
                    font-family: Courier, monospace;
                    font-size: 9pt;
                    background: #f2f2f2;
                    padding: 1px 3px;
                    border-radius: 2px;
                }

                pre.code-box {
                    font-family: Courier, monospace;
                    font-size: 9pt;
                    background-color: #282c34;
                    color: #ffffff;
                    border: 1px solid #ddd;
                    padding: 10px;
                    margin-top: 10px;
                    line-height: 1.25;
                    border-radius: 4px;
                    white-space: pre;
                }
            </style>
            """

            html_parts = []
            html_parts.append(f"<html><head>{css_style}</head><body>")
            html_parts.append(
                '<div id="footerContent" style="text-align:center; font-size:9pt; color:#777;">'
                "Page <pdf:pagenumber></div>"
            )

            html_parts.append("<h1>Project Final Report</h1><p>Generated by SaaS Launchpad Crew</p>")
            html_parts.append("<hr>")

            file_map = [
                ("lite_output/1_spec.md", "Phase 1: Product Specification"),
                ("lite_output/2_tech_stack.md", "Phase 2: Tech Stack"),
            ]

            for filepath, section_title in file_map:
                if os.path.exists(filepath):
                    with open(filepath, "r", encoding="utf-8") as f:
                        raw_content = f.read()

                    html_content = self._render_markdown_to_html(raw_content)

                    html_parts.append(f"<h2>{html.escape(section_title)}</h2>")
                    html_parts.append(html_content)
                    html_parts.append("<pdf:nextpage />")

            code_path = "lite_output/3_mvp_skeleton.md"
            if os.path.exists(code_path):
                with open(code_path, "r", encoding="utf-8") as f:
                    raw_code = f.read().strip()

                # strip code fences if any
                if raw_code.startswith("```"):
                    lines = raw_code.split("\n")
                    if len(lines) >= 2:
                        raw_code = "\n".join(lines[1:-1]).rstrip("\n")

                html_parts.append("<h2>Phase 3: MVP Skeleton Code</h2>")
                html_parts.append(f"<pre class='code-box'>{html.escape(raw_code)}</pre>")

            html_parts.append("</body></html>")
            full_html = "".join(html_parts)

            # Debug if needed:
            # with open("debug.html", "w", encoding="utf-8") as f:
            #     f.write(full_html)

            with open(output_filename, "wb") as pdf_file:
                pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)

            if pisa_status.err:
                return f"Error creating PDF: {pisa_status.err}"

            for fpath in [
                "lite_output/1_spec.md",
                "lite_output/2_tech_stack.md",
                "lite_output/3_mvp_skeleton.md",
            ]:
                if os.path.exists(fpath):
                    os.remove(fpath)

            return f"Successfully compiled PDF report at: {output_filename}"

        except Exception as e:
            return f"Exception during PDF creation: {str(e)}"


# --- Main Crew Logic (核心不變，只加強 output 規格) ---

def run_lite_crew(inputs=None):
    print("=== Initializing SaaS Launchpad Crew ===")

    search_tool = SerperDevTool()
    pdf_tool = PDFGeneratorTool()

    analyst = Agent(
        role="Lead Product Analyst",
        goal="Analyze the user idea and produce a clear PRD.",
        backstory="Expert at translating vague ideas into logical docs.",
        tools=[search_tool],
        verbose=False,
        memory=True,
    )

    resource_hunter = Agent(
        role="Tech Resource Scout",
        goal="Find suitable Python libraries and APIs.",
        backstory="Familiar with Python ecosystem.",
        tools=[search_tool],
        verbose=False,
        memory=True,
    )

    architect = Agent(
        role="Senior Technical Architect",
        goal="Produce the MVP code structure.",
        backstory="Excels at building functional MVPs.",
        tools=[pdf_tool],
        verbose=False,
        memory=True,
    )

    task_analysis = Task(
        description=(
            "Analyze the user's idea: '{task}'. Define overview, features, risks, and competitors.\n"
            "FORMAT RULES:\n"
            "- Output VALID Markdown with real line breaks.\n"
            "- Use headings like: # Title, ## Overview, ## Features, ## Risks, ## Competitors, ## Conclusion.\n"
            "- Each list item MUST be on its own line.\n"
            "- Section titles must be on their own line (standalone).\n"
            "- Do NOT hard-wrap lines at a fixed width. Do not insert manual line breaks inside a sentence.\n"
            "- DO NOT wrap the whole document in code fences (no ```markdown ... ``` and no ``markdown ... ``).\n"
        ),
        expected_output=(
            "A clean Markdown document with headings and lists, properly separated by blank lines. "
            "No code fences."
        ),
        agent=analyst,
        output_file="lite_output/1_spec.md",
    )

    task_resources = Task(
        description=(
            "Recommend tech stack based on the spec.\n"
            "FORMAT RULES (MUST FOLLOW):\n"
            "- Output VALID Markdown.\n"
            "- Use sections: # Title, ## APIs, ## Libraries and Tools, ## Considerations (as needed).\n"
            "- Under '## Libraries and Tools', EVERY item MUST be in this exact format:\n"
            "  - **<ToolName>**: <One-sentence description>\n"
            "- Do NOT use '•' bullets. Use '-' only.\n"
            "- Do NOT hard-wrap lines inside a sentence.\n"
            "- DO NOT wrap the document in code fences.\n"
        ),
        expected_output=(
            "A clean Markdown document. Especially: a consistent bullet list under '## Libraries and Tools' "
            "using '- **ToolName**: description'."
        ),
        agent=resource_hunter,
        output_file="lite_output/2_tech_stack.md",
    )

    task_coding = Task(
        description=("Write the core 'main.py' skeleton code. Output ONLY raw code."),
        expected_output="Raw Python code text.",
        agent=architect,
        output_file="lite_output/3_mvp_skeleton.md",
    )

    task_report = Task(
        description=("Call 'Generate PDF Report' tool with output_filename='lite_output/final_report.pdf'. Return 'DONE'."),
        expected_output="The word 'DONE'.",
        agent=architect,
    )

    crew = Crew(
        agents=[analyst, resource_hunter, architect],
        tasks=[task_analysis, task_resources, task_coding, task_report],
        process=Process.sequential,
        verbose=False,
    )

    return crew


if __name__ == "__main__":
    print("## Local Testing Mode ##")
    user_idea = input("\nPlease enter your project idea: ")
    if not user_idea:
        sys.exit()
    my_crew = run_lite_crew()
    result = my_crew.kickoff(inputs={"task": user_idea})
    print("\n[Done] Check the lite_output folder.")