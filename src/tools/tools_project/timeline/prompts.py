"""Prompt templates: Persona · COT · Few-Shots · Guidelines."""

from langchain.prompts import PromptTemplate

# ---- בסיס משותף ----
_PERSONA = (
    "You are a veteran Property-Insurance Claims Adjuster with 15 years of "
    "experience investigating burglary and theft losses."
)

_COT = (
    "Think step by step:\n"
    "0. Collect all KEYWORDS (names, dates, locations, items, monetary values).\n"
    "1. Extract every timestamp, date, location, actor, damage and follow-up.\n"
    "2. Standardise dates to DD-MM-YYYY and times to HH:MM.\n"
    "3. Sort chronologically (prefix unknown dates with 'Undated').\n"
    "4. Merge duplicates, flag contradictions with '⚠️'.\n"
    "5. Return only the final timeline."
)

_FEW_SHOTS = """        ### EXAMPLE 1
Input:
"At 18:05 on 07-04-2025 Ms. Jane Roe arrived home…"
Output:
• 18:05 07-04-2025  Victim discovered burglary

### EXAMPLE 2
Input:
"Early May 2024 thieves forced the balcony door…"
Output:
• Undated May 2024  Intruders forced balcony door
"""
_GUIDELINES = """
- Write in the third person; stay neutral and factual.
- Each line: '• HH:MM DD-MM-YYYY  Event'.
- Reuse the source wording for NAMES, AMOUNTS and PLACES (improves recall).
- ≤ 200 words
"""


# ----------  unified prompt template body ----------
PROMPT_HEADER = (
    f"## PERSONA\n{_PERSONA}\n\n"
    f"## CHAIN-OF-THOUGHT INSTRUCTIONS\n{_COT}\n\n"
    f"## FEW-SHOT EXAMPLES\n{_FEW_SHOTS}\n\n"
    f"## GUIDELINES\n{_GUIDELINES}\n\n"
)

def map_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["text"],
        template=PROMPT_HEADER + "## TASK INPUT\n{text}\n\n## YOUR OUTPUT\n",
    )

def reduce_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["text"],
        template=(
            PROMPT_HEADER.replace("## TASK INPUT", "## PARTIAL TIMELINES")
            + "{text}\n\n## YOUR OUTPUT\n"
        ),
    )

def initial_prompt() -> PromptTemplate:
    return map_prompt()

def refine_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=(
            PROMPT_HEADER
            + "## EXISTING TIMELINE\n{existing_answer}\n\n"
            "## NEW CHUNK\n{text}\n\n"
            "## UPDATE THE TIMELINE ACCORDING TO THE INSTRUCTIONS."
        ),
    )
