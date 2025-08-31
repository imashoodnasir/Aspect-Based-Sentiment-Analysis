from typing import Dict, List, Tuple

SCHEMA_HEADER = (
"Task: Extract a list of (aspect, polarity) tuples from the review.\n"
"Output format: [(aspect, polarity)]* (use exact spans from the input; "
"polarity in {POS, NEU, NEG}). If none, output [].\n"
)

def build_prompt(x: str,
                 demos: List[Tuple[str, List[Tuple[str,str]]]],
                 verbalizers: Dict[str,str],
                 include_counter: int = 2) -> str:
    V = {"POS": verbalizers.get("pos","positive"),
         "NEU": verbalizers.get("neu","neutral"),
         "NEG": verbalizers.get("neg","negative")}
    parts = [SCHEMA_HEADER]
    # Add demonstrations (some with neutral/no-aspect)
    for i,(dx,dy) in enumerate(demos):
        parts.append(f"Example {i+1}:\nReview: {dx}\nAnswer: {format_tuples(dy,V)}\n")
    parts.append(f"Review: {x}\nAnswer:")
    return "\n".join(parts)

def format_tuples(tuples: List[Tuple[str,str]], V: Dict[str,str]) -> str:
    if not tuples: return "[]"
    mapped=[]
    for a,s in tuples:
        key = {"positive":"POS","neutral":"NEU","negative":"NEG"}.get(s.lower(), "NEU")
        mapped.append((a, V[key]))
    return "[" + ", ".join([f"({a}, {pol})" for a,pol in mapped]) + "]"
