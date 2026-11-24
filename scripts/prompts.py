"""
The four meaning-preserving prompt wordings (the independent variable).
All ask the same question and allow the same three answers; only phrasing,
tone and order differ. Report these verbatim in your paper (e.g., in a table).
"""

TEMPLATES = {
    "v1_plain": (
        "What is the sentiment of the following Stack Overflow post? "
        "Answer with one word: positive, negative, or neutral.\n\n"
        "{text}"
    ),
    "v2_formal": (
        "You are given a text written by a software developer on Stack Overflow. "
        "Please classify the sentiment polarity expressed in the text as "
        "\"positive\", \"negative\", or \"neutral\". "
        "Respond with exactly one of these three words.\n\n"
        "Text: {text}"
    ),
    "v3_casual": (
        "hey, quick one - is this dev post positive, negative or neutral? "
        "just say the one word\n\n"
        "{text}"
    ),
    "v4_reordered": (
        "{text}\n\n"
        "The text above is a Stack Overflow post. Is its sentiment positive, "
        "negative, or neutral? Reply with a single word."
    ),
}
