# Timeline Summerize 
python cli_timeline.py report.pdf --method refine --token-based

# QnA
python cli_qna.py manual.txt --top-k 6 --persist

# Haystack Agent
python cli_agent.py docs.pdf -q "When did the event happen?"
#agent
python cli_agent.py docs.pdf --chat
