import re

def deidentify_text(text):
    text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)
    text = re.sub(r'\b\d{2}/\d{2}/\d{4}\b', '[DATE]', text)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    return text

def preprocess_text(text):
    text = deidentify_text(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,;:!?()-]', '', text)
    text = ' '.join(text.split())
    return text