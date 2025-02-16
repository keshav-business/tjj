def handle_userinput(user_question: str, user_id: str):
    user_state = get_user_state(user_id)
    conversation_chain = user_state['chain']
    if not conversation_chain:
        return None

    # Get answer directly without refinement
    answer = conversation_chain({'question': user_question})['answer'].strip()

    # Transform the answer: replace periods between numbers with "point", and other periods with commas
    transformed_answer = transform_ai_text(answer)

    # Log events and update history
    user_state['history'].append((user_question, transformed_answer))
    log_event("PromptSentToGPT", f"Original Prompt: {user_question}", user_id=user_id)
    log_event("UserQuestion", f"Q: {user_question}", user_id=user_id)
    log_event("AIAnswer", f"A: {transformed_answer}", user_id=user_id)

    return {'text': transformed_answer}

def transform_ai_text(text):
    import re
    
    # First, handle periods between numbers (replace with "point")
    text = re.sub(r'(\d)\.(\d)', r'\1point\2', text)
    
    # Then replace remaining periods with commas
    text = text.replace('.', ',')
    
    return text
