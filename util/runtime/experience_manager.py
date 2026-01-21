# Shared experience management module to avoid circular dependencies
knowledge_experiences = []


def get_experience():
    if len(knowledge_experiences) == 0:
        return "No failure experience has been summarized yet. Proceed with your initial exploration."

    result = ""
    for idx, experience in enumerate(knowledge_experiences):
        result += f'{idx+1}. {experience}\n'
    return result


def add_experience(experience):
    """Add a new experience to the shared experience list"""
    knowledge_experiences.append(experience)


def clear_experiences():
    """Clear all experiences"""
    knowledge_experiences.clear()


def get_experience_count():
    """Get the number of experiences"""
    return len(knowledge_experiences)