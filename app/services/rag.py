from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.services.booking import handle_interview_booking
from app.services.embeddings import get_embedding_service
from app.services.llm import get_llm_service
from app.services.memory import get_chat_memory
from app.services.vector_store import SearchHit, get_vector_store


_SYSTEM = """
You are MediAssist, a helpful administrative healthcare information assistant.

You have access to two separate information sources:

1. CONVERSATION HISTORY:
Use this to answer questions about information the user previously shared,
such as their name, preferences, and earlier messages.
Only use information that is actually present in the conversation history.
Never invent personal details.

2. UPLOADED DOCUMENT CONTEXT:
Use this source for healthcare information, clinic information, hospital
policies, services, and other factual healthcare-related questions.
Only provide healthcare information supported by the uploaded documents.
If the documents do not contain the answer, clearly say:
"I don't know based on the uploaded documents."

IMPORTANT RULES:
- Conversation history can answer questions about previous conversations.
- Uploaded documents must support healthcare and institutional information.
- Never invent hospital policies or medical instructions.
- Do not diagnose medical conditions or prescribe treatment.
- Keep answers clear, helpful, and concise.
"""


_FALLBACK = "I couldn't find anything relevant in the uploaded documents."


def _format_history(history: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history
    )


def _format_context(hits: list[SearchHit]) -> str:
    return "\n\n".join(
        f"[{hit.filename}] {hit.text}"
        for hit in hits
    )


def _rewrite_question(
    message: str,
    history: list[dict[str, str]],
) -> str:
    if not history:
        return message

    prompt = f"""
Rewrite the user's latest message as a standalone question using
the conversation history.

Do not add facts that are not present in the conversation.

Conversation:
{_format_history(history)}

Latest message:
{message}

Return only the rewritten question.
"""

    return (
        get_llm_service().generate(prompt, temperature=0.0)
        or message
    ).strip()


def answer_question(
    session_id: str,
    message: str,
    db: Session,
) -> str:

    settings = get_settings()
    memory = get_chat_memory()

    history = memory.get_history(session_id)

    booking_reply = handle_interview_booking(
        message,
        history,
        db,
    )

    if booking_reply is not None:
        memory.add_message(session_id, "user", message)
        memory.add_message(session_id, "assistant", booking_reply)
        return booking_reply

    standalone = _rewrite_question(message, history)

    embedding = get_embedding_service().embed_query(standalone)

    hits = get_vector_store().search(
        embedding,
        settings.retrieval_top_k,
    )

    hits = [
        hit for hit in hits
        if hit.score >= settings.min_score
    ]

    context = _format_context(hits)
    conversation = _format_history(history)

    prompt = f"""
CONVERSATION HISTORY:
{conversation or "No previous conversation."}

UPLOADED DOCUMENT CONTEXT:
{context or "No relevant uploaded documents were found."}

CURRENT USER MESSAGE:
{message}

INSTRUCTIONS:
First determine what kind of information the user is requesting.

If the user asks about something they previously told you:
Answer using the conversation history.

If the user asks about healthcare information, clinic information,
hospital policies, or other institutional facts:
Answer only when supported by the uploaded document context.
If the answer is missing, say:
"I don't know based on the uploaded documents."

Do not confuse information from the conversation history with
information from uploaded healthcare documents.

ANSWER:
"""

    reply = get_llm_service().generate(
        prompt,
        system=_SYSTEM,
    )

    if not reply:
        reply = _FALLBACK if not hits else (
            "I'm sorry, I couldn't generate a response."
        )

    memory.add_message(session_id, "user", message)
    memory.add_message(session_id, "assistant", reply)

    return reply
