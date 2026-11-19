import json
import re
from datetime import date, time
from typing import Literal

from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Booking
from app.services.llm import get_llm_service


class BookingDecision(BaseModel):
    status: Literal["not_booking", "collecting", "confirm", "confirmed"]
    reply: str
    name: str | None = None
    email: str | None = None
    booking_date: str | None = None
    booking_time: str | None = None


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    start, end = cleaned.find("{"), cleaned.rfind("}")

    if start < 0 or end < start:
        raise ValueError("No JSON object found.")

    return json.loads(cleaned[start:end + 1])


def _get_previous_details(
    history: list[dict[str, str]],
) -> dict[str, str | None]:
    """Recover details from the assistant's previous booking summary."""

    details = {
        "name": None,
        "email": None,
        "booking_date": None,
        "booking_time": None,
    }

    for item in reversed(history):
        if item.get("role") != "assistant":
            continue

        content = item.get("content", "")

        patterns = {
            "name": r"Name:\s*(.+?)(?:,?\s*Email:|$)",
            "email": r"Email:\s*([^\s,]+)",
            "booking_date": r"Date:\s*(\d{4}-\d{2}-\d{2})",
            "booking_time": r"Time:\s*(\d{1,2}:\d{2})",
        }

        for field, pattern in patterns.items():
            if details[field] is None:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    details[field] = match.group(1).strip().rstrip(",")

        if all(details.values()):
            break

    return details


def handle_interview_booking(
    message: str,
    history: list[dict[str, str]],
    db: Session,
) -> str | None:

    history_text = "\n".join(
        f"{item.get('role', 'user')}: {item.get('content', '')}"
        for item in history[-10:]
    )

    booking_context = any(
        item.get("role") == "assistant"
        and "interview" in item.get("content", "").lower()
        and any(
            word in item.get("content", "").lower()
            for word in ("email", "date", "time", "name")
        )
        for item in history[-6:]
    )

    booking_intent = any(
        word in message.lower()
        for word in ("interview", "book", "booking", "schedule", "reschedule")
    )

    if not booking_intent and not booking_context:
        return None

    previous = _get_previous_details(history)

    prompt = f"""
You manage interview bookings for MediAssist AI.

Extract only details explicitly provided by the user.
Use the conversation history to retain previously supplied details.

Return ONLY valid JSON with:
status, reply, name, email, booking_date, booking_time.

Allowed statuses:
collecting, confirm, confirmed.

Rules:
- collecting: one or more required details are missing.
- confirm: all details are present but the user has not confirmed.
- confirmed: all details are present and the user explicitly confirmed.
- Never invent information.
- Use YYYY-MM-DD for dates.
- Use HH:MM for times.
- Ask only for missing details.
- Never claim a booking is saved before confirmation.

Conversation:
{history_text}

Current message:
{message}
"""

    try:
        raw = get_llm_service().generate(prompt, temperature=0.0)
        decision = BookingDecision.model_validate(_extract_json(raw))
    except (ValueError, ValidationError, json.JSONDecodeError):
        return (
            "I can help arrange your interview. Please provide "
            "your full name, email, preferred date and preferred time."
        )

    if decision.status == "not_booking":
        return None

    name = decision.name or previous["name"]
    email = decision.email or previous["email"]
    booking_date_text = decision.booking_date or previous["booking_date"]
    booking_time_text = decision.booking_time or previous["booking_time"]

    try:
        if not all([name, email, booking_date_text, booking_time_text]):
            raise ValueError("Missing booking details.")

        name = name.strip()
        email = email.strip().lower()

        if not name or len(name) > 100:
            raise ValueError("Invalid name.")

        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            raise ValueError("Invalid email.")

        parsed_date = date.fromisoformat(booking_date_text)
        parsed_time = time.fromisoformat(booking_time_text)

    except (ValueError, TypeError, AttributeError):
        return (
            "I still need a valid name, email, date (YYYY-MM-DD), "
            "and time (HH:MM). Please provide the missing information."
        )

    confirmation = bool(
        re.search(
            r"\b(yes|confirm|confirmed|correct|proceed|go ahead)\b",
            message,
            re.IGNORECASE,
        )
    )

    if not confirmation:
        return (
            f"Please confirm your interview details:\n"
            f"Name: {name}\n"
            f"Email: {email}\n"
            f"Date: {parsed_date.isoformat()}\n"
            f"Time: {parsed_time.strftime('%H:%M')}\n"
            "Reply YES to confirm."
        )

    existing = db.scalar(
        select(Booking).where(
            Booking.email == email,
            Booking.booking_date == parsed_date,
            Booking.booking_time == parsed_time,
        )
    )

    if existing:
        return (
            f"Your interview is already recorded for "
            f"{parsed_date.isoformat()} at {parsed_time.strftime('%H:%M')}."
        )

    booking = Booking(
        name=name,
        email=email,
        booking_date=parsed_date,
        booking_time=parsed_time,
    )

    try:
        db.add(booking)
        db.commit()
    except Exception:
        db.rollback()
        raise

    return (
        f"Your interview details have been successfully saved!\n"
        f"Name: {name}\n"
        f"Email: {email}\n"
        f"Date: {parsed_date.isoformat()}\n"
        f"Time: {parsed_time.strftime('%H:%M')}\n\n"
        "Please note: interview availability has not been independently verified."
    )
