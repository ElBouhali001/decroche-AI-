"""Deterministic bilingual restaurant reservation demo used by the dashboard."""

from dataclasses import dataclass, field
import re
import secrets


@dataclass
class ReservationState:
    date: str | None = None
    time: str | None = None
    guests: int | None = None
    customer_name: str | None = None
    confirmed: bool = False
    table: int | None = None
    reference: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)


class BilingualReservationAgent:
    """Collect booking details and confirm only a complete demo reservation."""

    def __init__(self, language: str = "fr") -> None:
        if language not in {"fr", "en"}:
            raise ValueError("language must be 'fr' or 'en'")
        self.language = language
        self.state = ReservationState()

    def reply(self, customer_text: str) -> str:
        text = customer_text.strip()
        lowered = text.lower()
        self.state.messages.append({"role": "customer", "content": text})
        cancel_words = ("annuler", "annulation") if self.language == "fr" else ("cancel", "cancellation")
        if any(word in lowered for word in cancel_words):
            answer = (
                "Bien sûr. Quelle est la référence de la réservation à annuler ?"
                if self.language == "fr"
                else "Of course. What is the reservation reference you would like to cancel?"
            )
        else:
            self._extract(text)
            answer = self._next_reply()
        self.state.messages.append({"role": "agent", "content": answer})
        return answer

    def _extract(self, text: str) -> None:
        date = re.search(r"\b(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b", text)
        clock = re.search(
            r"\b([01]?\d|2[0-3])(?:h| heures?|\:([0-5]\d)|\s*(am|pm))\b",
            text.lower(),
        )
        guest_words = r"personnes?|couverts?|adultes?" if self.language == "fr" else r"people|guests?|persons?"
        guests = re.search(rf"\b(\d+)\s*(?:{guest_words})\b", text.lower())
        name_words = r"je m['’]appelle|au nom de|nom est" if self.language == "fr" else r"my name is|under the name|name is"
        name = re.search(rf"(?:{name_words})\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ -]{{1,39}})", text, re.I)
        if date:
            self.state.date = date.group(1)
        if clock:
            suffix = clock.group(3) or ""
            self.state.time = f"{clock.group(1)}h{clock.group(2) or '00'}{suffix}"
        if guests:
            self.state.guests = int(guests.group(1))
        if name:
            self.state.customer_name = name.group(1).strip(" .")

    def _next_reply(self) -> str:
        if self.state.confirmed:
            return f"Votre réservation est déjà confirmée à la table {self.state.table}. Votre référence est {self.state.reference}."
        if self.language == "en":
            if not self.state.date:
                return "Hello, welcome to the restaurant. What date would you like to book?"
            if not self.state.time:
                return "Thank you. What time would you like to come?"
            if not self.state.guests:
                return "How many people will be joining you?"
            if not self.state.customer_name:
                return "Perfect. May I have the name for the reservation?"
        else:
            if not self.state.date:
                return "Bonjour, bienvenue au restaurant. Pour quelle date souhaitez-vous réserver ?"
            if not self.state.time:
                return "Très bien. À quelle heure souhaitez-vous venir ?"
            if not self.state.guests:
                return "Pour combien de personnes, s’il vous plaît ?"
            if not self.state.customer_name:
                return "Parfait. Puis-je avoir votre nom pour la réservation ?"
        self.state.confirmed = True
        self.state.table = 32 + secrets.randbelow(2)
        self.state.reference = f"DEC-{secrets.randbelow(9000) + 1000}"
        if self.language == "en":
            return (
                f"It is confirmed! I reserved a table for {self.state.guests} people "
                f"on {self.state.date} at {self.state.time}, under the name {self.state.customer_name}. "
                f"Your table number is {self.state.table}. Your reference is {self.state.reference}."
            )
        return (
            f"C’est confirmé ! J’ai réservé une table pour {self.state.guests} personnes "
            f"le {self.state.date} à {self.state.time}, au nom de {self.state.customer_name}. "
            f"Votre table est la numéro {self.state.table}. Votre référence est {self.state.reference}."
        )


class FrenchReservationAgent(BilingualReservationAgent):
    """Backward-compatible French-only reservation agent."""

    def __init__(self) -> None:
        super().__init__("fr")
