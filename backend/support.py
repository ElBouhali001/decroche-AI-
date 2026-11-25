"""Deterministic bilingual telecom customer-support agent."""

from dataclasses import dataclass, field
import re
import secrets

from .text_numbers import extract_mobile_number


@dataclass
class SupportState:
    issue: str | None = None
    issue_category: str | None = None
    mobile_number: str | None = None
    customer_name: str | None = None
    location: str | None = None
    case_reference: str | None = None
    resolved: bool = False
    messages: list[dict[str, str]] = field(default_factory=list)


class TelecomSupportAgent:
    """Diagnose common mobile/SIM issues and create a demo support case."""

    def __init__(self, language: str = "fr") -> None:
        if language not in {"fr", "en"}:
            raise ValueError("language must be 'fr' or 'en'")
        self.language = language
        self.state = SupportState()

    def reply(self, customer_text: str) -> str:
        text = customer_text.strip()
        lowered = text.lower()
        self.state.messages.append({"role": "customer", "content": text})

        cancel_words = ("cancel", "cancel ticket", "cancel complaint", "annuler", "annulation")
        if any(word in lowered for word in cancel_words):
            answer = (
                "Bien sûr. Je peux annuler la demande. Quel est votre numéro de dossier ?"
                if self.language == "fr"
                else "Certainly. I can cancel the complaint. What is your case reference?"
            )
        elif self.state.resolved:
            thanks = ("thank", "thanks", "merci", "gratitude")
            goodbye = ("bye", "goodbye", "au revoir", "salut")
            if any(word in lowered for word in thanks):
                answer = (
                    "Je vous en prie. Votre demande reste enregistrée sous la référence "
                    f"{self.state.case_reference}. Passez une bonne journée."
                    if self.language == "fr"
                    else f"You’re welcome. Your request is registered under reference "
                    f"{self.state.case_reference}. Have a great day."
                )
            elif any(word in lowered for word in goodbye):
                answer = (
                    "Au revoir et bonne journée. Nous vous souhaitons une résolution rapide."
                    if self.language == "fr"
                    else "Goodbye. We hope to resolve this quickly."
                )
            else:
                answer = (
                    f"Votre dossier est déjà enregistré sous la référence {self.state.case_reference}. "
                    "Pouvons-nous vous aider avec autre chose ?"
                    if self.language == "fr"
                    else f"Your case is already registered under reference {self.state.case_reference}. "
                    "Is there anything else I can help you with?"
                )
        else:
            self._extract(text)
            answer = self._next_reply()

        self.state.messages.append({"role": "agent", "content": answer})
        return answer

    def _extract(self, text: str) -> None:
        lowered = text.lower()
        had_customer_name = self.state.customer_name is not None
        had_location = self.state.location is not None
        categories = (
            ("network", ("network", "signal", "coverage", "réseau", "antenne", "connexion")),
            ("data", ("internet", "data", "speed", "slow", "4g", "5g", "wifi", "lent", "données")),
            ("sim", ("sim", "sims", "card", "puce", "puk", "carte")),
            ("balance", ("balance", "recharge", "plan", "bill", "payment", "solde", "facture", "crédit")),
            ("calling", ("call", "outgoing", "incoming", "ring", "appel", "appels")),
        )
        if not self.state.issue_category:
            for category, keywords in categories:
                if any(keyword in lowered for keyword in keywords):
                    self.state.issue_category = category
                    break

        if not self.state.issue:
            if any(word in lowered for word in ("not working", "problem", "issue", "ne marche pas", "ne fonctionne pas", "problème", "bug")):
                self.state.issue = text.strip()

        # French numbers (06 12 34 56 78, +33..., spoken digits) and the original
        # Indian 10-digit format are both handled in text_numbers.
        mobile = extract_mobile_number(text, self.language)
        if mobile:
            self.state.mobile_number = mobile

        name_match = re.search(
            r"(?:my name is|name is|mon nom est|je m'appelle|appellez-moi|c'est)\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ .'-]{1,39})",
            text,
            re.I,
        )
        if name_match:
            self.state.customer_name = name_match.group(1).strip(" .")
        elif self.state.mobile_number and not self.state.customer_name:
            candidate = re.sub(
                r"^(?:name|naam|nom|my name is|my name|it's|c'est|it is)\s*",
                "",
                text.strip(),
                flags=re.I,
            ).strip(" .,!?")
            if re.fullmatch(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ .'-]{1,39}", candidate):
                self.state.customer_name = candidate.title()

        location_match = re.search(
            r"(?:i am in|i live in|location is|address is|area is|je suis à|j'habite à|je me trouve à|ma ville est)\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ .'-]{1,39})",
            text,
            re.I,
        )
        if location_match:
            self.state.location = location_match.group(1).strip(" .")
        elif had_customer_name and not had_location:
            candidate = re.sub(
                r"^(?:i am in|i live in|location is|address is|area is|je suis à|j'habite à|je me trouve à|ma ville est)\s*",
                "",
                text.strip(),
                flags=re.I,
            ).strip(" .,!?")
            if re.fullmatch(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ .'-]{1,39}", candidate):
                self.state.location = candidate.title()

    def _next_reply(self) -> str:
        if not self.state.issue and not self.state.issue_category:
            return (
                "Bonjour ! Vous êtes bien avec le service d’assistance mobile. Quel est exactement le problème rencontré ?"
                if self.language == "fr"
                else "Hello! This is mobile customer support. What problem are you facing?"
            )
        if not self.state.mobile_number:
            return (
                "Merci. Pouvez-vous me donner votre numéro de mobile à 10 chiffres afin que je vérifie le compte ?"
                if self.language == "fr"
                else "Thank you. Please provide your 10-digit mobile number so I can check the account."
            )
        if not self.state.customer_name:
            return (
                "Parfait. Quel est le nom associé à la ligne ?"
                if self.language == "fr"
                else "Perfect. Please tell me the name on the account."
            )
        if not self.state.location:
            return (
                "Dans quelle ville ou quel secteur rencontrez-vous ce problème ?"
                if self.language == "fr"
                else "In which city or area is the problem occurring?"
            )

        action_fr = "vérifier le service d’appel et l’état du compte"
        action_en = "calling service and account status check"
        if not self.state.resolved:
            self.state.resolved = True
            self.state.case_reference = f"SIM-{secrets.randbelow(900000) + 100000}"

            if self.state.issue_category == "network":
                action_fr = "réinitialiser la couverture réseau et vérifier le signal des antennes à proximité"
                action_en = "network coverage reset and nearby tower signal check"
            elif self.state.issue_category == "data":
                action_fr = "vérifier les données mobiles, les réglages APN et le forfait"
                action_en = "mobile data/APN settings and data allowance check"
            elif self.state.issue_category == "sim":
                action_fr = "vérifier l’activation de la SIM et l’emplacement de la carte"
                action_en = "SIM activation and SIM slot check"
            elif self.state.issue_category == "balance":
                action_fr = "vérifier le solde, la recharge et le forfait actif"
                action_en = "balance, recharge and active plan check"
            else:
                action_fr = "vérifier le service d’appel et l’état du compte"
                action_en = "calling service and account status check"

        if self.state.resolved:
            if self.language == "fr":
                return (
                    f"J’ai bien enregistré votre demande. "
                    f"Nous allons {action_fr}. "
                    f"Vous recevrez une mise à jour sous 30 minutes. "
                    f"Votre référence de dossier est {self.state.case_reference}."
                )
            return (
                f"I have registered your support request. "
                f"We will now perform a {action_en}. "
                f"You should receive an update within 30 minutes. "
                f"Your case reference is {self.state.case_reference}."
            )


class FrenchSupportAgent(TelecomSupportAgent):
    def __init__(self) -> None:
        super().__init__("fr")
