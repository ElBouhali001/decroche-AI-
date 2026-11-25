import pytest

from backend.support import TelecomSupportAgent
from backend.text_numbers import extract_mobile_number, words_to_digits


@pytest.mark.parametrize("text", [
    "Mon numéro est 0612345678",
    "Mon numéro est 06 12 34 56 78",
    "c'est le 06.12.34.56.78",
    "+33 6 12 34 56 78",
    "0033612345678",
    "zéro six douze trente-quatre cinquante-six soixante-dix-huit",
])
def test_french_mobile_formats(text):
    assert extract_mobile_number(text, "fr") == "0612345678"


def test_indian_format_still_works():
    assert extract_mobile_number("My number is 9876543210", "en") == "9876543210"
    assert extract_mobile_number("nine eight seven six five four three two one zero", "en") == "9876543210"


def test_no_false_positive_on_ordinary_sentence():
    assert extract_mobile_number("j'ai un problème depuis deux jours", "fr") is None


def test_french_compound_numbers():
    assert words_to_digits("soixante-dix-sept quatre-vingt-dix-neuf vingt et un", "fr") == "77 99 21"
    assert words_to_digits("quatre-vingts", "fr") == "80"


def test_agent_accepts_french_number():
    agent = TelecomSupportAgent("fr")
    agent.reply("Ma carte SIM ne marche pas")
    answer = agent.reply("Mon numéro est le 06 12 34 56 78")
    assert agent.state.mobile_number == "0612345678"
    assert "nom" in answer.lower()
