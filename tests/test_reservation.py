from backend.reservation import BilingualReservationAgent, FrenchReservationAgent


def test_french_agent_collects_and_confirms_booking():
    agent = FrenchReservationAgent()
    assert "date" in agent.reply("Bonjour, je voudrais réserver une table.")
    assert "heure" in agent.reply("Le 12/12")
    assert "personnes" in agent.reply("À 20h")
    assert "nom" in agent.reply("Pour 4 personnes")
    confirmation = agent.reply("Je m'appelle Marie Dupont")
    assert "confirmé" in confirmation
    assert "table" in confirmation
    assert agent.state.confirmed


def test_english_agent_replies_and_confirms_booking():
    agent = BilingualReservationAgent("en")
    for message in ("I want to book a table", "12/12 at 8pm", "For 4 people", "My name is John Smith"):
        answer = agent.reply(message)
    assert "confirmed" in answer
    assert "table" in answer
