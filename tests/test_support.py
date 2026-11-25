from backend.support import FrenchSupportAgent, TelecomSupportAgent


def test_french_agent_collects_and_registers_support_case():
    agent = FrenchSupportAgent()
    answer = agent.reply("Ma carte SIM n'a plus de réseau")
    assert "numéro" in answer.lower()

    answer = agent.reply("Mon numéro est 9876543210")
    assert "nom" in answer.lower()
    answer = agent.reply("Je m'appelle Pierre Martin")
    assert "ville" in answer.lower()
    answer = agent.reply("Je suis à Paris")
    assert "SIM-" in answer
    assert agent.state.resolved


def test_english_agent_registers_network_case():
    agent = TelecomSupportAgent("en")
    agent.reply("I have no network signal")
    agent.reply("My number is 9876543210")
    agent.reply("My name is Rahul Sharma")
    answer = agent.reply("I am in Delhi")
    assert "case reference" in answer
    assert "30 minutes" in answer


def test_agent_accepts_plain_name_answer_then_plain_location():
    agent = TelecomSupportAgent("en")
    agent.reply("I have no network")
    agent.reply("9876543210")
    answer = agent.reply("Meghan")
    assert answer == "In which city or area is the problem occurring?"
    answer = agent.reply("London")
    assert "registered" in answer.lower()
    assert "case reference" in answer.lower()


def test_resolved_agent_thanks_customer_without_repeating_registration():
    agent = TelecomSupportAgent("en")
    agent.reply("I have no network")
    agent.reply("9876543210")
    agent.reply("Meghan")
    agent.reply("London")
    answer = agent.reply("thank you")
    assert "welcome" in answer.lower()
    assert "reference" in answer.lower()
