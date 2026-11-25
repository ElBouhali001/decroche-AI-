from backend.llm import RestaurantAssistant


class FakeResponse:
    output_text = "Bonjour, comment puis-je vous aider ?"


class FakeClient:
    class responses:
        @staticmethod
        def create(**kwargs):
            return FakeResponse()


def test_llm_preserves_language():
    result = RestaurantAssistant(FakeClient()).respond([], "fr")
    assert result.language == "fr"
    assert result.text.startswith("Bonjour")
