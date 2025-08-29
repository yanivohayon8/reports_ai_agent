from backend.core.user_interface import ConsoleChat


def test_console_chat():
    func = lambda x: f"You said: {x}"
    chat = ConsoleChat(func)

    known_input = ["Hello World","Hello World 2"]
    chat.start(known_input=known_input)