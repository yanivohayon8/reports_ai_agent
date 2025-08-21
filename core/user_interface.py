import colorama
from colorama import Fore, Back, Style
import time
import os
from typing import Callable

# Initialize colorama for cross-platform colored output
colorama.init(autoreset=True)

class ConsoleChat:

    EXIT_TERMS = ["exit", "quit", "bye", "goodbye"]

    def __init__(self, processor_func: Callable[[str], str]):
        self.processor_func = processor_func
        self.next_known_input_index = None

    def start(self, known_input:list[str]= None):
        """
        Enhanced console chat interface with emojis, colors, and optional processing.
        
        Args:
            processor_func: function that takes user input and returns response.
        """
        if known_input:
            self.next_known_input_index = 0
        
        self._print_welcome_message()
        
        user_input = self._get_user_input(known_input)

        while user_input.lower() not in self.EXIT_TERMS:
            # Handle special commands
            if user_input.lower() in ["help", "?", "h"]:
                self._print_help_message()
            elif user_input.lower() in ["clear", "cls"]:
                os.system('cls' if os.name == 'nt' else 'clear')
                self._print_welcome_message()
            elif user_input.strip() == "":
                print(f"{Fore.YELLOW}💭 Please type something to continue...\n")
            else:
                try:
                    response = self.processor_func(user_input)
                    self._print_ai_response(response)
                except Exception as e:
                    self._print_error_message(f"Processing error: {str(e)}")
            
            user_input = self._get_user_input(known_input)
        
        self._print_goodbye_message()

    def _print_welcome_message(self):
        """Display a beautiful welcome message with emojis and colors."""
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
        
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.YELLOW}🤖 AI Assistant Console Chat {Fore.CYAN}🤖")
        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.GREEN}✨ Welcome! I'm here to help you with your questions! ✨")
        print(f"{Fore.BLUE}💡 Type your questions and I'll do my best to assist you!")
        print(f"{Fore.MAGENTA}🚪 To exit, type: {Fore.WHITE}exit, quit, bye, or goodbye")
        print(f"{Fore.CYAN}{'='*60}\n")

    def _print_goodbye_message(self):
        """Display a friendly goodbye message."""
        print(f"\n{Fore.YELLOW}{'='*60}")
        print(f"{Fore.GREEN}👋 Thanks for chatting with me! 👋")
        print(f"{Fore.BLUE}🌟 Have a wonderful day! 🌟")
        print(f"{Fore.YELLOW}{'='*60}\n")
        time.sleep(1.5)  # Brief pause for effect

    def _get_user_input(self,known_input:list[str]= None):
        """Get user input with a beautiful, animated prompt."""
        if known_input is not None:
            if self.next_known_input_index < len(known_input):
                user_input = known_input[self.next_known_input_index]
                self.next_known_input_index += 1
                return user_input
            else:
                return self.EXIT_TERMS[0]
        else:
            try:
                # Animated typing indicator
                prompt = f"You: "
                user_input = input(prompt)
                
                # Add a small delay for better UX
                if user_input.strip():
                    print(f"{Fore.CYAN}⏳ Processing your request...\n")
                    time.sleep(0.5)
                
                return user_input.strip()
            
            except KeyboardInterrupt:
                print(f"\n{Fore.RED}⚠️  Interrupted by user")
                return "exit"
            except EOFError:
                print(f"\n{Fore.RED}⚠️  End of input reached")
                return "exit"

    def _print_ai_response(self, response_text):
        """Display AI response with beautiful formatting."""
        print(f"{Fore.YELLOW}🤖 AI Assistant: {Fore.WHITE}{response_text}")
        print(f"{Fore.CYAN}{'─'*60}\n")

    def _print_error_message(self, error_msg):
        """Display error messages with appropriate styling."""
        print(f"{Fore.RED}❌ Error: {Fore.WHITE}{error_msg}")
        print(f"{Fore.CYAN}{'─'*60}\n")

    def _print_help_message(self):
        """Display help information."""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.YELLOW}📚 Help & Commands {Fore.CYAN}📚")
        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.GREEN}💬 Just type your questions naturally!")
        print(f"{Fore.BLUE}🔍 I can help with document analysis, Q&A, and more!")
        print(f"{Fore.MAGENTA}🚪 Exit commands: {Fore.WHITE} {self.EXIT_TERMS}")
        print(f"{Fore.CYAN}{'='*60}\n")


