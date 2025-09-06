import openai
import os

from dotenv import load_dotenv

load_dotenv()


class CotMonitor:
    """A chain-of-thought monitor based on https://arxiv.org/pdf/2505.23575."""

    def __init__(self):
        """Set the model, client, and prompts."""
        self.model = "gpt-4o-mini"
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        with open("prompts/action_monitor.txt", "r") as f:
            self.action_monitor_prompt = f.read()
        with open("prompts/cot_monitor.txt", "r") as f:
            self.cot_monitor_prompt = f.read()

    def monitor_action(self, action: str) -> float:
        """Monitor only the model outputs."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": self.action_monitor_prompt.format(action=action),
                }
            ],
        )
        return float(response.choices[0].message.content)

    def monitor_cot(self, cot: str, action: str) -> float:
        """Monitor the models chain-of-thought."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": self.cot_monitor_prompt.format(cot=cot, action=action),
                }
            ],
        )
        return float(response.choices[0].message.content)

    def monitor_hybrid(self, action_score: float, cot_score: float) -> bool:
        """Use the hybrid monitoring protocol from the paper."""
        w = 0.55
        return w * action_score + (1 - w) * cot_score
