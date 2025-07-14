import logging
from typing import Optional, Tuple
import typing
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.agent import AgentConfig, AgentType, ChatGPTAgentConfig
from vocode.streaming.agent.base_agent import BaseAgent, RespondAgent
from vocode.streaming.agent.factory import AgentFactory


class SpellerAgentConfig(AgentConfig, type="agent_speller"):
    pass


class SpellerAgent(RespondAgent[SpellerAgentConfig]):
    def __init__(self, agent_config: SpellerAgentConfig):
        super().__init__(agent_config=agent_config)

    async def respond(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> Tuple[Optional[str], bool]:
        return "".join(c + " " for c in human_input), False


class GeminiAgentConfig(AgentConfig, type="gemini"):
    api_key: str
    model_name: str = "gemini-pro"
    temperature: float = 0.7


class GeminiAgent(RespondAgent[GeminiAgentConfig]):
    def __init__(self, agent_config: GeminiAgentConfig):
        super().__init__(agent_config=agent_config)
        self.api_key = agent_config.api_key
        self.model_name = agent_config.model_name
        self.temperature = agent_config.temperature

    async def respond(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> Tuple[Optional[str], bool]:
        # This is fine, it just calls generate_response
        async for response, stop in self.generate_response(human_input, conversation_id):
            return response, stop

    async def generate_response(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ):
        import httpx

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": human_input}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                yield text, False
            else:
                yield "Sorry, I couldn't process your request.", False


class CustomAIAgentConfig(AgentConfig, type="custom_ai"):
    model_name: str
    api_key: str
    temperature: float = 0.7


class CustomAIAgent(RespondAgent[CustomAIAgentConfig]):
    def __init__(self, agent_config: CustomAIAgentConfig):
        super().__init__(agent_config=agent_config)
        # Initialize your AI model here
        self.model_name = agent_config.model_name
        self.api_key = agent_config.api_key

    async def respond(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> Tuple[Optional[str], bool]:
        # Implement your AI model's response generation here
        response = "Custom AI Response"  # Replace with actual AI call
        return response, False


class SpellerAgentFactory(AgentFactory):
    def create_agent(
        self, agent_config: AgentConfig, logger: Optional[logging.Logger] = None
    ) -> BaseAgent:
        if agent_config.type == AgentType.CHAT_GPT:
            return ChatGPTAgent(
                agent_config=typing.cast(ChatGPTAgentConfig, agent_config)
            )
        elif agent_config.type == "agent_speller":
            return SpellerAgent(
                agent_config=typing.cast(SpellerAgentConfig, agent_config)
            )
        elif agent_config.type == "custom_ai":
            return CustomAIAgent(
                agent_config=typing.cast(CustomAIAgentConfig, agent_config)
            )
        elif agent_config.type == "gemini":
            return GeminiAgent(
                agent_config=typing.cast(GeminiAgentConfig, agent_config)
            )
        raise Exception("Invalid agent config")
