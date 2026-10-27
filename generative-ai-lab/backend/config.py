import os
from typing import Dict, Optional

class AppConfig:
    def __init__(self):
        self.hf_token: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        self.gemini_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        self.openai_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.output_dir: str = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

    def update_keys(self, hf_token: Optional[str] = None, gemini_key: Optional[str] = None, openai_key: Optional[str] = None):
        if hf_token is not None:
            self.hf_token = hf_token.strip() if hf_token.strip() else None
        if gemini_key is not None:
            self.gemini_key = gemini_key.strip() if gemini_key.strip() else None
        if openai_key is not None:
            self.openai_key = openai_key.strip() if openai_key.strip() else None

    def get_status(self) -> Dict[str, bool]:
        return {
            "hf_configured": bool(self.hf_token),
            "gemini_configured": bool(self.gemini_key),
            "openai_configured": bool(self.openai_key),
        }

config = AppConfig()
