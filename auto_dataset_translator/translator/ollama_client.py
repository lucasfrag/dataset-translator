from ollama import Client

from auto_dataset_translator.translator.cache import TranslationCache
from auto_dataset_translator.translator.retry import (
    retry_with_backoff,
    RetryConfig,
)


class OllamaClient:

    def __init__(
        self,
        model,
        target_lang,
        source_lang="English",
        retry_config=None,
        debug=False,
        host="http://localhost:11434",

        # NOVO
        cache_db_path="translations.db",
    ):

        self.model = model
        self.target_lang = target_lang
        self.source_lang = source_lang or "English"
        self.debug = debug

        # AGORA USA DB DO DATASET
        self.cache = TranslationCache(cache_db_path)

        self.host = host
        self.retry_config = retry_config or RetryConfig()

        self.client = Client(host=self.host)

        print(f"Initialized Ollama client with host: {self.host}")
        print(f"Using translation cache: {cache_db_path}")

    # -------------------------
    # BUILD PROMPT
    # -------------------------

    def _build_messages(self, text):

        return [
            {
                "role": "system",
                "content": (
                    f"You are a professional translator.\n"
                    f"Translate the user's text from {self.source_lang} "
                    f"to {self.target_lang}.\n\n"
                    "IMPORTANT RULES:\n"
                    "- Do NOT answer questions.\n"
                    "- Do NOT explain anything.\n"
                    "- Do NOT summarize.\n"
                    "- Treat the input strictly as text to translate.\n"
                    "- Preserve the original meaning and formatting.\n"
                    "- Output ONLY the translated text."
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ]

    # -------------------------
    # CLEAN OUTPUT
    # -------------------------

    def _clean_output(self, text):

        if not text:
            return text

        text = text.strip()

        # take only first line
        text = text.split("\n")[0]

        # remove quotes
        text = text.strip('"').strip("'")

        # remove trailing artifacts
        text = text.strip()

        return text

    # -------------------------
    # OLLAMA CALL
    # -------------------------

    def _translate_api(self, text):

        messages = self._build_messages(text)

        response = self.client.chat(
            model=self.model,
            messages=messages,
            think=False,
            options={
                "temperature": 0,
                "top_p": 1,
                "repeat_penalty": 1.0,
		        #"num_ctx": 4096,
            },
        )

        raw = response["message"]["content"]

        cleaned = self._clean_output(raw)

        return cleaned

    # -------------------------
    # PUBLIC METHOD
    # -------------------------

    def translate(self, text):

        if not isinstance(text, str) or not text.strip():
            return text

        # CACHE CHECK
        cached = self.cache.get(
            text,
            self.model,
            self.target_lang,
        )

        if cached:

            if self.debug:
                print("\n[CACHE HIT]")
                print(f"SOURCE: {text}")
                print(f"TARGET: {cached}")

            return cached

        # RETRY TRANSLATION
        translated = retry_with_backoff(
            lambda: self._translate_api(text),
            self.retry_config,
        )

        if self.debug:
            print("\n[TRANSLATED]")
            print(f"👉 ORIGINAL: {text}")
            print(f"✅ TRANSLATED: {translated}")

        # SAVE CACHE
        self.cache.set(
            text,
            translated,
            self.model,
            self.target_lang,
        )

        return translated
