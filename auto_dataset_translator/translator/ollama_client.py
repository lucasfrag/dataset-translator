import ollama

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
    ):

        self.model = model
        self.target_lang = target_lang
        self.source_lang = source_lang or "English"
        self.debug = debug
        self.cache = TranslationCache()

        self.retry_config = retry_config or RetryConfig()

    # -------------------------
    # FEW-SHOT PROMPT (CRITICAL)
    # -------------------------

    def _build_messages(self, text):

        examples = [
            ("Hello world", "Olá mundo"),
            ("Good morning", "Bom dia"),
            ("Test statement", "Declaração de teste"),
            ("Opinion about ML", "Opinião sobre ML"),
            ("Machine learning is amazing", "Aprendizado de máquina é incrível"),
            ("Dataset statement", "Declaração do conjunto de dados"),
        ]

        few_shot = ""

        for src, tgt in examples:
            few_shot += (
                f"{self.source_lang}: {src}\n"
                f"{self.target_lang}: {tgt}\n\n"
            )

        prompt = (
            "Translate the following text.\n"
            "Return ONLY the translation.\n\n"
            f"{few_shot}"
            f"{self.source_lang}: {text}\n"
            f"{self.target_lang}:"
        )

        return [
            {
                "role": "system",
                "content": "You are a translation engine.",
            },
            {
                "role": "user",
                "content": prompt,
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

        # remove trailing punctuation artifacts
        text = text.strip()

        return text

    # -------------------------
    # OLLAMA CALL
    # -------------------------

    def _translate_api(self, text):

        messages = self._build_messages(text)

        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": 0,
                "top_p": 1,
                "repeat_penalty": 1.0,
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
            print(f"SOURCE: {text}")
            print(f"TARGET: {translated}")

        # SAVE CACHE
        self.cache.set(
            text,
            translated,
            self.model,
            self.target_lang,
        )

        return translated