from auto_dataset_translator.dataset.loader import load_dataset
from auto_dataset_translator.dataset.writer import write_dataset
from auto_dataset_translator.translator.ollama_client import OllamaClient
from auto_dataset_translator.translator.retry import RetryConfig
from auto_dataset_translator.checkpoint.manager import CheckpointManager

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path

import os


def translate_column(
    df,
    column,
    translator,
    workers,
    dataset_name,
    checkpoint_path,
):

    # AGORA USA CHECKPOINT DO DATASET
    checkpoint = CheckpointManager(checkpoint_path)

    texts = df[column].tolist()

    results = [None] * len(texts)

    def process_row(args):

        idx, text = args

        #if checkpoint.is_done(dataset_name, column, idx):
        #    return idx, text
        if checkpoint.is_done(dataset_name, column, idx):
            translated = translate_value(text, translator)
            return idx, translated

        translated = translate_value(text, translator)

        checkpoint.mark_done(dataset_name, column, idx)

        return idx, translated

    work_items = list(enumerate(texts))

    if workers == 1:

        iterator = map(process_row, work_items)

    else:

        executor = ThreadPoolExecutor(max_workers=workers)
        iterator = executor.map(process_row, work_items)

    for idx, translated in tqdm(iterator, total=len(texts)):
        results[idx] = translated

    return results


def translate_value(value, translator):

    if isinstance(value, list):
        return [translate_value(v, translator) for v in value]

    elif isinstance(value, str):
        return translator.translate(value)

    else:
        return value


def run(
    input_path,
    output_path,
    columns,
    model,
    target_lang,
    source_lang=None,
    workers=1,
    max_retries=5,
    retry_delay=1.0,
    force=False,
    reset_cache=False,
    reset_checkpoint=False,
    debug=False,
    host="http://localhost:11434",
):

    # PEGA NOME DO DATASET SEM EXTENSÃO
    dataset_name = Path(input_path).stem

    # ARQUIVOS SEPARADOS POR DATASET
    checkpoint_path = f"{dataset_name}.checkpoint.json"
    cache_db_path = f"{dataset_name}.translations.db"

    if force:
        reset_cache = True
        reset_checkpoint = True

    if reset_cache and os.path.exists(cache_db_path):
        print("Resetting cache...")
        os.remove(cache_db_path)

    if reset_checkpoint and os.path.exists(checkpoint_path):
        print("Resetting checkpoint...")
        os.remove(checkpoint_path)

    print("Loading dataset...")
    df = load_dataset(input_path)

    print("Initializing translator...")

    retry_config = RetryConfig(
        max_retries=max_retries,
        base_delay=retry_delay,
    )

    translator = OllamaClient(
        model=model,
        target_lang=target_lang,
        source_lang=source_lang,
        retry_config=retry_config,
        debug=debug,
        host=host,

        # CACHE DO DATASET
        cache_db_path=cache_db_path,
    )

    print(f"Using Ollama host: {host}")
    print(f"Using {workers} workers")

    for col in columns:

        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")

        print(f"Translating column: {col}")

        df[col] = translate_column(
            df,
            col,
            translator,
            workers,
            dataset_name,
            checkpoint_path,
        )

    print("Writing output dataset...")
    write_dataset(df, output_path)

    print("Done.")