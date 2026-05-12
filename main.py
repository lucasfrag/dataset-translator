from auto_dataset_translator.cli import parse_args
from auto_dataset_translator.orchestrator import run


def main():

    args = parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        columns=args.columns,
        model=args.model,
        target_lang=args.target_lang,
        source_lang=args.source_lang,
        workers=args.workers,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        force=args.force,
        reset_cache=args.reset_cache,
        reset_checkpoint=args.reset_checkpoint,
        debug=args.debug,
        host=args.host,
    )


if __name__ == "__main__":
    main()