# cli.py

import argparse

from compute_embeddings import compute_and_persist_embeddings
# we'll import get_character_info later


def main():
    parser = argparse.ArgumentParser(
        description="LangChain Character Info CLI"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- compute-embeddings ----
    compute_parser = subparsers.add_parser(
        "compute-embeddings",
        help="Compute embeddings for all stories and store in vector DB.",
    )
    compute_parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing story files.",
    )
    compute_parser.add_argument(
        "--persist-dir",
        type=str,
        default="chroma_db",
        help="Directory where Chroma DB will be stored.",
    )

    # ---- get-character-info (will implement later) ----
    get_parser = subparsers.add_parser(
        "get-character-info",
        help="Get structured info for a given character.",
    )
    get_parser.add_argument(
        "name",
        type=str,
        help="Name of the character to search for.",
    )
    get_parser.add_argument(
        "--persist-dir",
        type=str,
        default="chroma_db",
        help="Directory where Chroma DB is stored.",
    )

    args = parser.parse_args()

    if args.command == "compute-embeddings":
        compute_and_persist_embeddings(
            data_dir=args.data_dir,
            persist_dir=args.persist_dir,
        )

    elif args.command == "get-character-info":
        # We will implement this function in the next step
        from get_character_info import get_character_info_cli

        get_character_info_cli(
            character_name=args.name,
            persist_dir=args.persist_dir,
        )


if __name__ == "__main__":
    main()
