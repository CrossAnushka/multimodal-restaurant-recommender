"""Command-line interface.

    python -m restaurant_rec.cli generate-data
    python -m restaurant_rec.cli build-embeddings
    python -m restaurant_rec.cli train
    python -m restaurant_rec.cli evaluate
    python -m restaurant_rec.cli recommend --user-id 5
    python -m restaurant_rec.cli recommend --cold --cuisines Italian,Japanese --price 2 --city Bayport
    python -m restaurant_rec.cli all          # run the whole pipeline end-to-end
"""

from __future__ import annotations

import argparse

from .config import CONFIG


def _cmd_generate(args):
    from .data.generate import generate_dataset, summarize

    data = generate_dataset(CONFIG)
    print(summarize(data))


def _cmd_build_embeddings(args):
    from .pipeline import build_and_cache_embeddings

    print("building text + image embeddings (frozen encoders)...")
    build_and_cache_embeddings(cfg=CONFIG)
    print("done -> artifacts/")


def _cmd_train(args):
    from .train import train_all

    train_all(CONFIG)


def _cmd_evaluate(args):
    from .evaluate import run_full_evaluation

    run_full_evaluation(CONFIG)


def _cmd_recommend(args):
    from .recommend import recommend_cold_user, recommend_for_user

    if args.cold or args.user_id is None:
        cuisines = [c.strip() for c in (args.cuisines or "").split(",") if c.strip()]
        if not cuisines:
            raise SystemExit("cold-start recommend needs --cuisines (e.g. Italian,Thai)")
        print(f"Cold-start user | cuisines={cuisines} price={args.price} city={args.city}")
        df = recommend_cold_user(cuisines, args.price, args.city, k=args.k)
    else:
        print(f"Recommendations for user {args.user_id}")
        df = recommend_for_user(args.user_id, k=args.k)
    print(df.to_string(index=False))


def _cmd_all(args):
    _cmd_generate(args)
    _cmd_build_embeddings(args)
    _cmd_train(args)
    _cmd_evaluate(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="restaurant-rec",
                                description="Multi-modal restaurant recommender")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("generate-data", help="generate the synthetic dataset").set_defaults(
        func=_cmd_generate)
    sub.add_parser("build-embeddings", help="cache DistilBERT + ResNet embeddings"
                   ).set_defaults(func=_cmd_build_embeddings)
    sub.add_parser("train", help="train CF, ranker, and cold-start models"
                   ).set_defaults(func=_cmd_train)
    sub.add_parser("evaluate", help="run the full evaluation suite").set_defaults(
        func=_cmd_evaluate)
    sub.add_parser("all", help="run generate -> embeddings -> train -> evaluate"
                   ).set_defaults(func=_cmd_all)

    rec = sub.add_parser("recommend", help="rank restaurants for a user")
    rec.add_argument("--user-id", type=int, default=None,
                     help="existing user id (warm or cold)")
    rec.add_argument("--cold", action="store_true",
                     help="treat as a brand-new user described by preferences")
    rec.add_argument("--cuisines", type=str, default=None,
                     help="comma-separated preferred cuisines (cold user)")
    rec.add_argument("--price", type=int, default=2, help="preferred price band 1-4")
    rec.add_argument("--city", type=str, default=None, help="home city (cold user)")
    rec.add_argument("--k", type=int, default=10, help="number of recommendations")
    rec.set_defaults(func=_cmd_recommend)
    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
