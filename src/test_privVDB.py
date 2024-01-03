import argparse
import milvus


def test_privVDB_insert(vdb: milvus.PrivVDB, txt: str):
    embeddings = vdb.insert_text(txt)
    return embeddings


def test_privVDB_query(vdb: milvus.PrivVDB, txt: str):
    res_text, score = vdb.search_text(txt)
    return res_text, score


def test_privVDB_get_dp_text(vdb: milvus.PrivVDB, txt: str):
    dp_text = vdb.get_dp_text("agent123", txt, 1)
    return dp_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Description of your program.")

    parser.add_argument("--base_dataset_path", type=str,
                        help="Path to cbt_train.txt")

    parser.add_argument("--embedding_type", type=str, default="glove",
                        help="Type of embedding.")

    parser.add_argument("--non_sensitive_p", type=float, default=0.3,
                        help="Non-sensitive parameter.")

    parser.add_argument("--sensitive_word_percentage", type=float, default=0.9,
                        help="Percentage of sensitive words.")

    parser.add_argument("--epsilons", type=int, nargs="+", default=[1, 3],
                        help="List of epsilon values.")

    parser.add_argument("--dp_mech", type=str, default="base",
                        help="Differential Privacy mechanism.")

    parser.add_argument("--embedding_path", type=str,
                        help="Path to glove.840B.300d.txt.")

    parser.add_argument("--text_dp_type", type=str, default="santext",
                        help="Type of text differential privacy.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = {}
    dp_config = {
        "base_dataset_path": args.base_dataset_path,
        "embedding_type": args.embedding_type,
        "embedding_path": args.embedding_path,
        "non_sensitive_p": args.non_sensitive_p,
        "sensitive_word_percentage": args.sensitive_word_percentage,
        "epsilons": args.epsilons,
        "DP_mech": args.dp_mech,
        "args": args
    }
    config["text_dp_type"] = args.text_dp_type
    config["text_dp_config"] = dp_config
    privDB = milvus.PrivVDB(config)
    test_privVDB_insert("123")
