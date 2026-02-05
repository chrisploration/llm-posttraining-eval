from config import load_config


def main():
    cfg = load_config("configs/post.yaml")

    # Post-training pipeline
    model, tokenizer = build_model(cfg)
    train_dataset = load_data(cfg, tokenizer)
    run_training(cfg, model, tokenizer, train_dataset)