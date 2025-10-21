# bayes_opt.py
import os
import json
import tempfile
from pathlib import Path
import optuna
import argparse
from wsdan_convnext_plus import train

def objective(trial):
    outdir = Path(tempfile.mkdtemp(prefix="optuna_webfg400_"))

    # Suggest hyperparameters (tight around baseline)
    lr = trial.suggest_float("lr", 5e-5, 1.5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.03, 0.08)
    mixup = trial.suggest_float("mixup", 0.1, 0.3)
    cutmix = trial.suggest_float("cutmix", 0.2, 0.4)
    randaug_M = trial.suggest_int("randaug_M", 8, 12)
    margin = trial.suggest_float("margin", 0.25, 0.40)
    scale = trial.suggest_int("scale", 25, 35)
    elr_lambda = trial.suggest_float("elr_lambda", 2.5, 4.0)
    erase_p = trial.suggest_float("erase_p", 0.4, 0.7)

    # Fixed args (match your CLI)
    args = argparse.Namespace()
    args.mode = 'train'
    args.train_dir = "./data/WebFG-400/train"
    args.val_dir = None
    args.outdir = str(outdir)
    args.arch = "convnextv2_large"
    args.img_size = 384
    args.batch_size = 32
    args.workers = 8
    args.epochs = 20
    args.warmup_epochs = 3
    args.lr = lr
    args.weight_decay = weight_decay
    args.amp = True
    args.use_wsdan = True
    args.K = 8
    args.wsdan_warm = 5
    args.erase_p = erase_p
    args.loss = 'elrplus'
    args.label_smooth = 0.1
    args.elr_lambda = elr_lambda
    args.mixup = mixup
    args.cutmix = cutmix
    args.randaug_N = 2
    args.randaug_M = randaug_M
    args.arcface = True
    args.margin = margin
    args.scale = scale
    args.class_balanced = True
    args.pretrained = True
    args.freeze_backbone_epochs = 0
    args.backbone_lr_mult = 0.1
    args.channels_last = True
    args.grad_ckpt = True
    args.microbatch = 16
    args.self_clean = True
    args.clean_warmup = 5
    args.clean_thresh = 0.4
    args.clean_min_w = 0.3
    args.clean_momentum = 0.9
    args.class_aware = True
    args.consistency_lambda = 1.0
    args.curriculum_start = 15
    args.curriculum_epochs = 50
    args.keep_ratio_start = 0.9
    args.keep_ratio_final = 0.7
    args.seed = 42
    args.init_from = None
    args.pseudo_csv = None
    args.elr_mem = 0

    # 🔥 Enable Lookahead Optimizer
    args.use_lookahead = True
    args.lookahead_k = 5
    args.lookahead_alpha = 0.5

    proxy_acc = 0.0
    try:
        train(args)

        # Read proxy metric: average of last 5 *epoch-level* accuracies
        proxy_file = outdir / "proxy_train_acc.txt"
        if proxy_file.exists():
            proxy_acc = float(proxy_file.read_text().strip())

        # Save hparams.json alongside model
        hparams = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))}
        with open(outdir / "hparams.json", "w") as f:
            json.dump(hparams, f, indent=2)

    except Exception as e:
        print(f"Trial failed: {e}")
        proxy_acc = 0.0

    return proxy_acc

if __name__ == "__main__":
    if not os.path.isdir("./data/WebFG-400/train"):
        raise FileNotFoundError("Train dir not found!")

    study = optuna.create_study(direction="maximize")

    # Enqueue baseline (with lookahead enabled)
    study.enqueue_trial({
        "lr": 1e-4,
        "weight_decay": 0.05,
        "mixup": 0.2,
        "cutmix": 0.3,
        "randaug_M": 10,
        "margin": 0.25,
        "scale": 30,
        "elr_lambda": 3.0,
        "erase_p": 0.6,
    })

    # Only 3 trials total (baseline + 2 new) → ~40 hours
    study.optimize(objective, n_trials=3)

    print("\n" + "="*60)
    print("✅ Best proxy training accuracy:", study.best_value)
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    with open("optuna_webfg400_best.json", "w") as f:
        json.dump({
            "best_proxy_acc": study.best_value,
            "best_params": study.best_params,
            "baseline_params": {
                "lr": 1e-4,
                "weight_decay": 0.05,
                "mixup": 0.2,
                "cutmix": 0.3,
                "randaug_M": 10,
                "margin": 0.25,
                "scale": 30,
                "elr_lambda": 3.0,
                "erase_p": 0.6,
            }
        }, f, indent=2)

    print("\nSaved to: optuna_webfg400_best.json")