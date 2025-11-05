import os, json, shutil
from pathlib import Path
import sys

ART = Path("artifacts")
DEPLOY = Path("deployed_model")
DEPLOY_METRICS = DEPLOY / "deployed_metrics.json"

def read_json(p: Path):
    with open(p) as f:
        return json.load(f)

def load_new_metrics():
    p = ART / "metrics.json"
    if not p.exists():
        print("artifacts/metrics.json not found. Did you run training?")
        sys.exit(1)
    return read_json(p)

def get_prod_accuracy():
    # Priority 1: CI secret or user env baseline
    env_val = os.getenv("PROD_ACCURACY")
    if env_val:
        try:
            return float(env_val)
        except:
            pass
    # Priority 2: previously deployed metrics saved on disk
    if DEPLOY_METRICS.exists():
        try:
            return float(read_json(DEPLOY_METRICS).get("accuracy", 0.0))
        except:
            return 0.0
    # First deploy case
    return None

def deploy(new_metrics):
    DEPLOY.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ART / "model.pkl", DEPLOY / "model.pkl")
    (DEPLOY_METRICS).write_text(json.dumps(new_metrics, indent=2))
    print(f"Deployed to {DEPLOY}/ with metrics: {new_metrics}")

if __name__ == "__main__":
    new_metrics = load_new_metrics()
    new_acc = float(new_metrics.get("accuracy", 0.0))
    prod_acc = get_prod_accuracy()
    print(f"New accuracy={new_acc} | Prod accuracy={prod_acc}")

    if prod_acc is None or new_acc >= prod_acc:
        print("Accepted new model. Deploying...")
        deploy(new_metrics)
        sys.exit(0)
    else:
        print("Rejected: worse than production. No deploy.")
        sys.exit(2)
