"""
Demo: load trained models and price options via natural language.
Uses DeepSeek API (OpenAI-compatible endpoint).

Run: python src/demo.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.bsm_pinn import BSM_PINN
from src.models.cev_pinn import CEV_PINN
from src.models.heston_pinn import Heston_PINN
from src.llm.router import OptionPricingRouter

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

# DeepSeek API configuration
DEEPSEEK_API_KEY = "sk-834bd55a1e934ebb9e88a001784fbbcf"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"


def load_models():
    bsm = BSM_PINN(K=100, T=1.0, r=0.05, sigma=0.2, S_max=300)
    bsm.load(os.path.join(CKPT_DIR, "bsm_pinn.pt"))

    cev = CEV_PINN(K=100, T=1.0, r=0.05, sigma=0.25, beta=0.5, S_max=300)
    cev.load(os.path.join(CKPT_DIR, "cev_pinn.pt"))

    heston = Heston_PINN(K=100, T=1.0, r=0.05,
                         kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
    heston.load(os.path.join(CKPT_DIR, "heston_pinn.pt"))

    return bsm, cev, heston


EXAMPLE_QUERIES = [
    "欧式看涨期权，当前股价S=100，行权价K=100，到期时间T=1年，无风险利率r=0.05，波动率sigma=0.2",
    "CEV model call option: S=110, K=100, T=0.5, r=0.03, sigma=0.3, beta=0.5",
    "Heston stochastic volatility call: S=100, K=105, T=1, r=0.05, "
    "kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04",
]


if __name__ == "__main__":
    # use env vars if set, otherwise fall back to hardcoded DeepSeek config
    api_key = os.environ.get("OPENAI_API_KEY", DEEPSEEK_API_KEY)
    base_url = os.environ.get("OPENAI_BASE_URL", DEEPSEEK_BASE_URL)
    model_name = os.environ.get("LLM_MODEL", DEEPSEEK_MODEL)

    bsm, cev, heston = load_models()
    router = OptionPricingRouter(bsm, cev, heston,
                                 api_key=api_key,
                                 base_url=base_url,
                                 model_name=model_name)

    for query in EXAMPLE_QUERIES:
        print(f"\nQuery : {query}")
        result = router.price(query)
        print(f"Model : {result['model']}")
        print(f"Price : {result['price']}")
