"""
LLM router: parses natural-language option pricing queries, extracts
parameters, selects the appropriate PINN model, and returns the price.

Supported models:
  - BSM  (Black-Scholes-Merton): constant volatility
  - CEV  (Constant Elasticity of Variance): local volatility with beta param
  - Heston: stochastic volatility (requires v0, kappa, theta, xi, rho)

Usage:
  router = OptionPricingRouter(bsm_model, cev_model, heston_model,
                               api_key="sk-...")
  result = router.price("欧式看涨期权，S=100, K=100, T=1, r=0.05, sigma=0.2")
  print(result)
"""

import json
import os
from openai import OpenAI


# ── system prompt for parameter extraction ─────────────────────────────────
_SYSTEM_PROMPT = """
You are a financial option pricing assistant. Extract option parameters from
the user's message and decide which pricing model to use.

Rules for model selection:
- Use "BSM" if only sigma (constant volatility) is given.
- Use "CEV" if a beta parameter is mentioned, or if the user says
  "constant elasticity of variance" or "CEV".
- Use "Heston" if stochastic volatility parameters are given
  (kappa, theta, xi, rho, v0), or if the user says "Heston" or
  "stochastic volatility".
- Default to "BSM" if unclear.

Always respond with a JSON object (no markdown, no explanation):
{
  "model": "BSM" | "CEV" | "Heston",
  "option_type": "call" | "put",
  "S": <float>,
  "K": <float>,
  "T": <float>,
  "r": <float>,
  "sigma": <float or null>,
  "beta": <float or null>,
  "v0": <float or null>,
  "kappa": <float or null>,
  "theta": <float or null>,
  "xi": <float or null>,
  "rho": <float or null>,
  "t": <float, current time, default 0.0>
}
Use null for parameters not mentioned. For missing required parameters,
use these defaults: sigma=0.2, beta=0.5, v0=0.04, kappa=2.0, theta=0.04,
xi=0.3, rho=-0.7.
"""


class OptionPricingRouter:
    def __init__(self, bsm_model, cev_model, heston_model,
                 api_key=None, base_url=None, model_name="gpt-4o-mini"):
        self.models = {
            "BSM": bsm_model,
            "CEV": cev_model,
            "Heston": heston_model,
        }
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,   # set for non-OpenAI endpoints (e.g. DeepSeek)
        )
        self.model_name = model_name

    # ── LLM call ────────────────────────────────────────────────────────────
    def _extract_params(self, user_query: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        return json.loads(raw)

    # ── main entry point ─────────────────────────────────────────────────────
    def price(self, user_query: str) -> dict:
        """
        Parse query, select model, return pricing result.

        Returns:
            {
              "model": str,
              "option_type": str,
              "params": dict,
              "price": float,
              "query": str,
            }
        """
        params = self._extract_params(user_query)
        model_key = params.get("model", "BSM")
        pinn = self.models[model_key]

        S = float(params["S"])
        t = float(params.get("t") or 0.0)

        # update model parameters before pricing
        if model_key == "BSM":
            if params.get("sigma"):
                pinn.sigma = float(params["sigma"])
            if params.get("K"):
                pinn.K = float(params["K"])
            if params.get("T"):
                pinn.T = float(params["T"])
            if params.get("r"):
                pinn.r = float(params["r"])
            price = pinn.price(S, t)

        elif model_key == "CEV":
            if params.get("sigma"):
                pinn.sigma = float(params["sigma"])
            if params.get("beta"):
                pinn.beta = float(params["beta"])
            if params.get("K"):
                pinn.K = float(params["K"])
            if params.get("T"):
                pinn.T = float(params["T"])
            if params.get("r"):
                pinn.r = float(params["r"])
            price = pinn.price(S, t)

        elif model_key == "Heston":
            if params.get("v0"):
                pinn.v0 = float(params["v0"])
            if params.get("kappa"):
                pinn.kappa = float(params["kappa"])
            if params.get("theta"):
                pinn.theta = float(params["theta"])
            if params.get("xi"):
                pinn.xi = float(params["xi"])
            if params.get("rho"):
                pinn.rho = float(params["rho"])
            if params.get("K"):
                pinn.K = float(params["K"])
            if params.get("T"):
                pinn.T = float(params["T"])
            if params.get("r"):
                pinn.r = float(params["r"])
            v = float(params["v0"]) if params.get("v0") else pinn.v0
            price = pinn.price(S, v=v, t=t)

        return {
            "model": model_key,
            "option_type": params.get("option_type", "call"),
            "params": params,
            "price": round(price, 4),
            "query": user_query,
        }
