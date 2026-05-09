"""
LLM router: given an option contract code (or natural-language query),
fetch live market data, select the appropriate PINN model, and return
the PINN price alongside the analytical reference price.

Supported input modes:
  1. Contract code only:
       router.price("10004073")          # CN 50ETF option
       router.price("SPY250117C00580000") # US SPY option
       router.price("欧式看涨, S=100, K=100, T=1, r=0.05, sigma=0.2")  # manual

  2. The LLM's only job is model selection (BSM / CEV / Heston) based on
     market and option type. All numeric parameters come from the live
     data lookup, not from the user.

Supported models:
  - BSM    : constant volatility (European call/put, American call/put)
  - CEV    : local volatility with beta param
  - Heston : stochastic volatility (kappa, theta, xi, rho, v0)
"""

import json
import os
import re
import datetime
from openai import OpenAI


# ── contract-code market detector ─────────────────────────────────────────

def detect_market(code: str) -> str:
    """
    Infer market from contract code format.
      CN  50ETF: 8-digit numeric, e.g. "10004073"
      US  OCC:   ticker + date + C/P + strike, e.g. "SPY250117C00580000"
      HK  HKEX:  starts with "^HSI" or "HSI", e.g. "HSIO250131C22000"
    Returns "cn", "us", or "hk". Defaults to "us" if unclear.
    """
    code = code.strip()
    if re.fullmatch(r"\d{8}", code):
        return "cn"
    if re.match(r"^(\^?HSI)", code, re.IGNORECASE):
        return "hk"
    if re.match(r"^[A-Z]{1,5}\d{6}[CP]\d+$", code):
        return "us"
    return "us"


# CEV beta is fixed at 0.5 following Beckers (1980), who found the empirical
# mean across 17 US equities to be ~0.48, indistinguishable from 0.5.
# Schroder (1989) further showed beta=0.5 admits an exact closed-form solution
# via the non-central chi-squared distribution, making it numerically optimal.
CEV_BETA_DEFAULT = 0.5

# Tickers treated as indices/ETFs → route to Heston by default
_INDEX_ETFS = {"SPY", "QQQ", "IWM", "DIA", "VXX", "UVXY", "TQQQ", "SQQQ",
               "XLF", "XLE", "XLK", "GLD", "SLV", "TLT", "HYG"}


# ── live data lookup ───────────────────────────────────────────────────────

def lookup_option(code: str) -> dict | None:
    """
    Fetch live option parameters from market data APIs.
    Returns dict with keys: S, K, T, r, mid, market, option_type, code
    or None on failure.
    """
    market = detect_market(code)
    today = datetime.date.today()

    if market == "us":
        return _lookup_us(code, today)
    elif market == "cn":
        return _lookup_cn(code, today)
    elif market == "hk":
        return _lookup_hk(code, today)
    return None


def _lookup_us(code: str, today: datetime.date) -> dict | None:
    """
    Parse OCC option symbol: TICKER YYMMDD C/P STRIKE(x1000)
    e.g. SPY250117C00580000 -> SPY, 2025-01-17, call, K=580.0
    """
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Run: pip install yfinance")
        return None

    m = re.match(r"^([A-Z]{1,5})(\d{6})([CP])(\d+)$", code.strip())
    if not m:
        print(f"[US] Cannot parse OCC code: {code}")
        return None

    ticker, date_str, cp, strike_str = m.groups()
    exp_date = datetime.datetime.strptime(date_str, "%y%m%d").date()
    K = int(strike_str) / 1000.0
    option_type = "american_call" if cp == "C" else "american_put"
    T = max((exp_date - today).days / 365.0, 1 / 365)
    r = 0.05

    try:
        tk = yf.Ticker(ticker)
        S = tk.fast_info["last_price"]
        exp_str = exp_date.strftime("%Y-%m-%d")
        chain = tk.option_chain(exp_str)
        df = chain.calls if cp == "C" else chain.puts
        row = df[df["strike"] == K]
        if row.empty:
            row = df.iloc[(df["strike"] - K).abs().argsort()[:1]]
        mid = (row["bid"].values[0] + row["ask"].values[0]) / 2
    except Exception as e:
        print(f"[US] Data fetch failed: {e}")
        return None

    is_index = ticker in _INDEX_ETFS
    beta = None if is_index else CEV_BETA_DEFAULT
    return {"S": S, "K": K, "T": T, "r": r, "mid": mid,
            "market": "us", "option_type": option_type, "code": code,
            "is_index": is_index, "beta": beta}


def _lookup_cn(code: str, today: datetime.date) -> dict | None:
    """
    Look up CN 50ETF option by 8-digit contract code via akshare.
    """
    try:
        import akshare as ak
    except ImportError:
        print("akshare not installed. Run: pip install akshare")
        return None

    r = 0.02
    try:
        spot_df = ak.fund_etf_hist_em(symbol="510050", period="daily", adjust="qfq")
        S = float(spot_df["收盘"].iloc[-1])
    except Exception as e:
        print(f"[CN] Spot price fetch failed: {e}")
        return None

    try:
        opt_df = ak.option_finance_board(symbol="510050", end_month="")
    except Exception as e:
        print(f"[CN] Option chain fetch failed: {e}")
        return None

    row = opt_df[opt_df["期权代码"].astype(str) == str(code)]
    if row.empty:
        print(f"[CN] Contract {code} not found in option chain.")
        return None

    row = row.iloc[0]
    name = str(row.get("名称", ""))
    option_type = "european_call" if "购" in name else "european_put"
    K = float(row["行权价"])
    bid = float(row.get("买价", 0))
    ask = float(row.get("卖价", 0))
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else float(row.get("最新价", 0))

    exp_raw = str(row.get("到期日", "")).replace("-", "")
    if len(exp_raw) == 8:
        exp_date = datetime.datetime.strptime(exp_raw, "%Y%m%d").date()
        T = max((exp_date - today).days / 365.0, 1 / 365)
    else:
        T = 0.25  # fallback: 3 months

    return {"S": S, "K": K, "T": T, "r": r, "mid": mid,
            "market": "cn", "option_type": option_type, "code": code}


def _lookup_hk(code: str, today: datetime.date) -> dict | None:
    """
    Look up HK HSI option via yfinance.
    Accepts codes like "HSIO250131C22000" or "^HSIO250131C22000".
    """
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Run: pip install yfinance")
        return None

    r = 0.04
    try:
        tk = yf.Ticker("^HSI")
        S = tk.fast_info["last_price"]
    except Exception as e:
        print(f"[HK] Spot price fetch failed: {e}")
        return None

    # parse code: HSIO YYMMDD C/P STRIKE
    clean = code.lstrip("^").upper()
    m = re.match(r"HSI[A-Z]?(\d{6})([CP])(\d+)", clean)
    if not m:
        print(f"[HK] Cannot parse HK option code: {code}")
        return None

    date_str, cp, strike_str = m.groups()
    exp_date = datetime.datetime.strptime(date_str, "%y%m%d").date()
    K = float(strike_str)
    option_type = "european_call" if cp == "C" else "european_put"
    T = max((exp_date - today).days / 365.0, 1 / 365)

    # try to get live mid from yfinance; fall back to BSM if unavailable
    mid = None
    try:
        exp_str = exp_date.strftime("%Y-%m-%d")
        chain = yf.Ticker("^HSI").option_chain(exp_str)
        df = chain.calls if cp == "C" else chain.puts
        row = df[df["strike"] == K]
        if not row.empty:
            mid = (row["bid"].values[0] + row["ask"].values[0]) / 2
    except Exception:
        pass

    if mid is None or mid <= 0:
        from scipy.stats import norm
        import numpy as np
        sigma = 0.20
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        mid = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        print(f"[HK] Using BSM-synthetic mid price (σ=0.20)")

    return {"S": S, "K": K, "T": T, "r": r, "mid": mid,
            "market": "hk", "option_type": option_type, "code": code}


# ── LLM model selector ─────────────────────────────────────────────────────

_MODEL_SELECT_PROMPT = """
You are a financial option pricing assistant. Choose the pricing model
(BSM, CEV, or Heston) based on the following decision rules.

== Model selection rules ==

BSM (Black-Scholes-Merton) — use when:
  - User provides only sigma (a single constant volatility).
  - User asks for a quick price, implied vol, or Greeks.
  - Underlying is a single stock with no mention of leverage effect.
  - User says "BSM", "Black-Scholes", "flat vol", or "constant vol".
  - Suitable for: short-dated ATM options, liquid single-stock options.

CEV (Constant Elasticity of Variance) — use when:
  - Underlying is a single stock (NOT an index or ETF).
  - User mentions "leverage effect", "vol increases as price drops",
    "CEV", "beta", or provides a beta parameter.
  - Stock price is low (< 50) or the user mentions high financial leverage.
  - Suitable for: individual equities where vol-price relationship matters.
  - Default beta = 0.5 (Beckers 1980 empirical consensus).

Heston (stochastic volatility) — use when:
  - Underlying is an index or ETF (SPY, QQQ, 50ETF, HSI, SPX, NDX, etc.).
  - User mentions "vol smile", "skew", "term structure",
    "stochastic vol", or "Heston".
  - User provides any of: kappa, theta, xi, rho, v0.
  - User wants accurate pricing across multiple strikes or maturities.
  - Suitable for: index options, vol surface fitting, risk-neutral calibration.

== Default rules ==
- Index or ETF underlying → Heston
- Single stock, no beta info → BSM
- Single stock + leverage effect mentioned → CEV

Respond with a single JSON object only:
{"model": "BSM" | "CEV" | "Heston", "reason": "<one sentence explaining the choice>"}
"""


# ── main router ────────────────────────────────────────────────────────────

class OptionPricingRouter:
    def __init__(self, bsm_model, cev_model, heston_model,
                 api_key=None, base_url=None, model_name="gpt-4o-mini"):
        self.models = {"BSM": bsm_model, "CEV": cev_model, "Heston": heston_model}
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )
        self.model_name = model_name

    def _select_model(self, market: str, option_type: str, hint: str = "") -> str:
        """Ask LLM to choose BSM/CEV/Heston given market context."""
        user_msg = f"market={market}, option_type={option_type}"
        if hint:
            user_msg += f", user_hint={hint}"
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": _MODEL_SELECT_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = json.loads(response.choices[0].message.content)
        return raw.get("model", "BSM")

    def _apply_params(self, model_key: str, params: dict):
        """Push fetched parameters into the PINN model."""
        pinn = self.models[model_key]
        for attr in ["K", "T", "r"]:
            if params.get(attr):
                setattr(pinn, attr, float(params[attr]))
        if model_key == "BSM" and params.get("sigma"):
            pinn.sigma = float(params["sigma"])
        elif model_key == "CEV":
            if params.get("sigma"):
                pinn.sigma = float(params["sigma"])
            if params.get("beta"):
                pinn.beta = float(params["beta"])
        elif model_key == "Heston":
            for attr in ["v0", "kappa", "theta", "xi", "rho"]:
                if params.get(attr):
                    setattr(pinn, attr, float(params[attr]))

    def price(self, query: str, hint: str = "") -> dict:
        """
        Price an option from a contract code or natural-language query.

        Args:
            query: contract code (e.g. "10004073") or NL query with params
            hint:  optional model hint ("Heston", "CEV", etc.)

        Returns:
            {model, option_type, params, price, mid, market, code}
        """
        # try live lookup first
        live = lookup_option(query)

        if live is not None:
            market = live["market"]
            option_type = live["option_type"]
            params = {k: live[k] for k in ["S", "K", "T", "r"]}
            # pass beta=0.5 for single-stock US options (Beckers 1980)
            if live.get("beta") is not None:
                params["beta"] = live["beta"]
            mid = live["mid"]
            code = live["code"]

            # index/ETF → Heston by default; single stock → let LLM decide
            is_index = live.get("is_index", market in ("cn", "hk"))
            auto_hint = "Heston" if is_index else hint
            model_key = hint if hint in self.models else \
                self._select_model(market, option_type, auto_hint)

            self._apply_params(model_key, params)
            pinn = self.models[model_key]

            S = float(live["S"])
            t = 0.0
            if model_key == "Heston":
                price = pinn.price(S, v=pinn.v0, t=t)
            else:
                price = pinn.price(S, t=t)

            return {
                "model": model_key,
                "option_type": option_type,
                "params": params,
                "price": round(price, 4),
                "mid": round(mid, 4),
                "market": market,
                "code": code,
            }

        # fallback: parse natural-language query with full LLM extraction
        return self._price_from_nl(query)

    # ── natural-language fallback (original behaviour) ─────────────────────
    _NL_PROMPT = """
You are a financial option pricing assistant. Extract option parameters from
the user's message and decide which pricing model to use.

Rules for model selection:
- Use "BSM" if only sigma (constant volatility) is given.
- Use "CEV" if a beta parameter is mentioned.
- Use "Heston" if stochastic volatility parameters are given.
- Default to "BSM" if unclear.

Rules for option_type:
- "european_call" (default), "european_put", "american_call", "american_put"

Respond with JSON only:
{
  "model": "BSM"|"CEV"|"Heston",
  "option_type": "european_call"|"european_put"|"american_call"|"american_put",
  "S": float, "K": float, "T": float, "r": float,
  "sigma": float|null, "beta": float|null,
  "v0": float|null, "kappa": float|null, "theta": float|null,
  "xi": float|null, "rho": float|null, "t": float
}
Defaults: sigma=0.2, beta=0.5, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7
"""

    def _price_from_nl(self, query: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self._NL_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        params = json.loads(response.choices[0].message.content)
        model_key = params.get("model", "BSM")
        self._apply_params(model_key, params)
        pinn = self.models[model_key]
        S = float(params["S"])
        t = float(params.get("t") or 0.0)
        if model_key == "Heston":
            v = float(params["v0"]) if params.get("v0") else pinn.v0
            price = pinn.price(S, v=v, t=t)
        else:
            price = pinn.price(S, t=t)
        return {
            "model": model_key,
            "option_type": params.get("option_type", "european_call"),
            "params": params,
            "price": round(price, 4),
            "mid": None,
            "market": "manual",
            "code": None,
        }
