"""
ResolutionScalper — captures value as politics/election markets approach resolution.

Pipeline:
    NearResolutionSource     → Fetch politics markets resolving within 7 days
    ElectionOutcomeClassifier → LLM-based filter to keep only election outcome markets
    OrderbookEnricher        → Add orderbook depth + spread
    ResolutionAnalyzer       → Estimate fair value based on time to resolution
    DiscountScorer           → Score opportunities by discount to fair value
    QualityFilter            → Remove low-confidence, illiquid candidates

Hybrid mode:
  - Scan: periodically find near-resolution candidates
  - Stream: watch candidates in real-time for optimal entry timing

Agent skill: "What near-resolution election markets have uncertainty discounts?"
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from farsight.markets.clients.polymarket.clob_client import ClobClient
from farsight.markets.clients.polymarket.gamma_client import GammaClient
from farsight.markets.strategies.base import (
    Action,
    ActionType,
    Analyzer,
    Enricher,
    Filter,
    MarketContext,
    Opportunity,
    Scorer,
    Source,
    Strategy,
    StrategyMode,
)

logger = logging.getLogger(__name__)


# ── Pipeline Stages ──────────────────────────────────────────────────


class NearResolutionSource(Source):
    """Fetch politics markets resolving within N days.

    Filters to the 'politics' category using the Gamma API's category field,
    tags, and keyword inference from market questions.

    Agent skill: "What politics prediction markets are resolving soon?"
    """

    POLITICS_KEYWORDS = {
        "election", "president", "congress", "senate", "governor",
        "trump", "biden", "democrat", "republican", "gop", "dnc", "rnc",
        "electoral", "ballot", "impeach", "scotus", "supreme court",
        "midterm", "primary", "caucus", "mayor", "parliament", "minister",
    }

    POLITICS_TAGS = {
        "politics", "elections", "government", "congress", "senate",
        "democrat", "republican",
    }

    def __init__(self, gamma: GammaClient, max_days: int = 7, category: str = "politics",
                 min_yes_price: float = 0.70):
        self.gamma = gamma
        self.max_days = max_days
        self.category = category
        self.min_yes_price = min_yes_price

    def _is_politics(self, raw: dict, question: str) -> bool:
        """Check if a market belongs to the politics category."""
        cat = (raw.get("category") or "").lower()
        if cat == self.category:
            return True

        tags = raw.get("tags", [])
        tag_labels = set()
        for t in tags:
            if isinstance(t, dict):
                tag_labels.add(t.get("label", "").lower())
            elif isinstance(t, str):
                tag_labels.add(t.lower())
        if tag_labels & self.POLITICS_TAGS:
            return True

        q_lower = question.lower()
        slug_lower = (raw.get("slug") or "").lower()
        combined = f"{q_lower} {slug_lower}"
        return any(kw in combined for kw in self.POLITICS_KEYWORDS)

    async def fetch(self) -> list[MarketContext]:
        raw_markets = await self.gamma.get_markets(
            active=True, closed=False,
            limit=100,
            order="end_date", ascending=True,
        )

        now = datetime.utcnow()
        contexts = []

        for raw in raw_markets:
            market = GammaClient.normalize_market(raw)

            if not market.end_date:
                continue

            if not self._is_politics(raw, market.question):
                continue

            end_naive = market.end_date.replace(tzinfo=None) if market.end_date.tzinfo else market.end_date
            days_left = (end_naive - now).total_seconds() / 86400
            if days_left < 0 or days_left > self.max_days:
                continue

            slug = market.slug or ""
            if "updown-5m" in slug or "updown-15m" in slug:
                continue

            if not market.outcomes or len(market.outcomes) < 2:
                continue

            primary = market.outcomes[0]

            if primary.current_price < self.min_yes_price:
                continue

            ctx = MarketContext(
                market_id=market.condition_id,
                market_question=market.question,
                event_slug=market.slug,
                token_id=primary.token_id,
                outcome_label=primary.label,
                current_price=primary.current_price,
                volume_24h=float(raw.get("volume24hr") or 0),
                liquidity=market.liquidity,
                end_date=market.end_date,
                raw=raw,
            )
            ctx._days_left = days_left
            contexts.append(ctx)

        return contexts


class ElectionOutcomeClassifier:
    """Uses a local zero-shot classification model to determine whether a
    politics market is about an election outcome.

    Filters out non-election politics markets (e.g., legislation, court rulings,
    policy decisions) and keeps only markets that are directly about who wins
    an election, vote shares, seat counts, or electoral results.

    Uses facebook/bart-large-mnli via Hugging Face transformers — runs locally,
    no API key needed. The model is loaded lazily on first use.

    Results are cached per market_id within a scan cycle to avoid redundant inference.
    """

    ELECTION_LABEL = "winning a political election"
    NON_ELECTION_LABEL = "government policy or legislation"

    def __init__(self, model: str = "facebook/bart-large-mnli", threshold: float = 0.7):
        self._model_name = model
        self._threshold = threshold
        self._classifier = None
        self._cache: dict[str, bool] = {}

    def _get_classifier(self):
        if self._classifier is None:
            from transformers import pipeline
            logger.info(f"ElectionClassifier: loading model {self._model_name}...")
            self._classifier = pipeline(
                "zero-shot-classification", model=self._model_name,
            )
            logger.info("ElectionClassifier: model loaded")
        return self._classifier

    def is_election_outcome(self, market_id: str, question: str) -> bool:
        """Classify a single market question. Returns True if it's an election outcome."""
        if market_id in self._cache:
            return self._cache[market_id]

        try:
            classifier = self._get_classifier()
            result = classifier(
                question,
                candidate_labels=[self.ELECTION_LABEL, self.NON_ELECTION_LABEL],
            )
            top_label = result["labels"][0]
            top_score = result["scores"][0]
            is_election = top_label == self.ELECTION_LABEL and top_score >= self._threshold
            logger.debug(
                f"ElectionClassifier: '{question[:50]}' → "
                f"{top_label} ({top_score:.2f}) → {'KEEP' if is_election else 'SKIP'}"
            )
        except Exception as e:
            logger.warning(f"ElectionClassifier error for '{question[:50]}': {e}")
            is_election = False

        self._cache[market_id] = is_election
        return is_election

    def classify_batch(self, contexts: list[MarketContext]) -> list[MarketContext]:
        """Filter a list of MarketContexts to only election outcome markets."""
        classified = []
        for ctx in contexts:
            if self.is_election_outcome(ctx.market_id, ctx.market_question):
                classified.append(ctx)
            else:
                logger.debug(
                    f"ElectionClassifier: filtered out non-election market: "
                    f"{ctx.market_question[:60]}"
                )
        logger.info(
            f"ElectionClassifier: {len(contexts)} politics → {len(classified)} election outcomes"
        )
        return classified


class ResolutionAnalyzer(Analyzer):
    """Estimate fair value based on time to resolution.

    As resolution approaches, high-probability outcomes should trade
    closer to 100%. The gap between current price and fair value
    represents an uncertainty discount that shrinks over time.

    Agent skill: "What should this market be priced at given time to resolution?"
    """

    def __init__(self, min_certainty: float = 0.85):
        self.min_certainty = min_certainty

    def analyze(self, ctx: MarketContext) -> MarketContext:
        days_left = getattr(ctx, "_days_left", 999)
        price = ctx.current_price

        # Only works for near-certain outcomes
        if price < self.min_certainty and (1 - price) < self.min_certainty:
            ctx.features = {"fair_value": price, "discount": 0, "qualified": False}
            return ctx

        fair_value = self._estimate_fair_value(price, days_left)
        discount = fair_value - price

        ctx.features = {
            "fair_value": fair_value,
            "discount": discount,
            "days_left": days_left,
            "current_certainty": max(price, 1 - price),
            "qualified": discount > 0.03,
        }
        return ctx

    @staticmethod
    def _estimate_fair_value(current_price: float, days_left: float) -> float:
        """Estimate fair value — closer to resolution, closer to 100%."""
        if days_left <= 0:
            return current_price
        convergence = min(0.95, 1.0 - (days_left / 7.0) * 0.8)
        convergence = max(0.0, convergence)
        distance_to_certain = 1.0 - current_price
        adjustment = distance_to_certain * convergence
        return min(0.99, current_price + adjustment)


class DiscountScorer(Scorer):
    """Convert resolution analysis into scored opportunities.

    Agent skill: "What's the discount on this near-resolution market?"
    """

    def __init__(self, min_discount: float = 0.03):
        self.min_discount = min_discount

    def score(self, ctx: MarketContext) -> list[Opportunity]:
        if not ctx.features.get("qualified"):
            return []

        discount = ctx.features.get("discount", 0)
        if discount < self.min_discount:
            return []

        fair_value = ctx.features["fair_value"]
        days_left = ctx.features.get("days_left", 999)

        opp = Opportunity(
            market_id=ctx.market_id,
            market_question=ctx.market_question,
            event_slug=ctx.event_slug,
            token_id=ctx.token_id,
            outcome=ctx.outcome_label,
            strategy="resolution",
            reasoning=(
                f"Resolves in {days_left:.1f} days. "
                f"{ctx.outcome_label} at {ctx.current_price:.0%} but fair value ~{fair_value:.0%}. "
                f"Discount: {discount:.1%}."
            ),
            direction="buy",
            entry_price=ctx.current_price,
            model_price=fair_value,
            edge=discount,
            confidence=min(0.95, ctx.current_price + 0.05),
            horizon=f"{days_left:.0f}d",
            liquidity=ctx.liquidity,
            volume_24h=ctx.volume_24h,
            spread=ctx.spread or 0.02,
            risk_flags=_assess_risk(days_left, ctx.current_price, ctx.liquidity),
            resolution_date=ctx.end_date,
            context=ctx,
        )
        return [opp]


# ── Composed Strategy ────────────────────────────────────────────────


class ResolutionScalper(Strategy):
    """Find near-resolution election markets with uncertainty discounts.

    Scoped to politics category only, with LLM classification to keep
    only markets whose resolution depends on an election outcome.

    Pipeline: NearResolutionSource (politics) → ElectionOutcomeClassifier (LLM)
              → OrderbookEnricher → ResolutionAnalyzer → DiscountScorer → QualityFilter
    """

    name = "resolution"
    mode = StrategyMode.HYBRID
    scan_interval_seconds = 900

    def __init__(
        self,
        gamma: Optional[GammaClient] = None,
        clob: Optional[ClobClient] = None,
        max_days: int = 14,
        min_certainty: float = 0.75,
        min_discount: float = 0.02,
        min_liquidity: float = 5000,
        min_yes_price: float = 0.70,
        classifier_threshold: float = 0.7,
    ):
        gamma = gamma or GammaClient()
        clob = clob or ClobClient()

        self.source = NearResolutionSource(
            gamma, max_days=max_days, category="politics", min_yes_price=min_yes_price,
        )
        self.election_classifier = ElectionOutcomeClassifier(threshold=classifier_threshold)
        self.orderbook_enricher = None  # Import inline to avoid circular
        self._clob = clob
        self.resolution_analyzer = ResolutionAnalyzer(min_certainty=min_certainty)
        self.discount_scorer = DiscountScorer(min_discount=min_discount)
        self.quality_filter = Filter(min_edge=min_discount, min_liquidity=min_liquidity)

    async def scan(self) -> list[Opportunity]:
        from farsight.markets.strategies.opportunity_scanner import OrderbookEnricher

        orderbook_enricher = OrderbookEnricher(self._clob)

        # 1. Source — politics markets only
        contexts = await self.source.fetch()
        logger.info(f"ResolutionScalper: sourced {len(contexts)} near-resolution politics markets")

        # 2. Classify — keep only election outcome markets via LLM
        contexts = self.election_classifier.classify_batch(contexts)

        all_opportunities = []

        for ctx in contexts:
            try:
                # 3. Enrich
                ctx = await orderbook_enricher.enrich(ctx)
                if ctx.liquidity < 1000:
                    continue

                # 4. Analyze
                ctx = self.resolution_analyzer.analyze(ctx)

                # 5. Score
                opps = self.discount_scorer.score(ctx)
                all_opportunities.extend(opps)
            except Exception as e:
                logger.debug(f"Resolution: error on {ctx.market_question[:40]}: {e}")

        # 6. Filter
        filtered = self.quality_filter.filter(all_opportunities)
        logger.info(f"ResolutionScalper: {len(all_opportunities)} raw → {len(filtered)} after filter")
        return filtered

    async def monitor(self, open_positions: list[dict]) -> list[Action]:
        actions = []
        for pos in open_positions:
            if pos.get("strategy") != self.name:
                continue

            token_id = pos.get("token_id", "")
            if not token_id:
                continue

            book = await self._clob.get_orderbook(token_id)
            if not book:
                continue

            current_price = book.mid
            entry = pos.get("entry_price", 0)

            if current_price >= 0.97:
                actions.append(Action(
                    action_type=ActionType.CLOSE,
                    trade_id=pos["id"],
                    reason=f"Price reached {current_price:.0%} — taking profit",
                    exit_price=current_price,
                ))
            elif entry > 0 and current_price < entry - 0.05:
                actions.append(Action(
                    action_type=ActionType.STOP_LOSS,
                    trade_id=pos["id"],
                    reason=f"Stop loss: {current_price:.0%} < entry {entry:.0%} - 5%",
                    exit_price=current_price,
                ))

        return actions

    def get_source(self) -> Source:
        return self.source

    def get_analyzer(self):
        return self.resolution_analyzer


def _assess_risk(days_left: float, price: float, liquidity: float) -> list[str]:
    flags = []
    if days_left < 0.5:
        flags.append("imminent_resolution")
    if price > 0.95:
        flags.append("very_high_certainty")
    if liquidity < 10000:
        flags.append("low_liquidity")
    return flags
