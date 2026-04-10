"""tokonomics — Universal LLM token counting and cost management."""

from tokonomics._types import (
    BudgetExceededError,
    CostEstimate,
    ModelNotFoundError,
    ModelPricing,
    Provider,
    TokonomicsError,
    UsageRecord,
)
from tokonomics.budget import Budget
from tokonomics.charts import export_svg_chart, format_bar_chart, format_table
from tokonomics.compare import cheapest_model, compare_models, format_comparison
from tokonomics.cost import calculate_cost, cost_per_token, estimate_cost
from tokonomics.models import find_models, get_model, list_models
from tokonomics.streaming import (
    StreamingCostTracker,
    StreamingUsage,
    async_track_stream,
    track_stream,
)
from tokonomics.tokenizer import count_message_tokens, count_tokens, fits_context
from tokonomics.tracker import CostTracker, get_global_tracker, track_cost
from tokonomics.rate_limiter import (
    PROVIDER_DEFAULTS,
    RateLimitConfig,
    RateLimitState,
    RateLimiter,
    create_limiter,
    format_rate_status,
)
from tokonomics.usage_report import (
    UsageEntry,
    UsageReport,
    export_usage_json,
    format_usage_report,
)

__version__ = "0.2.0"

__all__ = [
    # Version
    "__version__",
    # Types
    "Provider",
    "ModelPricing",
    "UsageRecord",
    "CostEstimate",
    # Exceptions
    "TokonomicsError",
    "ModelNotFoundError",
    "BudgetExceededError",
    # Models
    "get_model",
    "list_models",
    "find_models",
    # Token counting
    "count_tokens",
    "count_message_tokens",
    "fits_context",
    # Cost calculation
    "estimate_cost",
    "calculate_cost",
    "cost_per_token",
    # Tracking
    "track_cost",
    "CostTracker",
    "get_global_tracker",
    # Streaming
    "StreamingCostTracker",
    "StreamingUsage",
    "async_track_stream",
    "track_stream",
    # Budget
    "Budget",
    # Comparison
    "compare_models",
    "cheapest_model",
    "format_comparison",
    # Charts
    "format_bar_chart",
    "export_svg_chart",
    "format_table",
    # Usage reporting
    "UsageEntry",
    "UsageReport",
    "format_usage_report",
    "export_usage_json",
    # Rate limiting
    "RateLimitConfig",
    "RateLimitState",
    "RateLimiter",
    "PROVIDER_DEFAULTS",
    "create_limiter",
    "format_rate_status",
]
