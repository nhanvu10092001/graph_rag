"""GetTimeTool — Get current date and time in any timezone.

Wraps the same logic used in mcp_sever/mcp_sever.py::get_time.
"""

from datetime import datetime
from typing import Any

import pytz
from pydantic import BaseModel, Field

from ..abs_interface.utility_tool import AbstractUtilityTool
from ..core.registry import ServiceToolRegistry


class TimeInput(BaseModel):
    timezone_str: str = Field(
        default="UTC",
        description="Timezone name, e.g. 'Asia/Ho_Chi_Minh', 'Asia/Tokyo', 'US/Eastern', 'UTC'",
    )


@ServiceToolRegistry.register
class GetTimeTool(AbstractUtilityTool):
    """Get current date and time in any timezone."""

    @property
    def name(self) -> str:
        return "get_time"

    @property
    def description(self) -> str:
        return (
            "Get the current date and time in a specific timezone. "
            "Use this when the user asks what time or date it is. "
            "Input: timezone name (default UTC). "
            "Returns: current datetime and timezone."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return TimeInput

    async def call_api(self, timezone_str: str = "UTC", **_) -> Any:
        try:
            tz = pytz.timezone(timezone_str)
            now = datetime.now(tz)
            return {
                "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                "timezone": timezone_str,
                "day_of_week": now.strftime("%A"),
            }
        except pytz.exceptions.UnknownTimeZoneError:
            return {"error": f"Unknown timezone: {timezone_str}. Use format like 'Asia/Ho_Chi_Minh'."}
