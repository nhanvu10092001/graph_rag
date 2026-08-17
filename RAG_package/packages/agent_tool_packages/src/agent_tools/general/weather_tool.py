"""WeatherTool — Get current weather for a city.

Wraps the same WeatherAPI.com integration used in mcp_sever/mcp_sever.py::get_weather.
"""

import os
from typing import Any

import requests
from pydantic import BaseModel, Field

from ..abs_interface.utility_tool import AbstractUtilityTool
from ..core.registry import ServiceToolRegistry


class WeatherInput(BaseModel):
    city: str = Field(description="City name to get weather for, e.g. 'Tokyo', 'Ho Chi Minh City'")


@ServiceToolRegistry.register
class WeatherTool(AbstractUtilityTool):
    """Get current weather for any city."""

    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return (
            "Get the current weather for a given city. "
            "Use this when the user asks about weather, temperature, or climate conditions. "
            "Input: city name (e.g. 'Tokyo', 'Hanoi'). "
            "Returns: temperature (°C), condition, humidity, wind speed."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return WeatherInput

    async def call_api(self, city: str, **_) -> Any:
        api_key = os.getenv("WEATHER_API_KEY", "")
        if not api_key:
            return {"error": "WEATHER_API_KEY not configured"}

        try:
            resp = requests.get(
                "http://api.weatherapi.com/v1/current.json",
                params={"key": api_key, "q": city},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            current = data.get("current", {})
            location = data.get("location", {})
            return {
                "city": location.get("name", city),
                "country": location.get("country", ""),
                "temp_c": current.get("temp_c"),
                "condition": current.get("condition", {}).get("text", ""),
                "humidity": current.get("humidity"),
                "wind_kph": current.get("wind_kph"),
                "feels_like_c": current.get("feelslike_c"),
            }
        except requests.RequestException as e:
            return {"error": f"Weather API request failed: {str(e)}"}
