"""Input validators with specific failure reasons for MCP tools."""

import re
import ipaddress
from typing import Any, Callable, List, Union
from urllib.parse import urlparse
from pathlib import Path
import json

from .exceptions import ValidationError


class ValidatorRegistry:
    """Registry of common validators with clear error messages."""

    @staticmethod
    def non_empty_string(value: Any) -> bool:
        """Validate that value is a non-empty string.

        Args:
            value: Value to validate

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(
                field="value", value=value, requirement="Must be a string"
            )

        if not value.strip():
            raise ValidationError(
                field="value",
                value=value,
                requirement="String cannot be empty or contain only whitespace",
            )

        return True

    @staticmethod
    def positive_integer(value: Any) -> bool:
        """Validate that value is a positive integer.

        Args:
            value: Value to validate

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                raise ValidationError(
                    field="value",
                    value=value,
                    requirement="Must be an integer or convertible to integer",
                )

        if value <= 0:
            raise ValidationError(
                field="value",
                value=value,
                requirement="Must be a positive integer (greater than 0)",
            )

        return True

    @staticmethod
    def positive_number(value: Any) -> bool:
        """Validate that value is a positive number.

        Args:
            value: Value to validate

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise ValidationError(
                    field="value",
                    value=value,
                    requirement="Must be a number or convertible to number",
                )

        if value <= 0:
            raise ValidationError(
                field="value",
                value=value,
                requirement="Must be a positive number (greater than 0)",
            )

        return True

    @staticmethod
    def valid_port(value: Any) -> bool:
        """Validate that value is a valid port number.

        Args:
            value: Value to validate

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                raise ValidationError(
                    field="port", value=value, requirement="Port must be an integer"
                )

        if not (1 <= value <= 65535):
            raise ValidationError(
                field="port",
                value=value,
                requirement="Port must be between 1 and 65535",
            )

        return True

    @staticmethod
    def valid_url(value: Any) -> bool:
        """Validate that value is a valid URL.

        Args:
            value: Value to validate

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(
                field="url", value=value, requirement="URL must be a string"
            )

        try:
            result = urlparse(value)
            if not all([result.scheme, result.netloc]):
                raise ValidationError(
                    field="url",
                    value=value,
                    requirement="URL must have both scheme (http/https) and domain",
                )
        except Exception:
            raise ValidationError(
                field="url",
                value=value,
                requirement="Must be a valid URL (e.g., https://example.com)",
            )

        return True

    @staticmethod
    def valid_email(value: Any) -> bool:
        """Validate that value is a valid email address.

        Args:
            value: Value to validate

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(
                field="email", value=value, requirement="Email must be a string"
            )

        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, value):
            raise ValidationError(
                field="email",
                value=value,
                requirement="Must be a valid email address (e.g., user@example.com)",
            )

        return True

    @staticmethod
    def valid_ip_address(value: Any) -> bool:
        """Validate that value is a valid IP address.

        Args:
            value: Value to validate

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(
                field="ip_address",
                value=value,
                requirement="IP address must be a string",
            )

        try:
            ipaddress.ip_address(value)
        except ValueError:
            raise ValidationError(
                field="ip_address",
                value=value,
                requirement="Must be a valid IPv4 or IPv6 address (e.g., 192.168.1.1 or ::1)",
            )

        return True

    @staticmethod
    def valid_file_path(value: Any) -> bool:
        """Validate that value is a valid file path.

        Args:
            value: Value to validate

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, (str, Path)):
            raise ValidationError(
                field="file_path",
                value=value,
                requirement="File path must be a string or Path object",
            )

        try:
            path = Path(value)
            if not path.exists():
                raise ValidationError(
                    field="file_path",
                    value=value,
                    requirement=f"File does not exist: {path}",
                )

            if not path.is_file():
                raise ValidationError(
                    field="file_path",
                    value=value,
                    requirement=f"Path exists but is not a file: {path}",
                )
        except Exception as e:
            raise ValidationError(
                field="file_path", value=value, requirement=f"Invalid file path: {e}"
            )

        return True

    @staticmethod
    def valid_directory_path(value: Any) -> bool:
        """Validate that value is a valid directory path.

        Args:
            value: Value to validate

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, (str, Path)):
            raise ValidationError(
                field="directory_path",
                value=value,
                requirement="Directory path must be a string or Path object",
            )

        try:
            path = Path(value)
            if not path.exists():
                raise ValidationError(
                    field="directory_path",
                    value=value,
                    requirement=f"Directory does not exist: {path}",
                )

            if not path.is_dir():
                raise ValidationError(
                    field="directory_path",
                    value=value,
                    requirement=f"Path exists but is not a directory: {path}",
                )
        except Exception as e:
            raise ValidationError(
                field="directory_path",
                value=value,
                requirement=f"Invalid directory path: {e}",
            )

        return True

    @staticmethod
    def valid_json(value: Any) -> bool:
        """Validate that value is valid JSON.

        Args:
            value: Value to validate

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(
                field="json",
                value=value,
                requirement="JSON must be provided as a string",
            )

        try:
            json.loads(value)
        except json.JSONDecodeError as e:
            raise ValidationError(
                field="json", value=value, requirement=f"Must be valid JSON. Error: {e}"
            )

        return True

    @staticmethod
    def length_between(min_length: int, max_length: int) -> Callable[[Any], bool]:
        """Create validator for string length within range.

        Args:
            min_length: Minimum allowed length
            max_length: Maximum allowed length

        Returns:
            Callable: Validator function
        """

        def validator(value: Any) -> bool:
            if not isinstance(value, str):
                raise ValidationError(
                    field="value", value=value, requirement="Must be a string"
                )

            length = len(value)
            if not (min_length <= length <= max_length):
                raise ValidationError(
                    field="value",
                    value=value,
                    requirement=f"Length must be between {min_length} and {max_length} characters (current: {length})",
                )

            return True

        return validator

    @staticmethod
    def regex_pattern(pattern: str, description: str = "") -> Callable[[Any], bool]:
        """Create validator for regex pattern matching.

        Args:
            pattern: Regex pattern to match
            description: Description of what pattern should match

        Returns:
            Callable: Validator function
        """
        compiled_pattern = re.compile(pattern)

        def validator(value: Any) -> bool:
            if not isinstance(value, str):
                raise ValidationError(
                    field="value", value=value, requirement="Must be a string"
                )

            if not compiled_pattern.match(value):
                requirement = f"Must match pattern: {pattern}"
                if description:
                    requirement += f" ({description})"

                raise ValidationError(
                    field="value", value=value, requirement=requirement
                )

            return True

        return validator

    @staticmethod
    def one_of(allowed_values: List[Any]) -> Callable[[Any], bool]:
        """Create validator for allowed values.

        Args:
            allowed_values: List of allowed values

        Returns:
            Callable: Validator function
        """

        def validator(value: Any) -> bool:
            if value not in allowed_values:
                raise ValidationError(
                    field="value",
                    value=value,
                    requirement=f"Must be one of: {allowed_values}",
                )

            return True

        return validator

    @staticmethod
    def range_validator(
        min_value: Union[int, float], max_value: Union[int, float]
    ) -> Callable[[Any], bool]:
        """Create validator for numeric range.

        Args:
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Callable: Validator function
        """

        def validator(value: Any) -> bool:
            if not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    raise ValidationError(
                        field="value", value=value, requirement="Must be a number"
                    )

            if not (min_value <= value <= max_value):
                raise ValidationError(
                    field="value",
                    value=value,
                    requirement=f"Must be between {min_value} and {max_value} (current: {value})",
                )

            return True

        return validator


# Convenient aliases for common validators
validate_non_empty_string = ValidatorRegistry.non_empty_string
validate_positive_integer = ValidatorRegistry.positive_integer
validate_positive_number = ValidatorRegistry.positive_number
validate_port = ValidatorRegistry.valid_port
validate_url = ValidatorRegistry.valid_url
validate_email = ValidatorRegistry.valid_email
validate_ip_address = ValidatorRegistry.valid_ip_address
validate_file_path = ValidatorRegistry.valid_file_path
validate_directory_path = ValidatorRegistry.valid_directory_path
validate_json = ValidatorRegistry.valid_json
