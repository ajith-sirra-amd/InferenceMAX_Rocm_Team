# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for FormattablePath and FormattableString."""

from pathlib import Path

import pytest

from srtctl.core.formatting import FormattablePath, FormattableString


class TestFormattableString:
    """Tests for FormattableString."""

    def test_no_placeholders(self):
        """Test string with no placeholders."""
        fs = FormattableString(template="Hello World!")
        result = fs.raw_string()
        assert result == "Hello World!"

    def test_with_placeholders(self):
        """Test string with placeholders."""
        fs = FormattableString(template="Hello {name}!")
        result = fs.raw_string({"name": "Test"})
        assert result == "Hello Test!"

    def test_frozen(self):
        """Test that FormattableString is immutable."""
        fs = FormattableString(template="test")

        with pytest.raises(AttributeError):
            fs.template = "new"  # type: ignore[misc]


class TestFormattablePath:
    """Tests for FormattablePath."""

    def test_no_placeholders(self):
        """Test path with no placeholders."""
        fp = FormattablePath(template="/static/path")
        result = fp.raw_path_no_context(make_absolute=False, ensure_exists=False)
        assert result == Path("/static/path")

    def test_with_placeholders(self):
        """Test path with placeholders."""
        fp = FormattablePath(template="/logs/{job_id}")
        result = fp.raw_path_no_context(make_absolute=False, ensure_exists=False, format_kwargs={"job_id": "12345"})
        assert result == Path("/logs/12345")

    def test_frozen(self):
        """Test that FormattablePath is immutable."""
        fp = FormattablePath(template="/test")

        with pytest.raises(AttributeError):
            fp.template = "/new"  # type: ignore[misc]

    def test_relative_path(self):
        """Test relative path handling."""
        fp = FormattablePath(template="./outputs/test")
        result = fp.raw_path_no_context(make_absolute=False, ensure_exists=False)
        assert not result.is_absolute()

    def test_absolute_path(self):
        """Test absolute path resolution."""
        fp = FormattablePath(template="./outputs/test")
        result = fp.raw_path_no_context(make_absolute=True, ensure_exists=False)
        assert result.is_absolute()
