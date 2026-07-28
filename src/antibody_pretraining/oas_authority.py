"""Build a canonical OAS data-unit authority from an official search response."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import json
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qs, quote, unquote, urlparse


FORMAT_VERSION = "oas_data_unit_authority_v1"
OFFICIAL_HOST = "opig.stats.ox.ac.uk"
SOURCE_SEARCH_URLS = {
    "unpaired": "https://opig.stats.ox.ac.uk/webapps/oas/oas_unpaired/",
    "paired": "https://opig.stats.ox.ac.uk/webapps/oas/oas_paired/",
}
DIRECT_PREFIXES = {
    "unpaired": "https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/",
    "paired": "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/",
}
_HEADLINE_PATTERN = re.compile(
    rb"Your search yielded\s*<b>([0-9,]+)</b>\s*unique sequences from\s*"
    rb"<b>([0-9,]+)</b>\s*studies\."
)
_WGET_PATTERN = re.compile(rb'"wget (https://[^"\s]+\.csv\.gz)"')
_SPACE_PATTERN = re.compile(r"\s+")


class OASAuthorityError(ValueError):
    """Report a fail-closed OAS authority parsing or validation failure."""


def canonical_json_bytes(payload: object) -> bytes:
    """Encode deterministic finite ASCII JSON with one terminal line feed."""
    try:
        return (
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise OASAuthorityError("Payload is not finite canonical JSON") from error


def _sha256(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest for bytes."""
    return hashlib.sha256(payload).hexdigest()


def _normalized_text(parts: Iterable[str]) -> str:
    """Collapse HTML text whitespace without changing other characters."""
    return _SPACE_PATTERN.sub(" ", "".join(parts)).strip()


@dataclass(frozen=True)
class _Cell:
    tag: str
    text: str
    href: str | None


class _ResultsTableParser(HTMLParser):
    """Extract one explicitly identified OAS results table."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_table = False
        self._table_depth = 0
        self._in_row = False
        self._cell_tag: str | None = None
        self._cell_parts: list[str] = []
        self._cell_href: str | None = None
        self._row: list[_Cell] = []
        self.rows: list[tuple[_Cell, ...]] = []
        self.table_count = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: Sequence[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if tag == "table":
            if self._in_table:
                self._table_depth += 1
            elif attributes.get("id") == "results_table":
                self._in_table = True
                self._table_depth = 1
                self.table_count += 1
            return
        if not self._in_table:
            return
        if tag == "tr":
            if self._in_row:
                raise OASAuthorityError("Nested table rows are not supported")
            self._in_row = True
            self._row = []
        elif tag in {"th", "td"} and self._in_row:
            if self._cell_tag is not None:
                raise OASAuthorityError("Nested table cells are not supported")
            self._cell_tag = tag
            self._cell_parts = []
            self._cell_href = None
        elif tag == "a" and self._cell_tag is not None:
            href = attributes.get("href")
            if href is not None:
                if self._cell_href is not None:
                    raise OASAuthorityError("A table cell contains multiple links")
                self._cell_href = href

    def handle_endtag(self, tag: str) -> None:
        if not self._in_table:
            return
        if tag == "table":
            self._table_depth -= 1
            if self._table_depth == 0:
                if self._cell_tag is not None or self._in_row:
                    raise OASAuthorityError("Results table ended inside a row or cell")
                self._in_table = False
            return
        if tag in {"th", "td"} and tag == self._cell_tag:
            self._row.append(
                _Cell(
                    tag=tag,
                    text=_normalized_text(self._cell_parts),
                    href=self._cell_href,
                )
            )
            self._cell_tag = None
            self._cell_parts = []
            self._cell_href = None
        elif tag == "tr" and self._in_row:
            if self._cell_tag is not None:
                raise OASAuthorityError("Results row ended inside a cell")
            if self._row:
                self.rows.append(tuple(self._row))
            self._row = []
            self._in_row = False

    def handle_data(self, data: str) -> None:
        if self._in_table and self._cell_tag is not None:
            self._cell_parts.append(data)

    def close(self) -> None:
        super().close()
        if self._in_table or self._in_row or self._cell_tag is not None:
            raise OASAuthorityError("Results HTML ended inside a table structure")


def _strict_source_class(source_class: str) -> str:
    """Validate and return one admitted OAS source class."""
    if source_class not in SOURCE_SEARCH_URLS:
        raise OASAuthorityError(
            f"source_class must be one of {sorted(SOURCE_SEARCH_URLS)}"
        )
    return source_class


def _canonical_official_url(value: str, source_class: str) -> str:
    """Validate one official direct data-unit URL and normalize its path."""
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != OFFICIAL_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise OASAuthorityError(f"Invalid official OAS data-unit URL: {value!r}")
    prefix_path = urlparse(DIRECT_PREFIXES[source_class]).path
    decoded_path = unquote(parsed.path)
    if not decoded_path.startswith(prefix_path) or not decoded_path.endswith(
        ".csv.gz"
    ):
        raise OASAuthorityError(
            f"URL is outside the {source_class} data-unit namespace: {value!r}"
        )
    relative = decoded_path[len(prefix_path) :]
    _validate_relative_unit(relative)
    canonical_path = prefix_path + quote(relative, safe="/-._")
    return f"https://{OFFICIAL_HOST}{canonical_path}"


def _validate_relative_unit(value: str) -> None:
    """Reject ambiguous or escaping data-unit paths."""
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or "//" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or not value.endswith(".csv.gz")
    ):
        raise OASAuthorityError(f"Invalid relative OAS data-unit path: {value!r}")


def _unit_from_details_href(value: str) -> str:
    """Extract one unambiguous relative unit path from a details link."""
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc or parsed.fragment:
        raise OASAuthorityError(f"Invalid relative details link: {value!r}")
    if not parsed.path.endswith("dataunit"):
        raise OASAuthorityError(f"Unexpected details endpoint: {value!r}")
    query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
    if set(query) != {"unit"} or len(query["unit"]) != 1:
        raise OASAuthorityError(f"Ambiguous details query: {value!r}")
    unit = query["unit"][0]
    _validate_relative_unit(unit)
    return unit


def _integer(value: str, label: str) -> int:
    """Parse a non-negative comma-formatted integer without coercion."""
    if not re.fullmatch(r"(?:0|[1-9][0-9]{0,2}(?:,[0-9]{3})*|[1-9][0-9]*)", value):
        raise OASAuthorityError(f"{label} is not a canonical integer: {value!r}")
    result = int(value.replace(",", ""))
    if result < 0:
        raise OASAuthorityError(f"{label} must be non-negative")
    return result


def _wget_urls(html_bytes: bytes, source_class: str) -> tuple[str, ...]:
    """Extract and validate the independently rendered bulk-download URL list."""
    matches = [
        _canonical_official_url(match.decode("ascii"), source_class)
        for match in _WGET_PATTERN.findall(html_bytes)
    ]
    if not matches:
        raise OASAuthorityError("Official response contains no bulk-download URLs")
    if len(matches) != len(set(matches)):
        raise OASAuthorityError("Official bulk-download URL list contains duplicates")
    return tuple(matches)


def _table_rows(
    html_bytes: bytes,
    source_class: str,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    """Parse the rendered table and independently derive every direct URL."""
    try:
        html_text = html_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OASAuthorityError("Official response is not valid UTF-8") from error
    parser = _ResultsTableParser()
    parser.feed(html_text)
    parser.close()
    if parser.table_count != 1 or not parser.rows:
        raise OASAuthorityError("Expected exactly one nonempty results table")

    header_cells = parser.rows[0]
    if not header_cells or any(cell.tag != "th" for cell in header_cells):
        raise OASAuthorityError("Results table has no canonical header row")
    headers = tuple(cell.text for cell in header_cells)
    required_prefix = ("Details", "DS Name", "#Unique Sequences")
    if headers[:3] != required_prefix or len(set(headers)) != len(headers):
        raise OASAuthorityError(f"Unexpected results table headers: {headers!r}")

    entries: list[dict[str, Any]] = []
    for row_number, cells in enumerate(parser.rows[1:], start=1):
        if any(cell.tag != "td" for cell in cells) or len(cells) != len(headers):
            raise OASAuthorityError(
                f"Malformed results row {row_number}: expected {len(headers)} cells"
            )
        detail = cells[0]
        if detail.href is None or detail.text != "Details":
            raise OASAuthorityError(f"Malformed details cell in row {row_number}")
        relative_unit = _unit_from_details_href(detail.href)
        direct_url = _canonical_official_url(
            DIRECT_PREFIXES[source_class] + relative_unit,
            source_class,
        )
        study = cells[1].text
        if not study or relative_unit.split("/", 1)[0] != study:
            raise OASAuthorityError(
                f"Study and unit path disagree in row {row_number}"
            )
        unique_sequences = _integer(
            cells[2].text,
            f"row {row_number} unique sequence count",
        )
        metadata = {
            header: cell.text for header, cell in zip(headers[3:], cells[3:])
        }
        entries.append(
            {
                "canonical_url": direct_url,
                "relative_unit": relative_unit,
                "study": study,
                "unique_sequences": unique_sequences,
                "metadata": metadata,
            }
        )
    if not entries:
        raise OASAuthorityError("Results table contains no data-unit rows")
    urls = [entry["canonical_url"] for entry in entries]
    if len(urls) != len(set(urls)):
        raise OASAuthorityError("Results table contains duplicate data units")
    return headers, entries


def build_oas_authority_manifest(
    html_bytes: bytes,
    *,
    source_class: str,
) -> dict[str, Any]:
    """Build and self-hash one canonical authority from official raw bytes."""
    source_class = _strict_source_class(source_class)
    if not isinstance(html_bytes, bytes) or not html_bytes:
        raise OASAuthorityError("html_bytes must be nonempty bytes")
    headline_matches = _HEADLINE_PATTERN.findall(html_bytes)
    if len(headline_matches) != 1:
        raise OASAuthorityError("Expected exactly one OAS result-count headline")
    headline_sequences = _integer(
        headline_matches[0][0].decode("ascii"),
        "headline unique sequence count",
    )
    headline_studies = _integer(
        headline_matches[0][1].decode("ascii"),
        "headline study count",
    )

    wget_urls = _wget_urls(html_bytes, source_class)
    headers, entries = _table_rows(html_bytes, source_class)
    table_urls = tuple(entry["canonical_url"] for entry in entries)
    if set(wget_urls) != set(table_urls) or len(wget_urls) != len(table_urls):
        missing_from_table = sorted(set(wget_urls) - set(table_urls))
        missing_from_script = sorted(set(table_urls) - set(wget_urls))
        raise OASAuthorityError(
            "Bulk script and table disagree: "
            f"script_only={missing_from_table[:3]!r}, "
            f"table_only={missing_from_script[:3]!r}"
        )
    entries.sort(key=lambda entry: entry["canonical_url"])
    observed_studies = sorted({entry["study"] for entry in entries})
    if len(observed_studies) != headline_studies:
        raise OASAuthorityError(
            "Headline study count disagrees with the rendered data-unit table"
        )
    table_sequence_sum = sum(entry["unique_sequences"] for entry in entries)
    if table_sequence_sum != headline_sequences:
        raise OASAuthorityError(
            "Headline unique sequence count disagrees with data-unit count sum"
        )

    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "source": {
            "database": "Observed Antibody Space",
            "discovery_method": "HTTP POST with an empty search form",
            "license": "CC-BY-4.0",
            "raw_response_sha256": _sha256(html_bytes),
            "search_url": SOURCE_SEARCH_URLS[source_class],
            "source_class": source_class,
        },
        "schema": {
            "rendered_table_headers": list(headers),
            "url_cross_check": "bulk_wget_script_equals_rendered_table",
        },
        "summary": {
            "data_unit_count": len(entries),
            "headline_study_count": headline_studies,
            "headline_unique_sequence_count": headline_sequences,
            "observed_study_count": len(observed_studies),
            "table_unique_sequence_count_sum": table_sequence_sum,
        },
        "studies": observed_studies,
        "data_units": entries,
    }
    manifest["manifest_sha256"] = _sha256(canonical_json_bytes(manifest))
    return manifest


def verify_manifest_self_hash(manifest: Mapping[str, Any]) -> bool:
    """Verify the embedded hash without accepting extra hash interpretations."""
    if not isinstance(manifest, Mapping):
        return False
    claimed = manifest.get("manifest_sha256")
    if not isinstance(claimed, str) or not re.fullmatch(r"[0-9a-f]{64}", claimed):
        return False
    unhashed = dict(manifest)
    del unhashed["manifest_sha256"]
    return _sha256(canonical_json_bytes(unhashed)) == claimed

