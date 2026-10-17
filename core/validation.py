"""Database-agnostic validation of user goals (rejects gibberish / noise)."""

import json
import re

from fastapi import HTTPException

_GOAL_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "is", "are", "was",
    "were", "be", "been", "being", "to", "of", "in", "on", "for", "with",
    "at", "by", "from", "it", "this", "that", "these", "those", "as", "how",
    "why", "when", "where", "who", "what", "which", "show", "tell", "give",
    "me", "please", "hi", "hello", "do", "does", "can", "could", "would",
    "should", "the", "your", "our", "there", "their", "you", "you're",
    "yours", "yourself", "am", "we", "we're", "us", "them", "they", "they're",
    "i", "i'm", "im", "he", "she", "it's", "its", "his", "her",
}

_GIBSMASH = {
    "asdf", "asdfg", "asdfgh", "asdfghj", "fdsa", "sadf", "dsf", "fds",
    "qwerty", "qwer", "wert", "poiu", "poiuy", "rewq", "ytrewq",
    "sdfg", "dfgh", "fghj", "ghjk", "hjkl", "jkl", "jkl;", "klj", "kjhg",
    "zxcv", "zxcvb", "xcvbn", "cvbnm", "mnbvc", "vbnm", "nbvc", "bvcx",
    "aaaa", "bbbb", "cccc", "dddd", "eeee", "ffff", "gggg", "hhhh", "iiii",
    "jjjj", "kkkk", "llll", "mmmm", "nnnn", "oooo", "pppp", "qqqq", "rrrr",
    "ssss", "tttt", "uuuu", "vvvv", "wwww", "xxxx", "yyyy", "zzzz",
    "asdfjkl", "qwe", "rty", "uio", "p", "sdf", "fgh", "jkl", "lkj",
    "xyz", "xy",
    "lorem", "ipsum", "loremipsum", "fuck", "shit", "poop", "butt",
}
# NOTE: real English words ("test", "testing", "abc", ...) were removed
# from the smash list — "Show test results by category" and "ABC analysis"
# are legitimate questions. Smash detection relies on the structural
# checks below (repetition, low diversity, no vowels) instead.

_GOAL_VOWELS = set("aeiouy")


def validate_goal_text(goal: str) -> str:
    """Lightweight, database-agnostic validation.

    Rejects empty input, pure noise, keyboard smashes and obvious gibberish,
    while accepting any real business question regardless of database domain.
    """
    cleaned = (goal or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Please write a business question, e.g. 'Show total sales by category'.")

    if not re.search(r"[A-Za-z]", cleaned):
        raise HTTPException(status_code=400, detail="That doesn't look like a real question. Please describe what you want in words.")

    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", cleaned.lower())
    if len(tokens) < 2:
        raise HTTPException(status_code=400, detail="Please write a fuller business question, e.g. 'Which product has the highest revenue?'")

    # Single-letter tokens ("a b c d e f g", "x y") add no meaning: a real
    # question needs at least a couple of proper words.
    real_words = [t for t in tokens if len(t) >= 2]
    if len(real_words) < 2:
        raise HTTPException(status_code=400, detail="That doesn't look like a real question. Please type a meaningful business goal.")

    for token in tokens:
        if re.fullmatch(r"(.)\1{3,}", token):
            raise HTTPException(status_code=400, detail="That doesn't look like a real question. Please type a meaningful business goal.")
        if len(token) >= 4 and len(set(token)) <= 2:
            raise HTTPException(status_code=400, detail="That doesn't look like a real question. Please type a meaningful business goal.")
        if len(token) >= 4 and not any(ch in _GOAL_VOWELS for ch in token):
            raise HTTPException(status_code=400, detail="That doesn't look like a real question. Please type a meaningful business goal.")
        if token in _GIBSMASH:
            raise HTTPException(status_code=400, detail="That doesn't look like a real question. Please type a meaningful business goal.")

    content_words = [t for t in real_words if t not in _GOAL_STOPWORDS]
    if not content_words:
        raise HTTPException(status_code=400, detail="That doesn't look like a real question. Please type a meaningful business goal.")

    return cleaned


# ---------------------------------------------------------------------------
# LLM output guardrail layer (provider-agnostic).
#
# Every agent that consumes raw model output (local Ollama or cloud Groq) must
# run it through these helpers BEFORE trusting it. Local models are weaker and
# more prone to echo, refusals, code-fence soup and gibberish, so a single
# shared layer keeps the behaviour identical on both providers.
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:sql|json|python|text)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

_REFUSAL_RE = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bi\s+(?:cannot|cant|can't|won't|am\s+unable|don't\s+know|am\s+sorry|apologize)",
        r"\bas\s+an\s+ai",
        r"i'm\s+(?:sorry|not\s+able)",
        r"i\s+don't\s+have\s+(?:access|enough)",
        r"cannot\s+(?:provide|help|assist)",
    )
]

_SQL_TABLE_RE = re.compile(
    r"\b(?:FROM|JOIN|UPDATE|INTO|DELETE\s+FROM|TABLE)\s+"
    r"[\"`\[]?([A-Za-z_][\w$]*)[\]\"`]?(?:\s*\.\s*[\"`\[]?([A-Za-z_][\w$]*)[\]\"`]?)?",
    re.IGNORECASE,
)


# PostgreSQL functions that read/write server files, kill sessions, change
# server state or tunnel to another host. Matched only when CALLED (`name(`)
# so a column such as `query_to_xml_count` or a bare mention of the name is
# never rejected. Deliberately an explicit list (NOT a broad `pg_*` rule) so
# legitimate catalog helpers like pg_typeof / pg_get_expr stay usable.
_READONLY_DANGEROUS_FNS = (
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "pg_terminate_backend", "pg_cancel_backend", "pg_reload_conf",
    "pg_read_server_files", "pg_write_server_files", "pg_ls_waldir",
    "pg_ls_logicalrepdir", "pg_promote", "pg_create_restore_point",
    "pg_switch_wal", "pg_execute_server_program", "pg_sleep",
    "query_to_xml_and_xmlschema", "query_to_xml", "schema_to_xml",
    "database_to_xml", "set_config", "lo_import", "lo_export", "lo_unlink",
    "dblink_connect", "dblink_send_query", "dblink_exec", "dblink",
)
_DANGEROUS_FN_ALT = "|".join(_READONLY_DANGEROUS_FNS)

_READONLY_DESTRUCTIVE_RE = re.compile(
    r"\b(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM|DELETE\s+\w+\s+FROM"
    r"|DROP|ALTER|CREATE|TRUNCATE\s+(?:TABLE\s+)?|GRANT|REVOKE|CALL|EXEC(?:UTE)?\b|MERGE\s+INTO"
    r"|COMMENT\s+ON|COPY\b"
    r"|\bVACUUM\b|\bANALYZE\b|\bEXPLAIN\b|\bSET\b|\bSHOW\b"
    r"|\bLISTEN\b|\bNOTIFY\b|\bLOAD\b|\bIMPORT\b|\bINTO\b"
    r"|\bPRAGMA\b|\bATTACH\b|\bDETACH\b|\bREINDEX\b|\bREPLACE\s+INTO\b|\bHANDLER\b"
    r"|\bDO\b|FOR\s+UPDATE\b|LOCK\s+IN\s+SHARE\s+MODE"
    rf"|\b(?:{_DANGEROUS_FN_ALT})\s*\()",
    re.IGNORECASE,
)
_READONLY_STRIP_STRINGS_RE = re.compile(r"'([^']|'')*'|\"([^\"]|\"\")*\"|`[^`]*`")
_READONLY_STRIP_COMMENTS_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)

# Whitelisted scalar/aggregate helpers: the ONLY calls allowed in a statement
# that references no table at all (see the FROM-less check below).
_READONLY_SAFE_FNS = frozenset({
    "count", "sum", "avg", "min", "max", "abs", "round", "floor", "ceil",
    "length", "lower", "upper", "coalesce", "nullif", "cast", "extract",
    "date_trunc", "now", "current_date", "current_timestamp", "substring",
    "concat", "trim", "replace", "split_part", "to_char", "to_date",
    "greatest", "least", "array_agg", "string_agg", "stddev", "variance",
    "percentile_cont", "rank", "dense_rank", "row_number", "ntile", "lag",
    "lead", "first_value", "last_value", "sqrt", "power", "mod", "trunc",
    "sign", "strpos", "position", "initcap", "md5",
})
# SQL keywords that may legally precede `(` (aliases, window/predicate
# syntax) but are NOT function calls.
_READONLY_CALL_KEYWORDS = frozenset({
    "as", "on", "and", "or", "not", "in", "is", "any", "all", "some",
    "exists", "values", "using", "over", "filter", "partition", "by",
    "when", "case", "between", "like", "ilike", "returning", "window",
    "only", "into", "from", "select", "with", "where", "group", "order",
    "having", "limit", "offset", "union", "except", "intersect", "table",
})
_READONLY_CALL_RE = re.compile(r"\b([A-Za-z_][\w$]*)\s*\(")


def is_read_only_sql(sql) -> bool:
    """Single shared read-only SQL guard for all agents.

    True only for SELECT / WITH ... SELECT statements with no destructive
    keywords. Strings and -- /* */ comments are stripped first so keywords
    inside literals/comments cannot cause false accepts or rejects.
    Returns False for empty/None SQL.
    """
    if not sql or not isinstance(sql, str):
        return False
    stripped = _READONLY_STRIP_STRINGS_RE.sub(" ", sql)
    stripped = _READONLY_STRIP_COMMENTS_RE.sub(" ", stripped)
    # Fail closed on stacked statements: any semicolon outside a string or
    # comment can smuggle a second statement past keyword checks.
    # A single trailing semicolon terminator is tolerated.
    _core = stripped.strip()
    if ";" in _core.rstrip(";").strip():
        return False
    if _READONLY_DESTRUCTIVE_RE.search(stripped):
        return False
    if not re.match(r"\s*(select|with)\b", stripped, re.IGNORECASE):
        return False
    # FROM-less statements: with no table to read from, the only legitimate
    # calls are the whitelisted scalar/aggregate helpers. The denylist above
    # cannot enumerate every file/session/server primitive, so any other
    # function call in a statement that references no table fails closed.
    if not sql_table_names(stripped):
        for m in _READONLY_CALL_RE.finditer(stripped):
            fn = m.group(1).lower()
            if fn in _READONLY_SAFE_FNS or fn in _READONLY_CALL_KEYWORDS:
                continue
            return False
    return True


def strip_code_fences(text):
    """Extract the content of any ```...``` block, otherwise return trimmed text."""
    text = (text or "").strip()
    if not text:
        return ""
    blocks = _CODE_FENCE_RE.findall(text)
    if blocks:
        return max((b.strip() for b in blocks), key=len)
    return text


def looks_like_refusal(text):
    low = (text or "").lower()
    return any(r.search(low) for r in _REFUSAL_RE)


def is_gibberish(text, min_letters=6):
    t = strip_code_fences(text or "")
    if not t:
        return True
    if re.fullmatch(r"([A-Za-z0-9])\1{4,}", t):
        return True
    letters = [ch for ch in t if ch.isalpha()]
    if len(letters) < min_letters:
        return True
    if len(set(letters)) / len(letters) < 0.35:
        return True
    return False


def clean_llm_text(raw):
    """Validate and normalise free-text model output.

    Returns (text, None) on success or (None, reason) when the output is
    empty / a refusal / gibberish and must not be trusted.
    """
    text = strip_code_fences(raw or "")
    if not text:
        return None, "empty output"
    if looks_like_refusal(text):
        return None, "refusal detected"
    if is_gibberish(text):
        return None, "gibberish output"
    return text, None


def parse_llm_json(raw):
    """Validate JSON model output (accepts code-fenced or embedded JSON).

    Returns (parsed, None) or (None, reason).
    """
    text = strip_code_fences(raw or "")
    if not text:
        return None, "empty output"
    try:
        return json.loads(text), None
    except Exception:
        pass
    block = re.search(r"\{.*\}", text, re.DOTALL)
    if block:
        try:
            return json.loads(block.group(0)), None
        except Exception:
            pass
    return None, "invalid JSON"


def sql_table_names(sql):
    """Return the set of table identifiers referenced by a SQL statement.

    Supports bare (`orders`) and schema-qualified (`sales.orders`) names,
    returning the fully-qualified name when a schema prefix is present.
    """
    names = set()
    for m in _SQL_TABLE_RE.finditer(sql or ""):
        first = m.group(1)
        second = m.group(2)
        name = f"{first}.{second}" if second else first
        if name.lower() not in ("dual", "public"):
            names.add(name)
    return names


def _known_table_aliases(schema_tables):
    """Known names + bare last-segments (sales.orders <-> orders).

    Multi-schema mappings key tables qualified (humanresources.employee)
    while SQL legitimately uses the bare name (resolved via search_path).
    Both directions are accepted; a bare name shared by several schemas
    resolves via the database search_path (the DB is the final arbiter).
    """
    known: set[str] = set()
    try:
        for t in (schema_tables or []):
            tl = str(t).lower()
            known.add(tl)
            if "." in tl:
                known.add(tl.split(".")[-1])
    except Exception:
        pass
    return known


_CTE_HEAD_RE = re.compile(r"\s*with\s+(?:recursive\s+)?", re.IGNORECASE)
_CTE_DEF_RE = re.compile(
    r"\s*[\"`\[]?([A-Za-z_][\w$]*)[\]\"`]?"
    r"\s*(?:\([^()]*\))?\s+AS\s*\(",
    re.IGNORECASE,
)


def _cte_names(sql) -> set:
    """Names declared by a leading ``WITH [RECURSIVE] ... `` clause (CTEs).

    Walks each ``name AS ( ... )`` definition with balanced parentheses (and
    string/identifier quoting) so multi-CTE and column-list forms such as
    ``c(i) AS (`` are recognised. Only a statement that STARTS with WITH can
    yield names: a `name AS (` inside the main statement (or in plain SELECT
    SQL) is never collected, so the unknown-table guard is not weakened.
    """
    text = sql or ""
    head = _CTE_HEAD_RE.match(text)
    if not head:
        return set()
    names: set = set()
    pos = head.end()
    while True:
        m = _CTE_DEF_RE.match(text, pos)
        if not m:
            break
        names.add(m.group(1).lower())
        i = m.end() - 1  # index of the '(' that opens the CTE body
        depth = 0
        quote = ""
        while i < len(text):
            ch = text[i]
            if quote:
                if ch == quote:
                    if text[i + 1:i + 2] == quote:  # '' / "" escape
                        i += 2
                        continue
                    quote = ""
            elif ch in ("'", '"', "`"):
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:  # unbalanced body: stop rather than guess
            break
        pos = i + 1
        nxt = re.match(r"\s*,", text[pos:])
        if not nxt:  # WITH clause finished; main statement follows
            break
        pos += nxt.end()
    return names


def unknown_sql_tables(sql, schema_tables):
    """Return referenced tables that do not exist in the schema.

    This catches hallucinations like `order_items` when the real table is
    `order_details` — deterministically, before a database round-trip.
    Schema-qualified keys also accept their bare segment (search_path).
    Names defined by a leading WITH clause are CTE aliases, not tables, so
    they are never reported as unknown (legitimate CTE SQL must not be
    rejected); every other referenced name is still checked.
    """
    known = _known_table_aliases(schema_tables)
    ctes = _cte_names(sql)
    return sorted(
        n for n in sql_table_names(sql)
        if n.lower() not in known and n.lower() not in ctes
    )


def empty_sql_tables(sql, table_counts):
    """Return tables referenced by `sql` that exist but contain zero rows."""
    counts = {t.lower(): c for t, c in (table_counts or {}).items()}
    try:
        for t, c in list(counts.items()):
            if "." in t:
                counts.setdefault(t.split(".")[-1], c)
    except Exception:
        pass
    return sorted(n for n in sql_table_names(sql) if counts.get(n.lower(), 1) == 0)


# ---------------------------------------------------------------------------
# Column-level SQL validation.
#
# Catches hallucinated COLUMN references (e.g. `orders.quantity` when the real
# column lives in `order_details`). Deterministic and provider-agnostic, so a
# weak local model and the cloud model are held to the same standard. FROM/JOIN
# aliases are resolved back to their real table names before checking.
# ---------------------------------------------------------------------------

_SQL_ALIAS_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+[\"`\[]?([A-Za-z_][\w$]*)[\]\"`]?"
    r"(?:\s+(?:AS\s+)?[\"`\[]?([A-Za-z_][\w$]*)[\]\"`]?)?",
    re.IGNORECASE,
)

_SQL_QUALIFIED_RE = re.compile(
    r"\b([A-Za-z_][\w$]*)\s*\.\s*[\"`\[]?([A-Za-z_][\w$]*)[\]\"`]?",
    re.IGNORECASE,
)


def sql_qualified_columns(sql):
    """Return the set of `table.column` references in a SQL statement.

    FROM/JOIN aliases are resolved back to their real table name first, so both
    `orders.quantity` and `o.quantity` (with `FROM orders o`) yield the same
    reference. Qualifiers that cannot be resolved to any alias are returned
    verbatim; bare columns (no qualifier) are not returned.
    """
    aliases = {}
    for m in _SQL_ALIAS_RE.finditer(sql or ""):
        table = m.group(1)
        alias = m.group(2)
        aliases[table.lower()] = table
        if alias:
            aliases[alias.lower()] = table
    refs = set()
    for m in _SQL_QUALIFIED_RE.finditer(sql or ""):
        qual = m.group(1)
        col = m.group(2)
        refs.add(f"{aliases.get(qual.lower(), qual)}.{col}")
    return refs


def unknown_sql_columns(sql, columns_by_table):
    """Return `table.column` references that do not exist in the schema.

    `columns_by_table` maps a table name to an iterable of its real columns.
    Only references whose qualifier resolves to a KNOWN table are validated;
    unresolvable qualifiers and bare columns are skipped (the former is the
    job of `unknown_sql_tables`). This catches hallucinations like
    `orders.quantity` when quantity only exists on `order_details`.
    """
    columns_by_table = {
        t.lower(): {c.lower() for c in cols}
        for t, cols in (columns_by_table or {}).items()
    }
    # Schema-qualified keys (sales.orders) also validate under their bare
    # segment (orders.col), since search_path resolves bare SQL names.
    try:
        for t, cols in list(columns_by_table.items()):
            if "." in t:
                columns_by_table.setdefault(t.split(".")[-1], cols)
    except Exception:
        pass
    bad = []
    for ref in sorted(sql_qualified_columns(sql)):
        table, _, col = ref.partition(".")
        if table.lower() not in columns_by_table:
            continue
        if col.lower() not in columns_by_table[table.lower()]:
            bad.append(ref)
    return bad


# ---------------------------------------------------------------------------
# Query-generation pipeline validation (n.txt item 2).
#
# Post-generation gate used by GoalAgent BEFORE execution: nonexistent
# table/column, invalid JOIN (missing FROM entry), bad aggregation (nested
# aggregates, bare SELECT * for an aggregating goal, SUM/AVG over an
# incompatible text/datetime type) and execution failure. Returns issues so
# the caller can repair/regenerate instead of sending a broken result to UI.
# ---------------------------------------------------------------------------

_NESTED_AGG_RE = re.compile(
    r"\b(AVG|SUM|COUNT|MIN|MAX)\s*\(\s*(?:DISTINCT\s+)?(AVG|SUM|COUNT|MIN|MAX)\s*\(",
    re.IGNORECASE,
)

_BARE_STAR_RE = re.compile(r"\bselect\b([\s\S]*?)\bfrom\b", re.IGNORECASE)
_AGG_FN_RE = re.compile(
    r"\b(?:count|sum|avg|average|min|max|stddev|std|variance|var_pop|var_samp"
    r"|median|percentile_cont|percentile_disc)\s*\(",
    re.IGNORECASE,
)

_AGG_CALL_RE = re.compile(
    r"\b(SUM|AVG|MIN|MAX|COUNT)\s*\(\s*(?:DISTINCT\s+)?"
    r"(?:([A-Za-z_][\w$]*)\s*\.\s*)?[\"`\[]?([A-Za-z_][\w$*]*)[\]\"`]?\s*\)",
    re.IGNORECASE,
)

_INCOMPATIBLE_AGG_TEXT_TYPES = (
    "char", "text", "varchar", "citext", "uuid", "ltree",
)
_INCOMPATIBLE_AGG_DT_TYPES = ("date", "time", "timestamp", "datetime", "smalldatetime")


def _classify_col_type(data_type) -> str:
    dt = str(data_type or "").lower().strip()
    if not dt:
        return "unknown"
    if any(k in dt for k in ("int", "decimal", "numeric", "float", "real",
                             "double", "money")):
        return "numeric"
    if any(k in dt for k in ("date", "time", "timestamp", "datetime", "smalldatetime")):
        return "datetime"
    if any(k in dt for k in ("char", "text", "varchar", "citext", "uuid", "ltree")):
        return "text"
    if any(k in dt for k in ("bytea", "blob", "binary", "varbinary")):
        return "binary"
    return "other"


def find_nested_aggregates(sql) -> list:
    """Return nested-aggregate snippets like AVG(SUM(...)); [] when clean."""
    return [f"{a}({b}" for a, b in _NESTED_AGG_RE.findall(sql or "")]


def is_bare_select_star(sql) -> bool:
    """True when sql is SELECT * with no aggregate fn and no GROUP BY."""
    if not sql:
        return False
    m = _BARE_STAR_RE.search(sql)
    if not m or "*" not in m.group(1):
        return False
    if _AGG_FN_RE.search(sql):
        return False
    return True


def find_incompatible_aggregations(sql, columns_by_type=None) -> list:
    """Flag SUM/AVG over text/datetime/binary columns (wrong-datatype math).

    `columns_by_type` maps table -> {column: data_type}. COUNT(*) is always
    valid and never flagged. Returns issue strings.
    """
    if not sql or not columns_by_type:
        return []
    type_map = {
        str(t).lower(): {str(c).lower(): _classify_col_type(dt)
                         for c, dt in (cols or {}).items()}
        for t, cols in columns_by_type.items()
    }
    aliases = {}
    for m in _SQL_ALIAS_RE.finditer(sql):
        aliases[m.group(1).lower()] = m.group(1)
        if m.group(2):
            aliases[m.group(2).lower()] = m.group(1)
    issues = []
    for m in _AGG_CALL_RE.finditer(sql):
        fn, qual, col = m.group(1).upper(), m.group(2), m.group(3)
        if col == "*" or fn in ("COUNT", "MIN", "MAX"):
            continue
        # Coerced text-numeric aggregates wrap a SAFE cast expression
        # (SUM(CAST(NULLIF(...)))) rather than a bare column: the value is
        # numeric at aggregation time, so this is valid and never flagged.
        # Bare SUM/AVG over text/datetime/binary is still flagged below.
        try:
            _call = m.group(0)
        except Exception:
            _call = ""
        if re.search(r"\b(CAST|REGEXP_REPLACE|NULLIF)\s*\(", _call,
                     re.IGNORECASE):
            continue
        # SUM/AVG over a text/datetime/binary column is a datatype error.
        table = aliases.get((qual or "").lower()) if qual else None
        candidates = [table] if table else list(type_map)
        for t in candidates:
            kind = type_map.get(str(t or "").lower(), {}).get(col.lower())
            if kind in ("text", "datetime", "binary"):
                issues.append(
                    f"{fn} over incompatible {kind} column "
                    f"{(table or qual or '?')}.{col}"
                )
                break
    return sorted(set(issues))


def validate_generated_sql(sql, *, schema_tables=None, columns_by_table=None,
                           columns_by_type=None, join_path=None) -> list:
    """Single post-generation gate. Returns ordered issue strings ([] = PASS).

    Checks: unknown tables, unknown columns, nested aggregates, incompatible
    aggregation datatypes. Missing-FROM / PK-FK JOIN checks live on the agent
    (they need the FK graph); execution/zero-row checks happen after execute.
    """
    issues = []
    for t in unknown_sql_tables(sql, schema_tables or []):
        issues.append(f"nonexistent table: {t}")
    for c in unknown_sql_columns(sql, columns_by_table or {}):
        issues.append(f"nonexistent column: {c}")
    for n in find_nested_aggregates(sql):
        issues.append(f"nested aggregate: {n} is invalid SQL")
    for inc in find_incompatible_aggregations(sql, columns_by_type):
        issues.append(f"incompatible datatype: {inc}")
    return issues


def enforce_readonly_guards(sql, mapping=None) -> str:
    """Fail-closed allowlist guard called before EVERY SQL execution.

    Asserts ``is_read_only_sql(sql)`` AND ``validate_generated_sql`` unknown
    table/column issues == []. ``mapping`` is ``{table: columns}`` (either
    ``{table: {col: meta}}`` or ``{table: [cols]}``); when omitted only the
    read-only check runs. Returns the cleaned SQL on pass; raises
    RuntimeError with a clear message on failure (never returns a forgiving
    default).
    """
    cleaned = (sql or "").strip()
    if not is_read_only_sql(cleaned):
        raise RuntimeError(
            "SQL guard rejected execution: statement is not read-only "
            "(only SELECT / WITH ... SELECT are allowed)."
        )
    schema_tables = list((mapping or {}).keys())
    columns_by_table: dict = {}
    if mapping:
        for table, cols in mapping.items():
            if isinstance(cols, dict):
                columns_by_table[table] = list(cols.keys())
            else:
                try:
                    columns_by_table[table] = list(cols)
                except TypeError:
                    columns_by_table[table] = []
    bad_tables = unknown_sql_tables(cleaned, schema_tables) if mapping is not None else []
    bad_columns = unknown_sql_columns(cleaned, columns_by_table) if mapping else []
    if bad_tables or bad_columns:
        details = []
        if bad_tables:
            details.append(f"unknown tables: {', '.join(bad_tables)}")
        if bad_columns:
            details.append(f"unknown columns: {', '.join(bad_columns)}")
        raise RuntimeError(
            "SQL guard rejected execution (" + "; ".join(details) + ")."
        )
    return cleaned


# ---------------------------------------------------------------------------
# Chart-type guardrail.
#
# The LLM (local or cloud) may PROPOSE a chart type after seeing the data, but
# a deterministic rule decides the final type based on the data shape. Both
# providers pass through the exact same check, so charts are identical whether
# the backend is Ollama or Groq and a weak local model can never force a chart
# that does not fit the data.
# ---------------------------------------------------------------------------

CHART_WHITELIST = {
    "bar", "bar_horizontal", "bar_stacked", "line", "area", "combo",
    "scatter", "bubble", "pie", "doughnut", "polar_area", "radar",
    "histogram", "kpi_card",
}

CHART_SHAPE_RULES = {
    "category_numeric": {
        "allowed": {"bar", "bar_horizontal", "line", "pie", "doughnut"},
        "default": "bar",
    },
    "category_numeric_multi": {
        "allowed": {"bar_stacked", "bar", "radar", "line", "combo"},
        "default": "bar_stacked",
    },
    "category_low_card": {
        "allowed": {"pie", "doughnut", "polar_area", "bar"},
        "default": "pie",
    },
    "category_counts": {
        "allowed": {"bar", "bar_horizontal", "pie", "doughnut"},
        "default": "bar",
    },
    "datetime_numeric": {
        "allowed": {"line", "area", "bar", "combo"},
        "default": "line",
    },
    "datetime_numeric_multi": {
        "allowed": {"combo", "line", "area", "bar_stacked"},
        "default": "combo",
    },
    "numeric_numeric": {
        "allowed": {"scatter", "bubble", "line"},
        "default": "scatter",
    },
    "numeric_numeric_numeric": {
        "allowed": {"bubble", "scatter"},
        "default": "bubble",
    },
    "distribution": {
        "allowed": {"histogram", "bar"},
        "default": "histogram",
    },
    "share": {
        "allowed": {"pie", "doughnut", "polar_area", "bar"},
        "default": "pie",
    },
}

_CHART_ALIASES = {
    "bar": "bar", "bars": "bar", "column": "bar", "columns": "bar",
    "horizontal bar": "bar_horizontal", "hbar": "bar_horizontal", "barh": "bar_horizontal",
    "stacked bar": "bar_stacked", "stacked": "bar_stacked",
    "line": "line", "lines": "line", "trend": "line", "line chart": "line",
    "area": "area", "area chart": "area",
    "combo": "combo", "combination": "combo", "dual axis": "combo", "dual-axis": "combo",
    "scatter": "scatter", "scatterplot": "scatter", "scatter plot": "scatter",
    "bubble": "bubble", "bubble chart": "bubble",
    "pie": "pie", "pie chart": "pie",
    "donut": "doughnut", "doughnut": "doughnut", "donut chart": "doughnut",
    "polar": "polar_area", "polar area": "polar_area", "radar": "radar",
    "histogram": "histogram", "hist": "histogram",
    "kpi": "kpi_card", "card": "kpi_card", "metric": "kpi_card", "cards": "kpi_card",
}


def _normalize_chart(suggested):
    """Map a free-form LLM suggestion (e.g. 'Pie chart!') to a whitelisted id."""
    if not suggested:
        return None
    low = str(suggested).strip().lower()
    if low in CHART_WHITELIST:
        return low
    if low in _CHART_ALIASES:
        return _CHART_ALIASES[low]
    for word in re.findall(r"[a-z]+", low):
        if word in _CHART_ALIASES:
            return _CHART_ALIASES[word]
    return None


# ---------------------------------------------------------------------------
# hyp.txt PHASE 1-2 — semantic pair helpers for Decision eligibility
# (additive only; existing guards above are unchanged).
#
# All checks operate on canonical FIELD_ROLES from core.semantic
# (never on raw column names), so no per-column exclusions exist here.
# ---------------------------------------------------------------------------

_MEASURE_FAMILY = frozenset({
    "MEASURE", "BUSINESS_MEASURE", "TECHNICAL_MEASURE", "DERIVED_METRIC",
})
_JOIN_ONLY_ROLES = frozenset({"FOREIGN_KEY", "IDENTIFIER"})


def is_measure_role_name(role) -> bool:
    return str(role or "").upper() in _MEASURE_FAMILY


def measure_used_as_dimension(metric_role, segment_role) -> bool:
    """True when a measure-family field is used as a segment/dimension.

    hyp #12: ``metric = Price Per Night, segment = Total Amt`` is invalid
    when both sides are measures — never turn one measure into a segment.
    """
    return (is_measure_role_name(metric_role)
            and is_measure_role_name(segment_role))


def fk_used_as_driver(driver_role) -> bool:
    """True when a join-context key is proposed as a driver/lever.

    Foreign keys are join/relational context only, never a business
    driver or lever (hyp #10/#95).
    """
    return str(driver_role or "").upper() in _JOIN_ONLY_ROLES


def semantic_pair_rejection(metric_role, segment_role,
                             driver_role=None) -> str | None:
    """Return the eligibility rejection code for a semantic pair, or None.

    Codes align with services.decision_eligibility reason vocabulary.
    """
    if measure_used_as_dimension(metric_role, segment_role):
        return "measure_used_as_dimension"
    if driver_role is not None and fk_used_as_driver(driver_role):
        return "no_actionable_lever"
    return None


def validate_chart_choice(suggested, shape):
    """Deterministic guardrail for an LLM-proposed chart type.

    Returns (chart_type, reason). A suggestion is used only if it is a real
    chart type AND fits the data shape; otherwise the shape's default wins.
    """
    rule = CHART_SHAPE_RULES.get(shape)
    if not rule:
        return "bar", "unknown shape -> default 'bar'"
    default = rule["default"]
    norm = _normalize_chart(suggested)
    if norm in rule["allowed"]:
        return norm, f"accepted '{norm}' for {shape}"
    if norm:
        return default, f"rejected '{norm}' for {shape} (does not fit) -> default '{default}'"
    return default, f"no usable suggestion for {shape} -> default '{default}'"
