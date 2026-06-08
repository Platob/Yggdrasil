"""Full SQL query parser → plan nodes.

Reuses the tokenizer from :mod:`yggdrasil.saga.expr.backends.sql`
and extends the grammar from predicate-only to full SELECT/INSERT/MERGE
statements with CTEs, JOINs, GROUP BY, ORDER BY, HAVING, LATERAL VIEW,
window functions, function calls, CASE WHEN, and set operations.

Dialect-aware: the base parser handles ANSI + common extensions.
:class:`DatabricksSQLParser` (in :mod:`databricks`) adds Databricks
specifics (LATERAL VIEW EXPLODE, date_trunc, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yggdrasil.enums import JoinType
from yggdrasil.enums.dialect import Dialect
from yggdrasil.saga.expr.backends.sql import (
    _CAST_DTYPE_ALIASES,
    _CAST_TEMPORAL_PARSERS,
    _Token,
    _coerce_number,
    _resolve_dialect,
    _tokenize,
)
from yggdrasil.saga.expr.nodes import (
    Alias,
    Arithmetic,
    Between,
    CaseWhen,
    Cast,
    Column,
    Comparison,
    Expression,
    FunctionCall,
    InList,
    IsNull,
    Like,
    Literal,
    Logical,
    Not,
    Predicate,
    SortOrder,
    Star,
    Subscript,
    WindowFunction,
    WindowSpec,
)
from yggdrasil.saga.expr.operators import ArithmeticOp, CompareOp, LogicalOp

from .nodes import InsertNode, MergeNode, PlanNode, ScanNode, SelectNode
from .ops import CTE, JoinClause, LateralViewItem, SetOp, SubqueryRef, TableRef

if TYPE_CHECKING:
    pass


__all__ = ["parse_sql", "SQLQueryParser"]


# Extended reserved keywords for full SQL parsing
_QUERY_RESERVED = frozenset({
    # Original predicate keywords
    "AND", "OR", "NOT", "IN", "BETWEEN", "IS", "NULL",
    "LIKE", "ILIKE", "TRUE", "FALSE", "CAST", "AS",
    "TIMESTAMP", "TIMESTAMPTZ", "TIMESTAMPNTZ", "TIMESTAMP_NTZ",
    "TIMESTAMP_LTZ", "DATETIME", "DATE", "TIME", "TIMETZ",
    # Statement keywords
    "SELECT", "FROM", "WHERE", "GROUP", "BY", "HAVING",
    "ORDER", "ASC", "DESC", "LIMIT", "OFFSET",
    "DISTINCT", "ALL",
    # Joins
    "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "CROSS", "ON",
    # Set operations
    "UNION", "INTERSECT", "EXCEPT",
    # CTE
    "WITH",
    # CASE
    "CASE", "WHEN", "THEN", "ELSE", "END",
    # Window
    "OVER", "PARTITION", "ROWS", "RANGE", "UNBOUNDED",
    "PRECEDING", "FOLLOWING", "CURRENT", "ROW",
    # Lateral
    "LATERAL", "VIEW",
    # DML
    "INSERT", "INTO", "VALUES", "MERGE", "USING", "MATCHED",
    "UPDATE", "DELETE", "SET",
    # Misc
    "EXISTS", "ANY", "NULLS", "FIRST", "LAST",
    "ARRAY", "MAP", "STRUCT",
    # Interval / Extract
    "INTERVAL", "EXTRACT",
    # Qualify
    "QUALIFY",
})

_CMP_OPS = {"=": CompareOp.EQ, "==": CompareOp.EQ, "!=": CompareOp.NE,
            "<>": CompareOp.NE, "<": CompareOp.LT, "<=": CompareOp.LE,
            ">": CompareOp.GT, ">=": CompareOp.GE}
_ARITH_ADD = frozenset({"+", "-"})
_ARITH_MUL = frozenset({"*", "/", "%"})


def _tokenize_query(sql: str, dialect: Dialect) -> list[_Token]:
    """Tokenize with extended keyword set."""
    tokens: list[_Token] = []
    from yggdrasil.saga.expr.backends.sql import (
        _DOUBLE_QUOTE_IS_STRING,
        _scan_number,
        _scan_quoted,
    )
    i = 0
    n = len(sql)
    dq_is_str = dialect in _DOUBLE_QUOTE_IS_STRING
    while i < n:
        c = sql[i]
        if c.isspace():
            i += 1
            continue
        if c == "-" and i + 1 < n and sql[i + 1] == "-":
            while i < n and sql[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and sql[i + 1] == "*":
            i += 2
            while i + 1 < n and not (sql[i] == "*" and sql[i + 1] == "/"):
                i += 1
            if i + 1 >= n:
                raise ValueError(f"Unterminated block comment in {sql!r}")
            i += 2
            continue
        if c == "`":
            text, i = _scan_quoted(sql, i, "`")
            tokens.append(_Token("ident", text, text.upper(), i))
            continue
        if c == '"':
            if dq_is_str:
                text, i = _scan_quoted(sql, i, '"')
                tokens.append(_Token("string", text, "", i))
            else:
                text, i = _scan_quoted(sql, i, '"')
                tokens.append(_Token("ident", text, text.upper(), i))
            continue
        if c == "'":
            text, i = _scan_quoted(sql, i, "'")
            tokens.append(_Token("string", text, "", i))
            continue
        if c.isdigit() or (c == "." and i + 1 < n and sql[i + 1].isdigit()):
            text, i = _scan_number(sql, i)
            tokens.append(_Token("number", text, "", i))
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (sql[j].isalnum() or sql[j] == "_"):
                j += 1
            text = sql[i:j]
            upper = text.upper()
            kind = "keyword" if upper in _QUERY_RESERVED else "ident"
            tokens.append(_Token(kind, text, upper, i))
            i = j
            continue
        if c in "!<>=":
            if i + 1 < n and sql[i:i + 2] in ("!=", "<>", "<=", ">=", "=="):
                op = sql[i:i + 2]
                tokens.append(_Token("op", op, op, i))
                i += 2
                continue
            tokens.append(_Token("op", c, c, i))
            i += 1
            continue
        if c in "+-*/%":
            tokens.append(_Token("op", c, c, i))
            i += 1
            continue
        if c == "[":
            tokens.append(_Token("lbracket", c, c, i))
            i += 1
            continue
        if c == "]":
            tokens.append(_Token("rbracket", c, c, i))
            i += 1
            continue
        if c in "().,;":
            kind = {"(": "lparen", ")": "rparen", ",": "comma",
                    ".": "dot", ";": "semicolon"}[c]
            tokens.append(_Token(kind, c, c, i))
            i += 1
            continue
        raise ValueError(f"Unexpected character {c!r} at position {i} in {sql!r}")
    tokens.append(_Token("eof", "", "", n))
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class SQLQueryParser:
    """Recursive-descent parser for full SQL queries."""

    __slots__ = ("sql", "dialect", "tokens", "pos")

    def __init__(self, sql: str, dialect: Dialect) -> None:
        self.sql = sql
        self.dialect = dialect
        self.tokens = _tokenize_query(sql, dialect)
        self.pos = 0

    # ---- token helpers ---------------------------------------------------

    @property
    def cur(self) -> _Token:
        return self.tokens[self.pos]

    def _peek(self, offset: int = 1) -> _Token:
        idx = self.pos + offset
        if idx >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[idx]

    def _eat(self) -> _Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def _expect_kw(self, *kws: str) -> _Token:
        t = self.cur
        if t.kind != "keyword" or t.upper not in kws:
            raise self._error(f"expected {', '.join(kws)}")
        return self._eat()

    def _expect_kind(self, kind: str) -> _Token:
        if self.cur.kind != kind:
            raise self._error(f"expected {kind}")
        return self._eat()

    def _accept_kw(self, *kws: str) -> "_Token | None":
        t = self.cur
        if t.kind == "keyword" and t.upper in kws:
            return self._eat()
        return None

    def _accept_op(self, *ops: str) -> "_Token | None":
        t = self.cur
        if t.kind == "op" and t.text in ops:
            return self._eat()
        return None

    def _is_kw(self, *kws: str) -> bool:
        return self.cur.kind == "keyword" and self.cur.upper in kws

    def _ident_or_kw(self) -> _Token:
        """Eat an identifier or a keyword used as a name."""
        if self.cur.kind in ("ident", "keyword"):
            return self._eat()
        raise self._error("expected identifier")

    def _error(self, what: str) -> ValueError:
        t = self.cur
        return ValueError(
            f"SQL parse error: {what} at position {t.pos} in "
            f"{self.sql!r}; got {t.kind} {t.text!r}."
        )

    # ==================================================================
    # Statement-level parsing
    # ==================================================================

    def parse(self) -> PlanNode:
        node = self._parse_statement()
        if self.cur.kind not in ("eof", "semicolon"):
            raise self._error("unexpected trailing token")
        return node

    def _parse_statement(self) -> PlanNode:
        if self._is_kw("WITH"):
            return self._parse_with()
        if self._is_kw("SELECT"):
            return self._parse_select_stmt()
        if self._is_kw("INSERT"):
            return self._parse_insert()
        if self._is_kw("MERGE"):
            return self._parse_merge()
        raise self._error("expected SELECT, INSERT, MERGE, or WITH")

    # ---- WITH (CTE) --------------------------------------------------

    def _parse_with(self) -> PlanNode:
        self._expect_kw("WITH")
        ctes: list[CTE] = []
        while True:
            name_tok = self._ident_or_kw()
            self._expect_kw("AS")
            self._expect_kind("lparen")
            sub = self._parse_select_stmt()
            self._expect_kind("rparen")
            ctes.append(CTE(name=name_tok.text, plan=sub))
            if self.cur.kind == "comma":
                self._eat()
                continue
            break
        body = self._parse_select_stmt()
        if isinstance(body, SelectNode):
            body.ctes = ctes
        return body

    # ---- SELECT -------------------------------------------------------

    def _parse_select_stmt(self) -> SelectNode:
        node = self._parse_select_core()
        # Set operations: UNION [ALL], INTERSECT, EXCEPT
        set_ops: list[SetOp] = []
        while self._is_kw("UNION", "INTERSECT", "EXCEPT"):
            kw = self._eat().upper
            all_flag = ""
            if kw == "UNION" and self._accept_kw("ALL"):
                all_flag = " ALL"
            elif kw == "UNION" and self._accept_kw("DISTINCT"):
                pass  # UNION DISTINCT = UNION
            kind = kw + all_flag
            right = self._parse_select_core()
            set_ops.append(SetOp(kind=kind, plan=right))
        if set_ops:
            node.set_ops = set_ops
        return node

    def _parse_select_core(self) -> SelectNode:
        self._expect_kw("SELECT")
        node = SelectNode()

        # DISTINCT
        if self._accept_kw("DISTINCT"):
            node.distinct = True
        elif self._accept_kw("ALL"):
            pass

        # Projection list
        node.projections = self._parse_select_list()

        # FROM
        if self._accept_kw("FROM"):
            node.from_clause = self._parse_from_clause()

        # LATERAL VIEW (Databricks)
        lateral_views: list[LateralViewItem] = []
        while self._is_kw("LATERAL"):
            lateral_views.append(self._parse_lateral_view())
        if lateral_views:
            node.lateral_views = lateral_views

        # WHERE
        if self._accept_kw("WHERE"):
            node.where = self._parse_expr()

        # GROUP BY
        if self._is_kw("GROUP"):
            self._eat()
            self._expect_kw("BY")
            node.group_by = self._parse_expr_list()

        # HAVING
        if self._accept_kw("HAVING"):
            node.having = self._parse_expr()

        # QUALIFY (Databricks window-function filter)
        if self._accept_kw("QUALIFY"):
            node.qualify = self._parse_expr()

        # ORDER BY
        if self._is_kw("ORDER"):
            self._eat()
            self._expect_kw("BY")
            node.order_by = self._parse_order_by_list()

        # LIMIT
        if self._accept_kw("LIMIT"):
            tok = self._expect_kind("number")
            node.limit = int(tok.text)

        # OFFSET
        if self._accept_kw("OFFSET"):
            tok = self._expect_kind("number")
            node.offset = int(tok.text)

        return node

    # ---- SELECT list --------------------------------------------------

    def _parse_select_list(self) -> list[Expression]:
        items = [self._parse_select_item()]
        while self.cur.kind == "comma":
            self._eat(); items.append(self._parse_select_item())
        return items

    def _parse_select_item(self) -> Expression:
        # Star
        if self.cur.kind == "op" and self.cur.text == "*":
            self._eat()
            return Star()
        # qualifier.* or expression [AS alias]
        expr = self._parse_expr()
        # Check for table.*
        if (isinstance(expr, Column) and self.cur.kind == "dot"
                and self._peek(1).kind == "op" and self._peek(1).text == "*"):
            self._eat()  # dot
            self._eat()  # *
            return Star(qualifier=expr.name)
        # AS alias
        alias = self._parse_optional_alias()
        if alias:
            return Alias(expr=expr, name=alias)
        return expr

    def _parse_optional_alias(self) -> str | None:
        if self._accept_kw("AS"):
            return self._ident_or_kw().text
        # Implicit alias: ident not followed by keyword that starts a clause
        if (self.cur.kind == "ident"
                and not self._is_kw(*self._CLAUSE_KW)):
            return self._eat().text
        return None

    _CLAUSE_KW = frozenset({"FROM", "WHERE", "GROUP", "HAVING", "QUALIFY", "ORDER", "LIMIT",
        "OFFSET", "UNION", "INTERSECT", "EXCEPT", "ON", "USING", "JOIN", "INNER", "LEFT",
        "RIGHT", "FULL", "CROSS", "LATERAL", "WHEN", "THEN", "ELSE", "END", "AND", "OR"})
    _JOIN_KW = frozenset({"JOIN", "INNER", "LEFT", "RIGHT", "FULL", "CROSS"})

    def _parse_from_clause(self) -> Any:
        left = self._parse_from_item()
        while self._is_kw(*self._JOIN_KW):
            left = self._parse_join(left)
        return left

    def _parse_from_item(self) -> Any:
        if self.cur.kind == "lparen":
            self._eat()
            if self._is_kw("SELECT", "WITH"):
                sub = self._parse_statement()
                self._expect_kind("rparen")
                alias = self._parse_optional_alias()
                return SubqueryRef(plan=sub, alias=alias or "_subquery")
            if self._is_kw("VALUES"):
                # (VALUES (...), (...)) [AS] alias [(col1, col2, ...)]
                return self._parse_values_from_item(in_parens=True)
            inner = self._parse_from_clause()
            self._expect_kind("rparen")
            return inner
        # Bare VALUES at FROM start (no parens, less common but accepted)
        if self._is_kw("VALUES"):
            return self._parse_values_from_item(in_parens=False)
        # String literal as path/URL source: FROM '/path/to/file.parquet'
        if self.cur.kind == "string":
            path = self._eat().text
            alias = self._parse_optional_alias()
            return TableRef(name=path, alias=alias)
        return self._parse_table_ref()

    def _parse_values_from_item(self, in_parens: bool) -> Any:
        """Parse ``VALUES (...), (...) AS alias [(col_aliases)]`` as a from-item.

        Returns a ValuesRef carrying the row data plus optional column
        aliases; the executor materialises it as an in-memory table.
        """
        from .ops import ValuesRef
        self._expect_kw("VALUES")
        values = self._parse_values_list()
        if in_parens:
            self._expect_kind("rparen")
        self._accept_kw("AS")
        alias = None
        col_aliases: list[str] = []
        if self.cur.kind in ("ident", "keyword") and not self._is_kw(*self._CLAUSE_KW):
            alias = self._eat().text
            if self.cur.kind == "lparen":
                self._eat()
                while self.cur.kind != "rparen":
                    col_aliases.append(self._ident_or_kw().text)
                    if self.cur.kind == "comma":
                        self._eat()
                self._expect_kind("rparen")
        return ValuesRef(values=values, alias=alias or "_values",
                         columns=col_aliases or None)

    def _parse_table_ref(self) -> TableRef:
        parts: list[str] = [self._ident_or_kw().text]
        while self.cur.kind == "dot" and self._peek(1).kind in ("ident", "keyword"):
            self._eat()  # dot
            parts.append(self._ident_or_kw().text)
        alias = self._parse_optional_alias()
        if len(parts) == 1:
            return TableRef(name=parts[0], alias=alias)
        elif len(parts) == 2:
            return TableRef(name=parts[1], schema=parts[0], alias=alias)
        else:
            return TableRef(name=parts[-1], schema=parts[-2],
                            catalog=".".join(parts[:-2]), alias=alias)

    def _parse_join(self, left: Any) -> JoinClause:
        jt = self._parse_join_type()
        self._expect_kw("JOIN")
        right = self._parse_from_item()
        on_pred = None
        if self._accept_kw("ON"):
            on_pred = self._parse_expr()
        elif self._accept_kw("USING"):
            self._expect_kind("lparen")
            cols: list[str] = []
            while self.cur.kind != "rparen":
                cols.append(self._ident_or_kw().text)
                if self.cur.kind == "comma":
                    self._eat()
            self._expect_kind("rparen")
            # Convert USING(a, b) into ON left.a = right.a AND left.b = right.b
            conditions = []
            for c in cols:
                conditions.append(Comparison(Column(name=c), CompareOp.EQ, Column(name=c)))
            if len(conditions) == 1:
                on_pred = conditions[0]
            else:
                on_pred = Logical(LogicalOp.AND, tuple(conditions))
        return JoinClause(left=left, right=right, join_type=jt, on=on_pred)

    def _parse_join_type(self) -> JoinType:
        if self._accept_kw("INNER"):
            return JoinType.INNER
        if self._accept_kw("LEFT"):
            self._accept_kw("OUTER")
            return JoinType.LEFT_OUTER
        if self._accept_kw("RIGHT"):
            self._accept_kw("OUTER")
            return JoinType.RIGHT_OUTER
        if self._accept_kw("FULL"):
            self._accept_kw("OUTER")
            return JoinType.FULL_OUTER
        if self._accept_kw("CROSS"):
            return JoinType.CROSS
        return JoinType.INNER

    # ---- LATERAL VIEW -------------------------------------------------

    def _parse_lateral_view(self) -> LateralViewItem:
        self._expect_kw("LATERAL")
        self._expect_kw("VIEW")
        # Optional OUTER
        self._accept_kw("OUTER")
        func = self._parse_function_call()
        table_alias = self._ident_or_kw().text
        col_aliases: list[str] = []
        if self._accept_kw("AS"):
            while True:
                col_aliases.append(self._ident_or_kw().text)
                if self.cur.kind == "comma":
                    self._eat()
                    continue
                break
        return LateralViewItem(
            function=func, table_alias=table_alias,
            column_aliases=col_aliases,
        )

    # ---- ORDER BY -----------------------------------------------------

    def _parse_order_by_list(self) -> list[Any]:
        items = [self._parse_sort_order()]
        while self.cur.kind == "comma":
            self._eat(); items.append(self._parse_sort_order())
        return items

    def _parse_sort_order(self) -> Any:
        expr = self._parse_expr()
        ascending = True
        if self._accept_kw("ASC"):
            ascending = True
        elif self._accept_kw("DESC"):
            ascending = False
        nulls_first = None
        if self._accept_kw("NULLS"):
            if self._accept_kw("FIRST"):
                nulls_first = True
            elif self._accept_kw("LAST"):
                nulls_first = False
            else:
                raise self._error("expected FIRST or LAST after NULLS")
        return SortOrder(expr=expr, ascending=ascending, nulls_first=nulls_first)

    # ---- INSERT -------------------------------------------------------

    def _parse_insert(self) -> InsertNode:
        self._expect_kw("INSERT")
        self._accept_kw("INTO")
        target = self._parse_table_ref()
        columns = None
        if self.cur.kind == "lparen" and not self._is_kw("SELECT"):
            self._eat()
            columns = []
            while self.cur.kind != "rparen":
                columns.append(self._ident_or_kw().text)
                if self.cur.kind == "comma":
                    self._eat()
            self._expect_kind("rparen")
        if self._is_kw("VALUES"):
            self._eat()
            values = self._parse_values_list()
            return InsertNode(target=target, columns=columns, values=values)
        source = self._parse_statement()
        return InsertNode(target=target, columns=columns, source=source)

    def _parse_values_list(self) -> list[list[Any]]:
        rows: list[list[Any]] = []
        while True:
            self._expect_kind("lparen")
            row: list[Any] = []
            while self.cur.kind != "rparen":
                row.append(self._parse_expr())
                if self.cur.kind == "comma":
                    self._eat()
            self._expect_kind("rparen")
            rows.append(row)
            if self.cur.kind == "comma":
                self._eat()
                continue
            break
        return rows

    # ---- MERGE --------------------------------------------------------

    def _parse_merge(self) -> MergeNode:
        self._expect_kw("MERGE")
        self._accept_kw("INTO")
        target = self._parse_table_ref()
        self._expect_kw("USING")
        if self.cur.kind == "lparen":
            self._eat()
            source = self._parse_statement()
            self._expect_kind("rparen")
            alias = self._parse_optional_alias()
            source = SubqueryRef(plan=source, alias=alias or "_source")
        else:
            source = self._parse_table_ref()
        self._expect_kw("ON")
        on = self._parse_expr()
        when_matched = []
        when_not_matched = []
        while self._accept_kw("WHEN"):
            negated = self._accept_kw("NOT") is not None
            self._expect_kw("MATCHED")
            cond = None
            if self._accept_kw("AND"):
                cond = self._parse_expr()
            self._expect_kw("THEN")
            action = self._parse_merge_action()
            entry = {"condition": cond, "action": action}
            if negated:
                when_not_matched.append(entry)
            else:
                when_matched.append(entry)
        return MergeNode(
            target=target, source=source, on=on,
            when_matched=when_matched or None,
            when_not_matched=when_not_matched or None,
        )

    def _parse_merge_action(self) -> dict:
        if self._accept_kw("UPDATE"):
            self._expect_kw("SET")
            assignments = {}
            while True:
                col_expr = self._parse_column()
                col_name = col_expr.name if isinstance(col_expr, Column) else str(col_expr)
                self._accept_op("=")
                val = self._parse_expr()
                assignments[col_name] = val
                if self.cur.kind == "comma":
                    self._eat()
                    continue
                break
            return {"type": "UPDATE", "set": assignments}
        if self._accept_kw("DELETE"):
            return {"type": "DELETE"}
        if self._accept_kw("INSERT"):
            columns = None
            if self.cur.kind == "lparen":
                self._eat()
                columns = []
                while self.cur.kind != "rparen":
                    columns.append(self._ident_or_kw().text)
                    if self.cur.kind == "comma":
                        self._eat()
                self._expect_kind("rparen")
            self._expect_kw("VALUES")
            self._expect_kind("lparen")
            values = []
            while self.cur.kind != "rparen":
                values.append(self._parse_expr())
                if self.cur.kind == "comma":
                    self._eat()
            self._expect_kind("rparen")
            return {"type": "INSERT", "columns": columns, "values": values}
        raise self._error("expected UPDATE, DELETE, or INSERT")

    # ==================================================================
    # Expression parsing (full SQL expressions, not just predicates)
    # ==================================================================

    def _parse_expr(self) -> Expression:
        return self._parse_or()

    def _parse_expr_list(self) -> list[Expression]:
        items = [self._parse_or()]
        while self.cur.kind == "comma":
            self._eat(); items.append(self._parse_or())
        return items

    def _try_parse_lambda(self) -> "Expression | None":
        """Detect ``param -> body`` or ``(p1, p2, ...) -> body`` lambdas.

        Returns the Lambda expression on match, or None if the current
        position isn't a lambda (caller falls through to normal parsing).
        Used by higher-order functions like TRANSFORM, FILTER, AGGREGATE.
        """
        from yggdrasil.saga.expr.nodes import Lambda
        saved = self.pos
        params: list[str] = []
        if self.cur.kind in ("ident", "keyword"):
            # Single-param: x -> body
            nxt, nxt2 = self._peek(1), self._peek(2)
            if (nxt.kind == "op" and nxt.text == "-"
                    and nxt2.kind == "op" and nxt2.text == ">"):
                params.append(self._eat().text)
                self._eat(); self._eat()  # - >
                body = self._parse_or()
                return Lambda(tuple(params), body)
        elif self.cur.kind == "lparen":
            # Multi-param: (p1, p2) -> body
            self._eat()
            if self.cur.kind not in ("ident", "keyword"):
                self.pos = saved
                return None
            while self.cur.kind in ("ident", "keyword"):
                params.append(self._eat().text)
                if self.cur.kind == "comma":
                    self._eat()
                    continue
                break
            if (self.cur.kind != "rparen"
                    or self._peek(1).kind != "op" or self._peek(1).text != "-"
                    or self._peek(2).kind != "op" or self._peek(2).text != ">"):
                self.pos = saved
                return None
            self._eat()  # )
            self._eat(); self._eat()  # - >
            body = self._parse_or()
            return Lambda(tuple(params), body)
        return None

    def _parse_or(self) -> Expression:
        # Lambda detection — has to happen before arithmetic so `x -> ...`
        # isn't parsed as `x - (>...)`.
        lam = self._try_parse_lambda()
        if lam is not None:
            return lam
        left = self._parse_and()
        operands: list[Expression] = []
        while self._accept_kw("OR") is not None:
            operands.append(self._parse_and())
        if not operands:
            return left
        return Logical(LogicalOp.OR, (left, *operands))

    def _parse_and(self) -> Expression:
        left = self._parse_not()
        operands: list[Expression] = []
        while self._accept_kw("AND") is not None:
            operands.append(self._parse_not())
        if not operands:
            return left
        return Logical(LogicalOp.AND, (left, *operands))

    def _parse_not(self) -> Expression:
        if self._accept_kw("NOT") is not None:
            return Not(self._parse_not())
        return self._parse_predicate()

    def _parse_predicate(self) -> Expression:
        left = self._parse_add()
        while True:
            negated = False
            if self.cur.kind == "keyword" and self.cur.upper == "NOT":
                nxt = self._peek(1)
                if nxt.kind == "keyword" and nxt.upper in (
                    "BETWEEN", "IN", "LIKE", "ILIKE",
                ):
                    self._eat()
                    negated = True
                else:
                    break
            t = self.cur
            if t.kind == "keyword":
                if t.upper == "BETWEEN":
                    self._eat()
                    low = self._parse_add()
                    self._expect_kw("AND")
                    high = self._parse_add()
                    left = Between(left, low, high, negated=negated)
                    continue
                if t.upper == "IN":
                    self._eat()
                    left = self._parse_in_tail(left, negated)
                    continue
                if t.upper in ("LIKE", "ILIKE"):
                    ci = t.upper == "ILIKE"
                    self._eat()
                    pattern = self._parse_primary()
                    if isinstance(pattern, Literal) and isinstance(pattern.value, str):
                        left = Like(target=left, pattern=pattern.value,
                                    case_insensitive=ci, negated=negated)
                    else:
                        raise self._error("LIKE pattern must be a string literal")
                    continue
                if t.upper == "IS" and not negated:
                    self._eat()
                    neg = self._accept_kw("NOT") is not None
                    self._expect_kw("NULL")
                    left = IsNull(target=left, negated=neg)
                    continue
            if t.kind == "op" and t.text in _CMP_OPS:
                if negated:
                    raise self._error("unexpected NOT before comparison")
                op = self._eat().text
                right = self._parse_add()
                left = Comparison(left, _CMP_OPS[op], right)
                continue
            if negated:
                raise self._error("unexpected NOT")
            break
        return left

    def _parse_in_tail(self, target: Expression, negated: bool) -> Expression:
        self._expect_kind("lparen")
        if self._is_kw("SELECT", "WITH"):
            # Subquery IN — not fully supported, store as FunctionCall placeholder
            sub = self._parse_statement()
            self._expect_kind("rparen")
            return FunctionCall(name="IN_SUBQUERY", args=(target, Literal(value=str(sub))))
        values: list[Any] = []
        has_null = False
        if self.cur.kind != "rparen":
            while True:
                v = self._parse_primary()
                if not isinstance(v, Literal):
                    raise self._error("IN with non-literal values not yet supported")
                if v.value is None:
                    has_null = True
                else:
                    values.append(v.value)
                if self.cur.kind == "comma":
                    self._eat()
                    continue
                break
        self._expect_kind("rparen")
        return InList(target=target, values=tuple(values),
                      negated=negated, includes_null=has_null)

    def _parse_add(self) -> Expression:
        left = self._parse_mul()
        while self.cur.kind == "op" and self.cur.text in _ARITH_ADD:
            op_text = self._eat().text
            right = self._parse_mul()
            left = Arithmetic(ArithmeticOp(op_text), left, right)
        return left

    def _parse_mul(self) -> Expression:
        left = self._parse_unary()
        while self.cur.kind == "op" and self.cur.text in _ARITH_MUL:
            op_text = self._eat().text
            right = self._parse_unary()
            left = Arithmetic(ArithmeticOp(op_text), left, right)
        return left

    def _parse_unary(self) -> Expression:
        if self.cur.kind == "op" and self.cur.text in ("+", "-"):
            sign = self._eat().text
            inner = self._parse_unary()
            if sign == "+":
                return inner
            from decimal import Decimal
            if isinstance(inner, Literal) and isinstance(
                inner.value, (int, float, Decimal),
            ):
                return Literal(value=-inner.value, dtype=inner.dtype)
            return Arithmetic(ArithmeticOp.SUB, Literal(value=0), inner)
        return self._parse_postfix()

    def _parse_postfix(self) -> Expression:
        expr = self._parse_primary()
        # Subscript: expr[index]
        while self.cur.kind == "lbracket":
            self._eat()
            idx = self._parse_expr()
            self._expect_kind("rbracket")
            expr = Subscript(expr=expr, index=idx)
        # Dot access: expr.field
        while self.cur.kind == "dot" and self._peek(1).kind in ("ident", "keyword"):
            if self._peek(1).kind == "op" and self._peek(1).text == "*":
                break  # qualifier.* handled in select list
            self._eat()
            field_name = self._ident_or_kw().text
            # Check if next is ( → method-like function call
            if self.cur.kind == "lparen":
                expr = self._parse_function_call_named(field_name, qualifier=expr)
            else:
                expr = Subscript(expr=expr, index=Literal(value=field_name))
        return expr

    def _parse_primary(self) -> Expression:
        t = self.cur
        if t.kind == "lparen":
            self._eat()
            expr = self._parse_or()
            self._expect_kind("rparen")
            return expr
        if t.kind == "string":
            self._eat()
            return Literal(value=t.text)
        if t.kind == "number":
            self._eat()
            return Literal(value=_coerce_number(t.text))
        if t.kind == "keyword":
            if t.upper == "NULL":
                self._eat()
                return Literal(value=None)
            if t.upper == "TRUE":
                self._eat()
                return Literal(value=True)
            if t.upper == "FALSE":
                self._eat()
                return Literal(value=False)
            if t.upper == "CAST":
                return self._parse_cast()
            if t.upper == "CASE":
                return self._parse_case_when()
            if t.upper == "EXISTS":
                return self._parse_exists()
            if t.upper in _CAST_TEMPORAL_PARSERS:
                return self._parse_typed_temporal(t)
            if t.upper in ("ARRAY", "MAP", "STRUCT"):
                if self._peek(1).kind == "lparen":
                    return self._parse_function_call()
                self._eat()
                return Column(name=t.text)
            if t.upper == "INTERVAL":
                return self._parse_interval()
            if t.upper == "EXTRACT":
                return self._parse_extract()
            if t.upper in ("CURRENT_DATE", "CURRENT_TIMESTAMP", "NOW"):
                self._eat()
                if self.cur.kind == "lparen":
                    self._eat()
                    self._expect_kind("rparen")
                return FunctionCall(name=t.upper, args=())
            # Generic function call check
            if self._peek(1).kind == "lparen":
                return self._parse_function_call()
            # Not a known keyword in expression context → treat as column
            return self._parse_column()
        if t.kind == "ident":
            if self._peek(1).kind == "lparen":
                return self._parse_function_call()
            return self._parse_column()
        if t.kind == "op" and t.text == "*":
            self._eat()
            return Star()
        raise self._error("unexpected token")

    def _parse_column(self) -> Expression:
        first = self._ident_or_kw()
        qualifier = None
        name = first.text
        if self.cur.kind == "dot" and self._peek(1).kind in ("ident", "keyword"):
            self._eat()
            second = self._ident_or_kw()
            qualifier, name = name, second.text
        return Column(name=name, qualifier=qualifier)

    def _parse_function_call(self, name: str | None = None) -> Expression:
        if name is None:
            name = self._ident_or_kw().text
        return self._parse_function_call_named(name)

    def _parse_function_call_named(
        self, name: str, qualifier: Expression | None = None,
    ) -> Expression:
        upper_name = name.upper()
        self._expect_kind("lparen")

        # COUNT(*)
        if self.cur.kind == "op" and self.cur.text == "*":
            self._eat()
            self._expect_kind("rparen")
            fc = FunctionCall(name=upper_name, args=(Star(),))
            return (WindowFunction(function=fc, window=self._parse_window_spec()) if self._accept_kw("OVER") else fc)

        # DISTINCT
        distinct = False
        if self._accept_kw("DISTINCT"):
            distinct = True

        args: list[Expression] = []
        if self.cur.kind != "rparen":
            while True:
                args.append(self._parse_or())
                if self.cur.kind == "comma":
                    self._eat()
                    continue
                break
        self._expect_kind("rparen")

        fc = FunctionCall(name=upper_name, args=tuple(args), distinct=distinct)
        return (WindowFunction(function=fc, window=self._parse_window_spec()) if self._accept_kw("OVER") else fc)

    def _parse_window_spec(self) -> Any:
        self._expect_kind("lparen")
        partition_by: list[Expression] = []
        order_by: list[Any] = []
        frame_start = None
        frame_end = None

        if self._is_kw("PARTITION"):
            self._eat()
            self._expect_kw("BY")
            partition_by = self._parse_expr_list()

        if self._is_kw("ORDER"):
            self._eat()
            self._expect_kw("BY")
            order_by = self._parse_order_by_list()

        if self._is_kw("ROWS", "RANGE"):
            frame_type = self._eat().upper
            if self._accept_kw("BETWEEN"):
                frame_start = self._parse_frame_bound()
                self._expect_kw("AND")
                frame_end = self._parse_frame_bound()
            else:
                frame_start = self._parse_frame_bound()

        self._expect_kind("rparen")
        return WindowSpec(
            partition_by=tuple(partition_by),
            order_by=tuple(order_by),
            frame_start=frame_start,
            frame_end=frame_end,
        )

    def _parse_frame_bound(self) -> str:
        if self._accept_kw("UNBOUNDED"):
            if self._accept_kw("PRECEDING"):
                return "UNBOUNDED PRECEDING"
            self._expect_kw("FOLLOWING")
            return "UNBOUNDED FOLLOWING"
        if self._accept_kw("CURRENT"):
            self._expect_kw("ROW")
            return "CURRENT ROW"
        # N PRECEDING / N FOLLOWING
        n = self._expect_kind("number").text
        if self._accept_kw("PRECEDING"):
            return f"{n} PRECEDING"
        self._expect_kw("FOLLOWING")
        return f"{n} FOLLOWING"

    def _parse_cast(self) -> Expression:
        self._expect_kw("CAST")
        self._expect_kind("lparen")
        inner = self._parse_or()
        self._expect_kw("AS")
        type_name = self._parse_type_head()
        self._expect_kind("rparen")
        return self._fold_cast(inner, type_name)

    def _parse_typed_temporal(self, head: _Token) -> Expression:
        nxt = self._peek(1)
        if nxt.kind == "string":
            self._eat()
            lit_t = self._eat()
            return self._fold_cast(Literal(value=lit_t.text), head.upper)
        if nxt.kind == "lparen":
            self._eat()
            self._eat()
            inner = self._parse_or()
            self._expect_kind("rparen")
            return self._fold_cast(inner, head.upper)
        # Fall back to treating it as a column name (e.g., `date` as column)
        return self._parse_column()

    def _parse_type_head(self) -> str:
        t = self.cur
        if t.kind not in ("ident", "keyword"):
            raise self._error("expected a type name after AS")
        self._eat()
        name = t.upper
        if (name == "DOUBLE" and self.cur.kind == "ident"
                and self.cur.upper == "PRECISION"):
            self._eat()
            name = "DOUBLE PRECISION"
        if self.cur.kind == "lparen":
            self._eat()
            depth = 1
            while depth > 0 and self.cur.kind != "eof":
                if self.cur.kind == "lparen":
                    depth += 1
                elif self.cur.kind == "rparen":
                    depth -= 1
                    if depth == 0:
                        break
                self._eat()
            self._expect_kind("rparen")
        return name

    def _fold_cast(self, inner: Expression, type_name: str) -> Expression:
        from yggdrasil.saga.expr.backends.sql import _fold_cast
        return _fold_cast(inner, type_name)

    def _parse_case_when(self) -> Expression:
        self._expect_kw("CASE")
        operand = None
        if not self._is_kw("WHEN"):
            operand = self._parse_expr()
        branches: list[tuple[Expression, Expression]] = []
        while self._accept_kw("WHEN"):
            condition = self._parse_expr()
            self._expect_kw("THEN")
            result = self._parse_expr()
            branches.append((condition, result))
        else_expr = None
        if self._accept_kw("ELSE"):
            else_expr = self._parse_expr()
        self._expect_kw("END")
        return CaseWhen(
            branches=tuple(branches),
            else_expr=else_expr,
            operand=operand,
        )

    def _parse_exists(self) -> Expression:
        self._expect_kw("EXISTS")
        self._expect_kind("lparen")
        # EXISTS(SELECT ...) → subquery existence check
        # EXISTS(arr, x -> x > 0) → Databricks higher-order function
        if self._is_kw("SELECT", "WITH"):
            sub = self._parse_statement()
            self._expect_kind("rparen")
            return FunctionCall(name="EXISTS", args=(Literal(value=str(sub)),))
        # Higher-order form: parse as a regular function call body
        args: list[Expression] = []
        if self.cur.kind != "rparen":
            while True:
                args.append(self._parse_or())
                if self.cur.kind == "comma":
                    self._eat()
                    continue
                break
        self._expect_kind("rparen")
        return FunctionCall(name="EXISTS", args=tuple(args))

    # ---- INTERVAL literal ------------------------------------------------

    _INTERVAL_UNITS = frozenset({
        "DAY", "DAYS", "HOUR", "HOURS", "MINUTE", "MINUTES",
        "SECOND", "SECONDS", "MONTH", "MONTHS", "YEAR", "YEARS",
        "WEEK", "WEEKS",
    })

    def _parse_interval(self) -> Expression:
        """Parse ``INTERVAL 'value' unit`` or ``INTERVAL N unit`` → FunctionCall("INTERVAL", ...)."""
        self._expect_kw("INTERVAL")
        # Accept either string literal ('value') or numeric literal (N)
        if self.cur.kind == "string":
            value_text = self._eat().text
        elif self.cur.kind == "number":
            value_text = _coerce_number(self._eat().text)
        else:
            raise self._error("expected string or number after INTERVAL")
        # Unit keyword — might be tokenized as ident or keyword
        t = self.cur
        if t.kind in ("ident", "keyword") and t.upper in self._INTERVAL_UNITS:
            unit = self._eat().upper
            unit = self._normalize_interval_unit(unit)
        else:
            raise self._error(
                f"expected interval unit (DAY, HOUR, MINUTE, SECOND, "
                f"MONTH, YEAR, WEEK) after INTERVAL literal"
            )
        return FunctionCall(name="INTERVAL",
            args=(Literal(value=value_text), Literal(value=unit)))

    @staticmethod
    def _normalize_interval_unit(unit: str) -> str:
        _PLURAL_MAP = {"DAYS": "DAY", "HOURS": "HOUR", "MINUTES": "MINUTE",
                        "SECONDS": "SECOND", "MONTHS": "MONTH", "YEARS": "YEAR",
                        "WEEKS": "WEEK"}
        return _PLURAL_MAP.get(unit, unit)

    # ---- EXTRACT(field FROM expr) ----------------------------------------

    _EXTRACT_FIELDS = frozenset({
        "YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND",
        "DOW", "DOY", "EPOCH", "QUARTER", "WEEK",
    })

    def _parse_extract(self) -> Expression:
        """Parse ``EXTRACT(field FROM expr)`` → FunctionCall("EXTRACT", ...)."""
        self._expect_kw("EXTRACT")
        self._expect_kind("lparen")
        # The field name — could be tokenized as ident or keyword
        t = self.cur
        if t.kind in ("ident", "keyword") and t.upper in self._EXTRACT_FIELDS:
            field = self._eat().upper
        else:
            raise self._error(
                f"expected EXTRACT field (YEAR, MONTH, DAY, HOUR, MINUTE, "
                f"SECOND, DOW, DOY, EPOCH, QUARTER, WEEK)"
            )
        self._expect_kw("FROM")
        source = self._parse_expr()
        self._expect_kind("rparen")
        return FunctionCall(name="EXTRACT", args=(Literal(value=field), source))



# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_sql(
    sql: str,
    *,
    dialect: "Dialect | str | None" = None,
    parser_class: type[SQLQueryParser] | None = None,
    default: "Any" = ...,
) -> "PlanNode | Any":
    """Parse a SQL query string into a :class:`PlanNode` tree.

    When *default* is ``...`` (the sentinel), parsing errors raise
    :class:`ValueError`. When *default* is anything else, parsing
    errors return *default* instead.
    """
    resolved = _resolve_dialect(dialect)
    cls = parser_class
    if cls is None:
        cls = _get_parser_class(resolved)
    try:
        return cls(sql, resolved).parse()
    except (ValueError, NotImplementedError) as exc:
        if default is not ...:
            return default
        raise


def _get_parser_class(dialect: Dialect) -> type[SQLQueryParser]:
    if dialect == Dialect.DATABRICKS:
        from .databricks import DatabricksSQLParser
        return DatabricksSQLParser
    return _DIALECT_PARSERS.get(dialect, SQLQueryParser)


# Dialect → parser class registry; extended by dialect-specific parsers
_DIALECT_PARSERS: dict[Dialect, type[SQLQueryParser]] = {}
