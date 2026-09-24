"""
Relógio de referência da memória: datas das passagens e âncoras de pergunta.

Cada registro conversacional carrega a data da própria sessão (``Session
date:`` e, com ``--temporal-annotations``, ``date=`` por fala e
``reference_time=`` nas normalizações). Essa data é o relógio de referência
τ(u) de cada fala. O relógio da memória inteira é o intervalo [t₀, t_now]:
t₀ é a primeira sessão e t_now a última, o "presente" em que as perguntas são
feitas. Nada aqui lê perguntas de avaliação, respostas ou categorias.

Tudo é determinístico e sem LLM. Em corpora sem datas (HotpotQA, por exemplo)
as funções devolvem listas vazias, e a lente temporal vira identidade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

_MONTHS = {name: index for index, name in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"), 1)}
_MONTHS.update({name[:3]: index for name, index in list(_MONTHS.items())})
_MONTHS["sept"] = 9
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))

_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\.?,?\s+(\d{{4}})\b", re.I)
_MDY = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MY = re.compile(rf"\b({_MONTH_RE})\.?,?\s+(?:of\s+)?(\d{{4}})\b", re.I)
_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
_SEASONS = {"spring": (3, 5), "summer": (6, 8), "fall": (9, 11), "autumn": (9, 11),
            "winter": (12, 2)}


def _month(name: str) -> int:
    key = (name or "").lower().rstrip(".")
    return _MONTHS.get(key, _MONTHS.get(key[:3], 0))


def _safe(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_dates(text: str) -> list[date]:
    """Todas as datas completas (dia, mês, ano) de um texto, em ordem."""
    found: list[tuple[int, date]] = []
    for match in _DMY.finditer(text or ""):
        value = _safe(int(match.group(3)), _month(match.group(2)), int(match.group(1)))
        if value:
            found.append((match.start(), value))
    for match in _MDY.finditer(text or ""):
        value = _safe(int(match.group(3)), _month(match.group(1)), int(match.group(2)))
        if value and all(value != other for _pos, other in found):
            found.append((match.start(), value))
    for match in _ISO.finditer(text or ""):
        value = _safe(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if value:
            found.append((match.start(), value))
    return [value for _pos, value in sorted(found)]


def parse_date(text: str) -> date | None:
    values = parse_dates(text)
    return values[0] if values else None


@dataclass(frozen=True)
class Interval:
    """Intervalo fechado de datas e sua granularidade em dias."""

    start: date
    end: date
    text: str = ""

    @property
    def width_days(self) -> int:
        return max(1, (self.end - self.start).days + 1)

    def distance_days(self, value: date) -> int:
        if value < self.start:
            return (self.start - value).days
        if value > self.end:
            return (value - self.end).days
        return 0

    def to_dict(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(),
                "text": self.text}


def _month_interval(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    end = (date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1))
    return start, end


def parse_anchor(text: str, now: date | None = None) -> Interval | None:
    """Interpreta a âncora temporal copiada da pergunta pelo planejador.

    Cobre as formas que aparecem em memória conversacional: data completa,
    "early/mid/late <mês> <ano>", "(first|last) week of <mês> <ano>",
    estação + ano, mês + ano, ano, e "last/this year" relativo a t_now.
    Uma âncora não interpretável devolve None e a lente fica desligada.
    """
    raw = (text or "").strip()
    low = raw.lower()
    if not low:
        return None
    exact = parse_date(raw)
    week = re.search(rf"\b(first|second|third|last)\s+week\s+of\s+({_MONTH_RE})\.?,?\s+(\d{{4}})",
                     low)
    if week:
        start, end = _month_interval(int(week.group(3)),
                                     _month(week.group(2)))
        offsets = {"first": 0, "second": 7, "third": 14}
        if week.group(1) == "last":
            return Interval(end - timedelta(days=6), end, raw)
        first = start + timedelta(days=offsets[week.group(1)])
        return Interval(first, first + timedelta(days=6), raw)
    if exact:
        if re.search(r"\bbefore\b", low):
            # "the week before 9 June 2023", "the Friday before 15 July 2023"
            return Interval(exact - timedelta(days=7), exact - timedelta(days=1), raw)
        if re.search(r"\bafter\b", low):
            return Interval(exact + timedelta(days=1), exact + timedelta(days=7), raw)
        if re.search(r"\b(week|weekend)\b", low):
            return Interval(exact, exact + timedelta(days=6), raw)
        return Interval(exact, exact, raw)
    part = re.search(rf"\b(early|mid|middle of|late|end of|beginning of|start of)\s+({_MONTH_RE})\.?,?\s+(\d{{4}})",
                     low)
    if part:
        start, end = _month_interval(int(part.group(3)),
                                     _month(part.group(2)))
        if part.group(1) in {"early", "beginning of", "start of"}:
            return Interval(start, start + timedelta(days=9), raw)
        if part.group(1) in {"late", "end of"}:
            return Interval(end - timedelta(days=9), end, raw)
        return Interval(start + timedelta(days=10), start + timedelta(days=19), raw)
    season = re.search(r"\b(spring|summer|fall|autumn|winter)\b(?:\s+of)?\s+(\d{4})", low)
    if season:
        first, last = _SEASONS[season.group(1)]
        year = int(season.group(2))
        start = date(year, first, 1)
        end_year = year + 1 if last < first else year
        return Interval(start, _month_interval(end_year, last)[1], raw)
    month_year = _MY.search(low)
    if month_year:
        start, end = _month_interval(int(month_year.group(2)), _month(month_year.group(1)))
        return Interval(start, end, raw)
    year = _YEAR.search(low)
    if year:
        value = int(year.group(1))
        return Interval(date(value, 1, 1), date(value, 12, 31), raw)
    if now is not None:
        if re.search(r"\blast year\b", low):
            return Interval(date(now.year - 1, 1, 1), date(now.year - 1, 12, 31), raw)
        if re.search(r"\bthis year\b", low):
            return Interval(date(now.year, 1, 1), now, raw)
        if re.search(r"\blast month\b", low):
            end = now.replace(day=1) - timedelta(days=1)
            return Interval(end.replace(day=1), end, raw)
    return None


_SESSION_LINE = re.compile(r"^Session date:\s*(.+)$", re.M)
_TURN_DATE = re.compile(r"\bdate=([^\]]+)\]")
_REFERENCE = re.compile(r"(?:reference_time|start|end)=(\d{4}-\d{2}-\d{2})")


def passage_dates(text: str, session_time: str = "") -> list[date]:
    """Datas de referência de uma passagem: sessões, falas e normalizações.

    Inclui os extremos ``start=``/``end=`` das normalizações determinísticas
    ("last week" → a semana anterior à sessão): são tempos de EVENTO, que é o
    que uma âncora como "a última semana de agosto" costuma designar.
    """
    values: list[date] = []
    for source in ([session_time] + _SESSION_LINE.findall(text or "") +
                   _TURN_DATE.findall(text or "") + _REFERENCE.findall(text or "")):
        value = parse_date(source)
        if value and value not in values:
            values.append(value)
    return sorted(values)


@dataclass(frozen=True)
class ReferenceClock:
    """t₀ (primeiro registro) e t_now (último registro, o presente da memória)."""

    start: date | None
    now: date | None

    @classmethod
    def from_dates(cls, dates: Iterable[date]) -> "ReferenceClock":
        values = sorted(dates)
        return cls(values[0] if values else None, values[-1] if values else None)

    @property
    def available(self) -> bool:
        return self.start is not None and self.now is not None

    def to_dict(self) -> dict[str, str]:
        return {"t0": self.start.isoformat() if self.start else "",
                "t_now": self.now.isoformat() if self.now else ""}


# ---------------------------------------------------------------------------
# Resolução de expressões de tempo de um FATO (etapa Registrar do desenho v3)
# ---------------------------------------------------------------------------
# A extração de fatos grava o tempo como o texto o diz: a data da sessão, uma
# data explícita ou uma expressão relativa ("last week", "two days ago"). Para
# que todo fato tenha um tempo comparável, a expressão é resolvida contra a
# data da sessão em que foi DITA. Tudo aqui é determinístico e sem LLM; uma
# expressão que não se reconhece devolve None, e quem chama usa a data da
# sessão (o fato foi relatado naquele dia).

_NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
                 "twelve": 12, "couple of": 2, "a couple of": 2, "few": 3, "a few": 3,
                 "several": 4}
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_DM = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\b", re.I)
_MD = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.I)


def _count(token: str) -> int | None:
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


def _week_of(day: date) -> tuple[date, date]:
    monday = day - timedelta(days=day.weekday())
    return monday, monday + timedelta(days=6)


def _shift_months(day: date, months: int) -> tuple[date, date]:
    index = day.year * 12 + (day.month - 1) + months
    return _month_interval(index // 12, index % 12 + 1)


def resolve_expression(expression: str, reference: date | None) -> Interval | None:
    """Intervalo de dias designado por uma expressão de tempo.

    Nunca levanta exceção: uma expressão fora do calendário ("3000 years ago")
    devolve None, e quem chama usa a data da sessão.
    """
    try:
        return _resolve(expression, reference)
    except (ValueError, OverflowError):
        return None


def _last_weekend(ref: date, weeks_back: int = 1) -> Interval:
    """O fim de semana anterior à fala; num domingo, o da semana passada
    (mesma convenção de locomo._temporal_annotation)."""
    sunday = ref - timedelta(days=(ref.weekday() - 6) % 7 or 7)
    sunday -= timedelta(days=7 * (max(1, weeks_back) - 1))
    return Interval(sunday - timedelta(days=1), sunday)


def _resolve(expression: str, reference: date | None) -> Interval | None:
    """Intervalo de dias designado por uma expressão de tempo.

    ``reference`` é a data da sessão em que a expressão foi dita. Datas
    absolutas não precisam dela; expressões relativas sem referência devolvem
    None. A convenção de semana é segunda a domingo, a mesma das anotações
    temporais do LoCoMo neste repositório ("last week" dito numa terça, 27/6,
    é a semana de 19 a 25/6).
    """
    raw = (expression or "").strip()
    low = re.sub(r"\s+", " ", raw.lower())
    if not low:
        return None
    exact = parse_date(raw)
    if exact and re.search(r"\b(before|after|week|weekend)\b", low):
        # "the week before 27 June 2023": o qualificador muda o intervalo.
        anchored = parse_anchor(raw, now=reference)
        if anchored is not None:
            return anchored
    if exact:
        return Interval(exact, exact, raw)
    if reference is not None:
        # dia e mês sem ano: "12 May", "9:59 pm on 19 June"
        match = _DM.search(low) or _MD.search(low)
        if match:
            groups = match.groups()
            day_text, month_text = ((groups[0], groups[1]) if groups[0].isdigit()
                                    else (groups[1], groups[0]))
            value = _safe(reference.year, _month(month_text), int(day_text))
            if value:
                if value > reference + timedelta(days=31):
                    value = _safe(reference.year - 1, value.month, value.day) or value
                return Interval(value, value, raw)
    anchored = parse_anchor(raw, now=reference)
    if anchored is not None:
        return anchored
    if reference is None:
        return None
    ref = reference
    month = re.fullmatch(rf"(?:(in|during|last|this|next|past|early|late|mid)\s+)?({_MONTH_RE})\.?", low)
    if month:
        # Mês sem ano: a ocorrência mais recente até a fala ("in October");
        # "last June" é a anterior ao mês corrente; "next June", a seguinte.
        qualifier, number = month.group(1) or "", _month(month.group(2))
        year = ref.year if number <= ref.month else ref.year - 1
        if qualifier in {"last", "past"} and number == ref.month:
            year -= 1
        if qualifier == "next":
            year = ref.year if number > ref.month else ref.year + 1
        start, end = _month_interval(year, number)
        if qualifier == "early":
            end = start + timedelta(days=9)
        elif qualifier == "late":
            start = end - timedelta(days=9)
        elif qualifier == "mid":
            start, end = start + timedelta(days=10), start + timedelta(days=19)
        return Interval(start, end, raw)
    if re.search(r"\b(today|tonight|this (morning|afternoon|evening)|right now|just now)\b", low):
        return Interval(ref, ref, raw)
    if re.search(r"\b(yesterday|last night)\b", low):
        day = ref - timedelta(days=1)
        return Interval(day, day, raw)
    if re.search(r"\btomorrow\b", low):
        day = ref + timedelta(days=1)
        return Interval(day, day, raw)
    if re.search(r"\b(the other day|a few days ago|few days ago|recently|lately)\b", low):
        return Interval(ref - timedelta(days=7), ref, raw)
    ago = re.search(r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
                    r"twelve|a couple of|couple of|a few|few|several)\s+(day|week|weekend|month|year)s?"
                    r"\s+(ago|back|earlier)\b", low)
    if ago:
        n = _count(ago.group(1)) or 1
        unit = ago.group(2)
        if unit == "day":
            day = ref - timedelta(days=n)
            return Interval(day, day, raw)
        if unit == "week":
            start, end = _week_of(ref - timedelta(days=7 * n))
            return Interval(start, end, raw)
        if unit == "weekend":
            weekend = _last_weekend(ref, n)
            return Interval(weekend.start, weekend.end, raw)
        if unit == "month":
            start, end = _shift_months(ref, -n)
            return Interval(start, end, raw)
        return Interval(date(ref.year - n, 1, 1), date(ref.year - n, 12, 31), raw)
    later = re.search(r"\b(?:in\s+)?(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
                      r"a couple of|a few)\s+(day|week|month|year)s?\s+(later|from now)\b", low)
    if later:
        n = _count(later.group(1)) or 1
        unit = later.group(2)
        if unit == "day":
            day = ref + timedelta(days=n)
            return Interval(day, day, raw)
        if unit == "week":
            start, end = _week_of(ref + timedelta(days=7 * n))
            return Interval(start, end, raw)
        if unit == "month":
            start, end = _shift_months(ref, n)
            return Interval(start, end, raw)
        return Interval(date(ref.year + n, 1, 1), date(ref.year + n, 12, 31), raw)
    for index, name in enumerate(_WEEKDAYS):
        if re.search(rf"\blast\s+({name}|{name[:3]})\b", low):
            day = ref - timedelta(days=(ref.weekday() - index) % 7 or 7)
            return Interval(day, day, raw)
        if re.search(rf"\bnext\s+({name}|{name[:3]})\b", low):
            day = ref + timedelta(days=(index - ref.weekday()) % 7 or 7)
            return Interval(day, day, raw)
        if re.fullmatch(rf"(on\s+)?(this\s+)?({name}|{name[:3]})", low):
            # Um dia da semana sem qualificador, dito na sessão, é o mais recente.
            day = ref - timedelta(days=(ref.weekday() - index) % 7)
            return Interval(day, day, raw)
    if re.search(r"\blast weekend\b|\bpast weekend\b", low):
        weekend = _last_weekend(ref)
        return Interval(weekend.start, weekend.end, raw)
    if re.search(r"\bthis weekend\b", low):
        saturday = ref + timedelta(days=(5 - ref.weekday()) % 7)
        if ref.weekday() == 6:
            saturday = ref - timedelta(days=1)
        return Interval(saturday, saturday + timedelta(days=1), raw)
    if re.search(r"\bnext weekend\b", low):
        saturday = ref + timedelta(days=(5 - ref.weekday()) % 7 or 7)
        return Interval(saturday, saturday + timedelta(days=1), raw)
    if re.search(r"\blast week\b|\bpast week\b|\bprevious week\b", low):
        start, end = _week_of(ref - timedelta(days=7))
        return Interval(start, end, raw)
    if re.search(r"\bthis week\b", low):
        start, end = _week_of(ref)
        return Interval(start, end, raw)
    if re.search(r"\bnext week\b", low):
        start, end = _week_of(ref + timedelta(days=7))
        return Interval(start, end, raw)
    if re.search(r"\blast month\b|\bpast month\b|\bprevious month\b", low):
        start, end = _shift_months(ref, -1)
        return Interval(start, end, raw)
    if re.search(r"\bthis month\b", low):
        start, end = _shift_months(ref, 0)
        return Interval(start, end, raw)
    if re.search(r"\bnext month\b", low):
        start, end = _shift_months(ref, 1)
        return Interval(start, end, raw)
    season = re.search(r"\b(last|this|next|past)?\s*(spring|summer|fall|autumn|winter)\b", low)
    if season:
        first, last = _SEASONS[season.group(2)]

        def season_of(year: int) -> tuple[date, date]:
            end_year = year + 1 if last < first else year
            return date(year, first, 1), _month_interval(end_year, last)[1]

        qualifier = season.group(1) or "this"
        if qualifier in {"last", "past"}:
            # a ocorrência mais recente que já terminou antes da fala
            year = ref.year if season_of(ref.year)[1] < ref else ref.year - 1
            if season_of(year)[1] >= ref:
                year -= 1
        elif qualifier == "next":
            year = ref.year if season_of(ref.year)[0] > ref else ref.year + 1
        else:
            year = ref.year
            if last < first and ref.month <= last:
                year -= 1   # "this winter" dito em janeiro
        start, end = season_of(year)
        return Interval(start, end, raw)
    if re.search(r"\blast year\b|\bpast year\b|\bprevious year\b", low):
        return Interval(date(ref.year - 1, 1, 1), date(ref.year - 1, 12, 31), raw)
    if re.search(r"\bthis year\b", low):
        return Interval(date(ref.year, 1, 1), date(ref.year, 12, 31), raw)
    if re.search(r"\bnext year\b", low):
        return Interval(date(ref.year + 1, 1, 1), date(ref.year + 1, 12, 31), raw)
    return None


def interval_distance(left: Interval, right: Interval) -> int:
    """Vão em dias entre dois intervalos; zero quando se sobrepõem."""
    if left.end < right.start:
        return (right.start - left.end).days
    if right.end < left.start:
        return (left.start - right.end).days
    return 0


def format_interval(value: Interval | None) -> str:
    """Forma legível e estável ("7 May 2023" ou "19 June 2023 - 25 June 2023")."""
    if value is None:
        return ""
    fmt = lambda day: f"{day.day} {day.strftime('%B %Y')}"
    if value.start == value.end:
        return fmt(value.start)
    return f"{fmt(value.start)} - {fmt(value.end)}"
