"""Auditable HTTP ingestion. No generated matches or silently imputed scores."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import re
import ssl
from urllib.request import Request, urlopen
from urllib.error import URLError
import warnings
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup

UEFA_URL = 'https://www.uefa.com/uefachampionsleague/news/02a8-2174c9e9019d-f909a77bd77a-1000--2026-27-champions-league-all-the-league-phase-fixtures/'
CURRENT_TEAMS = frozenset('AEK Athens|Arsenal|Aston Villa|Atlético de Madrid|Barcelona|Bayern München|Bodø/Glimt|Borussia Dortmund|Club Brugge|Como|Fenerbahçe|Feyenoord|Galatasaray|Inter|LASK|Leipzig|Lens|Lille|Liverpool|Manchester City|Manchester United|Napoli|Paris Saint-Germain|Porto|PSV Eindhoven|Real Betis|Real Madrid|Roma|Sabah|Shakhtar Donetsk|Slavia Praha|Slovan Bratislava|Sporting CP|Stuttgart|Viking|Villarreal'.split('|'))
FIXTURE_COLUMNS = ['match_id', 'date', 'season', 'stage', 'matchday', 'home', 'away', 'home_goals', 'away_goals', 'status', 'source']
HISTORY_COLUMNS = ['date', 'home', 'away', 'home_goals', 'away_goals', 'competition', 'source']
ALIASES = {
    'Man City': 'Manchester City', 'Manchester City FC': 'Manchester City',
    'Man United': 'Manchester United', 'Manchester United FC': 'Manchester United',
    'Bayern Munich': 'Bayern München', 'FC Bayern München': 'Bayern München',
    'Bayern': 'Bayern München', 'Dortmund': 'Borussia Dortmund',
    'RB Leipzig': 'Leipzig', 'RasenBallsport Leipzig': 'Leipzig',
    'VfB Stuttgart': 'Stuttgart', 'VfB Stuttgart 1893': 'Stuttgart',
    'Ath Madrid': 'Atlético de Madrid', 'Atletico Madrid': 'Atlético de Madrid',
    'Atlético Madrid': 'Atlético de Madrid', 'Club Atlético de Madrid': 'Atlético de Madrid',
    'FC Barcelona': 'Barcelona', 'Real Madrid CF': 'Real Madrid',
    'Betis': 'Real Betis', 'Real Betis Balompié': 'Real Betis',
    'Villarreal CF': 'Villarreal', 'Paris SG': 'Paris Saint-Germain',
    'Paris Saint-Germain FC': 'Paris Saint-Germain', 'PSG': 'Paris Saint-Germain',
    'LOSC Lille': 'Lille', 'Lille OSC': 'Lille', 'RC Lens': 'Lens',
    'Arsenal FC': 'Arsenal', 'Aston Villa FC': 'Aston Villa', 'Liverpool FC': 'Liverpool',
    'FC Internazionale Milano': 'Inter', 'Internazionale': 'Inter', 'Inter Milan': 'Inter',
    'SSC Napoli': 'Napoli', 'AS Roma': 'Roma', 'Como 1907': 'Como',
    'FC Porto': 'Porto', 'Sporting': 'Sporting CP', 'Sporting Lisbon': 'Sporting CP',
    'Sporting Clube de Portugal': 'Sporting CP', 'PSV': 'PSV Eindhoven',
    'Feyenoord Rotterdam': 'Feyenoord', 'Club Brugge KV': 'Club Brugge',
    'Fenerbahce': 'Fenerbahçe', 'Fenerbahçe SK': 'Fenerbahçe',
    'Galatasaray SK': 'Galatasaray', 'AEK': 'AEK Athens', 'AEK Athina': 'AEK Athens',
    'AEK Athens FC': 'AEK Athens', 'LASK Linz': 'LASK', 'Viking FK': 'Viking',
    'Bodo/Glimt': 'Bodø/Glimt', 'FK Bodø/Glimt': 'Bodø/Glimt',
    'Slavia Prague': 'Slavia Praha', 'SK Slavia Praha': 'Slavia Praha',
    'ŠK Slovan Bratislava': 'Slovan Bratislava', 'Slovan': 'Slovan Bratislava',
    'Shakhtar': 'Shakhtar Donetsk', 'FC Shakhtar Donetsk': 'Shakhtar Donetsk',
    'Sabah FK': 'Sabah', 'Sabah FC': 'Sabah',
    'Sp Lisbon': 'Sporting CP', 'FK Shakhtar Donetsk': 'Shakhtar Donetsk',
    'SL Benfica': 'Benfica', 'Sp Braga': 'Braga', 'Sporting Braga': 'Braga',
    'SC Braga': 'Braga', 'Vitória SC': 'Guimaraes',
    'Chelsea FC': 'Chelsea', 'Newcastle United FC': 'Newcastle',
    'Tottenham Hotspur FC': 'Tottenham', 'Nottingham Forest FC': "Nott'm Forest",
    'Crystal Palace FC': 'Crystal Palace', 'Brighton & Hove Albion FC': 'Brighton',
    'West Ham United FC': 'West Ham', 'Atalanta BC': 'Atalanta',
    'Bologna FC 1909': 'Bologna', 'AC Milan': 'Milan', 'Juventus FC': 'Juventus',
    'SS Lazio': 'Lazio', 'ACF Fiorentina': 'Fiorentina',
    'Bayer 04 Leverkusen': 'Leverkusen', 'Eintracht Frankfurt': 'Ein Frankfurt',
    'SC Freiburg': 'Freiburg', '1. FC Heidenheim 1846': 'Heidenheim',
    'Athletic Club': 'Ath Bilbao', 'Real Sociedad': 'Sociedad', 'Girona FC': 'Girona',
    'Stade Brestois 29': 'Brest', 'AS Monaco': 'Monaco',
    'Olympique de Marseille': 'Marseille', 'Olympique Lyonnais': 'Lyon',
    'OGC Nice': 'Nice', 'AFC Ajax': 'Ajax', 'FC Twente': 'Twente',
    'PAOK FC': 'PAOK', 'Panathinaikos FC': 'Panathinaikos',
    'Olympiacos FC': 'Olympiakos', 'SK Sturm Graz': 'Sturm Graz',
    'FC Salzburg': 'Salzburg', 'FC Red Bull Salzburg': 'Salzburg',
    'SK Rapid Wien': 'Rapid Vienna', 'FK Austria Wien': 'Austria Vienna',
    'Molde FK': 'Molde', 'SK Brann': 'Brann', 'Rosenborg BK': 'Rosenborg',
    'AEK Athen': 'AEK Athens', 'RB Salzburg': 'Salzburg', 'Rapid Wien': 'Rapid Vienna',
    'Austria Wien': 'Austria Vienna', 'PAOK Saloniki': 'PAOK', 'RC Strasbourg': 'Strasbourg',
    'FC Utrecht': 'Utrecht', 'Vitória Guimarães': 'Guimaraes', 'Beşiktaş': 'Besiktas',
    'Royale Union Saint-Gilloise': 'Union SG', 'Union Saint-Gilloise': 'Union SG',
    'RSC Anderlecht': 'Anderlecht', 'KRC Genk': 'Genk', 'KAA Gent': 'Gent',
    'Sporting Charleroi': 'Charleroi', '1. FSV Mainz 05': 'Mainz',
}


def normalize_team(name: str) -> str:
    name = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m[1], 16)), str(name))
    name = unicodedata.normalize('NFC', ' '.join(name.split()))
    return ALIASES.get(name, name)


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def _system_tls() -> None:
    # No global SSL monkey patch: shared requests contexts can race on Windows.
    import truststore


def _download(url: str, raw_dir: Path, filename: str) -> bytes:
    raw_dir.mkdir(parents=True, exist_ok=True)
    import truststore
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    request = Request(url, headers={'User-Agent': 'ChampionsLab/0.1 (research data ingestion)'})
    try:
        with urlopen(request, context=context, timeout=45) as response:
            body = response.read()
            resolved_url, status, headers = response.url, response.status, response.headers
    except (URLError, TimeoutError, OSError) as exc:
        raise requests.RequestException(str(exc)) from exc
    path = raw_dir / filename
    # Each response has an immutable content-addressed copy as well as the latest file.
    digest = sha256(body).hexdigest()
    archive = raw_dir / 'objects' / digest
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        archive.write_bytes(body)
    path.write_bytes(body)
    _json(path.with_suffix(path.suffix + '.meta.json'), {
        'url': url, 'resolved_url': resolved_url, 'retrieved_at': datetime.now(timezone.utc).isoformat(),
        'sha256': digest, 'bytes': len(body), 'http_status': status,
        'etag': headers.get('ETag'), 'last_modified': headers.get('Last-Modified'),
        'transport': 'HTTPS; system trust store; certificate verification enabled',
    })
    return body


def _parse_current(text: str) -> pd.DataFrame:
    if '<html' in text.lower() or '<p' in text.lower():
        text = BeautifulSoup(text, 'html.parser').get_text('\n', strip=True)
    rows, matchday, date = [], None, None
    lines = [' '.join(line.split()) for line in text.splitlines()]
    for i, line in enumerate(lines):
        if line.startswith(('Highlights:', 'Paris vs', 'Watch ', '©')):
            continue
        md = re.fullmatch(r'(?:## )?Matchday (\d+)', line)
        if md:
            matchday = int(md[1])
            continue
        day = re.fullmatch(r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) (\d{1,2} [A-Za-z]+ 20\d\d)', line)
        if day:
            date = datetime.strptime(day[1], '%d %B %Y')
            continue
        game = re.fullmatch(r'(.+?) (?:vs|(\d+)-(\d+)) (.+?)(?: \((\d{2}:\d{2})\))?', line)
        if not game or not matchday or date is None:
            continue
        home, hg, ag, away, time = game.groups()
        home, away = normalize_team(home), normalize_team(away)
        if home not in CURRENT_TEAMS or away not in CURRENT_TEAMS:
            raise ValueError(f'Unrecognized UEFA fixture: {line}')
        if time is None and i + 1 < len(lines) and re.fullmatch(r'\(\d{2}:\d{2}\)', lines[i+1]):
            time = lines[i+1].strip('()')
        hour, minute = map(int, (time or '21:00').split(':'))
        # Article explicitly labels every kickoff CET (UTC+1), not local CEST.
        kickoff = date.replace(hour=hour, minute=minute, tzinfo=timezone(timedelta(hours=1)))
        rows.append(dict(match_id=f'ucl-2026-27-{sha256((home+"|"+away).encode()).hexdigest()[:16]}',
            date=kickoff.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            season='2026/27', stage='league', matchday=matchday, home=home, away=away,
            home_goals=int(hg) if hg is not None else None, away_goals=int(ag) if ag is not None else None,
            status='finished' if hg is not None else 'scheduled', source=UEFA_URL))
    return pd.DataFrame(rows, columns=FIXTURE_COLUMNS)


def _required(df: pd.DataFrame, columns: list[str], nullable=()) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if df.empty:
        raise ValueError('Empty match dataset')
    for col in set(columns) - set(nullable):
        if df[col].isna().any() or df[col].astype(str).str.strip().eq('').any():
            raise ValueError(f'Missing values: {col}')
    if not df.date.astype(str).str.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z').all():
        raise ValueError('Dates must be ISO UTC timestamps ending Z')
    if pd.to_datetime(df.date, utc=True, format='%Y-%m-%dT%H:%M:%SZ', errors='coerce').isna().any():
        raise ValueError('Invalid calendar dates')
    if (df.home == df.away).any():
        raise ValueError('A team cannot play itself')


def _scores(df: pd.DataFrame) -> None:
    for col in ['home_goals', 'away_goals']:
        n = pd.to_numeric(df[col], errors='coerce')
        if n.isna().any() or (n < 0).any() or (n % 1 != 0).any():
            raise ValueError(f'{col} must contain nonnegative finite integers')


def validate_fixtures(df: pd.DataFrame) -> None:
    """Raise ValueError unless this is the complete valid 2026/27 league schedule."""
    _required(df, FIXTURE_COLUMNS, ('home_goals', 'away_goals'))
    if len(df) != 144 or set(df.home) | set(df.away) != CURRENT_TEAMS:
        raise ValueError('Expected 144 fixtures and all 36 official current teams')
    if not df.season.eq('2026/27').all() or not df.stage.eq('league').all():
        raise ValueError('Expected season 2026/27, stage league')
    if not df.status.isin(['finished', 'scheduled']).all():
        raise ValueError('Invalid status')
    dates = pd.to_datetime(df.date, utc=True)
    if not dates.between('2026-09-01', '2027-02-01', inclusive='left').all():
        raise ValueError('Dates outside the current league phase')
    md = pd.to_numeric(df.matchday, errors='coerce')
    if md.isna().any() or (md % 1 != 0).any() or not md.between(1, 8).all():
        raise ValueError('Invalid matchday')
    if df.match_id.duplicated().any() or df[['home', 'away']].duplicated().any():
        raise ValueError('Duplicate fixture')
    pairs = df.apply(lambda r: tuple(sorted((r.home, r.away))), axis=1)
    if pairs.duplicated().any():
        raise ValueError('Repeated opponent pair')
    if not df.home.value_counts().eq(4).all() or not df.away.value_counts().eq(4).all():
        raise ValueError('Each team must have four home and four away matches')
    for day, group in df.groupby('matchday'):
        if len(group) != 18 or pd.concat([group.home, group.away]).nunique() != 36:
            raise ValueError(f'Matchday {day} must contain each team exactly once')
    _scores(df[df.status == 'finished'])
    if df.loc[df.status == 'scheduled', ['home_goals', 'away_goals']].notna().any().any():
        raise ValueError('Scheduled games must have missing scores')


def fetch_current(raw_dir: Path) -> pd.DataFrame:
    """Fetch UEFA; explicitly report a same-day web-extraction fallback on HTTP block."""
    raw_dir = Path(raw_dir)
    _system_tls()
    fallback = None
    try:
        body = _download(UEFA_URL, raw_dir, 'uefa-2026-27.html').decode('utf-8-sig')
    except requests.RequestException as exc:
        path = raw_dir / 'uefa-2026-27.web.txt'
        meta_path = raw_dir / 'uefa-2026-27.web.meta.json'
        if not path.exists() or not meta_path.exists():
            raise RuntimeError('UEFA HTTP unavailable; no dated web extraction cache') from exc
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        today = datetime.now(timezone.utc).date().isoformat()
        if meta.get('url') != UEFA_URL or meta.get('retrieved_on') != today:
            raise RuntimeError('UEFA web extraction cache is stale; refresh it explicitly') from exc
        body = path.read_text(encoding='utf-8')
        fallback = f'UEFA direct HTTP failed: {exc}; using web extraction retrieved {today}'
        warnings.warn(fallback, RuntimeWarning, stacklevel=2)
        meta['sha256'] = sha256(path.read_bytes()).hexdigest()
        _json(meta_path, meta)
    frame = _parse_current(body)
    validate_fixtures(frame)
    report = {'rows': len(frame), 'teams': 36, 'finished': int(frame.status.eq('finished').sum()),
              'scheduled': int(frame.status.eq('scheduled').sum()), 'source': UEFA_URL,
              'fallback': fallback, 'kickoff_policy': 'Article explicitly says CET; interpreted UTC+01:00.',
              'checks': '144 matches; 36 official teams; 8 distinct opponents; 4H4A; 18 games per matchday'}
    _json(raw_dir / 'fixtures-quality.json', report)
    frame.attrs['quality_report'] = report
    return frame


def _parse_openfootball(text: str, source: str, competition: str, start_year: int) -> list[dict]:
    rows, date, year, previous_month = [], None, start_year, 7
    for line in text.splitlines():
        line = ' '.join(line.split())
        match_date = re.fullmatch(r'(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) ([A-Za-z]{3}) (\d{1,2})(?: (20\d\d))?', line)
        if match_date:
            month = datetime.strptime(match_date[1], '%b').month
            year = int(match_date[3]) if match_date[3] else year + int(month < previous_month)
            previous_month = month
            date = datetime(year, month, int(match_date[2]), 23, 59, 59, tzinfo=timezone.utc)
            continue
        game = re.fullmatch(r'(?:\d{2}:\d{2} )?(.+?) \([A-Z]{3}\) v (.+?) \([A-Z]{3}\) +(\d+)-(\d+)(.*)', line)
        if game is None:
            continue
        if date is None:
            raise ValueError(f'Missing date in {source}: {line}')
        home, away, hg, ag, suffix = game.groups()
        if 'aet' in suffix.lower() or 'a.e.t.' in suffix.lower() or 'pen' in suffix.lower():
            # Football.TXT lists regulation then half-time in parentheses after AET.
            regulation = re.search(r'\((\d+)-(\d+),', suffix)
            if regulation is None:
                continue  # Never use extra-time/penalty totals as 90-minute goals.
            hg, ag = regulation.groups()
        rows.append(dict(date=date.strftime('%Y-%m-%dT%H:%M:%SZ'), home=normalize_team(home),
            away=normalize_team(away), home_goals=int(hg), away_goals=int(ag), competition=competition, source=source))
    if not rows:
        raise ValueError(f'No scored matches parsed: {source}')
    return rows


def validate_history(df: pd.DataFrame) -> None:
    _required(df, HISTORY_COLUMNS)
    _scores(df)
    if df.duplicated(['date', 'home', 'away', 'competition']).any():
        raise ValueError('Duplicate historical match')


def _connected_teams(df: pd.DataFrame, start: str) -> set[str]:
    adjacency: dict[str, set[str]] = {}
    for row in df.itertuples():
        adjacency.setdefault(row.home, set()).add(row.away)
        adjacency.setdefault(row.away, set()).add(row.home)
    seen, pending = set(), [start]
    while pending:
        team = pending.pop()
        if team not in seen:
            seen.add(team)
            pending.extend(adjacency.get(team, set()) - seen)
    return seen


def fetch_history(raw_dir: Path) -> pd.DataFrame:
    """Download domestic and cross-league European results; require 36-team coverage."""
    raw_dir = Path(raw_dir)
    _system_tls()
    sources = []
    for season in ['2425', '2526', '2627']:
        for league in ['E0', 'D1', 'I1', 'SP1', 'F1', 'N1', 'B1', 'P1', 'T1', 'G1']:
            sources.append((f'https://www.football-data.co.uk/mmz4281/{season}/{league}.csv', f'fd-{season}-{league}.csv', league, None))
    for country in ['AUT', 'NOR']:
        sources.append((f'https://www.football-data.co.uk/new/{country}.csv', f'fd-{country}.csv', country, None))
    for year in [2024, 2025]:
        for filename, competition in [('cl', 'UCL'), ('clq', 'UCL qualifying'), ('el', 'UEL'), ('conf', 'UECL'), ('elq', 'UEL qualifying'), ('confq', 'UECL qualifying')]:
            season = f'{year}-{str(year+1)[-2:]}'
            sources.append((f'https://raw.githubusercontent.com/openfootball/champions-league/master/{season}/{filename}.txt', f'openfootball-{season}-{filename}.txt', competition, year))

    def ingest(spec):
        url, filename, competition, year = spec
        try:
            content = _download(url, raw_dir, filename).decode('utf-8-sig')
            if year:
                return _parse_openfootball(content, url, competition, year), None
            table = pd.read_csv(StringIO(content))
            table = table.rename(columns={'HomeTeam': 'home', 'AwayTeam': 'away', 'Home': 'home', 'Away': 'away',
                                          'FTHG': 'home_goals', 'FTAG': 'away_goals', 'HG': 'home_goals', 'AG': 'away_goals'})
            table = table.dropna(how='all')
            # Unplayed rows have both scores absent; partial/malformed scores fail validation.
            table = table[~table[['home_goals', 'away_goals']].isna().all(axis=1)].copy()
            dates = pd.to_datetime(table.Date, dayfirst=True, format='mixed', errors='raise')
            table['date'] = dates.dt.strftime('%Y-%m-%dT23:59:59Z')
            for column in ['home', 'away']:
                if table[column].isna().any():
                    raise ValueError('Missing team name')
                table[column] = table[column].map(normalize_team)
            table['competition'], table['source'] = competition, url
            table = table[HISTORY_COLUMNS]
            validate_history(table)
            return table.to_dict('records'), None
        except (requests.RequestException, ValueError, KeyError) as exc:
            return [], {'source': url, 'error': str(exc)}

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(ingest, sources))
    errors = [error for _, error in results if error]
    frame = pd.DataFrame([row for rows, _ in results for row in rows], columns=HISTORY_COLUMNS)
    # Keep recent domestic matches after Champions starts as well. The forecast
    # pipeline applies its own historical cutoff; the download never freezes form.
    retrieved = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    frame = frame[frame.date.between('2024-07-01T00:00:00Z', retrieved, inclusive='left')].copy()
    validate_history(frame)
    frame = frame.sort_values(['date', 'competition', 'home', 'away']).reset_index(drop=True)
    for col in ['home_goals', 'away_goals']:
        frame[col] = frame[col].astype(int)
    counts = pd.concat([frame.home, frame.away]).value_counts()
    coverage = {team: int(counts.get(team, 0)) for team in sorted(CURRENT_TEAMS)}
    disconnected = sorted(CURRENT_TEAMS - _connected_teams(frame, 'Arsenal'))
    report = {'rows': len(frame), 'date_min': frame.date.min(), 'date_max': frame.date.max(),
        'current_team_match_counts': coverage, 'missing_teams': [t for t, n in coverage.items() if n == 0],
        'disconnected_current_teams': disconnected, 'competitions': frame.competition.value_counts().to_dict(),
        'source_failures': errors, 'sources_succeeded': len(sources)-len(errors),
        'date_policy': 'Date-only history conservatively timestamped at 23:59:59 UTC; no exact kickoff claim.',
        'cutoff': f'Strictly before retrieval {retrieved}; caller additionally filters to its training cutoff.',
        'score_policy': '90-minute goals only; extra-time matches omitted unless regulation score is explicit.'}
    _json(raw_dir / 'history-quality.json', report)
    if report['missing_teams'] or disconnected or not frame.competition.eq('UCL').any():
        raise ValueError(f'Historical coverage/connectivity incomplete: {report}')
    if errors:
        warnings.warn(f'{len(errors)} history sources failed; see history-quality.json', RuntimeWarning, stacklevel=2)
    frame.attrs['quality_report'] = report
    return frame


if __name__ == '__main__':
    raw = Path('data/raw')
    processed = Path('data/processed')
    processed.mkdir(parents=True, exist_ok=True)
    for name, fetch in [('fixtures', fetch_current), ('history', fetch_history)]:
        dataset = fetch(raw)
        temporary = processed / f'{name}.csv.tmp'
        dataset.to_csv(temporary, index=False)
        temporary.replace(processed / f'{name}.csv')
        print(name, json.dumps(dataset.attrs['quality_report'], ensure_ascii=False))
