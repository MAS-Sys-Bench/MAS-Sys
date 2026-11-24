import json
import os
import re
import time
from functools import wraps
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import arxiv
import openreview
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader


def save_benchmark(benchmark: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark, f, indent=4, ensure_ascii=False)
    print(f'Benchmark saved to {output_path}')


def load_benchmark(input_path: str) -> Any:
    with open(input_path, 'r', encoding='utf-8') as file:
        raw = file.read()

    stripped = raw.lstrip()
    if not stripped:
        raise ValueError(f'Benchmark file is empty: {input_path}')

    if stripped.startswith('version https://git-lfs.github.com/spec/v1'):
        raise ValueError(
            f'Benchmark file is a Git LFS pointer, not the real JSON: {input_path}\n'
            'Please fetch the dataset files first, e.g. run `git lfs pull` in the repo root.'
        )

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        preview = stripped[:120].replace('\n', '\\n')
        raise ValueError(
            f'Invalid benchmark JSON in {input_path}: {exc}. '
            f'File starts with: {preview}'
        ) from exc


def with_cache(
    cache_dir: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(arxiv_id: str, *args: Any, **kwargs: Any) -> Any:
            if not cache_dir:
                return func(arxiv_id, *args, **kwargs)
            cache_path = Path(cache_dir) / f'{arxiv_id}.json'
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.exists():
                with cache_path.open('r') as f:
                    return json.load(f)
            result = func(arxiv_id, *args, **kwargs)
            with cache_path.open('w') as f:
                json.dump(result, f)
            return result

        return wrapper

    return decorator


def _perform_arxiv_search(
    search: arxiv.Search,
    max_retries: int = 5,
    delay: int = 2,
) -> Any:
    client = arxiv.Client()
    for attempt in range(max_retries):
        try:
            return client.results(search)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f'Connection error: {e}. Retrying ({attempt + 1}/{max_retries})...')
                time.sleep(delay)
            else:
                raise e


def _get_paper_by_arxiv_id(arxiv_id: str) -> Optional[arxiv.Result]:
    search = arxiv.Search(
        query=f'id:{arxiv_id}',
        max_results=1,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    for result in _perform_arxiv_search(search):
        if result.get_short_id().split('v')[0] == arxiv_id:
            return result
    return None


def _get_paper_by_keyword(
    keyword: str, existing_ids: Set[str], max_papers: int = 10
) -> List[arxiv.Result]:
    search = arxiv.Search(
        query=f'all:"{keyword}" AND (cat:cs.AI OR cat:cs.LG)',
        max_results=max_papers * 2,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    papers = []
    for paper in _perform_arxiv_search(search):
        short_id = paper.get_short_id().split('v')[0]
        if short_id not in existing_ids:
            papers.append(paper)
            existing_ids.add(short_id)
        if len(papers) >= max_papers:
            break
    return papers


def _get_paper_by_title(title: str) -> Optional[arxiv.Result]:
    search = arxiv.Search(
        query=f'ti:"{title}"',
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    for result in _perform_arxiv_search(search):
        if result.title.lower() == title.lower():
            return result
    return None


def _fetch_html_content(url: str) -> Optional[BeautifulSoup]:
    try:
        html_url = url.replace('/abs/', '/html/')
        response = requests.get(html_url, timeout=30)
        if response.status_code == 200:
            return BeautifulSoup(response.text, 'html.parser')
    except Exception:
        pass
    return None


def _get_section_contents(soup: BeautifulSoup) -> Optional[Dict[str, str]]:
    sections = {}
    for section in soup.find_all('section'):
        heading = section.find(['h1', 'h2', 'h3'])
        if heading:
            title = heading.get_text(strip=True)
            content = section.get_text(separator=' ', strip=True)
            sections[title] = content
    return sections if sections else None


def _get_paper_content_from_html(url: str) -> Optional[Dict[str, str]]:
    soup = _fetch_html_content(url)
    return _get_section_contents(soup) if soup else None


def _get_paper_content_from_pdf(url: str) -> Optional[Dict[str, str]]:
    try:
        pdf_url = url.replace('abs', 'pdf') if 'abs' in url else url.replace('html', 'pdf')
        response = requests.get(pdf_url, timeout=60)
        text = ''.join(
            page.extract_text() for page in PdfReader(BytesIO(response.content)).pages
        )
        if not text:
            return None
        section_titles = [
            'Abstract', 'Introduction', 'Related Work', 'Background', 'Methods',
            'Experiments', 'Results', 'Discussion', 'Conclusion', 'Conclusions',
            'Acknowledgments', 'References', 'Appendix', 'Materials and Methods',
        ]
        pattern = re.compile(
            r'\b(' + '|'.join(re.escape(t) for t in section_titles) + r')\b',
            re.IGNORECASE,
        )
        matches = list(pattern.finditer(text))
        sections = {}
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections[match.group()] = text[start:end].strip()
        return sections
    except Exception:
        return None


def _get_paper_introduction(url: str) -> Optional[str]:
    intro_length = 512
    sections = _get_paper_content_from_html(url) or _get_paper_content_from_pdf(url)
    if not sections:
        return None
    keys = list(sections.keys())
    on_and_after_intro = False
    intro_text = ''
    for key in keys:
        if 'introduction' in key.lower():
            on_and_after_intro = True
        if on_and_after_intro:
            intro_text += ' ' + sections[key]
    if on_and_after_intro:
        return ' '.join(intro_text.split()[:intro_length])
    fallback = ''.join(
        sections[key] for i, key in enumerate(keys) if i != 0 or len(keys) >= 2
    )
    return ' '.join(fallback.split()[:intro_length])


def get_references(arxiv_id: str, max_retries: int = 5) -> List[Dict[str, Any]]:
    url = f'https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}/references'
    params = {'limit': 100, 'offset': 0, 'fields': 'title,abstract'}
    headers = {'User-Agent': 'PaperProcessor/1.0'}
    for attempt in range(max_retries):
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return [
                {
                    'title': r['citedPaper'].get('title'),
                    'abstract': r['citedPaper'].get('abstract'),
                }
                for r in response.json().get('data', [])
                if r.get('citedPaper')
            ]
        time.sleep(2**attempt)
    return []


@with_cache(cache_dir='paper_data')
def get_paper_data(arxiv_id: str) -> Dict[str, Any]:
    paper = _get_paper_by_arxiv_id(arxiv_id)
    if not paper:
        raise ValueError(f'Paper with arXiv ID {arxiv_id} not found.')
    url = paper.entry_id
    if not url:
        raise ValueError(f'Paper with arXiv ID {arxiv_id} has no URL.')
    introduction = _get_paper_introduction(url)
    return {
        'title': paper.title,
        'url': url,
        'arxiv_id': arxiv_id,
        'authors': [a.name for a in paper.authors],
        'abstract': paper.summary.replace('\n', ' '),
        'introduction': introduction.replace('\n', ' ') if introduction else None,
        'references': get_references(arxiv_id),
        'categories': paper.categories,
    }


def get_arxiv_ids_from_keyword(
    keyword: str, existing_ids: Set[str], max_papers: int = 10
) -> List[str]:
    return [
        p.get_short_id().split('v')[0]
        for p in _get_paper_by_keyword(keyword, existing_ids, max_papers)
    ]


def get_arxiv_id_from_title(title: str) -> Optional[str]:
    paper = _get_paper_by_title(title)
    return str(paper.get_short_id().split('v')[0]) if paper else None


def get_url_from_title(title: str) -> Optional[str]:
    paper = _get_paper_by_title(title)
    return str(paper.entry_id) if paper else None


@with_cache(cache_dir='reference_proposal_data')
def get_proposal_from_paper(arxiv_id: str, intro: str, model: str = 'gpt-4o-mini') -> str:
    from openai import OpenAI

    prompt = (
        'Here is a high-level summarized insight of a research field Machine Learning.\n\n'
        'Here are the five core questions:\n\n'
        '[Question 1] - What is the problem?\n\n'
        'Formulate the specific research question you aim to address. Only output one question and do not include any more information.\n\n'
        '[Question 2] - Why is it interesting and important?\n\n'
        'Explain the broader implications of solving this problem for the research community.\n'
        'Discuss how such paper will affect the future research.\n'
        'Discuss how addressing this question could advance knowledge or lead to practical applications.\n\n'
        '[Question 3] - Why is it hard?\n\n'
        'Discuss the challenges and complexities involved in solving this problem.\n'
        'Explain why naive or straightforward approaches may fail.\n'
        'Identify any technical, theoretical, or practical obstacles that need to be overcome. MAKE IT CLEAR.\n\n'
        "[Question 4] - Why hasn't it been solved before?\n\n"
        'Identify gaps or limitations in previous research or existing solutions.\n'
        'Discuss any barriers that have prevented this problem from being solved until now.\n'
        'Explain how your approach differs from or improves upon prior work. MAKE IT CLEAR.\n\n'
        '[Question 5] - What are the key components of my approach and results?\n\n'
        'Outline your proposed methodology in detail, including the method, dataset, metric that you plan to use.\n'
        'Describe the expected outcomes. MAKE IT CLEAR.\n\n'
        f'Introduction:\n{intro}\n\n'
        'Please provide the five core questions contents based on the above introduction.'
    )
    response = OpenAI().chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
    )
    return response.choices[0].message.content


def get_all_reviews(venue_id: str) -> Dict[str, Any]:
    client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
    all_reviews = {}
    submissions = client.get_all_notes(
        content={'venueid': venue_id}, details='replies'
    )
    print('Fetching reviews...')
    for submission in submissions:
        url = get_url_from_title(submission.content['title']['value'])
        if not url:
            continue
        match = re.search(r'arxiv\.org/abs/([^\s/]+)', url)
        if not match:
            continue
        arxiv_id = match.group(1).split('v')[0]
        reviews = []
        for reply in submission.details['replies']:
            if 'summary' in reply['content']:
                review_content = {
                    k: v['value'] if isinstance(v, dict) and 'value' in v else v
                    for k, v in reply['content'].items()
                }
                reviews.append(review_content)
        all_reviews[arxiv_id] = {'reviews': reviews}
    print(len(all_reviews), 'reviews fetched.')
    return all_reviews
