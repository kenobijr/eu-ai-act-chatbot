from dataclasses import dataclass, field, asdict
import requests
import json
import re
import time
import random
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime
from ratelimit import limits, sleep_and_retry


# data Models as dataclasses with validation
@dataclass
class Recital:
    id: str
    text_content: str

    def __post_init__(self):
        if not self.id or not self.text_content:
            raise ValueError(f"Recital {self.id} requires id and text_content")


@dataclass
class Article:
    id: str
    title: str
    text_content: str
    chapter_title: str
    section_title: Optional[str] = None
    entry_date: Optional[str] = None
    related_recital_ids: List[str] = field(default_factory=list)
    related_annex_ids: List[str] = field(default_factory=list)
    related_article_ids: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not all([self.id, self.title, self.text_content, self.chapter_title]):
            raise ValueError(f"Article {self.id} missing required fields")
        # limit article cross-references to 5
        # if len(self.related_article_ids) > 5:
            # self.related_article_ids = self.related_article_ids[:5]


@dataclass
class Annex:
    id: str
    title: str
    text_content: str

    def __post_init__(self):
        if not all([self.id, self.title, self.text_content]):
            raise ValueError(f"Annex {self.id} missing required fields")


@dataclass
class Definition:
    id: str
    title: str
    text_content: str

    def __post_init__(self):
        if not all([self.id, self.title, self.text_content]):
            raise ValueError(f"Definition {self.id} missing required fields")


@dataclass
class ScrapeMeta:
    scraped_date: datetime
    source_url: str
    entity_counts: Dict[str, int] = field(default_factory=dict)


class Scraper:
    def __init__(self):
        self.base_url = "https://artificialintelligenceact.eu"
        # create session enabling tcp connection pooling, cookies over multiple requests
        self.session = requests.Session()
        # set http request header user agent to real browser instead of python-requests/2.x.x
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        # save the scraped entites as objects before serialising
        self.prey = {}
        # save scrape metadata
        self.metadata = None

    @sleep_and_retry  # if rate limit hit, sleep and try again
    @limits(calls=1, period=2)  # max 1 call every 2 seconds
    def _make_request(self, target_url: str):
        """ wrapper for requests against target-url adding rate-limiting and error handling """
        try:
            response = self.session.get(target_url, timeout=20)
            # convert http error status codes into exceptions
            response.raise_for_status()
            # add random delay between 10ms and 200ms
            time.sleep(random.uniform(0.01, 0.2))
        except requests.RequestException as e:
            # crash the app with passing through the http error with url and exception type
            raise requests.RequestException(f"request failed for {target_url} with {e}")
        else:
            return response

    @staticmethod
    def extract_url_digits(url_container: List[str], phrase: str) -> List[int]:
        """
        - receives list of raw url strings in form "https://artificialintelligenceact.eu/annex/3"
        - take phrase, e.g. "annex", after match, catch number and return as list of ints
        """
        return sorted(int(url.split((phrase + "/"))[1].rstrip('/')) for url in url_container)

    @staticmethod
    def validate_and_create(entity_class, **kwargs):
        """
        - helper to create entity with validation and error handling.
        - returns entity or None if validation fails.
        """
        try:
            return entity_class(**kwargs)
        except (ValueError, TypeError) as e:
            print(f"Validation failed for {entity_class.__name__}: {e}")
            print(f"Data: {kwargs}")
            return None

    def _scrape_recital(self, path_segment: str, range_start: int, range_end: int) -> Dict[str, Recital]:
        """
        - recitals consist of raw text without headline, external refs or internal relations
        - relationships from articles & annexes to recitals will be saved oneway in these entities
        """
        content_urls = [f"{self.base_url}/{path_segment}/{i+1}/" for i in range(range_start, range_end)]
        recital_dict = {}
        print(f"scraping {len(content_urls)} recitals...")
        for i, url in enumerate(content_urls, start=1):
            response = self._make_request(url)
            assert response, f"empty response for {url}"
            soup = BeautifulSoup(response.content, "html.parser")
            # fetch text content; assign none if empty / non existing to print error
            text_elem = soup.select_one("#aia-explorer-content .et_pb_post_content")
            # scrape text with sep = " "; otherwise no space around inline elements "underArticle"
            text_content = text_elem.get_text(separator=' ') if text_elem else None
            # clean up excess whitespace, linebreaks, ...
            if text_content:
                text_content = ' '.join(text_content.split())
            # validate + create with dataclass & central helper validation method
            new_recital = self.validate_and_create(
                Recital,
                id=f"recital_{i:03}",
                text_content=text_content,
            )
            if new_recital:
                recital_dict[f"recital_{i:03}"] = new_recital
            else:
                print(f"Skipping recital at {url}...")
        return recital_dict

    def _scrape_annex(self, path_segment: str, range_start: int, range_end: int) -> Dict[str, Annex]:
        content_urls = [f"{self.base_url}/{path_segment}/{i+1}/" for i in range(range_start, range_end)]
        annex_dict = {}
        print(f"scraping {len(content_urls)} annexes...")
        for i, url in enumerate(content_urls, start=1):
            response = self._make_request(url)
            assert response, f"empty response for {url}"
            soup = BeautifulSoup(response.content, "html.parser")
            # scrape title
            title_elem = soup.select_one("#aia-explorer-content .entry-title")
            title = str(title_elem.get_text(strip=True)) if title_elem else None
            # scrape text_content; remove in text span cital relationships & all urls
            content_div = soup.select_one("#aia-explorer-content .et_pb_post_content")
            for span in content_div.select("span.aia-recital-ref"):
                span.decompose()
            # scrape text with sep = " "; otherwise no space around inline elements "underArticle"
            text_content = content_div.get_text(separator=' ') if content_div else None
            if text_content:
                text_content = ' '.join(text_content.split())
            # validate + create with dataclass & central helper validation method
            new_annex = self.validate_and_create(
                Annex,
                id=f"annex_{i:02}",
                title=title,
                text_content=text_content,
            )
            if new_annex:
                annex_dict[f"annex_{i:02}"] = new_annex
            else:
                print(f"Skipping annex at {url}...")
        return annex_dict

    def _scrape_article(self, path_segment: str, range_start: int, range_end: int) -> Dict[str, Article]:
        """
        - special logic when article_id = 3:
          - extract definitions into sep entities
          - keep text content within article 3 itsself minimal to avoid duplication
        """
        content_urls = [f"{self.base_url}/{path_segment}/{i+1}/" for i in range(range_start, range_end)]
        article_dict = {}
        print(f"scraping {len(content_urls)} articles...")
        for i, url in enumerate(content_urls, start=1):
            response = self._make_request(url)
            assert response, f"empty response for {url}"
            soup = BeautifulSoup(response.content, "html.parser")
            # titles of attached chapter & section links (section link if existing)
            chapter_link = soup.select_one("#aia-explorer-content .aia-post-meta-wrapper a[href*='/chapter/']")
            section_link = soup.select_one("#aia-explorer-content .aia-post-meta-wrapper a[href*='/section/']")
            chapter_title = str(chapter_link.get_text(strip=True)) if chapter_link else None
            section_title = str(section_link.get_text(strip=True)) if section_link else None
            # article title
            article_elem = soup.select_one("#aia-explorer-content .et_pb_post_title")
            article_title = str(article_elem.get_text(strip=True)) if article_elem else None
            # date
            entry_date_elem = soup.select_one("#aia-explorer-content p.aia-eif-value")
            entry_date = str(entry_date_elem.get_text(strip=True)) if entry_date_elem else None
            # relationships
            relations = {}
            for entity in ["annex", "article", "recital"]:
                # catch relationships as unique urls from <a href tags
                # form: <a href="https://artificialintelligenceact.eu/annex/1">Annex I</a>
                url_tags = soup.select(f"#aia-explorer-content .et_pb_post_content a[href*='/{entity}/']")
                # extract unique urls from <a tags
                unique_urls = list(set(a["href"] for a in url_tags))
                # extract resource digits from urls
                digits = self.extract_url_digits(url_container=unique_urls, phrase=entity)
                # construct unique id's with proper padding
                padding = "03d" if entity in ["article", "recital"] else "02d"
                # each article may not have multiple relationships to same identical entity
                relations[entity] = list(set(f"{entity}_{i:{padding}}" for i in digits))
            # get text after removing in text span relationships
            text_div = soup.select_one("#aia-explorer-content .et_pb_post_content")
            for span in text_div.select("span.aia-recital-ref"):
                span.decompose()
            # scrape text with sep = " "; otherwise no space around inline elements "underArticle"   
            text_content = text_div.get_text(separator=' ') if text_div else None
            if text_content:
                text_content = ' '.join(text_content.split())
                text_content = re.sub(r'Related:\s*Recital\s*\d+', '', text_content).strip()
            # special handling for article 3 - extract definitions!!
            if i == 3:
                # For Article 3, store only intro text (before definitions)
                text_content = "This article contains definitions for terms used in the AI Act."
            # validate + create with dataclass & central helper validation method
            new_article = self.validate_and_create(
                Article,
                id=f"article_{i:03}",
                title=article_title,
                text_content=text_content,
                chapter_title=chapter_title,
                section_title=section_title,
                entry_date=entry_date,
                related_annex_ids=relations["annex"],
                related_article_ids=relations["article"],
                related_recital_ids=relations["recital"],
            )
            if new_article:
                article_dict[f"article_{i:03}"] = new_article
            else:
                print(f"Skipping article at {url}...")
        return article_dict

    def _scrape_definition(self, path_segment: str, article_id: int) -> Dict[str, Definition]:
        """
        - extract definition from text of article 3
        - text is delivered as str stripped off all html tags, spans, urls
        """
        url = f"{self.base_url}/{path_segment}/{article_id}/"
        definition_dict = {}
        print("scraping definitions...")
        response = self._make_request(url)
        assert response, f"empty response for {url}"
        soup = BeautifulSoup(response.content, "html.parser")
        # get text after removing in text span relationships
        text_div = soup.select_one("#aia-explorer-content .et_pb_post_content")
        for span in text_div.select("span.aia-recital-ref"):
            span.decompose()
        # scrape text with sep = " "; otherwise no space around inline elements "underArticle"   
        text_content = text_div.get_text(separator=' ') if text_div else None
        # remove "Related: Recital X" metadata from text
        if text_content:
            text_content = ' '.join(text_content.split())
            text_content = re.sub(r'Related:\s*Recital\s*\d+', '', text_content).strip()
        # \u2018 = left curly quote '
        # \u2019 = right curly quote '
        pattern = r"\((\d+)\)\s+[\u2018']([^\u2018\u2019',]+)[\u2019']?,?\s*(.*?)(?=;\s*\(\d+\)\s+[\u2018']|;\s*$|$)"
        matches = re.findall(pattern, text_content, re.DOTALL)
        for number, term, text_raw in matches:
            # concatenate term / title to text to make text cohesive
            text_final = f"{term} {text_raw}"
            # validate + create with dataclass & central helper validation method
            new_definition = self.validate_and_create(
                Definition,
                id=f"definition_{int(number):02}",
                title=term,
                text_content=text_final,
            )
            if new_definition:
                definition_dict[f"definition_{int(number):02}"] = new_definition
            else:
                print(f"Skipping article at {number}, {term}...")
        return definition_dict

    def execute_scraping(self):
        """
        - central method to steer scraping: entities, range; saves results as objects
        - track stats and save them to scrape_metadata
        """
        # init scraping meta_data obj
        self.metadata = ScrapeMeta(scraped_date=datetime.now(), source_url=self.base_url)
        print(f"...started scraping at {datetime.now()}")
        # scrape recitals: p only with no relations within text
        self.prey["recitals"] = self._scrape_recital(path_segment="recital", range_start=0, range_end=180)
        self.metadata.entity_counts["recitals"] = len(self.prey["recitals"])
        # scrape annexes: h, p with 0-n relations to articles & recitals within text
        self.prey["annexes"] = self._scrape_annex(path_segment="annex", range_start=0, range_end=13)
        self.metadata.entity_counts["annexes"] = len(self.prey["annexes"])
        # scrape articles as "central hub": content & all releavnt relationships
        self.prey["articles"] = self._scrape_article(path_segment="article", range_start=0, range_end=113)
        self.metadata.entity_counts["articles"] = len(self.prey["articles"])
        # special case: scrape definitons from article 3 as sep entitiy
        self.prey["definitions"] = self._scrape_definition(path_segment="article", article_id=3)
        self.metadata.entity_counts["definitions"] = len(self.prey["definitions"])
        print(f"...ended scraping at {datetime.now()} scraped {self.metadata.entity_counts}")

    def save_to_disc(self) -> None:
        """ take the scraped data and serialize it to json"""
        save_dir = Path(__file__).parent.parent / "data" / "raw"
        save_dir.mkdir(parents=True, exist_ok=True)
        # save each entity type separately
        for entity_type, entities_dict in self.prey.items():
            file_path = save_dir / f"{entity_type}.json"
            # convert dataclass instances to dict using asdict
            data_to_save = {entity_id: asdict(entity) for entity_id, entity in entities_dict.items()}
            # serialise to json
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            print(f"saved {len(entities_dict)} {entity_type} to {file_path}")
        # save metadata
        metadata_path = save_dir / "metadata.json"
        metadata_dict = asdict(self.metadata)
        # convert datetime to ISO string for JSON serialization
        metadata_dict['scraped_date'] = self.metadata.scraped_date.isoformat()
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        print(f"saved metadata to {metadata_path}")
