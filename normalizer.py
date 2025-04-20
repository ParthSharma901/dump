import re
import json
import logging
import os.path
import pickle
from typing import Dict, List, Tuple, Any
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from transformers import AutoModelForSequenceClassification, AutoConfig
import requests
from bs4 import BeautifulSoup
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HinglishNormalizer:
    def __init__(self, use_cached_dicts=True, use_transformers=True):
        self.hindi_mappings = {}
        self.short_forms = {}
        self.acronyms = {}
        self.use_transformers = use_transformers
        
        # File paths for cached dictionaries
        self.hindi_cache_file = "hindi_mappings_cache.pkl"
        self.short_forms_cache_file = "short_forms_cache.pkl"
        self.acronyms_cache_file = "acronyms_cache.pkl"
        
        # Load or build dictionaries
        if use_cached_dicts and self._cached_dicts_exist():
            self._load_cached_dicts()
        else:
            self._build_dictionaries()
            self._save_cached_dicts()
            
        # Add some common mappings that might be missed by scraping
        self._add_common_mappings()
        
        # Initialize transformers if required
        if self.use_transformers:
            self._init_transformers()
    
    def _init_transformers(self):
        """Initialize transformer models for language identification and normalization."""
        logging.info("Initializing transformer models...")
        
        try:
            # Load language identification model
            # This model can identify whether a word is Hindi, English, or another language
            self.tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBERT-MLM-Transliteration")
            self.model = AutoModelForSequenceClassification.from_pretrained("ai4bharat/IndicBERT-MLM-Transliteration")
            
            # Alternative: Use IndicNLP or similar models specifically trained for code-mixed text
            # You can use a different model if it works better for your specific task
            
            # Named Entity Recognition pipeline for identifying proper nouns, which might not need normalization
            self.ner_pipeline = pipeline(
                "ner",
                model="monsoon-nlp/hindi-bert",
                tokenizer="monsoon-nlp/hindi-bert"
            )
            
            # Text normalization model - specifically for Hinglish text
            # Note: You might need to fine-tune this model on your specific normalization task
            self.norm_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
            self.norm_model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large")
            
            logging.info("Transformer models initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing transformer models: {str(e)}")
            logging.warning("Falling back to dictionary-based normalization only.")
            self.use_transformers = False
    
    def _cached_dicts_exist(self) -> bool:
        """Check if cached dictionary files exist."""
        return (os.path.isfile(self.hindi_cache_file) and 
                os.path.isfile(self.short_forms_cache_file) and 
                os.path.isfile(self.acronyms_cache_file))
    
    def _load_cached_dicts(self) -> None:
        """Load dictionaries from cached files."""
        logging.info("Loading cached dictionaries...")
        with open(self.hindi_cache_file, 'rb') as f:
            self.hindi_mappings = pickle.load(f)
        
        with open(self.short_forms_cache_file, 'rb') as f:
            self.short_forms = pickle.load(f)
            
        with open(self.acronyms_cache_file, 'rb') as f:
            self.acronyms = pickle.load(f)
        logging.info(f"Loaded {len(self.hindi_mappings)} Hindi mappings, {len(self.short_forms)} short forms, and {len(self.acronyms)} acronyms.")
    
    def _save_cached_dicts(self) -> None:
        """Save dictionaries to cache files."""
        logging.info("Saving dictionaries to cache...")
        with open(self.hindi_cache_file, 'wb') as f:
            pickle.dump(self.hindi_mappings, f)
            
        with open(self.short_forms_cache_file, 'wb') as f:
            pickle.dump(self.short_forms, f)
            
        with open(self.acronyms_cache_file, 'wb') as f:
            pickle.dump(self.acronyms, f)
        logging.info("Dictionaries cached successfully.")
    
    def _add_common_mappings(self) -> None:
        """Add common mappings that might be missed by scraping."""
        common_hindi_mappings = {
            'hu': 'hoo', 'abhi': 'abhee', 'bhee': 'bhee', 'hi': 'hee', 'ki': 'kee',
            'degii': 'degee', 'bhai': 'bhaee', 'nai': 'nahin', 'muh': 'munh',
            'kam': 'kaam', 'he': 'hai', 'ahae': 'he', 'wahi': 'vahee',
            'jaldi': 'jaldee', 'bahar': 'baahar', 'bta': 'bata', 'koee': 'koee',
            'koi': 'koee', 'mahol': 'maahaul', 'mei': 'mein', 'karte': 'karate',
            'or': 'aur', 'yarr': 'yaar', 'kay': 'kya', 'bnane': 'banaane',
            'yr': 'yaar', 'dekhlenge': 'dekhalenge', 'lagayegi': 'lagaegee',
            'bolbi': 'bolabi', 'kitna': 'kitana', 'bhi': 'bhee', 'jo': 'jo',
            'ho': 'ho', 'na': 'na', 'mast': 'mast', 'tum': 'tum', 'lo': 'lo',
            'sab': 'sab', 'batao': 'batao', 'pe': 'pe', 'kar': 'kar'
        }
        
        common_short_forms = {
            'plz': 'please', 'plzz': 'please', 'plzzz': 'please',
            'm': 'i am', 'fv': 'favourite', 'u': 'you',
            'kro': 'karo', 'n': 'na', 'r': 'are', 'ur': 'your',
            'ppl': 'people', 'thx': 'thanks', 'k': 'okay',
            'y': 'why', 'bc': 'because', 'bf': 'boyfriend',
            'gf': 'girlfriend', 'tc': 'take care', 'omg': 'oh my god'
        }
        
        common_acronyms = {
            'fb': 'facebook', 'ig': 'instagram', 'yt': 'youtube',
            'lol': 'laughing out loud', 'brb': 'be right back',
            'btw': 'by the way', 'tbh': 'to be honest',
            'idk': 'i do not know', 'imo': 'in my opinion',
            'asap': 'as soon as possible', 'fyi': 'for your information',
            'gate': 'graduate aptitude test in engineering',
            'iit': 'indian institute of technology',
            'jee': 'joint entrance examination',
            'ssc': 'staff selection commission'
        }
        
        # Update with common mappings (don't overwrite scraped ones)
        for k, v in common_hindi_mappings.items():
            if k not in self.hindi_mappings:
                self.hindi_mappings[k] = v
                
        for k, v in common_short_forms.items():
            if k not in self.short_forms:
                self.short_forms[k] = v
                
        for k, v in common_acronyms.items():
            if k not in self.acronyms:
                self.acronyms[k] = v
    
    def _build_dictionaries(self) -> None:
        """Build dictionaries by scraping online resources."""
        logging.info("Building dictionaries from online resources...")
        
        # Scrape Hindi transliteration data
        self._scrape_hindi_mappings()
        
        # Scrape short forms
        self._scrape_short_forms()
        
        # Scrape acronyms
        self._scrape_acronyms()
        
        logging.info(f"Dictionary building complete. Hindi mappings: {len(self.hindi_mappings)}, Short forms: {len(self.short_forms)}, Acronyms: {len(self.acronyms)}")
    
    def _scrape_hindi_mappings(self) -> None:
        """Scrape Hindi transliteration mappings from online resources."""
        try:
            # Example: Scrape common Hindi words and their standardized transliterations
            url = "https://www.learnsanskrit.cc/index.php?mode=3&direct=au&script=hk&tran_input=hindi"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Find tables containing Hindi words
                tables = soup.find_all('table', class_='grammar')
                
                for table in tables:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            hindi_word = cells[0].text.strip().lower()
                            transliteration = cells[1].text.strip().lower()
                            if hindi_word and transliteration:
                                self.hindi_mappings[hindi_word] = transliteration
            
            # Additional URLs for transliteration standards
            urls = [
                "https://en.wiktionary.org/wiki/Category:Hindi_transliterations",
                "https://www.indifferentlanguages.com/words/english-hindi"
            ]
            
            for url in urls:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        # Extract content based on the specific structure of each site
                        if "wiktionary" in url:
                            word_items = soup.find_all('div', class_='CategoryTreeItem')
                            for item in word_items:
                                text = item.text.strip()
                                if " → " in text:
                                    parts = text.split(" → ")
                                    if len(parts) >= 2:
                                        hinglish = parts[0].lower()
                                        standard = parts[1].lower()
                                        self.hindi_mappings[hinglish] = standard
                        elif "indifferentlanguages" in url:
                            word_pairs = soup.find_all('div', class_='table-cell')
                            for i in range(0, len(word_pairs), 2):
                                if i+1 < len(word_pairs):
                                    english = word_pairs[i].text.strip().lower()
                                    hindi = word_pairs[i+1].text.strip().lower()
                                    if english and hindi:
                                        # Here we would need transliteration, not translation
                                        # This is just to demonstrate the approach
                                        pass
                    time.sleep(1)  # Be respectful with rate limiting
                except Exception as e:
                    logging.warning(f"Error scraping {url}: {str(e)}")
                    
        except Exception as e:
            logging.error(f"Error in scraping Hindi mappings: {str(e)}")
    
    def _scrape_short_forms(self) -> None:
        """Scrape short forms and their expansions from online resources."""
        try:
            # Example URL for short forms/text speak
            url = "https://en.wiktionary.org/wiki/Appendix:English_internet_slang"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Find tables containing short forms
                tables = soup.find_all('table')
                
                for table in tables:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            short_form = cells[0].text.strip().lower()
                            meaning = cells[1].text.strip().lower()
                            if short_form and meaning:
                                self.short_forms[short_form] = meaning
            
            # Additional URLs for short forms
            urls = [
                "https://www.urbandictionary.com/popular.php",
                "https://www.netlingo.com/acronyms.php"
            ]
            
            for url in urls:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        # Extract content based on the specific structure of each site
                        if "urbandictionary" in url:
                            word_items = soup.find_all('a', class_='word')
                            for item in word_items:
                                short_form = item.text.strip().lower()
                                # We'd need to click through to get the meaning
                                # This is simplified for example purposes
                                if short_form and len(short_form) <= 5:
                                    # Placeholder - would need actual meanings
                                    pass
                        elif "netlingo" in url:
                            acronym_items = soup.find_all('tr')
                            for item in acronym_items:
                                cells = item.find_all('td')
                                if len(cells) >= 2:
                                    short_form = cells[0].text.strip().lower()
                                    meaning = cells[1].text.strip().lower()
                                    if short_form and meaning:
                                        self.short_forms[short_form] = meaning
                    time.sleep(1)  # Be respectful with rate limiting
                except Exception as e:
                    logging.warning(f"Error scraping {url}: {str(e)}")
        
        except Exception as e:
            logging.error(f"Error in scraping short forms: {str(e)}")

    def _scrape_acronyms(self) -> None:
        """Scrape acronyms and their expansions from online resources."""
        try:
            # Example URL for acronyms
            url = "https://www.abbreviations.com/abbreviations/Hindi"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Find tables containing acronyms
                tables = soup.find_all('table', class_='table')
                
                for table in tables:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            acronym = cells[0].text.strip().lower()
                            meaning = cells[1].text.strip().lower()
                            if acronym and meaning:
                                self.acronyms[acronym] = meaning
            
            # Additional URLs for acronyms
            urls = [
                "https://www.acronymfinder.com/Hindi/",
                "https://www.allacronyms.com/tag/indian"
            ]
            
            for url in urls:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        # Extract content based on the specific structure of each site
                        if "acronymfinder" in url:
                            items = soup.find_all('div', class_='result-list')
                            for item in items:
                                acronym_elem = item.find('span', class_='acronym')
                                meaning_elem = item.find('span', class_='meaning')
                                if acronym_elem and meaning_elem:
                                    acronym = acronym_elem.text.strip().lower()
                                    meaning = meaning_elem.text.strip().lower()
                                    if acronym and meaning:
                                        self.acronyms[acronym] = meaning
                        elif "allacronyms" in url:
                            items = soup.find_all('div', class_='acronym_result')
                            for item in items:
                                acronym_elem = item.find('div', class_='acronym')
                                meaning_elem = item.find('div', class_='meaning')
                                if acronym_elem and meaning_elem:
                                    acronym = acronym_elem.text.strip().lower()
                                    meaning = meaning_elem.text.strip().lower()
                                    if acronym and meaning:
                                        self.acronyms[acronym] = meaning
                    time.sleep(1)  # Be respectful with rate limiting
                except Exception as e:
                    logging.warning(f"Error scraping {url}: {str(e)}")
                    
        except Exception as e:
            logging.error(f"Error in scraping acronyms: {str(e)}")

    def _identify_language(self, word: str) -> str:
        """
        Use transformer model to identify the language of a word.
        
        Args:
            word: The word to identify
            
        Returns:
            Language tag ('Hindi', 'English', 'Unknown')
        """
        if not self.use_transformers:
            # Fallback to simple heuristics if transformers aren't available
            if word.lower() in self.hindi_mappings:
                return 'Hindi'
            elif word.lower() in self.short_forms:
                return 'Short Form'
            elif word.lower() in self.acronyms:
                return 'Acronym'
            else:
                return 'Looks Good'
        
        try:
            # Encode the word for the model
            inputs = self.tokenizer(word, return_tensors="pt")
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits.argmax(dim=-1)
            
            # Map the prediction to a language tag
            # This mapping depends on how the model was trained
            # You might need to adjust based on the specific model you use
            lang_map = {0: "Hindi", 1: "English", 2: "Unrecognizable or other language"}
            
            # Get the predicted language
            pred_lang_id = predictions.item()
            
            # Map to language tag
            if pred_lang_id in lang_map:
                return lang_map[pred_lang_id]
            else:
                return "Unknown"
                
        except Exception as e:
            logging.warning(f"Error in language identification for word '{word}': {str(e)}")
            # Fallback to dictionary lookup on error
            if word.lower() in self.hindi_mappings:
                return 'Hindi'
            elif word.lower() in self.short_forms:
                return 'Short Form'
            elif word.lower() in self.acronyms:
                return 'Acronym'
            else:
                return 'Looks Good'

    def _normalize_word_transformer(self, word: str, lang_tag: str) -> str:
        """
        Use transformer model to normalize a word based on its language tag.
        
        Args:
            word: The word to normalize
            lang_tag: The language tag ('Hindi', 'English', 'Unknown')
            
        Returns:
            Normalized word
        """
        if not self.use_transformers:
            # Fallback to dictionary if transformers aren't available
            return self._normalize_word_dictionary(word)
            
        try:
            word_lower = word.lower()
            
            # For short forms and acronyms, use dictionary lookup
            if lang_tag == 'Short Form' and word_lower in self.short_forms:
                return self.short_forms[word_lower]
            elif lang_tag == 'Acronym' and word_lower in self.acronyms:
                return self.acronyms[word_lower]
            
            # For Hindi words, use transformer-based normalization
            # This would ideally be a model fine-tuned for Hindi transliteration normalization
            if lang_tag == 'Hindi':
                # If the word is in our dictionary, use that first (more reliable)
                if word_lower in self.hindi_mappings:
                    return self.hindi_mappings[word_lower]
                
                # Otherwise use the transformer model
                # Note: This would be improved by using a model specifically fine-tuned for Hinglish normalization
                inputs = self.norm_tokenizer(word, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.norm_model.generate(**inputs)
                    
                normalized = self.norm_tokenizer.decode(outputs[0], skip_special_tokens=True)
                return normalized
            
            # For other languages/unknown, return as is
            return word
            
        except Exception as e:
            logging.warning(f"Error in transformer normalization for word '{word}': {str(e)}")
            # Fallback to dictionary normalization on error
            return self._normalize_word_dictionary(word)
            
    def _normalize_word_dictionary(self, word: str) -> str:
        """
        Use dictionary lookup to normalize a word.
        
        Args:
            word: The word to normalize
            
        Returns:
            Normalized word
        """
        word_lower = word.lower()
        
        if word_lower in self.short_forms:
            return self.short_forms[word_lower]
        elif word_lower in self.acronyms:
            return self.acronyms[word_lower]
        elif word_lower in self.hindi_mappings:
            return self.hindi_mappings[word_lower]
        else:
            return word

    def normalize_text(self, text: str) -> Tuple[str, List[str]]:
        """
        Normalize Hinglish text using transformers for language identification.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Tuple of (normalized_text, tags)
        """
        words = text.strip().split()
        normalized_words = []
        tags = []
        
        # Enhanced approach with transformers
        if self.use_transformers:
            # First, try to identify named entities which might not need normalization
            try:
                ner_results = self.ner_pipeline(text)
                entities = {}
                for entity in ner_results:
                    word = entity['word']
                    entities[word] = True
            except Exception as e:
                logging.warning(f"Error in NER pipeline: {str(e)}")
                entities = {}
                
            # Process each word
            for word in words:
                # Skip normalization for named entities
                if word in entities:
                    normalized_words.append(word)
                    tags.append('Named Entity')
                    continue
                
                # Identify the language of the word
                lang_tag = self._identify_language(word)
                tags.append(lang_tag)
                
                # Normalize based on language
                normalized_word = self._normalize_word_transformer(word, lang_tag)
                normalized_words.append(normalized_word)
        else:
            # Fallback to dictionary-based approach
            for word in words:
                word_lower = word.lower()
                
                if word_lower in self.acronyms:
                    normalized_words.append(self.acronyms[word_lower])
                    tags.append('Acronym')
                elif word_lower in self.short_forms:
                    normalized_words.append(self.short_forms[word_lower])
                    tags.append('Short Form')
                elif word_lower in self.hindi_mappings:
                    normalized_words.append(self.hindi_mappings[word_lower])
                    tags.append('Hindi')
                else:
                    normalized_words.append(word)
                    tags.append('Looks Good')
        
        return ' '.join(normalized_words), tags

    def process_text_file(self, input_file: str, output_file: str) -> None:
        """
        Process a text file containing Hinglish sentences.
        
        Args:
            input_file: Path to input text file (one sentence per line)
            output_file: Path to output JSON file
        """
        result = []
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for idx, line in enumerate(lines, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    normalized_text, tags = self.normalize_text(line)
                    
                    result.append({
                        'id': idx,
                        'inputText': line,
                        'tags': str(tags),
                        'normalizedText': normalized_text
                    })
                    
            # Save result to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4)
                
            logging.info(f"Processed {len(result)} sentences. Results saved to {output_file}")
            
        except Exception as e:
            logging.error(f"Error processing text file: {str(e)}")


def main():
    # Configure these paths
    input_file = "hinglish_sentences.txt"  # Path to your input text file (one sentence per line)
    output_file = "normalized_hinglish.json"  # Path for the output JSON file
    
    # Initialize the normalizer with transformers
    # Set use_transformers to False if you want to use only dictionary-based approach
    normalizer = HinglishNormalizer(use_cached_dicts=True, use_transformers=True)
    
    # Process the text file
    normalizer.process_text_file(input_file, output_file)
    
    # You can also normalize individual sentences directly
    sample_sentences = [
        "bahar hu abhi",
        "tarika hai bolne ka",
        "so break hai",
        "bihar you",
        "then plzz",
        "m saswat",
        "sab batao",
        "yaar jo bhi ho na mast ho tum",
        "or jo ahae dekhlenge",
        "bta degii"
    ]
    
    print("Sample normalizations:")
    for sentence in sample_sentences:
        normalized_text, tags = normalizer.normalize_text(sentence)
        print(f"Original: {sentence}")
        print(f"Normalized: {normalized_text}")
        print(f"Tags: {tags}")
        print("-" * 50)


if __name__ == "__main__":
    main()