import streamlit as st
import json
import os
import re
import sys
import time
from datetime import datetime
import io
from typing import Optional, Dict, List, Any, Tuple

# Constants
DEFAULT_MAX_TOKENS = 16000
DEFAULT_STANDARD_TOKENS = 4096
MIN_TOKENS = 100
DEFAULT_AUTOPILOT_SEGMENTS = 20
DEFAULT_MANUAL_INITIAL_SEGMENTS = 10  # Default number of segments for initial manual processing
DEFAULT_AUTOPILOT_DELAY = 1.0
BINARY_PARTS_COUNT = 2
DEFAULT_MODEL = "openai/gpt-5"
LOG_TIMESTAMP_PRECISION = 3

# Quality thresholds
SIMILARITY_THRESHOLD_MAJOR = 0.7  # Below this is considered major mismatch
SIMILARITY_THRESHOLD_MINOR = 0.9  # Below this is considered minor variation
 # MODERATE_WARNING_THRESHOLD is no longer used; moderate warnings will not cause stopping
CROSS_LANGUAGE_SIMILARITY_FACTOR = 0.7  # Reduced similarity for cross-language comparison

# Source validation thresholds
DEFAULT_SOURCE_OVERLAP_THRESHOLD = 0.7  # Minimum word overlap for source validation
LOW_SOURCE_OVERLAP_THRESHOLD = 0.3      # Threshold for severe vs moderate warnings
PHRASE_MIN_LENGTH = 3                    # Minimum phrase length for phrase matching

# Try to import requests, display helpful error if not installed
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    st.error("Error: The 'requests' module is not installed.")
    st.error("Please install it using: pip install requests")
    st.stop()

# Try to import python-docx for DOCX file support
try:
    from docx import Document
except ImportError:
    st.error("Error: The 'python-docx' module is not installed.")
    st.error("Please install it using: pip install python-docx")
    st.stop()

# Try to import openai for OpenAI API support (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

# Try to import google.generativeai for Gemini API support (optional)
try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    GenerationConfig = None

# OpenRouter support (uses requests which is already required)
OPENROUTER_AVAILABLE = REQUESTS_AVAILABLE

# Try to import additional libraries for semantic matching
try:
    from difflib import SequenceMatcher
    SEMANTIC_MATCHING_AVAILABLE = True
except ImportError:
    SEMANTIC_MATCHING_AVAILABLE = False
    SequenceMatcher = None

# Source validation classes for ensuring content matching (not generation)
class SourceValidationResult:
    def __init__(self, is_from_source: bool, word_overlap_score: float, 
                 phrase_match: bool, substring_match: bool, target_text: str):
        self.is_from_source = is_from_source
        self.word_overlap_score = word_overlap_score
        self.phrase_match = phrase_match
        self.substring_match = substring_match
        self.target_text = target_text

class GenerationDetectionResult:
    def __init__(self, likely_generated: bool, indicators_found: bool, source_validation):
        self.likely_generated = likely_generated
        self.indicators_found = indicators_found
        self.source_validation = source_validation

class SourceValidationIssue:
    def __init__(self, segment, issue_type: str, severity: str, details: str):
        self.segment = segment
        self.issue_type = issue_type
        self.severity = severity
        self.details = details

class SimpleFormatValidator:
    """Lightweight format enforcement - no complex logic"""
    
    REQUIRED_SEPARATOR = " || "
    
    def quick_format_check(self, segment_text: str) -> List[str]:
        """Fast format validation"""
        issues = []
        
        if self.REQUIRED_SEPARATOR not in segment_text:
            issues.append("missing_separator")
        
        if segment_text.count(self.REQUIRED_SEPARATOR) > 1:
            issues.append("multiple_separators")
        
        if self.REQUIRED_SEPARATOR in segment_text:
            english_part = segment_text.split(self.REQUIRED_SEPARATOR)[0]
            if "\n" in english_part:
                issues.append("english_linebreaks")
            
        return issues
    
    def auto_fix_simple_issues(self, segment_text: str) -> str:
        """Auto-fix obvious problems"""
        if "||" in segment_text and self.REQUIRED_SEPARATOR not in segment_text:
            segment_text = segment_text.replace("||", self.REQUIRED_SEPARATOR)
        
        # Remove internal line breaks in English part
        if self.REQUIRED_SEPARATOR in segment_text:
            parts = segment_text.split(self.REQUIRED_SEPARATOR, 1)
            english_part = parts[0].replace("\n", " ").strip()
            target_part = parts[1] if len(parts) > 1 else ""
            segment_text = f"{english_part}{self.REQUIRED_SEPARATOR}{target_part}"
        
        return segment_text

class SourceTextValidator:
    def __init__(self, source_translation: str):
        self.source_translation = source_translation.lower()
        self.source_words = set(word.strip('.,!?;:"()[]{}') for word in source_translation.lower().split())
        self.source_phrases = self._extract_phrases(source_translation)
    
    def _extract_phrases(self, text: str, min_length: int = PHRASE_MIN_LENGTH) -> List[str]:
        """Extract meaningful phrases from text"""
        words = text.lower().split()
        phrases = []
        for i in range(len(words) - min_length + 1):
            phrase = ' '.join(words[i:i + min_length])
            phrases.append(phrase)
        return phrases
    
    def validate_text_is_from_source(self, target_text: str, threshold: float = DEFAULT_SOURCE_OVERLAP_THRESHOLD) -> SourceValidationResult:
        """Verify the target text comes from source translation"""
        if not target_text.strip():
            return SourceValidationResult(False, 0.0, False, False, target_text)
        
        target_words = set(word.strip('.,!?;:"()[]{}') for word in target_text.lower().split())
        
        # Check word overlap
        word_overlap = len(target_words.intersection(self.source_words)) / len(target_words) if target_words else 0
        
        # Check for exact phrase matches
        phrase_found = any(phrase in self.source_translation for phrase in self._extract_phrases(target_text))
        
        # Check for substring presence (more lenient)
        clean_target = target_text.lower().strip()
        substring_match = clean_target in self.source_translation or any(
            clean_target in sentence for sentence in self.source_translation.split('.')
        )
        
        is_from_source = word_overlap >= threshold or phrase_found or substring_match
        
        return SourceValidationResult(
            is_from_source=is_from_source,
            word_overlap_score=word_overlap,
            phrase_match=phrase_found,
            substring_match=substring_match,
            target_text=target_text
        )
    
    def detect_generated_content(self, target_text: str) -> GenerationDetectionResult:
        """Detect if content appears to be AI-generated rather than matched"""
        # Common AI generation indicators
        generation_indicators = [
            "based on the context",
            "this appears to be",
            "the text suggests",
            "[translation]",
            "[generated]",
            "approximately translates to",
            "no_match_found",
            "[no_match_found]"
        ]
        
        text_lower = target_text.lower()
        has_indicators = any(indicator in text_lower for indicator in generation_indicators)
        
        # Check source validation
        source_validation = self.validate_text_is_from_source(target_text)
        
        return GenerationDetectionResult(
            likely_generated=has_indicators or not source_validation.is_from_source,
            indicators_found=has_indicators,
            source_validation=source_validation
        )

def debug_log(message: str, level: str = "INFO"):
    """Log debug messages to the terminal output"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-LOG_TIMESTAMP_PRECISION]  # Include milliseconds
    level_emoji = {
        "INFO": "ℹ️",
        "SUCCESS": "✅", 
        "WARNING": "⚠️",
        "ERROR": "❌",
        "DEBUG": "🔍",
        "API": "🌐",
        "PROCESS": "⚙️"
    }.get(level, "📝")
    
    # Print to stdout so it appears in terminal
    print(f"[{timestamp}] {level_emoji} {level}: {message}")
    
    # Also flush to ensure immediate output
    sys.stdout.flush()

# Configure page
st.set_page_config(
    page_title="Bilingual Subtitle Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced prompt with source matching protection
DEFAULT_PROMPT = """You are a TEXT MATCHER, not a translator. Your job is to MATCH existing translation text to English segments.

🚨 FORBIDDEN: Do not generate, create, or translate any new text.
✅ REQUIRED: ONLY use text that exists in the provided translation below.

🚨 CRITICAL SOURCE MATCHING RULES:
- ONLY use text that appears in the translation below
- Do NOT translate, generate, or create any new text
- Each English segment maps to EXACTLY ONE portion of the existing translation
- Use translation text in sequential order - NO jumping around
- NO reusing translation words - each word used exactly once
- STOP immediately when approaching word limit, even mid-sentence
- If you cannot find matching text, use [NO_MATCH_FOUND]

🎯 PROCESSING REQUIREMENTS:
- Process ONLY the segments specified in "FIRST X SEGMENTS TO MAP" section
- Generate bilingual output for EXACTLY that number of segments
- Keep English text on SINGLE LINES (no line breaks within English text)
- Preserve exact timestamps and English content

REQUIRED OUTPUT FORMAT:
Use standard SRT format with exact timestamps:
1
00:00:01,020 --> 00:00:05,550
English text || Target text from translation (CALCULATED WORD LIMIT APPLIES)

2
00:00:05,551 --> 00:00:09,200
Next English text || Next target text from translation (CALCULATED WORD LIMIT APPLIES)

CRITICAL FORMATTING REQUIREMENTS:
- Number each segment on its own line: "1" then newline, "2" then newline, etc.
- Use exact timestamp format: HH:MM:SS,mmm --> HH:MM:SS,mmm
- Separate languages with " || " (space-pipe-pipe-space)
- Keep target segments within CALCULATED WORD LIMITS (based on English segment length and language ratio)
- Preserve ALL original English text exactly but on SINGLE LINES
- Keep each subtitle on a SINGLE LINE (no line breaks)
- Maintain chronological order

🎯 SOURCE MATCHING ENFORCEMENT:
- Count target words as you write: 1, 2, 3... STOP at calculated limit
- Use ONLY words that exist in the translation source below
- Prefer complete phrases but respect the calculated word limit absolutely
- Better to end mid-sentence than exceed calculated word limit OR use non-source text
- Sequential processing: use next unused portion of translation for each segment

Translation Source (ONLY use text from here):
{{TRANSLATION}}

Process ONLY the specified English segments by MATCHING them to portions of the translation source above."""


def initialize_session_state():
    """Initialize session state variables"""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""
    
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = ""
    
    if 'openrouter_api_key' not in st.session_state:
        st.session_state.openrouter_api_key = ""
    
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = "openrouter"
    
    if 'model' not in st.session_state:
        st.session_state.model = DEFAULT_MODEL
    
    if 'max_tokens' not in st.session_state:
        st.session_state.max_tokens = DEFAULT_MAX_TOKENS
    
    if 'english_sub_content' not in st.session_state:
        st.session_state.english_sub_content = ""
    
    if 'translation_content' not in st.session_state:
        st.session_state.translation_content = ""
    
    if 'prompt_template' not in st.session_state:
        st.session_state.prompt_template = DEFAULT_PROMPT
    
    if 'show_api_key' not in st.session_state:
        st.session_state.show_api_key = False
    
    if 'show_openai_key' not in st.session_state:
        st.session_state.show_openai_key = False
    
    if 'show_gemini_key' not in st.session_state:
        st.session_state.show_gemini_key = False
    
    if 'show_openrouter_key' not in st.session_state:
        st.session_state.show_openrouter_key = False
    
    if 'inputs_validated' not in st.session_state:
        st.session_state.inputs_validated = False
    
    if 'target_language' not in st.session_state:
        st.session_state.target_language = ""
    
    if 'language_confirmed' not in st.session_state:
        st.session_state.language_confirmed = False
    
    if 'autopilot_running' not in st.session_state:
        st.session_state.autopilot_running = False
    
    if 'autopilot_continue_next_batch' not in st.session_state:
        st.session_state.autopilot_continue_next_batch = False
    
    if 'autopilot_segments' not in st.session_state:
        st.session_state.autopilot_segments = DEFAULT_AUTOPILOT_SEGMENTS  # Standard batch size for balanced speed and quality
    
    if 'autopilot_delay' not in st.session_state:
        st.session_state.autopilot_delay = DEFAULT_AUTOPILOT_DELAY
    
    if 'autopilot_status' not in st.session_state:
        st.session_state.autopilot_status = {"last_batch": 0, "total_processed": 0, "warnings_count": 0, "last_processed": 0}
    
    if 'show_full_conversation' not in st.session_state:
        st.session_state.show_full_conversation = False
    
    if 'conversation_needs_refresh' not in st.session_state:
        st.session_state.conversation_needs_refresh = False
    
    if 'quality_report' not in st.session_state:
        st.session_state.quality_report = {
            'warnings': [],
            'severe_warnings': [],
            'moderate_warnings': [],
            'total_segments_processed': 0,
            'segments_with_issues': 0,
            'last_updated': None
        }

def setup_sidebar():
    """Setup the sidebar with API configuration"""
    with st.sidebar:
        st.header("🔧 LLM Configuration")
        
        # LLM Provider selection
        st.subheader("AI Provider")
        providers = ["openrouter", "claude", "openai", "gemini"]
        provider_labels = {
            "claude": "Claude (Anthropic)",
            "openai": "GPT (OpenAI)", 
            "gemini": "Gemini (Google)",
            "openrouter": "OpenRouter (Multi-Model)"
        }
        
        # Check availability and create options
        available_providers = []  # Start with empty list
        if OPENROUTER_AVAILABLE:
            available_providers.append("openrouter")
        available_providers.append("claude")  # Claude is always available via requests
        if OPENAI_AVAILABLE:
            available_providers.append("openai")
        if GEMINI_AVAILABLE:
            available_providers.append("gemini")
        
        # Show availability warnings
        if not OPENAI_AVAILABLE:
            st.warning("⚠️ OpenAI support unavailable. Install: `pip install openai`")
        if not GEMINI_AVAILABLE:
            st.warning("⚠️ Gemini support unavailable. Install: `pip install google-generativeai`")
        if not OPENROUTER_AVAILABLE:
            st.warning("⚠️ OpenRouter support unavailable. Install: `pip install requests`")
        
        selected_provider = st.selectbox(
            "Choose AI Provider",
            available_providers,
            format_func=lambda x: provider_labels[x],
            index=available_providers.index(st.session_state.llm_provider) if st.session_state.llm_provider in available_providers else 0
        )
        st.session_state.llm_provider = selected_provider
        
        # API Key inputs based on selected provider
        st.subheader("API Configuration")
        
        if selected_provider == "claude":
            show_key = st.checkbox("Show Claude API Key", value=st.session_state.show_api_key)
            st.session_state.show_api_key = show_key
            
            api_key_type = "default" if show_key else "password"
            api_key = st.text_input(
                "Claude API Key",
                value=st.session_state.api_key,
                type=api_key_type,
                help="Get your API key from: https://console.anthropic.com/settings/keys"
            )
            st.session_state.api_key = api_key
            
        elif selected_provider == "openai":
            show_key = st.checkbox("Show OpenAI API Key", value=st.session_state.show_openai_key)
            st.session_state.show_openai_key = show_key
            
            api_key_type = "default" if show_key else "password"
            openai_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.openai_api_key,
                type=api_key_type,
                help="Get your API key from: https://platform.openai.com/api-keys"
            )
            st.session_state.openai_api_key = openai_key
            
        elif selected_provider == "gemini":
            show_key = st.checkbox("Show Gemini API Key", value=st.session_state.show_gemini_key)
            st.session_state.show_gemini_key = show_key
            
            api_key_type = "default" if show_key else "password"
            gemini_key = st.text_input(
                "Gemini API Key",
                value=st.session_state.gemini_api_key,
                type=api_key_type,
                help="Get your API key from: https://makersuite.google.com/app/apikey"
            )
            st.session_state.gemini_api_key = gemini_key
            
        elif selected_provider == "openrouter":
            show_key = st.checkbox("Show OpenRouter API Key", value=st.session_state.show_openrouter_key)
            st.session_state.show_openrouter_key = show_key
            
            api_key_type = "default" if show_key else "password"
            openrouter_key = st.text_input(
                "OpenRouter API Key",
                value=st.session_state.openrouter_api_key,
                type=api_key_type,
                help="Get your API key from: https://openrouter.ai/keys"
            )
            st.session_state.openrouter_api_key = openrouter_key
        
        # Model selection based on provider
        st.subheader("Model Settings")
        
        models = []
        default_model = ""
        max_tokens_range = (MIN_TOKENS, 4096)  # Default range
        default_tokens = DEFAULT_STANDARD_TOKENS
        
        if selected_provider == "claude":
            models = [
                "claude-3-7-sonnet-20250219",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-sonnet-20240620",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229", 
                "claude-3-haiku-20240307"
            ]
            default_model = DEFAULT_MODEL
            max_tokens_range = (MIN_TOKENS, 8192)
            default_tokens = DEFAULT_STANDARD_TOKENS
            
        elif selected_provider == "openai":
            models = [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4.1-nano",  # New nano model
                "gpt-5",         # New GPT-5 model
                "o1-preview",
                "o1-mini",
                "gpt-4-turbo",
                "gpt-4-turbo-preview",
                "gpt-4",
                "gpt-3.5-turbo"
            ]
            default_model = "gpt-4o"
            max_tokens_range = (MIN_TOKENS, 16384)
            default_tokens = DEFAULT_STANDARD_TOKENS
            
        elif selected_provider == "gemini":
            models = [
                "gemini-2.0-flash-exp",
                "gemini-1.5-pro",
                "gemini-1.5-pro-002",
                "gemini-1.5-flash",
                "gemini-1.5-flash-002",
                "gemini-1.5-flash-8b",
                "gemini-1.0-pro"
            ]
            default_model = "gemini-2.0-flash-exp"
            max_tokens_range = (MIN_TOKENS, 8192)
            default_tokens = DEFAULT_STANDARD_TOKENS
            
        elif selected_provider == "openrouter":
            models = [
                # OpenAI models via OpenRouter                
                "openai/gpt-4o",
                "openai/gpt-4o-mini",
                "openai/gpt-4.1-nano",
                "openai/gpt-5",
                "openai/o1-preview",
                "openai/o1-mini",
                "openai/gpt-4-turbo",
                "openai/gpt-4",
                "openai/gpt-3.5-turbo",
                
                # Anthropic models via OpenRouter
                "anthropic/claude-3.5-sonnet",
                "anthropic/claude-3-opus",
                "anthropic/claude-3-sonnet",
                "anthropic/claude-3-haiku",
                
                # Google models via OpenRouter
                "google/gemini-pro-1.5",
                "google/gemini-flash-1.5",
                
                # Other popular models
                "meta-llama/llama-3.1-405b-instruct",
                "meta-llama/llama-3.1-70b-instruct",
                "meta-llama/llama-3.1-8b-instruct",
                "mistralai/mistral-large",
                "mistralai/mistral-medium",
                "cohere/command-r-plus",
                "qwen/qwen-2.5-72b-instruct",
                "qwen/qwen-2.5-coder-32b-instruct",
                "qwen/qwen3-235b-a22b-2507",
                "deepseek/deepseek-chat",
                "perplexity/llama-3.1-sonar-large-128k-online",
                "openai/gpt-oss-120b",
                "openai/gpt-5-nano",
            ]
            default_model = "openai/gpt-5"
            max_tokens_range = (MIN_TOKENS, 16384)
            default_tokens = DEFAULT_STANDARD_TOKENS
        
        # Update model if provider changed
        if st.session_state.model not in models:
            st.session_state.model = default_model
        
        selected_model = st.selectbox(
            f"{provider_labels[selected_provider]} Model",
            models,
            index=models.index(st.session_state.model) if st.session_state.model in models else 0
        )
        st.session_state.model = selected_model
        
        # Show model info for OpenRouter
        if selected_provider == "openrouter":
            st.info("💡 **OpenRouter Info:** Access to multiple AI providers through one API. Pricing varies by model.")
        
        # Max tokens (adjust range based on provider)
        # Reset max_tokens if out of range
        if st.session_state.max_tokens < max_tokens_range[0] or st.session_state.max_tokens > max_tokens_range[1]:
            st.session_state.max_tokens = default_tokens
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=max_tokens_range[0],
            max_value=max_tokens_range[1],
            value=st.session_state.max_tokens,
            step=100,
            help="Maximum number of tokens for the response"
        )
        st.session_state.max_tokens = max_tokens
        
        # Save configuration button
        if st.button("💾 Save Configuration", type="primary"):
            current_key = get_current_api_key()
            if not current_key:
                st.error(f"API Key is required for {provider_labels[selected_provider]}!")
            else:
                st.success("Configuration saved!")
                
        # Show current configuration status
        current_key = get_current_api_key()
        if current_key:
            st.success(f"✅ {provider_labels[selected_provider]} API Key configured")
        else:
            st.warning(f"⚠️ Please configure your {provider_labels[selected_provider]} API key")

def get_current_api_key():
    """Get the current API key based on selected provider with validation"""
    key = None
    
    if st.session_state.llm_provider == "claude":
        key = st.session_state.api_key
        # Basic validation for Claude API key format
        if key and not (key.startswith("sk-ant-") and len(key) > 50):
            return None
    elif st.session_state.llm_provider == "openai":
        key = st.session_state.openai_api_key
        # Basic validation for OpenAI API key format
        if key and not (key.startswith("sk-") and len(key) > 40):
            return None
    elif st.session_state.llm_provider == "gemini":
        key = st.session_state.gemini_api_key
        # Basic validation for Gemini API key format
        if key and len(key) < 30:
            return None
    elif st.session_state.llm_provider == "openrouter":
        key = st.session_state.openrouter_api_key
        # Basic validation for OpenRouter API key format
        if key and not (key.startswith("sk-or-") and len(key) > 30):
            return None
    
    return key if key and key.strip() else None

def call_llm_api(prompt: str) -> Dict[str, Any]:
    """Call the selected LLM API with the given prompt"""
    debug_log(f"Starting API call with provider: {st.session_state.llm_provider}", "API")
    debug_log(f"Model: {st.session_state.model}, Max tokens: {st.session_state.max_tokens}", "API")
    debug_log(f"Prompt length: {len(prompt)} characters", "API")
    
    start_time = time.time()
    
    if st.session_state.llm_provider == "claude":
        result = call_claude_api(prompt)
    elif st.session_state.llm_provider == "openai":
        result = call_openai_api(prompt)
    elif st.session_state.llm_provider == "gemini":
        result = call_gemini_api(prompt)
    elif st.session_state.llm_provider == "openrouter":
        result = call_openrouter_api(prompt)
    else:
        debug_log(f"Unknown LLM provider: {st.session_state.llm_provider}", "ERROR")
        return {"error": {"message": "Unknown LLM provider"}}
    
    elapsed_time = time.time() - start_time
    
    if "error" in result:
        debug_log(f"API call failed after {elapsed_time:.2f}s: {result['error'].get('message', 'Unknown error')}", "ERROR")
    else:
        response_text = extract_api_response(result)
        debug_log(f"API call successful after {elapsed_time:.2f}s, response length: {len(response_text)} characters", "SUCCESS")
    
    return result

def call_llm_api_with_messages(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Call the selected LLM API with a full message history"""
    debug_log(f"Starting API call with messages - provider: {st.session_state.llm_provider}", "API")
    debug_log(f"Model: {st.session_state.model}, Max tokens: {st.session_state.max_tokens}", "API")
    debug_log(f"Message history length: {len(messages)} messages", "API")
    
    # Log message summary for debugging
    total_chars = sum(len(msg.get("content", "")) for msg in messages)
    debug_log(f"Total conversation length: {total_chars} characters", "API")
    
    start_time = time.time()
    
    if st.session_state.llm_provider == "claude":
        result = call_claude_api_with_messages(messages)
    elif st.session_state.llm_provider == "openai":
        result = call_openai_api_with_messages(messages)
    elif st.session_state.llm_provider == "gemini":
        result = call_gemini_api_with_messages(messages)
    elif st.session_state.llm_provider == "openrouter":
        result = call_openrouter_api_with_messages(messages)
    else:
        debug_log(f"Unknown LLM provider: {st.session_state.llm_provider}", "ERROR")
        return {"error": {"message": "Unknown LLM provider"}}
    
    elapsed_time = time.time() - start_time
    
    if "error" in result:
        debug_log(f"API call with messages failed after {elapsed_time:.2f}s: {result['error'].get('message', 'Unknown error')}", "ERROR")
    else:
        response_text = extract_api_response(result)
        debug_log(f"API call with messages successful after {elapsed_time:.2f}s, response length: {len(response_text)} characters", "SUCCESS")
    
    return result

def call_claude_api(prompt: str) -> Dict[str, Any]:
    """Call the Claude API with the given prompt"""
    debug_log("Making Claude API request...", "API")
    url = "https://api.anthropic.com/v1/messages"
    
    headers = {
        "x-api-key": st.session_state.api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    data = {
        "model": st.session_state.model,
        "max_tokens": st.session_state.max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    debug_log(f"Claude API URL: {url}", "DEBUG")
    debug_log(f"Claude API data size: {len(json.dumps(data))} bytes", "DEBUG")
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=120)
        debug_log(f"Claude API response status: {response.status_code}", "DEBUG")
        response.raise_for_status()
        
        result = response.json()
        debug_log("Claude API request completed successfully", "SUCCESS")
        return result
    except requests.exceptions.RequestException as e:
        debug_log(f"Claude API request failed: {str(e)}", "ERROR")
        return {"error": {"message": f"Request failed: {str(e)}"}}

def call_claude_api_with_messages(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Call the Claude API with a full message history"""
    debug_log(f"Making Claude API request with {len(messages)} messages...", "API")
    url = "https://api.anthropic.com/v1/messages"
    
    headers = {
        "x-api-key": st.session_state.api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    data = {
        "model": st.session_state.model,
        "max_tokens": st.session_state.max_tokens,
        "messages": messages
    }
    
    debug_log(f"Claude API URL: {url}", "DEBUG")
    debug_log(f"Claude API data size: {len(json.dumps(data))} bytes", "DEBUG")
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=120)
        debug_log(f"Claude API response status: {response.status_code}", "DEBUG")
        response.raise_for_status()
        
        result = response.json()
        debug_log("Claude API request with messages completed successfully", "SUCCESS")
        return result
    except requests.exceptions.RequestException as e:
        debug_log(f"Claude API request with messages failed: {str(e)}", "ERROR")
        return {"error": {"message": f"Request failed: {str(e)}"}}

def call_openai_api(prompt: str) -> Dict[str, Any]:
    """Call the OpenAI API with the given prompt"""
    if not OPENAI_AVAILABLE:
        debug_log("OpenAI library not available", "ERROR")
        return {"error": {"message": "OpenAI library not installed. Please install with: pip install openai"}}
    
    debug_log("Making OpenAI API request...", "API")
    
    try:
        if openai is None:
            debug_log("OpenAI library not available", "ERROR")
            return {"error": {"message": "OpenAI library not available"}}
            
        client = openai.OpenAI(api_key=st.session_state.openai_api_key)
        debug_log(f"OpenAI API request with model: {st.session_state.model}", "DEBUG")
        
        response = client.chat.completions.create(
            model=st.session_state.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=st.session_state.max_tokens,
            timeout=120
        )
        
        debug_log("OpenAI API request completed successfully", "SUCCESS")
        
        # Convert to standard format
        return {
            "content": [{"text": response.choices[0].message.content}]
        }
    except Exception as e:
        debug_log(f"OpenAI API error: {str(e)}", "ERROR")
        return {"error": {"message": f"OpenAI API error: {str(e)}"}}

def call_openai_api_with_messages(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Call the OpenAI API with a full message history"""
    if not OPENAI_AVAILABLE:
        debug_log("OpenAI library not available", "ERROR")
        return {"error": {"message": "OpenAI library not installed. Please install with: pip install openai"}}
    
    debug_log(f"Making OpenAI API request with {len(messages)} messages...", "API")
    
    try:
        if openai is None:
            debug_log("OpenAI library not available", "ERROR")
            return {"error": {"message": "OpenAI library not available"}}
            
        client = openai.OpenAI(api_key=st.session_state.openai_api_key)
        debug_log(f"OpenAI API request with model: {st.session_state.model}", "DEBUG")
        
        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            openai_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        debug_log(f"OpenAI API converted {len(openai_messages)} messages", "DEBUG")
        
        response = client.chat.completions.create(
            model=st.session_state.model,
            messages=openai_messages,
            max_tokens=st.session_state.max_tokens,
            timeout=120
        )
        
        debug_log("OpenAI API request with messages completed successfully", "SUCCESS")
        
        # Convert to standard format
        return {
            "content": [{"text": response.choices[0].message.content}]
        }
    except Exception as e:
        debug_log(f"OpenAI API error: {str(e)}", "ERROR")
        return {"error": {"message": f"OpenAI API error: {str(e)}"}}

def call_gemini_api(prompt: str) -> Dict[str, Any]:
    """Call the Gemini API with comprehensive error handling"""
    if not GEMINI_AVAILABLE:
        debug_log("Gemini library not available", "ERROR")
        return {"error": {"message": "Gemini library not installed. Please install with: pip install google-generativeai"}}
    
    debug_log("Making Gemini API request...", "API")
    
    try:
        if not GEMINI_AVAILABLE or genai is None:
            debug_log("Gemini library not available", "ERROR")
            return {"error": {"message": "Gemini library not available"}}
        
        # Comprehensive API availability checking
        if not hasattr(genai, 'configure'):
            debug_log("genai.configure not available in current version", "ERROR")
            return {"error": {"message": "Gemini configure function not available. Please update google-generativeai package"}}
        
        if not hasattr(genai, 'GenerativeModel'):
            debug_log("genai.GenerativeModel not available in current version", "ERROR")  
            return {"error": {"message": "Gemini GenerativeModel not available. Please update google-generativeai package"}}
            
        # Safe API usage - guaranteed to work
        genai.configure(api_key=st.session_state.gemini_api_key)
        model = genai.GenerativeModel(st.session_state.model)
        debug_log(f"Gemini API request with model: {st.session_state.model}", "DEBUG")
        
        # Configure generation settings
        generation_config = GenerationConfig(
            max_output_tokens=st.session_state.max_tokens,
            temperature=0.1
        ) if GenerationConfig else None
        
        if generation_config:
            debug_log("Using Gemini generation config", "DEBUG")
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
        else:
            debug_log("Using default Gemini settings", "DEBUG")
            response = model.generate_content(prompt)
        
        debug_log("Gemini API request completed successfully", "SUCCESS")
        
        # Convert to standard format
        return {
            "content": [{"text": response.text}]
        }
    except Exception as e:
        debug_log(f"Gemini API error: {str(e)}", "ERROR")
        return {"error": {"message": f"Gemini API error: {str(e)}"}}

def call_gemini_api_with_messages(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Call the Gemini API with a full message history"""
    if not GEMINI_AVAILABLE:
        debug_log("Gemini library not available", "ERROR")
        return {"error": {"message": "Gemini library not installed. Please install with: pip install google-generativeai"}}
    
    debug_log(f"Making Gemini API request with {len(messages)} messages...", "API")
    
    try:
        if not GEMINI_AVAILABLE or genai is None:
            debug_log("Gemini library not available", "ERROR")
            return {"error": {"message": "Gemini library not available"}}
        
        # Comprehensive API availability checking
        if not hasattr(genai, 'configure'):
            debug_log("genai.configure not available in current version", "ERROR")
            return {"error": {"message": "Gemini configure function not available. Please update google-generativeai package"}}
        
        if not hasattr(genai, 'GenerativeModel'):
            debug_log("genai.GenerativeModel not available in current version", "ERROR")  
            return {"error": {"message": "Gemini GenerativeModel not available. Please update google-generativeai package"}}
            
        # Safe API usage - guaranteed to work
        genai.configure(api_key=st.session_state.gemini_api_key)
        model = genai.GenerativeModel(st.session_state.model)
        debug_log(f"Gemini API request with model: {st.session_state.model}", "DEBUG")
        
        # Configure generation settings
        generation_config = GenerationConfig(
            max_output_tokens=st.session_state.max_tokens,
            temperature=0.1
        ) if GenerationConfig else None
        
        # Convert messages to Gemini format
        chat = model.start_chat(history=[])
        debug_log(f"Starting Gemini chat with {len(messages)-1} history messages", "DEBUG")
        
        # Build conversation history
        for i, msg in enumerate(messages[:-1]):  # All but the last message
            if msg["role"] == "user":
                if generation_config:
                    chat.send_message(msg["content"], generation_config=generation_config)
                else:
                    chat.send_message(msg["content"])
        
        # Send the final message and get response
        final_message = messages[-1]["content"]
        debug_log(f"Sending final message to Gemini ({len(final_message)} chars)", "DEBUG")
        
        if generation_config:
            response = chat.send_message(final_message, generation_config=generation_config)
        else:
            response = chat.send_message(final_message)
        
        debug_log("Gemini API request with messages completed successfully", "SUCCESS")
        
        # Convert to standard format
        return {
            "content": [{"text": response.text}]
        }
    except Exception as e:
        debug_log(f"Gemini API error: {str(e)}", "ERROR")
        return {"error": {"message": f"Gemini API error: {str(e)}"}}

def call_openrouter_api(prompt: str) -> Dict[str, Any]:
    """Call the OpenRouter API with the given prompt"""
    debug_log("Making OpenRouter API request...", "API")
    
    if not OPENROUTER_AVAILABLE:
        debug_log("OpenRouter library not available", "ERROR")
        return {"error": {"message": "Requests library not available. Please install with: pip install requests"}}
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {st.session_state.openrouter_api_key}",
        "HTTP-Referer": "https://github.com/subtitle-generator",  # Optional
        "X-Title": "Bilingual Subtitle Generator",  # Optional
        "Content-Type": "application/json"
    }
    
    data = {
        "model": st.session_state.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": st.session_state.max_tokens,
        "temperature": 0.1,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    
    debug_log(f"OpenRouter API URL: {url}", "DEBUG")
    debug_log(f"OpenRouter API model: {st.session_state.model}", "DEBUG")
    debug_log(f"OpenRouter API data size: {len(json.dumps(data))} bytes", "DEBUG")
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=120)
        debug_log(f"OpenRouter API response status: {response.status_code}", "DEBUG")
        
        if response.status_code == 200:
            result = response.json()
            debug_log("OpenRouter API request completed successfully", "SUCCESS")
            
            # Convert to standard format
            if "choices" in result and len(result["choices"]) > 0:
                return {
                    "content": [{"text": result["choices"][0]["message"]["content"]}]
                }
            else:
                return {"error": {"message": "No response content from OpenRouter"}}
        else:
            error_detail = response.text
            debug_log(f"OpenRouter API error {response.status_code}: {error_detail}", "ERROR")
            return {"error": {"message": f"OpenRouter API error {response.status_code}: {error_detail}"}}
            
    except requests.exceptions.RequestException as e:
        debug_log(f"OpenRouter API request failed: {str(e)}", "ERROR")
        return {"error": {"message": f"Request failed: {str(e)}"}}

def call_openrouter_api_with_messages(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Call the OpenRouter API with a full message history"""
    debug_log(f"Making OpenRouter API request with {len(messages)} messages...", "API")
    
    if not OPENROUTER_AVAILABLE:
        debug_log("OpenRouter library not available", "ERROR")
        return {"error": {"message": "Requests library not available. Please install with: pip install requests"}}
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {st.session_state.openrouter_api_key}",
        "HTTP-Referer": "https://github.com/subtitle-generator",  # Optional
        "X-Title": "Bilingual Subtitle Generator",  # Optional
        "Content-Type": "application/json"
    }
    
    # Convert messages to OpenRouter format (same as OpenAI format)
    openrouter_messages = []
    for msg in messages:
        openrouter_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    data = {
        "model": st.session_state.model,
        "messages": openrouter_messages,
        "max_tokens": st.session_state.max_tokens,
        "temperature": 0.1,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    
    debug_log(f"OpenRouter API URL: {url}", "DEBUG")
    debug_log(f"OpenRouter API model: {st.session_state.model}", "DEBUG")
    debug_log(f"OpenRouter API converted {len(openrouter_messages)} messages", "DEBUG")
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=120)
        debug_log(f"OpenRouter API response status: {response.status_code}", "DEBUG")
        
        if response.status_code == 200:
            result = response.json()
            debug_log("OpenRouter API request with messages completed successfully", "SUCCESS")
            
            # Convert to standard format
            if "choices" in result and len(result["choices"]) > 0:
                return {
                    "content": [{"text": result["choices"][0]["message"]["content"]}]
                }
            else:
                return {"error": {"message": "No response content from OpenRouter"}}
        else:
            error_detail = response.text
            debug_log(f"OpenRouter API error {response.status_code}: {error_detail}", "ERROR")
            return {"error": {"message": f"OpenRouter API error {response.status_code}: {error_detail}"}}
            
    except requests.exceptions.RequestException as e:
        debug_log(f"OpenRouter API request with messages failed: {str(e)}", "ERROR")
        return {"error": {"message": f"Request failed: {str(e)}"}}

def calculate_progressive_word_limit(eng_words: int, language: str = "default") -> int:
    """
    Calculate progressive word limit with percentage-based buffer reduction.
    
    Model C: Percentage-Based Progressive System
    - For eng_words ≤ 15: buffer = +5 words (current system)
    - For eng_words > 15: buffer = max(1, (35 - eng_words) / 4)
    - Absolute maximum: 30 words to prevent severe bleeding
    
    Language-specific adjustments:
    - German: +20% buffer (compound words need more space)
    - Chinese/Japanese: -20% buffer (more compact languages)
    - Romance languages: Standard buffer
    
    Examples:
    - 5 words → +5 buffer → max 10 words
    - 15 words → +5 buffer → max 20 words
    - 20 words → +3.75 buffer → max 23-24 words
    - 25 words → +2.5 buffer → max 27-28 words  
    - 30 words → +1.25 buffer → max 31-32 words (capped at 30)
    """
    debug_log(f"Calculating progressive word limit for {eng_words} English words, language: {language}", "DEBUG")
    
    # Minimum word limit (always allow at least 3 words)
    min_limit = 3
    
    # Language-specific buffer multipliers
    language_multipliers = {
        'german': 1.2,       # German compound words need more space
        'finnish': 1.2,      # Agglutinative language
        'hungarian': 1.2,    # Agglutinative language
        'chinese': 0.8,      # More compact
        'japanese': 0.8,     # More compact
        'korean': 0.8,       # More compact
        'default': 1.0       # Standard multiplier
    }
    
    multiplier = language_multipliers.get(language.lower(), language_multipliers['default'])
    
    # Standard range (1-15 words): Use current +5 buffer system
    if eng_words <= 15:
        base_buffer = 5
        adjusted_buffer = int(base_buffer * multiplier)
        # Allow full buffer (up to 20 words for 15 English words), not capped at 15
        standard_limit = max(min_limit, eng_words + adjusted_buffer)
        debug_log(f"Standard range: {eng_words} words + {adjusted_buffer} buffer (×{multiplier}) → {standard_limit} limit", "DEBUG")
        return standard_limit
    
    # Extended range (16-30 words): Progressive percentage-based buffer
    elif eng_words <= 30:
        # Progressive buffer formula: max(1, (35 - eng_words) / 4)
        base_progressive_buffer = max(1, (35 - eng_words) / 4)
        adjusted_progressive_buffer = base_progressive_buffer * multiplier
        extended_limit = min(30, int(eng_words + adjusted_progressive_buffer))
        debug_log(f"Extended range: {eng_words} words + {adjusted_progressive_buffer:.2f} buffer (×{multiplier}) → {extended_limit} limit", "DEBUG")
        return extended_limit
    
    # Very long segments (30+ words): Hard cap at 30 words (or language-adjusted)
    else:
        # Allow slight adjustment for language-specific needs but maintain control
        max_cap = min(35, int(30 * multiplier)) if multiplier > 1.0 else 30
        debug_log(f"Hard cap: {eng_words} words → {max_cap} limit (language-adjusted maximum)", "WARNING")
        return max_cap

def extract_api_response(response: Dict[str, Any]) -> str:
    """Extract the content from any LLM API response"""
    if isinstance(response, dict) and 'error' in response:
        return f"API Error: {response['error'].get('message', 'Unknown error')}"
    
    try:
        # Standard format used by all providers
        if 'content' in response and isinstance(response['content'], list):
            return response['content'][0]['text']
        # Fall back to checking 'completion' for older API versions
        elif 'completion' in response:
            return response['completion']
        else:
            return f"Could not parse response format. Raw response: {json.dumps(response, indent=2)}"
    except (KeyError, IndexError) as e:
        return f"Error extracting content from response: {str(e)}\nRaw response: {json.dumps(response, indent=2)}"

def get_all_assistant_messages() -> str:
    """Helper function to extract all assistant messages from conversation"""
    return "\n".join([entry['message'] for entry in st.session_state.conversation if entry['speaker'] == 'Assistant'])

def parse_and_update_conversation(conversation_text: str) -> None:
    """
    Parse edited conversation text and update session state
    
    Args:
        conversation_text: The edited conversation text from the UI
    """
    try:
        # Split conversation into entries based on separator lines
        sections = conversation_text.split('=' * 80)
        
        new_conversation = []
        
        for section in sections[1:]:  # Skip first empty section
            section = section.strip()
            if not section:
                continue
                
            # Find speaker and message
            lines = section.split('\n', 1)
            if len(lines) < 1:
                continue
                
            speaker_line = lines[0].strip().rstrip(':')
            
            # Skip processing status and end markers
            if ('PROCESSING COMPLETE' in speaker_line or 
                'End of conversation' in speaker_line or
                speaker_line.startswith('💡') or
                speaker_line.startswith('─')):
                continue
                
            # Handle special cases for the initial prompt
            if speaker_line == 'User' and len(lines) > 1 and '[Initial prompt with files submitted' in lines[1]:
                # Keep original first message unchanged for file template
                if st.session_state.conversation:
                    new_conversation.append(st.session_state.conversation[0])
                continue
                
            # Parse normal conversation entries
            if speaker_line in ['User', 'Assistant'] and len(lines) > 1:
                message = lines[1].strip()
                if message:
                    new_conversation.append({
                        'speaker': speaker_line,
                        'message': message
                    })
        
        # Only update if we successfully parsed some entries and it's not dramatically smaller
        if new_conversation:
            original_count = len(st.session_state.conversation)
            new_count = len(new_conversation)
            
            # Safety check: don't allow updates that lose more than half the conversation
            if new_count >= max(1, original_count * 0.5):
                # Store backup before updating
                st.session_state.conversation_backup = st.session_state.conversation.copy()
                st.session_state.conversation = new_conversation
                debug_log(f"Updated conversation with {new_count} entries from user edits (was {original_count})", "SUCCESS")
            else:
                debug_log(f"Rejecting update: would reduce conversation from {original_count} to {new_count} entries (>50% loss)", "WARNING")
        else:
            debug_log("Failed to parse edited conversation - keeping original", "WARNING")
            
    except Exception as e:
        debug_log(f"Error parsing edited conversation: {str(e)} - keeping original", "ERROR")

def validate_and_extract_segments() -> Tuple[List[Dict[str, Any]], bool]:
    """
    Helper function to validate conversation and extract bilingual segments
    
    Returns:
        tuple: (segments_list, is_valid)
    """
    all_assistant_text = get_all_assistant_messages()
    segments = extract_bilingual_segments(all_assistant_text)
    debug_log(f"Validation found {len(segments)} bilingual segments", "PROCESS")
    
    is_valid = len(segments) > 0
    if not is_valid:
        debug_log("No valid bilingual segments detected", "WARNING")
    
    return segments, is_valid

def extract_srt_content(conversation: List[Dict[str, str]]) -> str:
    """Extract bilingual SRT content from the conversation with comprehensive pattern matching"""
    srt_segments = []
    
    # Define multiple patterns in order of preference (same as extract_bilingual_segments)
    patterns = [
        # Pattern 1: Standard SRT format (preferred) - matches standard format
        re.compile(
            r'^(\d+)\s*\n'                                # Subtitle number on its own line
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'          # Start time
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*\n'              # End time
            r'(.*?\|\|.*?)(?=\n\n|\n\d+\s*\n|\Z)',         # Multi-line bilingual text with ||
            re.MULTILINE | re.DOTALL
        ),
        # Pattern 2: Numbered list format with dots (fallback)
        re.compile(
            r'^(\d+)\.\s*'                                 # Subtitle number with dot
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'          # Start time
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*\n'              # End time
            r'(.*?\|\|.*?)(?=\n\n|\n\d+\.\s*\d{2}:\d{2}:\d{2}|\Z)',  # Multi-line bilingual text with ||
            re.MULTILINE | re.DOTALL
        ),
        # Pattern 3: Flexible spacing and optional dots
        re.compile(
            r'(\d+)\.?\s*\n?\s*'                        # Subtitle number with optional dot and flexible spacing
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'       # Start time
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*\n?\s*'       # End time with flexible spacing
            r'(.*?\|\|.*?)(?=\n\n|\n\d+\.?\s*\d{2}:\d{2}:\d{2}|\Z)',  # Multi-line bilingual text with ||
            re.MULTILINE | re.DOTALL
        ),
        # Pattern 4: Line-by-line format (single line bilingual)
        re.compile(
            r'(?:^|\n)(\d+)(?:\s*\n|\s+)'              # Subtitle number
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'      # Start time
            r'(\d{2}:\d{2}:\d{2},\d{3})(?:\s*\n|\s+)' # End time
            r'([^\n\r]*\|\|[^\n\r]*)',                 # Single line bilingual text with ||
            re.MULTILINE
        )
    ]
    
    for entry in conversation:
        if entry['speaker'] in {"Assistant", "Screen"}:
            # Try each pattern until we find matches
            matches = []
            for pattern in patterns:
                matches = pattern.findall(entry['message'])
                if matches:
                    break
            
            # If no pattern works, try fallback
            if not matches:
                matches = extract_fallback_bilingual_segments(entry['message'])
            
            for num, start, end, text in matches:
                # Normalize whitespace and clean up the text
                clean_text = re.sub(r'\s+', ' ', text.strip())
                block = f"{num}\n{start} --> {end}\n{clean_text}"
                srt_segments.append(block)
    
    return "\n\n".join(srt_segments).strip()

def extract_vtt_content(conversation: List[Dict[str, str]]) -> str:
    """Extract bilingual VTT content from the conversation with comprehensive pattern matching"""
    # Use the same patterns as extract_srt_content (prioritizing standard SRT format)
    patterns = [
        # Pattern 1: Standard SRT format (preferred) - matches standard format
        re.compile(
            r'^(\d+)\s*\n'                                # Subtitle number on its own line
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'          # Start time
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*\n'              # End time
            r'(.*?\|\|.*?)(?=\n\n|\n\d+\s*\n|\Z)',         # Multi-line bilingual text with ||
            re.MULTILINE | re.DOTALL
        ),
        # Pattern 2: Numbered list format with dots (fallback)
        re.compile(
            r'^(\d+)\.\s*'                                 # Subtitle number with dot
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'          # Start time
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*\n'              # End time
            r'(.*?\|\|.*?)(?=\n\n|\n\d+\.\s*\d{2}:\d{2}:\d{2}|\Z)',  # Multi-line bilingual text with ||
            re.MULTILINE | re.DOTALL
        ),
        # Pattern 3: Flexible spacing and optional dots
        re.compile(
            r'(\d+)\.?\s*\n?\s*'                        # Subtitle number with optional dot and flexible spacing
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'       # Start time
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*\n?\s*'       # End time with flexible spacing
            r'(.*?\|\|.*?)(?=\n\n|\n\d+\.?\s*\d{2}:\d{2}:\d{2}|\Z)',  # Multi-line bilingual text with ||
            re.MULTILINE | re.DOTALL
        ),
        # Pattern 4: Line-by-line format (single line bilingual)
        re.compile(
            r'(?:^|\n)(\d+)(?:\s*\n|\s+)'              # Subtitle number
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'      # Start time
            r'(\d{2}:\d{2}:\d{2},\d{3})(?:\s*\n|\s+)' # End time
            r'([^\n\r]*\|\|[^\n\r]*)',                 # Single line bilingual text with ||
            re.MULTILINE
        )
    ]
    
    vtt_lines = ["WEBVTT\n"]
    
    for entry in conversation:
        if entry['speaker'] in {"Assistant", "Screen"}:
            # Try each pattern until we find matches
            matches = []
            for pattern in patterns:
                matches = pattern.findall(entry['message'])
                if matches:
                    break
            
            # If no pattern works, try fallback
            if not matches:
                matches = extract_fallback_bilingual_segments(entry['message'])
            
            for _, start, end, text in matches:
                # Convert commas to dots for VTT
                start = start.replace(",", ".")
                end = end.replace(",", ".")
                # Normalize whitespace and clean up the text
                clean_text = re.sub(r'\s+', ' ', text.strip())
                vtt_lines.append(f"{start} --> {end}")
                vtt_lines.append(clean_text)
                vtt_lines.append("")  # Blank line between cues
    
    return "\n".join(vtt_lines).strip()

def extract_target_only_srt_content(conversation: List[Dict[str, str]]) -> str:
    """Extract target language only SRT content from the conversation (removes English text before ||) with comprehensive pattern matching"""
    debug_log("Extracting target-language-only SRT content", "PROCESS")
    
    srt_segments = []
    
    # Use the same patterns as other extraction functions (prioritizing standard SRT format)
    patterns = [
        # Pattern 1: Standard SRT format (preferred) - matches standard format
        re.compile(
            r'^(\d+)\s*\n'                                # Subtitle number on its own line
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'          # Start time
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*\n'              # End time
            r'(.*?\|\|.*?)(?=\n\n|\n\d+\s*\n|\Z)',         # Multi-line bilingual text with ||
            re.MULTILINE | re.DOTALL
        ),
        # Pattern 2: Numbered list format with dots (fallback)
        re.compile(
            r'^(\d+)\.\s*'                                 # Subtitle number with dot
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'          # Start time
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*\n'              # End time
            r'(.*?\|\|.*?)(?=\n\n|\n\d+\.\s*\d{2}:\d{2}:\d{2}|\Z)',  # Multi-line bilingual text with ||
            re.MULTILINE | re.DOTALL
        ),
        # Pattern 3: Flexible spacing and optional dots
        re.compile(
            r'(\d+)\.?\s*\n?\s*'                        # Subtitle number with optional dot and flexible spacing
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'       # Start time
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*\n?\s*'       # End time with flexible spacing
            r'(.*?\|\|.*?)(?=\n\n|\n\d+\.?\s*\d{2}:\d{2}:\d{2}|\Z)',  # Multi-line bilingual text with ||
            re.MULTILINE | re.DOTALL
        ),
        # Pattern 4: Line-by-line format (single line bilingual)
        re.compile(
            r'(?:^|\n)(\d+)(?:\s*\n|\s+)'              # Subtitle number
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'      # Start time
            r'(\d{2}:\d{2}:\d{2},\d{3})(?:\s*\n|\s+)' # End time
            r'([^\n\r]*\|\|[^\n\r]*)',                 # Single line bilingual text with ||
            re.MULTILINE
        )
    ]
    
    for entry in conversation:
        if entry['speaker'] in {"Assistant", "Screen"}:
            # Try each pattern until we find matches
            matches = []
            for pattern in patterns:
                matches = pattern.findall(entry['message'])
                if matches:
                    break
            
            # If no pattern works, try fallback
            if not matches:
                matches = extract_fallback_bilingual_segments(entry['message'])
            
            for num, start, end, text in matches:
                # Normalize whitespace and extract only the target language part (after ||)
                clean_text = re.sub(r'\s+', ' ', text.strip())
                parts = clean_text.split('||')
                if len(parts) == BINARY_PARTS_COUNT:
                    target_text = parts[1].strip()
                    block = f"{num}\n{start} --> {end}\n{target_text}"
                    srt_segments.append(block)
                else:
                    debug_log(f"Invalid bilingual format in segment {num}: '{clean_text}'", "WARNING")
    
    debug_log(f"Extracted {len(srt_segments)} target-language-only segments", "PROCESS")
    return "\n\n".join(srt_segments).strip()

def extract_target_only_vtt_content(conversation: List[Dict[str, str]]) -> str:
    """Extract target language only VTT content from the conversation (removes English text before ||) with comprehensive pattern matching"""
    debug_log("Extracting target-language-only VTT content", "PROCESS")
    
    # Use the same patterns as other extraction functions (prioritizing standard SRT format)
    patterns = [
        # Pattern 1: Standard SRT format (preferred) - matches standard format
        re.compile(
            r'^(\d+)\s*\n'                                # Subtitle number on its own line
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'          # Start time
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*\n'              # End time
            r'(.*?\|\|.*?)(?=\n\n|\n\d+\s*\n|\Z)',         # Multi-line bilingual text with ||
            re.MULTILINE | re.DOTALL
        ),
        # Pattern 2: Numbered list format with dots (fallback)
        re.compile(
            r'^(\d+)\.\s*'                                 # Subtitle number with dot
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'          # Start time
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*\n'              # End time
            r'(.*?\|\|.*?)(?=\n\n|\n\d+\.\s*\d{2}:\d{2}:\d{2}|\Z)',  # Multi-line bilingual text with ||
            re.MULTILINE | re.DOTALL
        ),
        # Pattern 3: Flexible spacing and optional dots
        re.compile(
            r'(\d+)\.?\s*\n?\s*'                        # Subtitle number with optional dot and flexible spacing
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'       # Start time
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*\n?\s*'       # End time with flexible spacing
            r'(.*?\|\|.*?)(?=\n\n|\n\d+\.?\s*\d{2}:\d{2}:\d{2}|\Z)',  # Multi-line bilingual text with ||
            re.MULTILINE | re.DOTALL
        ),
        # Pattern 4: Line-by-line format (single line bilingual)
        re.compile(
            r'(?:^|\n)(\d+)(?:\s*\n|\s+)'              # Subtitle number
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'      # Start time
            r'(\d{2}:\d{2}:\d{2},\d{3})(?:\s*\n|\s+)' # End time
            r'([^\n\r]*\|\|[^\n\r]*)',                 # Single line bilingual text with ||
            re.MULTILINE
        )
    ]
    
    vtt_lines = ["WEBVTT\n"]
    
    for entry in conversation:
        if entry['speaker'] in {"Assistant", "Screen"}:
            # Try each pattern until we find matches
            matches = []
            for pattern in patterns:
                matches = pattern.findall(entry['message'])
                if matches:
                    break
            
            # If no pattern works, try fallback
            if not matches:
                matches = extract_fallback_bilingual_segments(entry['message'])
            
            for _, start, end, text in matches:
                # Normalize whitespace and extract only the target language part (after ||)
                clean_text = re.sub(r'\s+', ' ', text.strip())
                parts = clean_text.split('||')
                if len(parts) == BINARY_PARTS_COUNT:
                    target_text = parts[1].strip()
                    # Convert commas to dots for VTT
                    start = start.replace(",", ".")
                    end = end.replace(",", ".")
                    vtt_lines.append(f"{start} --> {end}")
                    vtt_lines.append(target_text)
                    vtt_lines.append("")  # Blank line between cues
                else:
                    debug_log(f"Invalid bilingual format in segment: '{clean_text}'", "WARNING")
    
    debug_log(f"Extracted target-language-only VTT with {len(vtt_lines)} lines", "PROCESS")
    return "\n".join(vtt_lines).strip()

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from a DOCX file"""
    try:
        # Create a BytesIO object from the file content
        docx_file = io.BytesIO(file_content)
        # Load the document
        doc = Document(docx_file)
        # Extract text from all paragraphs
        text_content = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():  # Only add non-empty paragraphs
                text_content.append(paragraph.text.strip())
        return '\n\n'.join(text_content)
    except Exception as e:
        raise Exception(f"Error reading DOCX file: {str(e)}")

def clean_timestamp_patterns(text: str) -> str:
    """Remove common timestamp patterns from text including partial timestamps"""
    debug_log(f"Cleaning timestamp patterns from text ({len(text)} chars)", "PROCESS")
    
    # Only match true timestamps (hh:mm:ss or h:mm:ss), not Bible verses
    patterns = [
        # SRT/VTT timestamps: 00:00:00,000 --> 00:00:00,000
        r'\d{1,2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,\.]\d{3}',
        # SRT/VTT timestamps: 00:00:00.000 --> 00:00:00.000
        r'\d{1,2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}\.\d{3}',
        # Simple timestamps: 00:00:00,000 or 00:00:00.000
        r'\b\d{1,2}:\d{2}:\d{2}(?:[,\.]\d{3})?\b',
        # Standalone timestamps (hh:mm:ss) on their own line
        r'^\d{1,2}:\d{2}:\d{2}$',
        # Simple minute:second timestamps - ONLY at start of line or after newline (not Bible verses)
        r'(?:^|\n)(\d{1,2}:\d{2})(?=\s*\n|\s*$)',
        # Subtitle sequence numbers (standalone numbers on their own lines)
        r'^\d+\s*$',
        # Time codes in brackets: [00:00:00] [0:00:00] [00:00] [02:12]
        r'\[\d{1,2}:\d{2}(?::\d{2})?(?:[,\.]\d{3})?\]',
        # Time codes in parentheses: (00:00:00) (0:00:00) (00:00) (02:12)
        r'\(\d{1,2}:\d{2}(?::\d{2})?(?:[,\.]\d{3})?\)',
        # Timecode with frame numbers: 01:23:45:12
        r'\d{1,2}:\d{2}:\d{2}:\d{2}',
        # Time markers: 00h00m00s, 1h23m45s
        r'\d{1,2}h\d{1,2}m\d{1,2}s',
        # Time markers: 00m00s, 23m45s
        r'\d{1,2}m\d{1,2}s',
        # Simple seconds: 30s, 45s, 123s
        r'\b\d{1,3}s\b',
        # Time with 'min' abbreviation: 1min, 23min
        r'\d{1,2}min\b',
        # Timestamps with milliseconds only: ,000 .123 ,500
        r'[,\.]\d{1,3}(?=\s|$|\n)',
        # Arrow indicators: --> -> =>
        r'\s*(?:-->|->|=>)\s*',
        # Double dashes often used in transcripts: --
        r'\s*--\s*',
    ]
    
    cleaned_text = text
    patterns_found = []
    
    # Apply each pattern and track what was found
    for i, pattern in enumerate(patterns):
        matches = re.findall(pattern, cleaned_text, flags=re.MULTILINE)
        if matches:
            patterns_found.append(f"Pattern {i+1}: {len(matches)} matches")
            debug_log(f"Found timestamp pattern {i+1}: {len(matches)} matches", "DEBUG")
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE)
    
    if patterns_found:
        debug_log(f"Timestamp patterns cleaned: {', '.join(patterns_found)}", "PROCESS")
    
    # Clean up extra whitespace and empty lines
    lines = cleaned_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line:  # Only keep non-empty lines
            cleaned_lines.append(line)
    
    # Join lines and clean up multiple consecutive newlines
    result = '\n'.join(cleaned_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)  # Replace 3+ newlines with 2
    
    debug_log(f"Text cleaning complete: {len(text)} -> {len(result)} chars", "SUCCESS")
    return result.strip()




def parse_srt_segments(srt_content: str) -> List[Dict[str, Any]]:
    """Parse SRT content into segments with timestamps and text"""
    segments = []
    
    if not srt_content or not srt_content.strip():
        debug_log("No SRT content provided to parse", "WARNING")
        return segments
    
    debug_log(f"Parsing SRT content: {len(srt_content)} characters", "PROCESS")
    
    # Multiple patterns to handle different SRT format variations
    patterns = [
        # Standard SRT format
        re.compile(
            r'^(\d+)\s*\n'                                # Subtitle number
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'          # Start time
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*\n'              # End time
            r'(.*?)(?=\n\n|\n\d+\s*\n|\Z)',                # Text content
            re.MULTILINE | re.DOTALL
        ),
        # Alternative format with possible extra spaces
        re.compile(
            r'(\d+)\s*\n+'                               # Subtitle number with flexible newlines
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'        # Start time
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*\n+'           # End time with flexible newlines
            r'(.*?)(?=\n\s*\n|\n\s*\d+\s*\n|\Z)',        # Text content
            re.MULTILINE | re.DOTALL
        ),
        # Format with periods after numbers
        re.compile(
            r'^(\d+)\.\s*\n'                             # Subtitle number with period
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'        # Start time
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*\n'            # End time
            r'(.*?)(?=\n\n|\n\d+\.\s*\n|\Z)',            # Text content
            re.MULTILINE | re.DOTALL
        ),
    ]
    
    matches = []
    pattern_used = None
    
    # Try each pattern until we find matches
    for i, pattern in enumerate(patterns):
        matches = pattern.findall(srt_content)
        if matches:
            pattern_used = f"pattern_{i+1}"
            debug_log(f"SRT parsing successful with {pattern_used}, found {len(matches)} matches", "SUCCESS")
            break
    
    if not matches:
        debug_log("No matches found with any SRT pattern, trying line-by-line parsing", "WARNING")
        # Fallback: try manual parsing line by line
        lines = srt_content.strip().split('\n')
        current_segment: Dict[str, Any] = {}
        text_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:  # Empty line, end of segment
                if current_segment and text_lines:
                    current_segment['text'] = '\n'.join(text_lines).strip()
                    segments.append(current_segment)
                    current_segment = {}
                    text_lines = []
                continue
            
            # Check if line is a number
            if line.isdigit():
                if current_segment and text_lines:
                    current_segment['text'] = '\n'.join(text_lines).strip()
                    segments.append(current_segment)
                current_segment = {'number': int(line)}
                text_lines = []
                continue
            
            # Check if line is a timestamp
            if '-->' in line and ':' in line:
                if current_segment:
                    parts = line.split('-->')
                    if len(parts) == 2:
                        current_segment['start_time'] = parts[0].strip()
                        current_segment['end_time'] = parts[1].strip()
                continue
            
            # Otherwise, it's text content
            if current_segment and 'start_time' in current_segment:
                text_lines.append(line)
        
        # Don't forget the last segment
        if current_segment and text_lines:
            current_segment['text'] = '\n'.join(text_lines).strip()
            segments.append(current_segment)
        
        if segments:
            debug_log(f"Fallback parsing successful, found {len(segments)} segments", "SUCCESS")
        else:
            debug_log("All SRT parsing methods failed", "ERROR")
    else:
        # Process regex matches
        for num, start, end, text in matches:
            segments.append({
                'number': int(num),
                'start_time': start,
                'end_time': end,
                'text': text.strip()
            })
    
    debug_log(f"Final SRT parsing result: {len(segments)} segments", "PROCESS")
    return segments



def extract_bilingual_segments(response_text: str) -> List[Dict[str, Any]]:
    """Extract bilingual segments from AI response with comprehensive fallback patterns"""
    debug_log(f"Extracting bilingual segments from text ({len(response_text)} chars)", "PROCESS")
    
    segments = []
    
    # Define multiple patterns in order of preference
    patterns = [
        # Pattern 1: Standard SRT format (preferred) - matches your examples
        {
            'name': 'standard_srt',
            'pattern': re.compile(
                r'^(\d+)\s*\n'                                # Subtitle number on its own line
                r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'          # Start time
                r'(\d{2}:\d{2}:\d{2},\d{3})\s*\n'              # End time
                r'(.*?\|\|.*?)(?=\n\n|\n\d+\s*\n|\Z)',         # Multi-line bilingual text with ||
                re.MULTILINE | re.DOTALL
            )
        },
        # Pattern 2: Numbered list format with dots (fallback)
        {
            'name': 'numbered_list',
            'pattern': re.compile(
                r'^(\d+)\.\s*'                                 # Subtitle number with dot
                r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'          # Start time
                r'(\d{2}:\d{2}:\d{2},\d{3})\s*\n'              # End time
                r'(.*?\|\|.*?)(?=\n\n|\n\d+\.\s*\d{2}:\d{2}:\d{2}|\Z)',  # Multi-line bilingual text with ||
                re.MULTILINE | re.DOTALL
            )
        },
        # Pattern 3: Flexible spacing and optional dots
        {
            'name': 'flexible_spacing',
            'pattern': re.compile(
                r'(\d+)\.?\s*\n?\s*'                        # Subtitle number with optional dot and flexible spacing
                r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'       # Start time
                r'(\d{2}:\d{2}:\d{2},\d{3})\s*\n?\s*'       # End time with flexible spacing
                r'(.*?\|\|.*?)(?=\n\n|\n\d+\.?\s*\d{2}:\d{2}:\d{2}|\Z)',  # Multi-line bilingual text with ||
                re.MULTILINE | re.DOTALL
            )
        },
        # Pattern 4: Line-by-line format (single line bilingual)
        {
            'name': 'single_line',
            'pattern': re.compile(
                r'(?:^|\n)(\d+)(?:\s*\n|\s+)'              # Subtitle number
                r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*'      # Start time
                r'(\d{2}:\d{2}:\d{2},\d{3})(?:\s*\n|\s+)' # End time
                r'([^\n\r]*\|\|[^\n\r]*)',                 # Single line bilingual text with ||
                re.MULTILINE
            )
        }
    ]
    
    # Try each pattern until we find matches
    matches = []
    pattern_used = None
    
    for pattern_info in patterns:
        matches = pattern_info['pattern'].findall(response_text)
        if matches:
            pattern_used = pattern_info['name']
            debug_log(f"Found {len(matches)} matches using pattern: {pattern_used}", "PROCESS")
            break
    
    # If no standard patterns work, try fallback extraction
    if not matches:
        debug_log("Standard patterns failed, trying fallback extraction", "PROCESS")
        matches = extract_fallback_bilingual_segments(response_text)
        pattern_used = 'fallback'
    
    # Process matches
    for num, start, end, bilingual_text in matches:
        # Clean up the bilingual text and handle multi-line content
        bilingual_text = bilingual_text.strip()
        
        # Normalize whitespace within the bilingual text while preserving the structure
        bilingual_text = re.sub(r'\s+', ' ', bilingual_text)
        
        # Split the bilingual text
        if '||' in bilingual_text:
            parts = bilingual_text.split('||', 1)  # Split only on first ||
            if len(parts) == BINARY_PARTS_COUNT:
                # Normalize whitespace in individual components to remove line breaks
                english_text = re.sub(r'\s+', ' ', parts[0].strip())
                target_text = re.sub(r'\s+', ' ', parts[1].strip())
                
                # Apply format validation and auto-fixing
                format_validator = SimpleFormatValidator()
                original_bilingual = f"{english_text} || {target_text}"
                fixed_bilingual = format_validator.auto_fix_simple_issues(original_bilingual)
                
                # Update texts if format was fixed
                if fixed_bilingual != original_bilingual:
                    debug_log(f"Auto-fixed format issues in segment {num}", "DEBUG")
                    if ' || ' in fixed_bilingual:
                        fixed_parts = fixed_bilingual.split(' || ', 1)
                        english_text = fixed_parts[0].strip()
                        target_text = fixed_parts[1].strip()
                
                # Validate that we have reasonable content
                if len(english_text) > 0 and len(target_text) > 0:
                    segments.append({
                        'number': int(num),
                        'start_time': start,
                        'end_time': end,
                        'english_text': english_text,
                        'target_text': target_text
                    })
                    debug_log(f"Extracted segment {num}: '{english_text[:30]}...' || '{target_text[:30]}...'", "DEBUG")
                elif len(english_text) == 0 and len(target_text) == 0:
                    debug_log(f"Both English and target text empty in segment {num}", "WARNING")
                    # Still add the segment with empty content so validation can catch it
                    segments.append({
                        'number': int(num),
                        'start_time': start,
                        'end_time': end,
                        'english_text': english_text,
                        'target_text': target_text
                    })
                elif len(target_text) == 0:
                    debug_log(f"Empty target text found in segment {num}", "WARNING") 
                    # Still add the segment with empty target so validation can catch it
                    segments.append({
                        'number': int(num),
                        'start_time': start,
                        'end_time': end,
                        'english_text': english_text,
                        'target_text': target_text
                    })
                else:
                    debug_log(f"Empty English text found in segment {num}", "WARNING")
                    # Still add the segment so validation can handle it
                    segments.append({
                        'number': int(num),
                        'start_time': start,
                        'end_time': end,
                        'english_text': english_text,
                        'target_text': target_text
                    })
            else:
                debug_log(f"Invalid bilingual format in segment {num}: '{bilingual_text}'", "WARNING")
        else:
            debug_log(f"No || separator found in segment {num}: '{bilingual_text}'", "WARNING")
    
    # Sort segments by number to ensure proper order
    segments.sort(key=lambda x: x['number'])
    
    debug_log(f"Successfully extracted {len(segments)} bilingual segments using {pattern_used} pattern", "SUCCESS")
    
    # If we still have no segments, log the response text for debugging
    if len(segments) == 0:
        debug_log("No segments extracted, logging response for debugging:", "ERROR")
        debug_log(f"Response preview (first 500 chars): {response_text[:500]}", "DEBUG")
        debug_log(f"Response preview (last 500 chars): {response_text[-500:]}", "DEBUG")
    
    return segments

def extract_fallback_bilingual_segments(response_text: str) -> List[tuple]:
    """Fallback extraction for when standard patterns fail"""
    debug_log("Attempting fallback bilingual segment extraction", "PROCESS")
    
    fallback_matches = []
    lines = response_text.split('\n')
    
    # Method 1: Look for || separated text with nearby timestamps
    for i, line in enumerate(lines):
        if '||' in line and len(line.strip()) > 5:
            # Look for nearby timestamp and number
            timestamp_pattern = re.compile(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})')
            number_pattern = re.compile(r'^\s*(\d+)\.?\s*$')
            
            # Search nearby lines for number and timestamp
            seg_number = None
            start_time = None
            end_time = None
            
            # Check previous lines for number and timestamp
            for j in range(max(0, i-3), i):
                prev_line = lines[j].strip()
                
                # Look for segment number
                num_match = number_pattern.match(prev_line)
                if num_match and seg_number is None:
                    seg_number = num_match.group(1)
                
                # Look for timestamp
                time_match = timestamp_pattern.search(prev_line)
                if time_match and start_time is None:
                    start_time = time_match.group(1)
                    end_time = time_match.group(2)
            
            # Check current line for timestamp if not found
            if start_time is None:
                time_match = timestamp_pattern.search(line)
                if time_match:
                    start_time = time_match.group(1)
                    end_time = time_match.group(2)
            
            # Check next lines for missing components
            if seg_number is None or start_time is None:
                for j in range(i+1, min(len(lines), i+3)):
                    next_line = lines[j].strip()
                    
                    if seg_number is None:
                        num_match = number_pattern.match(next_line)
                        if num_match:
                            seg_number = num_match.group(1)
                    
                    if start_time is None:
                        time_match = timestamp_pattern.search(next_line)
                        if time_match:
                            start_time = time_match.group(1)
                            end_time = time_match.group(2)
            
            # If we found all components, add to matches
            if seg_number and start_time and end_time:
                # Clean the bilingual text
                clean_text = re.sub(r'\s+', ' ', line.strip())
                fallback_matches.append((seg_number, start_time, end_time, clean_text))
                debug_log(f"Fallback: Found segment {seg_number} with text '{clean_text[:50]}...'", "DEBUG")
    
    # Method 2: Look for timestamp patterns and try to find associated text
    if not fallback_matches:
        timestamp_lines = []
        for i, line in enumerate(lines):
            timestamp_match = re.search(r'(\d+)\.?\s*(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', line)
            if timestamp_match:
                seg_num = timestamp_match.group(1)
                start_t = timestamp_match.group(2)
                end_t = timestamp_match.group(3)
                
                # Look for bilingual text in nearby lines
                for j in range(i+1, min(len(lines), i+4)):
                    next_line = lines[j].strip()
                    if '||' in next_line and len(next_line) > 5:
                        clean_text = re.sub(r'\s+', ' ', next_line)
                        fallback_matches.append((seg_num, start_t, end_t, clean_text))
                        debug_log(f"Fallback method 2: Found segment {seg_num}", "DEBUG")
                        break
    
    debug_log(f"Fallback extraction found {len(fallback_matches)} segments", "PROCESS")
    return fallback_matches

def calculate_semantic_similarity(text1: str, text2: str, language: str = "english") -> float:
    """Calculate semantic similarity between two texts using enhanced string matching"""
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts for comparison
    text1_norm = re.sub(r'[^\w\s]', ' ', text1.lower().strip())
    text2_norm = re.sub(r'[^\w\s]', ' ', text2.lower().strip())
    text1_norm = re.sub(r'\s+', ' ', text1_norm)
    text2_norm = re.sub(r'\s+', ' ', text2_norm)
    
    if not text1_norm or not text2_norm:
        return 0.0
    
    # Multiple similarity metrics for robustness
    similarities = []
    
    # 1. Word overlap similarity (Jaccard index)
    words1 = set(text1_norm.split())
    words2 = set(text2_norm.split())
    if words1 and words2:
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard_sim = len(intersection) / len(union) if union else 0.0
        similarities.append(jaccard_sim)
    
    # 2. Character n-gram similarity (if available)
    if SEMANTIC_MATCHING_AVAILABLE and SequenceMatcher is not None:
        # Character-level similarity
        char_sim = SequenceMatcher(None, text1_norm, text2_norm).ratio()
        similarities.append(char_sim)
        
        # Word-level similarity
        words1_str = ' '.join(sorted(words1))
        words2_str = ' '.join(sorted(words2))
        word_sim = SequenceMatcher(None, words1_str, words2_str).ratio()
        similarities.append(word_sim)
    
    # 3. Length-based similarity (for cross-language validation)
    len1, len2 = len(text1_norm), len(text2_norm)
    if len1 > 0 and len2 > 0:
        length_sim = min(len1, len2) / max(len1, len2)
        # Weight length similarity less for different languages
        if language.lower() != "english":
            length_sim *= 0.5  # Reduce weight for cross-language comparison
        similarities.append(length_sim)
    
    # 4. Common word stems (simple stemming for better matching)
    def simple_stem(word):
        """Simple stemming by removing common suffixes"""
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 's', 'es']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word
    
    stems1 = set(simple_stem(word) for word in words1)
    stems2 = set(simple_stem(word) for word in words2)
    if stems1 and stems2:
        stem_intersection = stems1.intersection(stems2)
        stem_union = stems1.union(stems2)
        stem_sim = len(stem_intersection) / len(stem_union) if stem_union else 0.0
        similarities.append(stem_sim)
    
    # 5. Keyword density similarity (focus on content words)
    def extract_content_words(text, min_length=3):
        """Extract meaningful content words"""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        words = text.split()
        content_words = [word for word in words if len(word) >= min_length and word not in stop_words]
        return set(content_words)
    
    content1 = extract_content_words(text1_norm)
    content2 = extract_content_words(text2_norm)
    if content1 and content2:
        content_intersection = content1.intersection(content2)
        content_union = content1.union(content2)
        content_sim = len(content_intersection) / len(content_union) if content_union else 0.0
        similarities.append(content_sim * 1.5)  # Weight content words higher
    
    # Calculate weighted average of similarities
    if not similarities:
        return 0.0
    
    # Use harmonic mean for more conservative similarity scoring
    reciprocals = [1/sim if sim > 0 else 0 for sim in similarities]
    valid_reciprocals = [r for r in reciprocals if r > 0]
    
    if not valid_reciprocals:
        return 0.0
    
    harmonic_mean = len(valid_reciprocals) / sum(valid_reciprocals)
    
    # Apply language-specific adjustments
    if language.lower() != "english":
        # For cross-language comparison, we expect lower similarity
        # but should still detect complete semantic mismatches
        harmonic_mean *= CROSS_LANGUAGE_SIMILARITY_FACTOR  # Reduce threshold for different languages
    
    return min(harmonic_mean, 1.0)  # Cap at 1.0

def validate_segment_quality(original_segment: Dict[str, Any], generated_segment: Dict[str, Any], target_language: str) -> Dict[str, Any]:
    """Validate the quality of a generated bilingual segment with enhanced semantic checking"""
    issues = []
    warnings = []
    
    # Check timestamp matching
    if (original_segment['start_time'] != generated_segment['start_time'] or 
        original_segment['end_time'] != generated_segment['end_time']):
        issues.append(f"Timestamp mismatch: Expected {original_segment['start_time']} --> {original_segment['end_time']}, got {generated_segment['start_time']} --> {generated_segment['end_time']}")
    
    # Check English text preservation with enhanced similarity
    # Handle both 'text' (from SRT parsing) and 'english' (from UI preview) keys
    original_text_content = original_segment.get('text') or original_segment.get('english', '')
    original_text_clean = re.sub(r'\s+', ' ', original_text_content.strip())
    generated_english_clean = re.sub(r'\s+', ' ', generated_segment['english_text'].strip())
    
    if original_text_clean.lower() != generated_english_clean.lower():
        # Check for truncation specifically (generated text is subset of original)
        if generated_english_clean.lower() in original_text_clean.lower() and len(generated_english_clean) < len(original_text_clean) * 0.8:
            issues.append(f"English text truncation detected: Expected '{original_text_clean}', got truncated '{generated_english_clean}'")
        else:
            # Use enhanced semantic similarity for other mismatches
            english_similarity = calculate_semantic_similarity(original_text_clean, generated_english_clean, "english")
            
            if english_similarity < SIMILARITY_THRESHOLD_MINOR:  # Allow for minor differences
                if english_similarity < SIMILARITY_THRESHOLD_MAJOR:  # Major mismatch
                    issues.append(f"Major English text mismatch (similarity: {english_similarity:.2f}): Expected '{original_text_clean}', got '{generated_english_clean}'")
                else:  # Minor mismatch
                    warnings.append(f"Minor English text variation (similarity: {english_similarity:.2f})")
    
    # Enhanced target language validation
    target_text = generated_segment['target_text'].strip()
    
    # Calculate progressive word limit based on English text length
    original_text_content = original_segment.get('text') or original_segment.get('english', '')
    eng_words = len(original_text_content.split()) if original_text_content else 0
    expected_max_words = calculate_progressive_word_limit(eng_words, target_language)
    
    # Anti-bleeding word count validation with progressive limits
    target_word_count = len(target_text.split())
    
    # Critical word count limits to prevent bleeding (now progressive)
    if target_word_count > expected_max_words:
        issues.append(f"BLEEDING DETECTED: Target segment has {target_word_count} words (max: {expected_max_words} for {eng_words} English words) - '{target_text[:50]}...'")
    elif target_word_count > expected_max_words * 0.8:  # Warning at 80% of limit
        warnings.append(f"Target segment is long: {target_word_count} words (limit: {expected_max_words} for {eng_words} English words)")
    elif target_word_count < 3:
        warnings.append(f"Target segment is very short: {target_word_count} words")
    
    # Basic length checks
    if len(target_text) < 3:
        issues.append("Target language text too short (less than 3 characters)")
    
    # Check for completely empty target text
    if not target_text or target_text.strip() == "":
        issues.append("Empty target text - no translation provided")
    
    # Enhanced length ratio validation with language-specific thresholds
    language_thresholds = {
        'chinese': {'min': 0.2, 'max': 2.0},  # Chinese is more compact
        'japanese': {'min': 0.3, 'max': 2.5},
        'korean': {'min': 0.3, 'max': 2.5},
        'german': {'min': 0.8, 'max': 4.0},   # German can be longer
        'russian': {'min': 0.6, 'max': 3.5},
        'portuguese': {'min': 0.7, 'max': 3.0},
        'spanish': {'min': 0.7, 'max': 3.0},
        'french': {'min': 0.7, 'max': 3.0},
        'italian': {'min': 0.7, 'max': 3.0},
        'default': {'min': 0.5, 'max': 3.0}
    }
    
    thresholds = language_thresholds.get(target_language.lower(), language_thresholds['default'])
    english_len = len(generated_english_clean)
    target_len = len(target_text)
    
    if english_len > 0:
        length_ratio = target_len / english_len
        if length_ratio < thresholds['min']:
            issues.append(f"Target text too short for {target_language}: ratio {length_ratio:.2f} < {thresholds['min']}")
        elif length_ratio > thresholds['max']:
            issues.append(f"Target text too long for {target_language}: ratio {length_ratio:.2f} > {thresholds['max']}")
        elif length_ratio < thresholds['min'] * 1.2 or length_ratio > thresholds['max'] * 0.8:
            warnings.append(f"Unusual length ratio for {target_language}: {length_ratio:.2f}")
    
    # Semantic relationship validation between English and target text
    # This is a cross-language check to detect obvious mismatches
    cross_lang_similarity = calculate_semantic_similarity(generated_english_clean, target_text, target_language)
    
    # Language-specific similarity thresholds
    expected_similarity = {
        'portuguese': 0.15,  # Romance languages might have some cognates
        'spanish': 0.15,
        'italian': 0.15,
        'french': 0.12,
        'german': 0.08,      # Germanic language, fewer cognates
        'chinese': 0.02,     # Very different writing system
        'japanese': 0.02,
        'korean': 0.02,
        'russian': 0.05,     # Cyrillic script, some borrowed words
        'default': 0.08
    }
    
    min_similarity = expected_similarity.get(target_language.lower(), expected_similarity['default'])
    
    if cross_lang_similarity < min_similarity * 0.5:  # Less than half expected
        warnings.append(f"Very low cross-language similarity ({cross_lang_similarity:.3f}) - possible translation mismatch")
    
    # Check for obvious copy-paste errors (target same as English)
    if target_text.lower() == generated_english_clean.lower():
        issues.append("Target text identical to English text - translation not performed")
    
    # Check for repeated words or phrases (possible AI artifacts)
    target_words = target_text.lower().split()
    if len(target_words) > 3:
        word_counts = {}
        for word in target_words:
            if len(word) > 2:  # Only count meaningful words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated_words = [word for word, count in word_counts.items() if count > len(target_words) // 3]
        if repeated_words:
            warnings.append(f"Excessive word repetition detected: {', '.join(repeated_words)}")
    
    # Check for placeholder text or common AI failure patterns
    placeholder_patterns = [
        r'\[.*\]',  # Bracketed text
        r'\{.*\}',  # Curly bracketed text
        r'\.\.\.+', # Multiple dots
        r'xxx+',    # Multiple x's
        r'translation|translate|target|source',  # Meta-language
        r'error|fail|cannot|unable',  # Error indicators
    ]
    
    for pattern in placeholder_patterns:
        if re.search(pattern, target_text.lower()):
            issues.append(f"Placeholder or error text detected in target: '{target_text}'")
            break
    
    # Check for proper encoding (detect mojibake or encoding issues)
    try:
        target_text.encode('utf-8').decode('utf-8')
    except UnicodeError:
        issues.append("Text encoding issues detected in target text")
    
    # Language-specific script validation
    script_checks = {
        'chinese': r'[\u4e00-\u9fff]',
        'japanese': r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]',
        'korean': r'[\uac00-\ud7af]',
        'russian': r'[\u0400-\u04ff]'
    }
    
    if target_language.lower() in script_checks:
        pattern = script_checks[target_language.lower()]
        if not re.search(pattern, target_text):
            issues.append(f"No {target_language} script characters found in target text")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'segment_number': generated_segment['number'],
        'quality_score': max(0, 1.0 - len(issues) * 0.3 - len(warnings) * 0.1)  # 0-1 quality score
    }

def classify_warning_severity(warning_type: str, details: Optional[Dict] = None) -> str:
    """
    Classify warning severity as 'severe' or 'moderate'
    
    Severe warnings (should stop autopilot):
    - Multiple content bleeding incidents
    - Completely incorrect English text preservation
    - Missing required bilingual format
    - Severe formatting violations
    - Empty or missing target text
    
    Moderate warnings (continue with caution):
    - Minor English text differences (typos, punctuation)
    - Slight word limit exceedances
    - Minor formatting inconsistencies
    - Single content bleeding incident
    """
    if details is None:
        details = {}
    
    # Severe warning conditions
    severe_conditions = [
        # Content bleeding is severe if multiple segments affected
        warning_type == "content_bleeding" and details.get('affected_segments', 1) > 1,
        
        # English text preservation issues are severe if similarity is very low
        warning_type == "english_preservation" and details.get('similarity', 1.0) < SIMILARITY_THRESHOLD_MAJOR,
        
        # Missing bilingual format is always severe
        warning_type == "missing_bilingual_format",
        
        # Timestamp issues are severe
        warning_type == "timestamp_mismatch",
        
        # Word limit violations are severe if drastically over limit (50% more than allowed)
        warning_type == "word_limit_exceeded" and details.get('excess_words', 0) > 15,
        
        # Format violations are severe if they break structure
        warning_type == "format_violation" and details.get('structural_damage', False),
        
        # Empty target text is severe
        warning_type == "empty_target_text" or warning_type == "format_violation" and "empty" in details.get('message', '').lower(),
        
        # Multiple issues in single segment indicate severe problems
        details.get('issue_count', 1) > 2
    ]
    
    if any(severe_conditions):
        return "severe"
    
    return "moderate"

def add_quality_warning(warning_type: str, message: str, segment_number: Optional[int] = None, details: Optional[Dict] = None):
    """Add a warning to the quality report with severity classification"""
    severity = classify_warning_severity(warning_type, details)
    
    warning_entry = {
        'type': warning_type,
        'message': message,
        'severity': severity,
        'segment': segment_number,
        'details': details or {},
        'timestamp': datetime.now().strftime("%H:%M:%S")
    }
    
    # Add to appropriate severity list
    if severity == "severe":
        st.session_state.quality_report['severe_warnings'].append(warning_entry)
    else:
        st.session_state.quality_report['moderate_warnings'].append(warning_entry)
    
    # Add to general warnings list
    st.session_state.quality_report['warnings'].append(warning_entry)
    st.session_state.quality_report['last_updated'] = datetime.now()
    
    debug_log(f"Quality warning added: {severity.upper()} - {warning_type} - {message}", "WARNING")

def should_stop_autopilot() -> Tuple[bool, str]:
    """
    Determine if autopilot should stop based on severity-based criteria
    
    Returns:
        tuple: (should_stop, reason)
    """
    severe_warnings = st.session_state.quality_report['severe_warnings']
    moderate_warnings = st.session_state.quality_report['moderate_warnings']
    
    # Stop if we have any severe warnings, but make exception for single empty segment
    if severe_warnings:
        # Count empty segment warnings
        empty_segment_warnings = [w for w in severe_warnings if 'empty' in w.get('message', '').lower() or 'too short' in w.get('message', '').lower()]
        other_severe_warnings = [w for w in severe_warnings if w not in empty_segment_warnings]
        
        # If we only have 1 empty segment warning and no other severe warnings, continue
        if len(empty_segment_warnings) == 1 and len(other_severe_warnings) == 0:
            debug_log("Single empty segment detected - continuing autopilot", "WARNING")
            return False, ""
        
        # Stop for any other severe warnings or multiple empty segments
        severe_count = len(severe_warnings)
        latest_severe = severe_warnings[-1]
        return True, f"Severe quality issue detected: {latest_severe['message']} ({severe_count} severe warning{'s' if severe_count > 1 else ''} total)"
    
    # Continue regardless of moderate warning count (only stop on severe warnings)
    return False, ""

def generate_quality_report() -> str:
    """Generate a comprehensive quality report for user review with source validation details"""
    report = st.session_state.quality_report
    
    if not report['warnings']:
        return "✅ **Quality Report: No Issues Detected**\n\nAll processed segments meet quality standards with full source text compliance."
    
    # Build comprehensive report
    lines = ["📊 **Comprehensive Quality Report**\n"]
    
    # Summary statistics
    total_segments = report['total_segments_processed']
    segments_with_issues = report['segments_with_issues']
    success_rate = ((total_segments - segments_with_issues) / total_segments * 100) if total_segments > 0 else 100
    
    lines.append(f"**Processing Summary:**")
    lines.append(f"- Total segments processed: {total_segments}")
    lines.append(f"- Segments with issues: {segments_with_issues}")
    lines.append(f"- Success rate: {success_rate:.1f}%")
    lines.append(f"- Total warnings: {len(report['warnings'])}")
    lines.append(f"- Severe warnings: {len(report['severe_warnings'])}")
    lines.append(f"- Moderate warnings: {len(report['moderate_warnings'])}\n")
    
    # Source validation summary
    source_issues = [w for w in report['warnings'] if w['type'] in ['ai_content_generation', 'source_text_mismatch']]
    if source_issues:
        lines.append(f"**Source Text Compliance:**")
        generation_issues = [w for w in source_issues if w['type'] == 'ai_content_generation']
        mismatch_issues = [w for w in source_issues if w['type'] == 'source_text_mismatch']
        
        if generation_issues:
            lines.append(f"- AI generation detected: {len(generation_issues)} segment(s)")
        if mismatch_issues:
            lines.append(f"- Source text mismatches: {len(mismatch_issues)} segment(s)")
        
        compliance_rate = ((total_segments - len(source_issues)) / total_segments * 100) if total_segments > 0 else 100
        lines.append(f"- Source compliance rate: {compliance_rate:.1f}%\n")
    else:
        lines.append(f"**Source Text Compliance:** ✅ 100% - All text properly matched from source\n")
    
    # Severe warnings section
    if report['severe_warnings']:
        lines.append("🚨 **Severe Issues (Require Immediate Attention):**")
        for warning in report['severe_warnings']:
            segment_info = f" (Segment {warning['segment']})" if warning['segment'] else ""
            lines.append(f"- **{warning['type'].replace('_', ' ').title()}**{segment_info}: {warning['message']}")
            
            # Add source validation details for relevant issues
            if warning['type'] in ['ai_content_generation', 'source_text_mismatch'] and 'details' in warning:
                details = warning['details']
                if 'word_overlap_score' in details:
                    lines.append(f"  → Source overlap: {details['word_overlap_score']:.2f}")
                if 'target_text' in details:
                    lines.append(f"  → Text preview: {details['target_text']}")
        lines.append("")
    
    # Moderate warnings section
    if report['moderate_warnings']:
        lines.append("⚠️ **Moderate Issues (Review Recommended):**")
        for warning in report['moderate_warnings']:
            segment_info = f" (Segment {warning['segment']})" if warning['segment'] else ""
            lines.append(f"- **{warning['type'].replace('_', ' ').title()}**{segment_info}: {warning['message']}")
        lines.append("")
    
    # Enhanced recommendations based on issue types
    lines.append("💡 **Recommendations:**")
    if any(w['type'] == 'ai_content_generation' for w in report['severe_warnings']):
        lines.append("- **CRITICAL**: AI is generating text instead of matching source translation")
        lines.append("- Check translation content and prompt clarity")
        lines.append("- Ensure translation file contains all required content")
    elif any(w['type'] == 'source_text_mismatch' for w in report['warnings']):
        lines.append("- **Source matching issues detected**: Review translation alignment")
        lines.append("- Consider adjusting segment boundaries or translation content")
    elif report['severe_warnings']:
        lines.append("- **Immediate action required**: Review and manually correct segments with severe issues")
        lines.append("- Consider adjusting translation approach or prompt for better results")
    elif len(report['moderate_warnings']) > 2:
        lines.append("- **Review recommended**: Check moderate issues for potential improvements")
        lines.append("- Consider fine-tuning word limits or validation criteria")
    else:
        lines.append("- **Quality acceptable**: Minor issues detected, review at your discretion")
    
    lines.append(f"\n*Report generated at: {datetime.now().strftime('%H:%M:%S')}*")
    
    return "\n".join(lines)

def calculate_simple_quality_score() -> float:
    """Calculate a simple quality score based on warnings and processing success"""
    if 'quality_report' not in st.session_state:
        return 0.0
    
    # Start with base score
    base_score = 1.0
    
    # Penalize for severe warnings
    severe_warnings = len(st.session_state.quality_report.get('severe_warnings', []))
    moderate_warnings = len(st.session_state.quality_report.get('moderate_warnings', []))
    
    # Calculate penalty
    severe_penalty = severe_warnings * 0.2  # 20% penalty per severe warning
    moderate_penalty = moderate_warnings * 0.05  # 5% penalty per moderate warning
    
    total_penalty = severe_penalty + moderate_penalty
    final_score = max(0.0, base_score - total_penalty)
    
    return final_score

def check_content_bleeding(generated_segments: List[Dict[str, Any]], original_segments: List[Dict[str, Any]]) -> List[str]:
    """Check for content bleeding between segments with enhanced detection"""
    issues = []
    
    if len(generated_segments) != len(original_segments):
        issues.append(f"Segment count mismatch: Expected {len(original_segments)}, got {len(generated_segments)}")
        return issues
    
    for i, gen_seg in enumerate(generated_segments):
        current_target = gen_seg['target_text'].lower().strip()
        current_english = gen_seg['english_text'].lower().strip()
        current_target_words = set(current_target.split())
        
        # Check forward bleeding (content from next segment)
        if i < len(original_segments) - 1:
            next_original = original_segments[i + 1]['text'].lower().strip()
            next_words = set(next_original.split())
            
            # Enhanced bleeding detection with multiple criteria
            overlap = len(next_words.intersection(current_target_words))
            overlap_ratio = overlap / len(next_words) if next_words else 0
            
            # Check for significant word overlap
            if overlap_ratio > 0.3:  # More than 30% overlap
                issues.append(f"Forward content bleeding in segment {gen_seg['number']}: {overlap_ratio:.1%} overlap with next segment")
            
            # Check for phrase-level bleeding
            next_bigrams = set()
            next_words_list = next_original.split()
            for j in range(len(next_words_list) - 1):
                next_bigrams.add(f"{next_words_list[j]} {next_words_list[j+1]}")
            
            current_bigrams = set()
            current_words_list = current_target.split()
            for j in range(len(current_words_list) - 1):
                current_bigrams.add(f"{current_words_list[j]} {current_words_list[j+1]}")
            
            bigram_overlap = len(next_bigrams.intersection(current_bigrams))
            if bigram_overlap > 2:  # More than 2 shared bigrams
                issues.append(f"Phrase-level bleeding in segment {gen_seg['number']}: {bigram_overlap} shared phrases with next segment")
        
        # Check backward bleeding (content from previous segment)
        if i > 0:
            prev_original = original_segments[i - 1]['text'].lower().strip()
            prev_words = set(prev_original.split())
            
            overlap = len(prev_words.intersection(current_target_words))
            overlap_ratio = overlap / len(prev_words) if prev_words else 0
            
            if overlap_ratio > 0.3:  # More than 30% overlap
                issues.append(f"Backward content bleeding in segment {gen_seg['number']}: {overlap_ratio:.1%} overlap with previous segment")
        
        # Check for English text bleeding into target text
        english_words = set(current_english.split())
        
        # Allow some overlap for cognates and proper nouns
        english_in_target = len(english_words.intersection(current_target_words))
        english_overlap_ratio = english_in_target / len(english_words) if english_words else 0
        
        # High overlap might indicate untranslated content
        if english_overlap_ratio > 0.7:  # More than 70% of English words in target
            issues.append(f"Possible untranslated content in segment {gen_seg['number']}: {english_overlap_ratio:.1%} English words in target")
        
        # Check for segment concatenation (multiple sentences combined)
        # Look for multiple sentence markers
        sentence_markers = ['.', '!', '?', '。', '！', '？']  # Include some non-Latin markers
        marker_count = sum(current_target.count(marker) for marker in sentence_markers)
        
        # If target has many more sentences than English, it might be concatenated
        english_markers = sum(current_english.count(marker) for marker in sentence_markers[:3])
        
        if marker_count > english_markers + 2:  # Allow some variance
            issues.append(f"Possible segment concatenation in segment {gen_seg['number']}: {marker_count} vs {english_markers} sentence markers")
        
        # Check for extremely long translations (possible multiple segments combined)
        if len(current_target) > len(current_english) * 4:  # 4x longer than English
            issues.append(f"Suspiciously long translation in segment {gen_seg['number']}: {len(current_target)} vs {len(current_english)} characters")
    
    return issues

def check_autopilot_prerequisites() -> Dict[str, Any]:
    """Check if all prerequisites for autopilot are met"""
    issues = []
    
    # Check if API key is configured
    current_key = get_current_api_key()
    if not current_key:
        provider_name = {"claude": "Claude", "openai": "OpenAI", "gemini": "Gemini"}[st.session_state.llm_provider]
        issues.append(f"{provider_name} API key not configured")
    
    # Check if files are uploaded and validated
    if not st.session_state.inputs_validated:
        issues.append("Input files not validated")
    
    if not st.session_state.english_sub_content:
        issues.append("English subtitle file not uploaded")
    
    if not st.session_state.translation_content:
        issues.append("Translation file not uploaded")
    
    # Check if language is confirmed
    if not st.session_state.language_confirmed:
        issues.append("Target language not confirmed")
    
    # Check if conversation has started
    if not st.session_state.conversation:
        issues.append("No conversation started - submit initial prompt first")
    
    return {
        'all_met': len(issues) == 0,
        'issues': issues
    }

def start_autopilot():
    """Start the autopilot process"""
    prerequisites = check_autopilot_prerequisites()
    if not prerequisites['all_met']:
        st.error("🚨 **CANNOT START AUTOPILOT - MISSING REQUIREMENTS**")
        st.warning("**📋 PLEASE RESOLVE THE FOLLOWING ISSUES:**")
        for issue in prerequisites['issues']:
            st.write(f"• ❌ {issue}")
        
        st.info("""
**🔧 HOW TO FIX:**
1. **Upload Files:** Go to File Upload tab and load English SRT + Translation
2. **Set API Key:** Configure your Claude API key in Settings
3. **Confirm Language:** Select and confirm target language
4. **Submit Initial Prompt:** Process at least one segment manually first

**💡 TIP:** Complete these steps in order, then return to start autopilot.
        """)
        return
    
    # Reset severe warnings when user manually restarts autopilot
    # This allows the user to restart after fixing issues or choosing to continue despite warnings
    if 'quality_report' not in st.session_state:
        st.session_state.quality_report = {
            'warnings': [],
            'severe_warnings': [],
            'moderate_warnings': [],
            'total_segments_processed': 0,
            'segments_with_issues': 0,
            'last_updated': None
        }
        debug_log("Quality report initialized for new autopilot session", "PROCESS")
    else:
        # Clear severe warnings to allow restart, but keep moderate warnings for reference
        previous_severe = len(st.session_state.quality_report.get('severe_warnings', []))
        st.session_state.quality_report['severe_warnings'] = []
        if previous_severe > 0:
            debug_log(f"Cleared {previous_severe} previous severe warnings - autopilot can restart", "PROCESS")
    
    st.session_state.autopilot_running = True
    st.success(f"🚁 Autopilot started! Processing {st.session_state.autopilot_segments} segments per batch.")
    
    # Start the autopilot process
    run_autopilot_batch()

def run_autopilot_batch():
    """Run a single batch of autopilot processing"""
    if not st.session_state.autopilot_running:
        debug_log("Autopilot not running, skipping batch", "WARNING")
        return
    
    debug_log("Starting autopilot batch processing", "PROCESS")
    
    try:
        # Parse original English segments
        original_segments = parse_srt_segments(st.session_state.english_sub_content)
        debug_log(f"Parsed {len(original_segments)} original segments", "PROCESS")
        
        # Count how many segments have been processed
        processed_segments = count_processed_segments()
        debug_log(f"Already processed {processed_segments} segments", "PROCESS")
        
        # DEBUG: Check why autopilot might be completing immediately
        debug_log(f"AUTOPILOT DEBUG - Total segments: {len(original_segments)}, Processed: {processed_segments}, Should continue: {processed_segments < len(original_segments)}", "DEBUG")
        
        if processed_segments >= len(original_segments):
            st.session_state.autopilot_running = False
            debug_log("All segments processed, autopilot completed", "SUCCESS")
            
            # Comprehensive completion message
            st.success("🎉 **AUTOPILOT COMPLETED SUCCESSFULLY!**")
            st.info(f"""
**📊 COMPLETION SUMMARY:**
• ✅ **All {len(original_segments)} segments processed**
• 🎯 **Total batches completed:** {(processed_segments + st.session_state.autopilot_segments - 1) // st.session_state.autopilot_segments}
• 🌐 **Language:** {st.session_state.target_language}
• 🔧 **Processing mode:** Comprehensive Full-Context Approach
            
**🎬 READY FOR EXPORT:**
Your bilingual subtitles are ready! Switch to the **Conversation tab** to download in SRT/VTT format.
            """)
            handle_autopilot_completion(True)
            return
        
        # Request next batch
        remaining_segments = len(original_segments) - processed_segments
        batch_size = min(st.session_state.autopilot_segments, remaining_segments)
        debug_log(f"Processing batch: {batch_size} segments, {remaining_segments} remaining", "PROCESS")
        
        # Update autopilot status
        st.session_state.autopilot_status['last_processed'] = processed_segments
        
        # Get the specific English segments for this batch
        batch_segments = original_segments[processed_segments:processed_segments + batch_size]
        
        # Generate progressive word limits (calculated based on English text length)
        segment_word_limits = []
        for i, segment in enumerate(batch_segments):
            # Handle both 'text' (from SRT parsing) and 'english' (from UI preview) keys
            text_content = segment.get('text') or segment.get('english', '')
            eng_words = len(text_content.split())
            # Progressive word limit calculation with extended range support
            max_words = calculate_progressive_word_limit(eng_words)
            segment_word_limits.append(f"Segment {processed_segments + i + 1}: {eng_words} English words → MAX {max_words} target words")
        
        # Unified prompt system that works optimally for all batch sizes (1-30)
        format_enforcement = f"""
🎯 UNIVERSAL SUBTITLE ALIGNMENT SYSTEM
Processing {batch_size} segments with progressive word limits for {st.session_state.target_language.upper()}

📊 SEGMENT-SPECIFIC LIMITS (MANDATORY):
{chr(10).join(segment_word_limits)}

� CORE ALIGNMENT PRINCIPLES:
• Each English segment maps to EXACTLY ONE target language portion
• Progressive word limits scale intelligently with English segment length
• Sequential processing: use translation text in exact order provided
• Zero word reuse: each translation word used exactly once
• Natural boundaries: end at sentence breaks when possible

🎯 PRECISION PROTOCOL:
1. Read English segment + its calculated word limit
2. Extract NEXT unused portion from translation (in order)
3. Count target words as you write - STOP at calculated limit
4. End at natural boundary (., !, ?) when within limit
5. Mark extracted text as USED, move to next segment
6. Maintain semantic correspondence throughout

🚨 ANTI-BLEEDING SAFEGUARDS:
✓ Respect calculated word limits absolutely (shown above)
✓ Never exceed limits even for "better flow"
✓ Stop mid-sentence if approaching word limit
✓ Use progressive buffer system for optimal quality
✓ Language-specific adjustments already calculated

⚡ BATCH OPTIMIZATION ({batch_size} segments):
• Consistent quality regardless of batch size
• Efficient processing for large batches
• Maintained precision for small batches
• No format degradation at any scale
• Zero bleeding tolerance across all sizes

🎬 AUTOPILOT EXECUTION:
- Generate complete mapping immediately without questions
- NO confirmations, clarifications, or interaction requests
- Handle complex phrases and parenthetical content automatically
- Process entire batch in single response
- Provide direct output in specified format only
"""

        # Build detailed request with exact English segments and DYNAMIC FORMAT ENFORCEMENT
        english_segments_text = ""
        for i, segment in enumerate(batch_segments, start=processed_segments + 1):
            # Normalize English text to single line (remove line breaks)
            english_text = segment['text'].replace('\n', ' ').replace('\r', ' ').strip()
            # Remove multiple spaces
            english_text = ' '.join(english_text.split())
            english_segments_text += f"\n{i}. {segment['start_time']} --> {segment['end_time']}\n{english_text}\n"
        
        # Fix context message for initial batch vs continuation
        if processed_segments == 0:
            context_message = "CONTEXT: You have the complete translation and are starting with the first segments."
            action_verb = "Map"
        else:
            context_message = f"CONTEXT: You have the complete translation and have already mapped segments 1-{processed_segments}."
            action_verb = "Continue mapping"
        
        request_message = f"""{action_verb} segments {processed_segments + 1}-{processed_segments + batch_size}.

{context_message}

SEGMENTS TO MAP:
{english_segments_text}

PROGRESSIVE WORD LIMITS (CALCULATED):
{chr(10).join(segment_word_limits)}

OUTPUT: Map these {batch_size} segments using semantic alignment with progressive word limits. Use STANDARD SRT format:

{processed_segments + 1}
[timestamp]
[English] || [Target within calculated limit]

{processed_segments + 2}
[timestamp]
[English] || [Target within calculated limit]

CRITICAL FORMAT REQUIREMENTS:
- Number each segment on its own line: "{processed_segments + 1}" then newline, "{processed_segments + 2}" then newline, etc.
- Use exact timestamp format: HH:MM:SS,mmm --> HH:MM:SS,mmm on next line
- Separate languages with " || " (space-pipe-pipe-space)
- Keep all original English text exactly as provided
- Respect calculated word limits for each segment individually
- Maintain chronological order and semantic correspondence

Generate immediately without questions using the universal alignment system."""
        
        st.session_state.conversation.append({"speaker": "User", "message": request_message})
        debug_log(f"Added detailed request message with {batch_size} English segments", "PROCESS")
        
        # Prepare messages for API
        messages = []
        for entry in st.session_state.conversation:
            role = "user" if entry['speaker'] == "User" else "assistant"
            messages.append({"role": role, "content": entry['message']})
        
        debug_log(f"Prepared {len(messages)} messages for API call", "PROCESS")
        
        # Make API call with progress indicator
        with st.spinner(f"🤔 Processing batch {processed_segments//st.session_state.autopilot_segments + 1}..."):
            debug_log("Calling LLM API with messages", "API")
            response = call_llm_api_with_messages(messages)
            assistant_message = extract_api_response(response)
            
            # Add response to conversation
            st.session_state.conversation.append({"speaker": "Assistant", "message": assistant_message})
            debug_log(f"Received API response: {len(assistant_message)} characters", "PROCESS")
            
            if "API Error:" in assistant_message:
                st.session_state.autopilot_running = False
                st.session_state.autopilot_continue_next_batch = False  # Reset flag on API error
                debug_log(f"API error occurred: {assistant_message}", "ERROR")
                
                # Enhanced API error message with context
                st.error("🚨 **AUTOPILOT STOPPED - API ERROR**")
                st.warning(f"""
**📡 API ERROR DETAILS:**
• ❌ **Error:** {assistant_message}
• 🔄 **Processed segments:** {processed_segments} of {len(original_segments)}
• 📊 **Completion:** {(processed_segments/len(original_segments)*100):.1f}%

**🔧 TROUBLESHOOTING STEPS:**
1. **Check API Key:** Ensure your API key is valid and has credits
2. **Network Issues:** Verify internet connection
3. **Rate Limits:** Wait a few minutes if hitting rate limits
4. **Resume:** You can restart autopilot to continue from segment {processed_segments + 1}

**💡 TIP:** Your progress has been saved. Switch to **Conversation tab** to export current results or resume processing manually.
                """)
                return
            
            # Check for AI refusal or clarification requests
            refusal_patterns = [
                "apologize",
                "already mapped", 
                "clarify",
                "clarification",
                "questions",
                "would you like me to",
                "could you please",
                "unclear",
                "not sure",
                "which option",
                "please specify"
            ]
            
            assistant_lower = assistant_message.lower()
            ai_refused = any(pattern in assistant_lower for pattern in refusal_patterns)
            
            # Try to extract segments first before declaring refusal
            extracted_segments = extract_bilingual_segments(assistant_message)
            
            if ai_refused and len(extracted_segments) == 0:
                debug_log("AI refused to process or asked for clarification", "WARNING")
                
                # Remove the refusal response from conversation to avoid confusion
                st.session_state.conversation.pop()
                
                # Create a more direct retry prompt
                retry_message = f"""AUTOPILOT DIRECTIVE: Process segments {processed_segments + 1}-{processed_segments + batch_size} immediately.

{context_message}

SEGMENTS TO PROCESS:
{english_segments_text}

STRICT REQUIREMENTS:
- Output ONLY the bilingual mapping format
- NO questions, clarifications, or confirmations
- Use the translation provided in the initial context
- Follow word limits: calculated progressive limits per target segment (based on English length and language ratio)

REQUIRED STANDARD SRT FORMAT:
{processed_segments + 1}
HH:MM:SS,mmm --> HH:MM:SS,mmm
[English] || [Target]

{processed_segments + 2}
HH:MM:SS,mmm --> HH:MM:SS,mmm
[English] || [Target]

FORMAT RULES:
- Number on its own line: "{processed_segments + 1}" then newline
- Timestamp on next line: "HH:MM:SS,mmm --> HH:MM:SS,mmm"
- Bilingual text on next line with " || " separator
- Preserve exact English text and timestamps

PROCESS NOW."""
                
                # Add retry message and make second API call
                st.session_state.conversation.append({"speaker": "User", "message": retry_message})
                debug_log("Added retry directive message", "PROCESS")
                
                # Prepare retry messages for API
                retry_messages = []
                for entry in st.session_state.conversation:
                    role = "user" if entry['speaker'] == "User" else "assistant"
                    retry_messages.append({"role": role, "content": entry['message']})
                
                # Make retry API call
                with st.spinner(f"🔄 Retrying batch {processed_segments//st.session_state.autopilot_segments + 1}..."):
                    debug_log("Making retry API call", "API")
                    retry_response = call_llm_api_with_messages(retry_messages)
                    retry_assistant_message = extract_api_response(retry_response)
                    
                    # Update conversation with retry response
                    st.session_state.conversation.append({"speaker": "Assistant", "message": retry_assistant_message})
                    debug_log(f"Received retry response: {len(retry_assistant_message)} characters", "PROCESS")
                    
                    if "API Error:" in retry_assistant_message:
                        st.session_state.autopilot_running = False
                        st.session_state.autopilot_continue_next_batch = False
                        st.error("🚨 **AUTOPILOT STOPPED - RETRY FAILED**")
                        st.warning(f"API error on retry: {retry_assistant_message}")
                        return
                    
                    # Use retry response for validation
                    assistant_message = retry_assistant_message
            
            # Validate the response quality with severity-based assessment
            debug_log("Starting response validation", "PROCESS")
            validation_result = validate_autopilot_response(assistant_message, original_segments, processed_segments, batch_size)
            
            # Display quality metrics
            quality_score = validation_result.get('quality_score', 0.0)
            segments_processed = validation_result.get('segments_processed', 0)
            warnings = validation_result.get('warnings', [])
            should_stop = validation_result.get('should_stop_autopilot', False)
            stop_reason = validation_result.get('stop_reason', '')
            
            debug_log(f"Validation complete: quality={quality_score:.2f}, processed={segments_processed}, warnings={len(warnings)}, should_stop={should_stop}", "PROCESS")
            
            # Handle severity-based stopping
            if should_stop:
                st.session_state.autopilot_running = False
                st.session_state.autopilot_continue_next_batch = False
                
                # Show severity-based stopping message
                st.error("🚨 **AUTOPILOT STOPPED - QUALITY THRESHOLD EXCEEDED**")
                
                # Progress context
                progress_pct = (processed_segments/len(original_segments)*100)
                st.warning(f"""
**📊 PROCESSING STATUS:**
• ⏸️ **Stopped at:** Segment {processed_segments + 1} of {len(original_segments)}
• 📈 **Progress:** {progress_pct:.1f}% complete
• 🎯 **Last successful batch:** Segments {max(1, processed_segments - batch_size + 1)}-{processed_segments}
• 🔍 **Quality score:** {quality_score:.2f}/1.0

**🛑 STOPPING REASON:**
{stop_reason}
                """)
                
                # Show detailed quality report
                quality_report = generate_quality_report()
                st.markdown("### 📊 Quality Report")
                st.markdown(quality_report)
                
                # Enhanced next steps with severity-specific recommendations
                severe_count = len(st.session_state.quality_report['severe_warnings'])
                moderate_count = len(st.session_state.quality_report['moderate_warnings'])
                
                if severe_count > 0:
                    st.error(f"""
**🚨 SEVERE ISSUES DETECTED ({severe_count}):**
These require immediate attention before continuing.

**🔧 RECOMMENDED ACTIONS:**
1. **Review Severe Issues:** Check the quality report above for specific problems
2. **Adjust Settings:** Consider changing model, batch size, or word limits
3. **Manual Correction:** Fix problematic segments manually
4. **Restart Carefully:** Only restart after addressing severe issues
                    """)
                else:
                    st.warning(f"""
**⚠️ MODERATE ISSUES THRESHOLD REACHED ({moderate_count}):**
Quality concerns suggest process review is needed.

**💡 RECOMMENDED ACTIONS:**
1. **Review Quality Report:** Check moderate issues above
2. **Fine-tune Settings:** Consider adjusting parameters
3. **Continue Manually:** Process remaining segments individually
4. **Export Current:** Download successful results so far
                    """)
                
                st.info("💾 **YOUR PROGRESS IS SAVED:** All successfully processed segments are available for export in the Conversation tab.")
                return
            
            # Show warnings if any (but continue processing since no severe issues)
            if warnings:
                st.warning("⚠️ Quality warnings detected (processing continues):")
                for warning in warnings[:5]:  # Show only first 5 warnings
                    st.write(f"• {warning}")
                if len(warnings) > 5:
                    st.write(f"... and {len(warnings) - 5} more warnings")
            
            # Success message with quality metrics
            quality_emoji = "🟢" if quality_score > 0.8 else "🟡" if quality_score > 0.6 else "🟠"
            st.success(f"✅ Batch completed! Processed segments {processed_segments + 1} to {processed_segments + batch_size}")
            st.info(f"{quality_emoji} Quality Score: {quality_score:.2f}/1.0 | Warnings: {len(warnings)}")
            
            # Recalculate processed segments AFTER successful batch processing
            current_processed_segments = count_processed_segments()
            total_segments = len(original_segments)
            debug_log(f"UPDATED CONTINUATION CHECK: processed_before={processed_segments}, processed_now={current_processed_segments}, total={total_segments}, should_continue={current_processed_segments < total_segments}", "DEBUG")
            
            # Continue to next batch automatically (no delay/countdown)
            if st.session_state.autopilot_running and current_processed_segments < len(original_segments):
                # Show quick status and continue immediately
                debug_log("Continuing to next batch automatically", "PROCESS")
                st.info("🚀 Continuing to next batch...")
                # Set flag to continue processing after rerun
                st.session_state.autopilot_continue_next_batch = True
                time.sleep(st.session_state.autopilot_delay)  # Use configured delay
                st.rerun()
            else:
                # Autopilot completion - stop and prepare for export
                debug_log(f"Autopilot stopping: running={st.session_state.autopilot_running}, current_processed={current_processed_segments}, total={total_segments}, more_segments={current_processed_segments < total_segments}", "PROCESS")
                st.session_state.autopilot_running = False
                st.session_state.autopilot_continue_next_batch = False  # Reset flag
                
                # Determine why autopilot stopped and show appropriate message
                if not st.session_state.autopilot_running:
                    st.info("🛑 **AUTOPILOT STOPPED - USER REQUESTED**")
                    st.warning(f"""
**⏸️ MANUAL STOP DETECTED:**
• 🎯 **Processed:** {processed_segments + batch_size} of {len(original_segments)} segments
• 📊 **Progress:** {((processed_segments + batch_size)/len(original_segments)*100):.1f}% complete
• 🔄 **Status:** Stopped by user action

**▶️ RESUME OPTIONS:**
• **Continue:** Restart autopilot to process remaining {len(original_segments) - (processed_segments + batch_size)} segments
• **Export:** Download current results from Conversation tab
• **Manual:** Process remaining segments individually
                    """)
                elif processed_segments + batch_size >= len(original_segments):
                    st.success("🎉 **ALL SEGMENTS COMPLETED!**")
                else:
                    st.info("⏹️ **AUTOPILOT REACHED END CONDITION**")
                    st.warning(f"""
**🏁 PROCESSING COMPLETE:**
• ✅ **Completed:** {processed_segments + batch_size} of {len(original_segments)} segments
• 📊 **Final progress:** {((processed_segments + batch_size)/len(original_segments)*100):.1f}%
• 🎬 **Ready for export**
                    """)
                
                handle_autopilot_completion(current_processed_segments >= len(original_segments))
    
    except Exception as e:
        st.session_state.autopilot_running = False
        st.session_state.autopilot_continue_next_batch = False  # Reset flag on error
        debug_log(f"Autopilot stopped due to exception: {str(e)}", "ERROR")
        
        # Enhanced exception error message with context
        processed_segments = count_processed_segments()
        original_segments = parse_srt_segments(st.session_state.english_sub_content)
        progress_pct = (processed_segments/len(original_segments)*100) if original_segments else 0
        
        st.error("🚨 **AUTOPILOT STOPPED - UNEXPECTED ERROR**")
        st.error(f"**💥 Error Details:** {str(e)}")
        st.warning(f"""
**📊 STATUS AT ERROR:**
• 🎯 **Processed:** {processed_segments} of {len(original_segments) if original_segments else 0} segments
• 📈 **Progress:** {progress_pct:.1f}% complete
• ⚠️ **Error Type:** System exception

**🔧 RECOVERY OPTIONS:**
1. **Restart Autopilot:** Try running autopilot again (progress is saved)
2. **Check Settings:** Verify API key, model, and batch size settings
3. **Manual Processing:** Continue with manual segment processing
4. **Export Current:** Download processed segments from Conversation tab

**💡 TROUBLESHOOTING:**
• If error persists, try reducing batch size to 3-10 segments
• Check internet connection and API service status
• Review conversation in debug mode for more details
        """)
        
        # Force conversation refresh on error
        st.session_state.conversation_needs_refresh = True

def handle_autopilot_completion(all_completed: bool = True):
    """Handle autopilot completion and prepare for export"""
    debug_log("Handling autopilot completion", "PROCESS")
    
    # Mark that conversation needs refresh for export
    st.session_state.conversation_needs_refresh = True
    
    if all_completed:
        debug_log("All segments processed - autopilot completed successfully", "SUCCESS")
        
        # Comprehensive success message with export guidance
        st.balloons()
        st.success("🏆 **AUTOPILOT MISSION ACCOMPLISHED!**")
        
        # Show completion summary
        all_assistant_content = "\n".join([entry['message'] for entry in st.session_state.conversation if entry['speaker'] == 'Assistant'])
        segments = extract_bilingual_segments(all_assistant_content)
        
        if segments:
            st.success(f"""
**🎬 BILINGUAL SUBTITLES GENERATED SUCCESSFULLY!**

**📊 FINAL STATISTICS:**
• ✅ **Total segments created:** {len(segments)}
• 🌐 **Target language:** {st.session_state.target_language}
• 🧠 **Processing method:** Comprehensive Full-Context Approach
• 📝 **Format:** Bilingual SRT/VTT ready

**📁 EXPORT YOUR RESULTS:**
1. **Switch to Conversation tab** (click tab above)
2. **Click "Export Subtitles"** button
3. **Choose format:** SRT or VTT
4. **Download** your bilingual subtitle file

**🎯 YOUR SUBTITLES ARE READY FOR USE!**
            """)
        else:
            st.warning("⚠️ **COMPLETION WITH ISSUES**")
            st.warning("""
**📊 PROCESSING COMPLETED BUT:**
• ❌ No bilingual segments were detected in the final output
• 🔍 This might indicate format issues in AI responses
• 💾 Raw conversation data is still available

**🔧 NEXT STEPS:**
1. **Review Conversation:** Check responses in Conversation tab
2. **Manual Export:** Try extracting segments manually
3. **Restart:** Consider reprocessing with smaller batch sizes
            """)
    else:
        debug_log("Autopilot stopped by user", "INFO")
        
        # Enhanced stopped-by-user message
        processed_segments = count_processed_segments()
        original_segments = parse_srt_segments(st.session_state.english_sub_content)
        remaining = len(original_segments) - processed_segments if original_segments else 0
        progress_pct = (processed_segments/len(original_segments)*100) if original_segments else 0
        
        st.info("🛑 **AUTOPILOT PAUSED BY USER**")
        st.warning(f"""
**📊 CURRENT STATUS:**
• ✅ **Completed:** {processed_segments} segments
• ⏳ **Remaining:** {remaining} segments
• 📈 **Progress:** {progress_pct:.1f}% complete

**▶️ YOUR OPTIONS:**
1. **Resume:** Restart autopilot to continue processing
2. **Export Partial:** Download current results from Conversation tab
3. **Manual Mode:** Process remaining segments individually
4. **Review:** Check quality of current results before continuing

**💾 PROGRESS SAVED:** All completed work is preserved and ready for export.
        """)
    
    # Add navigation hint
    st.info("💡 **Ready to export?** Switch to the **Conversation tab** above to download your bilingual subtitles in SRT/VTT format.")

def count_processed_segments() -> int:
    """Count how many segments have been processed in the conversation"""
    processed_count = 0
    
    debug_log(f"Counting processed segments from {len(st.session_state.conversation)} conversation entries", "DEBUG")
    
    for i, entry in enumerate(st.session_state.conversation):
        if entry['speaker'] == "Assistant":
            segments = extract_bilingual_segments(entry['message'])
            debug_log(f"Entry {i}: Found {len(segments)} bilingual segments", "DEBUG")
            processed_count += len(segments)
    
    debug_log(f"Total processed segments counted: {processed_count}", "DEBUG")
    return processed_count

def validate_autopilot_response(response_text: str, original_segments: List[Dict[str, Any]], start_index: int, batch_size: int) -> Dict[str, Any]:
    """Validate the quality of an autopilot response with severity-based quality assessment"""
    debug_log(f"Validating autopilot response: start_index={start_index}, batch_size={batch_size}", "PROCESS")
    
    issues = []
    warnings = []
    quality_scores = []
    
    # Update quality report statistics
    st.session_state.quality_report['total_segments_processed'] += batch_size
    
    # Extract generated segments from response
    generated_segments = extract_bilingual_segments(response_text)
    debug_log(f"Extracted {len(generated_segments)} bilingual segments from response", "PROCESS")
    
    if len(generated_segments) == 0:
        debug_log("No bilingual segments found in response", "ERROR")
        
        # Add severe warning for missing bilingual format
        add_quality_warning(
            'missing_bilingual_format',
            'No bilingual segments detected - check SRT format with || separators',
            details={'response_length': len(response_text), 'has_pipes': '||' in response_text}
        )
        
        # Provide more detailed debugging information
        debug_info = []
        debug_info.append("🔍 DEBUG: No bilingual segments detected")
        debug_info.append(f"Response length: {len(response_text)} characters")
        
        # Check if response contains || patterns
        if '||' in response_text:
            pipe_count = response_text.count('||')
            debug_info.append(f"Found {pipe_count} || separators in response")
        else:
            debug_info.append("No || separators found in response")
        
        # Check if response contains timestamps
        timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2},\d{3}')
        timestamps = timestamp_pattern.findall(response_text)
        if timestamps:
            debug_info.append(f"Found {len(timestamps)} timestamp patterns")
        else:
            debug_info.append("No timestamp patterns found")
        
        # Show a preview of the response
        preview = response_text[:300] + "..." if len(response_text) > 300 else response_text
        debug_info.append(f"Response preview: {preview}")
        
        # Log detailed debug information
        for info in debug_info:
            debug_log(info, "DEBUG")
        
        issues.append("No bilingual segments found in response - check that AI is responding in the correct SRT format with || separators")
        
        # Check if we should stop autopilot
        should_stop, stop_reason = should_stop_autopilot()
        return {
            'valid': False, 
            'should_stop_autopilot': should_stop,
            'stop_reason': stop_reason,
            'issues': issues, 
            'warnings': warnings, 
            'quality_score': 0.0, 
            'debug_info': debug_info
        }
    
    # Check if we got the expected number of segments
    if len(generated_segments) != batch_size:
        if len(generated_segments) < batch_size:
            debug_log(f"Missing segments: expected {batch_size}, got {len(generated_segments)}", "WARNING")
            add_quality_warning(
                'segment_count_mismatch',
                f"Missing segments: Expected {batch_size}, got {len(generated_segments)}",
                details={'expected': batch_size, 'actual': len(generated_segments)}
            )
            issues.append(f"Missing segments: Expected {batch_size}, got {len(generated_segments)}")
        else:
            debug_log(f"Extra segments: expected {batch_size}, got {len(generated_segments)}", "WARNING")
            add_quality_warning(
                'segment_count_mismatch',
                f"Extra segments: Expected {batch_size}, got {len(generated_segments)}",
                details={'expected': batch_size, 'actual': len(generated_segments)}
            )
            warnings.append(f"Extra segments: Expected {batch_size}, got {len(generated_segments)}")
    
    # Get the corresponding original segments
    end_index = min(start_index + batch_size, len(original_segments))
    expected_segments = original_segments[start_index:end_index]
    
    # Track segments with issues for quality report
    segments_with_issues = 0
    
    # DYNAMIC validation based on actual English segments and target language
    if len(generated_segments) > 0 and start_index < len(original_segments):
        # Get the English segments for this batch
        batch_english_segments = original_segments[start_index:end_index]
        
        # Progressive word limit validation using the unified system
        try:
            # Validate each segment against its calculated progressive limit
            for i, gen_seg in enumerate(generated_segments):
                if i < len(batch_english_segments):
                    eng_segment = batch_english_segments[i]
                    eng_words = len(eng_segment['text'].split())
                    # Use the same progressive calculation as the prompt
                    expected_max_words = calculate_progressive_word_limit(eng_words, st.session_state.target_language)
                    actual_words = len(gen_seg['target_text'].split())
                    
                    if actual_words > expected_max_words:
                        excess_words = actual_words - expected_max_words
                        add_quality_warning(
                            'word_limit_exceeded',
                            f"Segment {gen_seg['number']} exceeds progressive limit: {actual_words} words (max: {expected_max_words})",
                            segment_number=gen_seg['number'],
                            details={'excess_words': excess_words, 'english_words': eng_words}
                        )
                        issues.append(f"PROGRESSIVE LIMIT EXCEEDED: Segment {gen_seg['number']} has {actual_words} words (max: {expected_max_words} for {eng_words} English words)")
                        segments_with_issues += 1
                    elif actual_words > expected_max_words * 0.85:  # Warning at 85% of limit
                        add_quality_warning(
                            'word_limit_warning',
                            f"Segment {gen_seg['number']} near limit: {actual_words} words (limit: {expected_max_words})",
                            segment_number=gen_seg['number'],
                            details={'words': actual_words, 'limit': expected_max_words}
                        )
                        warnings.append(f"Near limit: Segment {gen_seg['number']} has {actual_words} words (limit: {expected_max_words})")
            
            debug_log(f"Progressive word limit validation completed for {st.session_state.get('target_language', 'language')}", "PROCESS")
        except Exception as e:
            debug_log(f"Progressive validation failed, using basic fallback: {e}", "WARNING")
            # Fallback to basic validation if progressive calculation fails
            for gen_seg in generated_segments:
                word_count = len(gen_seg['target_text'].split())
                if word_count > 30:  # Universal hard cap
                    add_quality_warning(
                        'word_limit_exceeded',
                        f"Segment {gen_seg['number']} exceeds universal limit: {word_count} words (max: 30)",
                        segment_number=gen_seg['number'],
                        details={'excess_words': word_count - 30}
                    )
                    issues.append(f"UNIVERSAL LIMIT EXCEEDED: Segment {gen_seg['number']} has {word_count} words (max: 30 universal hard cap)")
                    segments_with_issues += 1
                elif word_count > 20:  # Universal warning
                    add_quality_warning(
                        'word_limit_warning',
                        f"Segment {gen_seg['number']} has high word count: {word_count} words",
                        segment_number=gen_seg['number'],
                        details={'words': word_count}
                    )
                    warnings.append(f"High word count: Segment {gen_seg['number']} has {word_count} words (consider optimizing)")
    
    # Validate each segment with enhanced quality checking
    for i, gen_seg in enumerate(generated_segments):
        if i < len(expected_segments):
            original_seg = expected_segments[i]
            
            # Validate segment quality with enhanced validation
            segment_validation = validate_segment_quality(original_seg, gen_seg, st.session_state.target_language)
            
            # NEW: Add source text validation to detect AI generation
            if hasattr(st.session_state, 'translation_content') and st.session_state.translation_content:
                source_validator = SourceTextValidator(st.session_state.translation_content)
                
                if 'target_text' in gen_seg and gen_seg['target_text']:
                    source_validation = source_validator.validate_text_is_from_source(gen_seg['target_text'])
                    generation_check = source_validator.detect_generated_content(gen_seg['target_text'])
                    
                    debug_log(f"Source validation for segment {gen_seg['number']}: overlap={source_validation.word_overlap_score:.2f}, from_source={source_validation.is_from_source}, generated={generation_check.likely_generated}", "DEBUG")
                    
                    # Handle severe source violations (AI generation detected)
                    if generation_check.likely_generated or not source_validation.is_from_source:
                        if generation_check.likely_generated:
                            issue_type = 'ai_content_generation'
                            severity = 'severe'
                            message = f"AI generation detected in segment {gen_seg['number']} - text not from source translation"
                        else:
                            severity = 'severe' if source_validation.word_overlap_score < LOW_SOURCE_OVERLAP_THRESHOLD else 'moderate'
                            issue_type = 'source_text_mismatch'
                            message = f"Segment {gen_seg['number']} text doesn't match source translation (overlap: {source_validation.word_overlap_score:.2f})"
                        
                        add_quality_warning(
                            issue_type,
                            message,
                            segment_number=gen_seg['number'],
                            details={
                                'word_overlap_score': source_validation.word_overlap_score,
                                'phrase_match': source_validation.phrase_match,
                                'substring_match': source_validation.substring_match,
                                'target_text': gen_seg['target_text'][:100] + '...' if len(gen_seg['target_text']) > 100 else gen_seg['target_text']
                            }
                        )
                        
                        if severity == 'severe':
                            issues.append(f"SEVERE: {message}")
                            segments_with_issues += 1
                        else:
                            warnings.append(f"MODERATE: {message}")
            
            # Process segment-level issues and warnings with quality tracking
            segment_has_issues = False
            if not segment_validation['valid'] or segment_validation['issues']:
                segment_has_issues = True
                for issue in segment_validation['issues']:
                    # Classify issue type for quality tracking
                    if 'text mismatch' in issue.lower():
                        similarity_match = re.search(r'similarity: ([\d.]+)', issue)
                        similarity = float(similarity_match.group(1)) if similarity_match else 0.0
                        add_quality_warning(
                            'english_preservation',
                            f"Segment {gen_seg['number']}: {issue}",
                            segment_number=gen_seg['number'],
                            details={'similarity': similarity}
                        )
                    elif 'timestamp' in issue.lower():
                        add_quality_warning(
                            'timestamp_mismatch',
                            f"Segment {gen_seg['number']}: {issue}",
                            segment_number=gen_seg['number']
                        )
                    elif 'empty' in issue.lower() or 'too short' in issue.lower():
                        # Empty or very short segments are severe warnings
                        add_quality_warning(
                            'empty_target_text',
                            f"Segment {gen_seg['number']}: {issue}",
                            segment_number=gen_seg['number'],
                            details={'message': issue}
                        )
                    else:
                        add_quality_warning(
                            'format_violation',
                            f"Segment {gen_seg['number']}: {issue}",
                            segment_number=gen_seg['number']
                        )
                    issues.append(f"Segment {gen_seg['number']}: {issue}")
            
            if 'warnings' in segment_validation and segment_validation['warnings']:
                for warning in segment_validation['warnings']:
                    add_quality_warning(
                        'format_warning',
                        f"Segment {gen_seg['number']}: {warning}",
                        segment_number=gen_seg['number']
                    )
                    warnings.append(f"Segment {gen_seg['number']}: {warning}")
            
            if segment_has_issues:
                segments_with_issues += 1
            
            if 'quality_score' in segment_validation:
                quality_scores.append(segment_validation['quality_score'])
        else:
            add_quality_warning(
                'format_violation',
                f"Unexpected extra segment {gen_seg['number']}",
                segment_number=gen_seg['number']
            )
            issues.append(f"Unexpected extra segment {gen_seg['number']}")
            segments_with_issues += 1
    
    # Enhanced content bleeding detection
    if len(generated_segments) > 1:
        bleeding_issues = check_content_bleeding(generated_segments, expected_segments)
        if bleeding_issues:
            add_quality_warning(
                'content_bleeding',
                f"Content bleeding detected in batch",
                details={'affected_segments': len(bleeding_issues)}
            )
            issues.extend(bleeding_issues)
            segments_with_issues = min(segments_with_issues + len(bleeding_issues), len(generated_segments))
    
    # Update quality report with segments that had issues
    st.session_state.quality_report['segments_with_issues'] += segments_with_issues
    
    # Calculate overall quality metrics
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    # Semantic consistency check across the batch
    if len(generated_segments) > 1:
        semantic_issues = check_batch_semantic_consistency(generated_segments, expected_segments, st.session_state.target_language)
        warnings.extend(semantic_issues)
    
    # Determine if autopilot should stop based on severity
    should_stop, stop_reason = should_stop_autopilot()
    
    # Determine overall validation result with severity considerations
    critical_issue_count = len([issue for issue in issues if any(keyword in issue.lower() for keyword in ['mismatch', 'missing', 'bleeding', 'identical'])])
    
    # Allow continuation if only moderate warnings
    is_valid = not should_stop and critical_issue_count == 0 and avg_quality > 0.6
    
    # Log validation summary
    debug_log(f"Validation complete: valid={is_valid}, should_stop={should_stop}, issues={len(issues)}, warnings={len(warnings)}", "PROCESS")
    if should_stop:
        debug_log(f"Autopilot stopping reason: {stop_reason}", "WARNING")
    
    return {
        'valid': is_valid,
        'should_stop_autopilot': should_stop,
        'stop_reason': stop_reason,
        'issues': issues,
        'warnings': warnings,
        'quality_score': avg_quality,
        'segments_processed': len(generated_segments),
        'expected_segments': batch_size
    }

def check_batch_semantic_consistency(generated_segments: List[Dict[str, Any]], original_segments: List[Dict[str, Any]], target_language: str) -> List[str]:
    """Check semantic consistency across a batch of segments"""
    warnings = []
    
    if len(generated_segments) != len(original_segments):
        return warnings  # Skip if segments don't match
    
    # Check for consistent translation quality across the batch
    quality_scores = []
    length_ratios = []
    
    for i, (gen_seg, orig_seg) in enumerate(zip(generated_segments, original_segments)):
        # Calculate quality metrics for each segment
        english_len = len(gen_seg['english_text'])
        target_len = len(gen_seg['target_text'])
        
        if english_len > 0:
            length_ratio = target_len / english_len
            length_ratios.append(length_ratio)
            
            # Simple quality heuristic based on length and content
            content_density = len(gen_seg['target_text'].split()) / max(len(gen_seg['english_text'].split()), 1)
            quality_scores.append(content_density)
    
    # Check for outliers in length ratios
    if len(length_ratios) > 2:
        avg_ratio = sum(length_ratios) / len(length_ratios)
        for i, ratio in enumerate(length_ratios):
            if abs(ratio - avg_ratio) > avg_ratio * 0.5:  # 50% deviation from average
                warnings.append(f"Segment {generated_segments[i]['number']} has unusual length ratio: {ratio:.2f} vs batch average {avg_ratio:.2f}")
    
    # Check for consistent translation style
    # Look for sudden changes in formality or style within the batch
    formal_indicators = ['please', 'thank you', 'sir', 'madam', 'kindly', 'respectfully']
    informal_indicators = ['hey', 'yeah', 'ok', 'gonna', 'wanna', 'cool']
    
    formal_count = 0
    informal_count = 0
    
    for gen_seg in generated_segments:
        target_lower = gen_seg['target_text'].lower()
        formal_count += sum(1 for indicator in formal_indicators if indicator in target_lower)
        informal_count += sum(1 for indicator in informal_indicators if indicator in target_lower)
    
    # If there's a strong mix of formal and informal, it might indicate inconsistency
    if formal_count > 0 and informal_count > 0 and abs(formal_count - informal_count) < 2:
        warnings.append(f"Mixed formality levels detected in batch: {formal_count} formal vs {informal_count} informal indicators")
    
    return warnings

def clean_conversation_encoding():
    """Clean up any corrupted Unicode characters in the conversation"""
    if not st.session_state.conversation:
        return
    
    cleaned_conversation = []
    for entry in st.session_state.conversation:
        try:
            # Ensure proper encoding
            message = entry['message'].encode('utf-8', errors='replace').decode('utf-8')
            cleaned_entry = {
                'speaker': entry['speaker'],
                'message': message
            }
            cleaned_conversation.append(cleaned_entry)
        except Exception as e:
            # If there's any encoding issue, create a fallback entry
            cleaned_entry = {
                'speaker': entry['speaker'],
                'message': f"[Content encoding error: {str(e)}]"
            }
            cleaned_conversation.append(cleaned_entry)
    
    st.session_state.conversation = cleaned_conversation

def show_autopilot_dashboard():
    """Enhanced autopilot dashboard with centralized warnings and improved readability"""
    st.markdown("# 🚁 Autopilot Dashboard")
    
    try:
        # Get current progress data
        original_segments = parse_srt_segments(st.session_state.english_sub_content) if st.session_state.english_sub_content else []
        processed_segments = count_processed_segments()
        total_segments = len(original_segments)
        
        if total_segments > 0:
            progress_percentage = min(processed_segments / total_segments, 1.0)
            remaining_segments = max(0, total_segments - processed_segments)
            
            # 🎯 CENTRALIZED STATUS OVERVIEW (Top Priority Section)
            st.markdown("## 📊 Status Overview")
            
            # Create compact status cards in a grid
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Progress Card
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h3 style="margin: 0; color: #1f77b4;">📈 Progress</h3>
                    <h2 style="margin: 0.5rem 0;">{processed_segments}/{total_segments}</h2>
                    <p style="margin: 0; color: #666;">{progress_percentage:.1%} Complete</p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(progress_percentage)
                
            with col2:
                # Status Card
                status_color = "#28a745" if st.session_state.get('autopilot_running', False) else "#ffc107"
                status_text = "🟢 RUNNING" if st.session_state.get('autopilot_running', False) else "⏸️ PAUSED"
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h3 style="margin: 0; color: {status_color};">🔄 Status</h3>
                    <h2 style="margin: 0.5rem 0;">{status_text}</h2>
                    <p style="margin: 0; color: #666;">{remaining_segments} remaining</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Control button
                if st.session_state.get('autopilot_running', False):
                    if st.button("⏹️ **STOP**", type="secondary", use_container_width=True):
                        st.session_state.autopilot_running = False
                        st.session_state.autopilot_continue_next_batch = False
                        st.success("🛑 Autopilot stopped")
                        st.rerun()
                else:
                    if st.button("▶️ **START**", type="primary", use_container_width=True):
                        if remaining_segments > 0:
                            st.session_state.autopilot_running = True
                            st.session_state.autopilot_continue_next_batch = True
                            st.success("🚀 Autopilot started")
                            st.rerun()
                        else:
                            st.info("✅ All segments processed!")
            
            with col3:
                # Quality & Warnings Card (CENTRALIZED WARNINGS)
                severe_warnings = len(st.session_state.quality_report.get('severe_warnings', []))
                moderate_warnings = len(st.session_state.quality_report.get('moderate_warnings', []))
                total_warnings = severe_warnings + moderate_warnings
                
                quality_score = calculate_simple_quality_score()
                
                if total_warnings > 0:
                    warning_color = "#dc3545" if severe_warnings > 0 else "#fd7e14"
                    warning_text = f"🚨 {severe_warnings} Severe" if severe_warnings > 0 else f"⚠️ {moderate_warnings} Moderate"
                else:
                    warning_color = "#28a745"
                    warning_text = "✅ All Clear"
                
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h3 style="margin: 0; color: {warning_color};">🎯 Quality</h3>
                    <h2 style="margin: 0.5rem 0;">{quality_score:.2f}</h2>
                    <p style="margin: 0; color: #666;">{warning_text}</p>
                </div>
                """, unsafe_allow_html=True)

            
            # 🚨 CENTRALIZED WARNINGS PANEL (Only show if warnings exist)
            if total_warnings > 0:
                st.markdown("---")
                st.markdown("## ⚠️ Warnings & Issues")
                
                # Create tabs for different warning types
                if severe_warnings > 0 and moderate_warnings > 0:
                    warning_tab1, warning_tab2 = st.tabs([f"🚨 Severe ({severe_warnings})", f"⚠️ Moderate ({moderate_warnings})"])
                    
                    with warning_tab1:
                        for warning in st.session_state.quality_report.get('severe_warnings', [])[:5]:
                            st.error(f"**Segment #{warning.get('segment_number', 'N/A')}:** {warning.get('message', 'No details')}")
                    
                    with warning_tab2:
                        for warning in st.session_state.quality_report.get('moderate_warnings', [])[:5]:
                            st.warning(f"**Segment #{warning.get('segment_number', 'N/A')}:** {warning.get('message', 'No details')}")
                            
                elif severe_warnings > 0:
                    st.markdown("### 🚨 Severe Issues")
                    for warning in st.session_state.quality_report.get('severe_warnings', [])[:5]:
                        st.error(f"**Segment #{warning.get('segment_number', 'N/A')}:** {warning.get('message', 'No details')}")
                        
                elif moderate_warnings > 0:
                    st.markdown("### ⚠️ Moderate Issues")
                    for warning in st.session_state.quality_report.get('moderate_warnings', [])[:5]:
                        st.warning(f"**Segment #{warning.get('segment_number', 'N/A')}:** {warning.get('message', 'No details')}")
                
                # Show total if more warnings exist
                if total_warnings > 5:
                    st.info(f"📋 **{total_warnings - 5} more warnings** available in detailed quality report")
            
            # 📊 COMPACT ACTIVITY LOG
            st.markdown("---")
            st.markdown("## 📝 Recent Activity")
            
            recent_responses = st.session_state.conversation[-3:] if len(st.session_state.conversation) >= 3 else st.session_state.conversation
            
            if recent_responses:
                # Show recent batches in a more compact table format
                activity_data = []
                for i, entry in enumerate(reversed(recent_responses)):
                    if entry['speaker'] == 'Assistant':
                        segments_in_response = len(extract_bilingual_segments(entry['message']))
                        segments = extract_bilingual_segments(entry['message'])
                        if segments:
                            batch_range = f"#{segments[0]['number']}-#{segments[-1]['number']}"
                            activity_data.append({
                                "Batch": f"#{len(recent_responses)-i}",
                                "Segments": f"{segments_in_response} segments",
                                "Range": batch_range,
                                "Status": "✅ Completed"
                            })
                
                if activity_data:
                    import pandas as pd
                    df = pd.DataFrame(activity_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Expandable section for detailed view
                    with st.expander("🔍 View Detailed Activity"):
                        for i, entry in enumerate(reversed(recent_responses)):
                            if entry['speaker'] == 'Assistant':
                                segments = extract_bilingual_segments(entry['message'])
                                if segments:
                                    st.markdown(f"**Batch #{len(recent_responses)-i}** - {len(segments)} segments")
                                    # Show first and last segment as examples
                                    st.code(f"#{segments[0]['number']}: {segments[0]['english_text'][:50]}... || {segments[0]['target_text'][:50]}...")
                                    if len(segments) > 1:
                                        st.code(f"#{segments[-1]['number']}: {segments[-1]['english_text'][:50]}... || {segments[-1]['target_text'][:50]}...")
                                    st.markdown("---")
            else:
                st.info("🔄 No recent activity. Start autopilot to begin processing.")
            
            # ⚙️ SETTINGS PANEL
            st.markdown("---")
            st.markdown("## ⚙️ Configuration")
            
            settings_col1, settings_col2 = st.columns(2)
            
            with settings_col1:
                st.markdown("**Processing Settings:**")
                new_segments = st.slider(
                    "Segments per batch",
                    min_value=3,
                    max_value=30,
                    value=st.session_state.autopilot_segments,
                    help="Smaller batches = higher quality, larger batches = faster processing"
                )
                if new_segments != st.session_state.autopilot_segments:
                    st.session_state.autopilot_segments = new_segments
                    st.success(f"✅ Batch size updated to {new_segments}")
                
                new_delay = st.slider(
                    "Delay between batches (seconds)",
                    min_value=1.0,
                    max_value=5.0,
                    value=st.session_state.autopilot_delay,
                    step=0.5,
                    help="Delay between API requests to prevent rate limiting"
                )
                if new_delay != st.session_state.autopilot_delay:
                    st.session_state.autopilot_delay = new_delay
                    st.success(f"✅ Delay updated to {new_delay}s")
            
            with settings_col2:
                st.markdown("**Current Configuration:**")
                st.write(f"• **Provider:** {st.session_state.llm_provider.title()}")
                st.write(f"• **Model:** {st.session_state.model}")
                st.write(f"• **Language:** {st.session_state.target_language.title()}")
                st.write(f"• **Batch Size:** {st.session_state.autopilot_segments} segments")
                st.write(f"• **Delay:** {st.session_state.autopilot_delay}s between batches")
                
                # Estimate time remaining
                avg_time_per_segment = 8  # seconds
                est_minutes = (remaining_segments * avg_time_per_segment) / 60
                st.write(f"• **Est. Time:** {est_minutes:.1f} minutes")
                
                # Quality report management
                st.markdown("---")
                total_warnings = len(st.session_state.quality_report.get('warnings', []))
                if total_warnings > 0:
                    st.markdown("**Quality Report:**")
                    st.write(f"• **Total Warnings:** {total_warnings}")
                    if st.button("🗑️ Reset Quality Report", help="Clear all warnings and start fresh"):
                        st.session_state.quality_report = {
                            'warnings': [],
                            'severe_warnings': [],
                            'moderate_warnings': [],
                            'total_segments_processed': 0,
                            'segments_with_issues': 0,
                            'last_updated': None
                        }
                        st.success("✅ Quality report reset!")
                        st.rerun()
        
        else:
            st.warning("📂 **No segments loaded.** Please upload subtitle files in the 'Prompt & Files' tab.")
        
    except Exception as e:
        st.error(f"❌ Error loading autopilot dashboard: {str(e)}")

def setup_tab():
    """App Instructions tab with comprehensive usage guide"""
    st.header("🎬 Bilingual Subtitle Generator")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### How to Use This Tool:
        
        **🔧 1. Configure AI Provider**
        - Use the **sidebar** (← left panel) to select your AI provider
        - Choose from Claude, OpenAI GPT, Google Gemini, or OpenRouter
        - Enter your API key and select a model
        
        **📁 2. Upload Files**
        - Go to **"Prompt & Files"** tab
        - Upload your English SRT/VTT subtitle file
        - Upload your translation file (TXT or DOCX)
        - Confirm the target language
        
        **🎯 3. Generate Subtitles**
        - Submit your initial prompt to start
        - Use **Autopilot** for fully automated processing
        - Or process manually with batch buttons
        
        **💾 4. Export Results**
        - Download bilingual SRT/VTT files
        - Export target-language-only files
        - Save conversation history
        
        **✨ Key Features:**
        - **Latest AI Models**: GPT-5, GPT-4.1 Nano, Claude 3.5, and 50+ models via OpenRouter
        - **Smart Quality Control**: Prevents content bleeding and format errors
        - **Multi-Format Support**: SRT, VTT, TXT, and DOCX files
        - **Autopilot Mode**: Hands-free batch processing with quality monitoring
        
        Create professional bilingual subtitles with advanced AI and quality validation.
        """)
    
    with col2:
        st.markdown("### 🔑 API Setup Guide")
        
        st.info("💡 **Configure your AI provider in the sidebar** (← left panel)")
        
        # Simplified API key information
        st.markdown("""
        **Get your API keys:**
        
        🤖 **Claude** - [Get Key](https://console.anthropic.com/settings/keys)
        - High-quality translations, moderate cost
        
        🧠 **OpenAI** - [Get Key](https://platform.openai.com/api-keys)  
        - GPT-5, GPT-4.1 Nano, GPT-4o models
        
        🌟 **Gemini** - [Get Key](https://makersuite.google.com/app/apikey)
        - Fast and cost-effective processing
        
        🚀 **OpenRouter** - [Get Key](https://openrouter.ai/keys)
        - Access 50+ models: GPT-5, Claude, Llama, Mistral
        - Pay-per-use, competitive pricing
        """)
        
        # Show current provider status
        current_provider = st.session_state.llm_provider
        provider_names = {"claude": "Claude", "openai": "OpenAI", "gemini": "Gemini", "openrouter": "OpenRouter"}
        current_name = provider_names.get(current_provider, "Unknown")
        
        st.markdown(f"**Current Provider:** {current_name}")
        
        current_key = get_current_api_key()
        if current_key:
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ Configure your API key in the sidebar →")

def prompt_files_tab():
    """Prompt & Files tab implementation"""
    st.header("📁 File Upload & Prompt Configuration")
    
    # File upload section
    st.subheader("📄 Input Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**English SRT/VTT File**")
        english_file = st.file_uploader(
            "Choose English subtitle file",
            type=['srt', 'vtt'],
            help="Upload your English SRT or VTT subtitle file"
        )
        
        if english_file is not None:
            try:
                # Read file content
                content = english_file.read().decode('utf-8')
                st.session_state.english_sub_content = content
                st.success(f"✅ English file loaded: {len(content)} characters")
            except UnicodeDecodeError:
                st.error("❌ Error reading file. Please ensure it's a valid text file with UTF-8 encoding.")
    
    with col2:
        st.markdown("**Translation File**")
        translation_file = st.file_uploader(
            "Choose translation file",
            type=['txt', 'docx'],
            help="Upload your translation text file (TXT) or Word document (DOCX)"
        )
        
        if translation_file is not None:
            try:
                # Handle different file types
                if translation_file.name.lower().endswith('.docx'):
                    # Read DOCX file
                    file_content = translation_file.read()
                    content = extract_text_from_docx(file_content)
                    # Clean timestamp patterns from translation
                    content = clean_timestamp_patterns(content)
                    st.session_state.translation_content = content
                    
                    st.success(f"✅ DOCX file loaded and cleaned: {len(content)} characters")
                    st.info("� Please select the target language below")
                else:
                    # Read text file
                    content = translation_file.read().decode('utf-8')
                    # Clean timestamp patterns from translation
                    content = clean_timestamp_patterns(content)
                    st.session_state.translation_content = content
                    
                    st.success(f"✅ Text file loaded and cleaned: {len(content)} characters")
                    st.info("� Please select the target language below")
                
            except UnicodeDecodeError:
                st.error("❌ Error reading text file. Please ensure it's a valid text file with UTF-8 encoding.")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    # Language selection section (always visible when translation content exists)
    if st.session_state.translation_content:
        st.subheader("🌐 Language Configuration")
        
        if not st.session_state.language_confirmed:
            st.markdown("**Please select the target language for translation:**")
            
            # Language selection dropdown
            available_languages = ['english', 'portuguese', 'spanish', 'french', 'german', 'chinese', 'russian', 'italian', 'japanese', 'korean', 'other']
            
            # Get currently selected language or default to first option
            if 'target_language' in st.session_state and st.session_state.target_language in available_languages:
                default_index = available_languages.index(st.session_state.target_language)
            else:
                default_index = 0  # Default to 'english'
            
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_language = st.selectbox(
                    "Target language:",
                    available_languages,
                    index=default_index,
                    format_func=lambda x: x.title() if x != 'other' else 'Other Language',
                    key="target_language_selectbox"
                )
                
                # Confirm button
                if st.button("✅ Confirm Language", type="primary", key="confirm_language_btn"):
                    st.session_state.target_language = selected_language
                    st.session_state.language_confirmed = True
                    debug_log(f"User manually selected and confirmed language: '{selected_language}'", "INFO")
                    st.success(f"✅ Language confirmed: **{selected_language.title()}**")
                    st.rerun()
            
            with col2:
                st.info("🎯 Manual Selection")
                st.caption(f"Selected: **{selected_language.title()}**")
                st.caption("Click 'Confirm Language' to proceed")
        else:
            # Language confirmed - show status and allow changes
            st.success(f"✅ **Language Confirmed:** {st.session_state.target_language.title()}")
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔄 Change Language", type="secondary", key="change_language_btn"):
                    st.session_state.language_confirmed = False
                    debug_log("User clicked 'Change Language' - resetting confirmation", "INFO")
                    st.rerun()
    
    # Validation checkpoint section
    st.subheader("✅ Validation Checkpoint")
    
    # Check if both files are uploaded
    files_uploaded = bool(st.session_state.english_sub_content and st.session_state.translation_content)
    
    if files_uploaded:
        st.info("📋 **Please review your uploaded files before proceeding:**")
        
        # Create tabs for file previews
        preview_tab1, preview_tab2 = st.tabs(["📄 English Subtitles Preview", "🌐 Translation Preview"])
        
        with preview_tab1:
            st.markdown("**English SRT/VTT Content:**")
            st.text_area(
                "English Subtitles Full Preview",
                value=st.session_state.english_sub_content,
                height=300,
                disabled=True,
                help="Review your English subtitle file content"
            )
            
            # Show some stats
            lines = st.session_state.english_sub_content.count('\n') + 1
            chars = len(st.session_state.english_sub_content)
            st.caption(f"📊 Stats: {lines} lines, {chars} characters")
        
        with preview_tab2:
            st.markdown("**Translation Content (Timestamps Automatically Cleaned):**")
            st.text_area(
                "Translation Full Preview",
                value=st.session_state.translation_content,
                height=300,
                disabled=True,
                help="Review your translation file content (timestamp patterns have been automatically removed)"
            )
            
            # Show some stats
            lines = st.session_state.translation_content.count('\n') + 1
            chars = len(st.session_state.translation_content)
            st.caption(f"📊 Stats: {lines} lines, {chars} characters")
        
        # Validation buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("✅ Validate & Approve Files", type="primary"):
                st.session_state.inputs_validated = True
                st.success("🎉 Files validated and approved! You can now submit to AI.")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset Validation"):
                st.session_state.inputs_validated = False
                st.session_state.english_sub_content = ""
                st.session_state.translation_content = ""
                st.info("🔄 Validation reset. Please upload your files again.")
                st.rerun()
        
        with col3:
            if st.session_state.inputs_validated:
                st.success("✅ **Files validated and ready for processing!**")
            else:
                st.warning("⚠️ **Please validate your files before submitting to AI**")
    else:
        st.info("📁 **Upload both files above to proceed with validation**")
        if st.session_state.inputs_validated:
            st.session_state.inputs_validated = False  # Reset validation if files are missing
    
    # Prompt configuration section
    st.subheader("📝 AI Prompt Template")
    
    # Reset to default button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Reset to Default"):
            st.session_state.prompt_template = DEFAULT_PROMPT
            st.success("Prompt reset to default!")
    
    # Prompt editor
    prompt_template = st.text_area(
        "Edit the prompt template (use {{TRANSLATION}} as placeholder for translation content)",
        value=st.session_state.prompt_template,
        height=400,
        help="This template will be used to generate the prompt sent to the AI model. The placeholders will be replaced with your uploaded file contents."
    )
    st.session_state.prompt_template = prompt_template
    
    # Submit button
    st.subheader("🚀 Submit to AI")
    
    if st.button("Submit Prompt to AI", type="primary", use_container_width=True):
        debug_log("Initial prompt submission started", "PROCESS")
        
        # Validation
        current_key = get_current_api_key()
        if not current_key:
            provider_name = {"claude": "Claude", "openai": "OpenAI", "gemini": "Gemini"}[st.session_state.llm_provider]
            debug_log(f"API key missing for provider: {st.session_state.llm_provider}", "ERROR")
            st.error(f"❌ Please configure your {provider_name} API key in the sidebar first!")
            return
        
        if not st.session_state.inputs_validated:
            debug_log("Input files not validated", "ERROR")
            st.error("❌ Please validate your input files first using the validation checkpoint above!")
            return
        
        if not st.session_state.language_confirmed:
            # Enhanced debugging for language confirmation issue
            debug_log(f"Language not confirmed: target='{st.session_state.get('target_language', 'None')}', confirmed={st.session_state.language_confirmed}", "ERROR")
            st.error("❌ Please confirm the target language first!")
            st.error(f"🐛 DEBUG: target_language='{st.session_state.get('target_language', 'None')}', language_confirmed={st.session_state.language_confirmed}, inputs_validated={st.session_state.inputs_validated}, english_sub_content_exists={bool(st.session_state.english_sub_content)}, translation_content_exists={bool(st.session_state.translation_content)}")
            return
        
        if not st.session_state.english_sub_content or not st.session_state.translation_content:
            debug_log("Missing content files", "ERROR")
            st.error("❌ Please upload both English subtitle and translation files first!")
            return
        
        # Prepare prompt for COMPREHENSIVE FULL-CONTEXT approach
        debug_log("Preparing prompt from template for comprehensive full-context processing", "PROCESS")
        
        # Include the full translation content in the base prompt
        prompt = st.session_state.prompt_template.replace("{{TRANSLATION}}", st.session_state.translation_content)
        
        # Parse English segments and add simple context
        original_segments = parse_srt_segments(st.session_state.english_sub_content)
        debug_log(f"Parsed {len(original_segments)} segments from English SRT content", "PROCESS")
        batch_size = 0  # Initialize for scope
        
        if original_segments:
            # Get first 10 segments for manual processing (using autopilot batch size for consistency)
            first_batch = original_segments[:DEFAULT_MANUAL_INITIAL_SEGMENTS]
            debug_log(f"Selected first batch of {len(first_batch)} segments from {len(original_segments)} total segments", "PROCESS")
            
            # Build simple English segments text
            english_segments_text = ""
            for i, segment in enumerate(first_batch, start=1):
                english_segments_text += f"""Segment {i}:
{segment['start_time']} --> {segment['end_time']}
{segment['text']}

"""

            # Simple request for the first 10 segments
            simple_prompt = f"""

Please create bilingual subtitles for the first {len(first_batch)} English segments using the translation provided above.

English segments to process:
{english_segments_text}

Instructions:
- Create bilingual format: [English] || [Translation]
- Use the translation text in sequential order
- Keep target language segments reasonably short (around 15-20 words)
- Output in standard SRT format

Example format:
1
00:01:23,456 --> 00:01:27,890
[English text] || [Translation text]

2
00:01:28,000 --> 00:01:32,500
[English text] || [Translation text]

Please process all {len(first_batch)} segments."""
            
            # Combine base prompt with simple instruction
            prompt += simple_prompt
            debug_log(f"Added simple instruction for {len(first_batch)} segments", "PROCESS")
        
        debug_log(f"Generated simple prompt length: {len(prompt)} characters", "PROCESS")
        
        # Clear conversation and add user prompt
        st.session_state.conversation = []
        st.session_state.conversation.append({"speaker": "User", "message": prompt})
        debug_log("Added initial prompt to conversation", "PROCESS")
        
        # Make API call
        with st.spinner("🤔 AI is thinking..."):
            debug_log("Making initial API call", "API")
            response = call_llm_api(prompt)
            assistant_message = extract_api_response(response)
            
            # Add response to conversation
            st.session_state.conversation.append({"speaker": "Assistant", "message": assistant_message})
            debug_log(f"Added API response to conversation: {len(assistant_message)} characters", "PROCESS")
            
            if "API Error:" in assistant_message:
                debug_log(f"Initial API call failed: {assistant_message}", "ERROR")
                st.error(f"❌ {assistant_message}")
            else:
                debug_log("Initial prompt submission completed successfully", "SUCCESS")
                st.success("✅ Response received! Check the Conversation tab to continue processing.")
                st.balloons()

def autopilot_tab():
    """Dedicated autopilot tab for processing and monitoring"""
    
    # Check for autopilot continuation flag and automatically run next batch
    if st.session_state.get('autopilot_continue_next_batch', False):
        st.session_state.autopilot_continue_next_batch = False  # Reset flag immediately
        debug_log("Auto-continuing autopilot batch due to continuation flag", "PROCESS")
        run_autopilot_batch()
        st.rerun()
        return
    
    # Always show the autopilot dashboard
    show_autopilot_dashboard()
    
    # Show severe warning notice in conversation tab if autopilot stopped due to severe errors
    if (not st.session_state.autopilot_running and 
        st.session_state.quality_report.get('severe_warnings')):
        
        st.markdown("---")
        st.warning("🔔 **Notice:** Autopilot stopped due to severe quality issues. Check warnings above for details.")
        if st.button("📄 **View Conversation Results**", type="secondary"):
            st.info("💡 Switch to the **'Conversation'** tab to view results and export files.")

def conversation_tab():
    # Conversation tab implementation - clean and focused on conversation management
    
    # Clean up any Unicode encoding issues
    clean_conversation_encoding()
    
    st.header("💬 Conversation Management")
    
    if not st.session_state.conversation:
        st.info("📝 No conversation yet. Go to 'Prompt & Files' tab to start a conversation with AI.")
        return
    
    # Show brief autopilot status if it was running but stopped due to severe errors
    if (not st.session_state.autopilot_running and 
        st.session_state.quality_report.get('severe_warnings')):
        
        severe_count = len(st.session_state.quality_report.get('severe_warnings', []))
        st.warning(f"⚠️ **Autopilot stopped** due to {severe_count} severe quality issue{'s' if severe_count > 1 else ''}. Check the **Autopilot tab** for details.")
    
    # Action buttons
    st.subheader("🎛️ Conversation Actions")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        if st.button("➕ Add Custom Message"):
            st.session_state.show_custom_input = True
    
    with col2:
        if st.button("⏭️ Next 5 Segments"):
            continue_with_preset("continue with next 5 segments")
    
    with col3:
        if st.button("⏭️ Next 20 Segments"):
            continue_with_preset("continue with next 20 segments")
    
    with col4:
        if not st.session_state.autopilot_running:
            if st.button("🚁 Start Autopilot"):
                start_autopilot()
        else:
            col_stop, col_continue = st.columns(2)
            with col_stop:
                if st.button("⏹️ Stop Autopilot"):
                    debug_log("User manually stopped autopilot", "INFO")
                    st.session_state.autopilot_running = False
                    st.session_state.autopilot_continue_next_batch = False  # Reset flag
                    st.session_state.conversation_needs_refresh = True
                    
                    # Enhanced manual stop message from conversation tab
                    processed_segments = count_processed_segments()
                    original_segments = parse_srt_segments(st.session_state.english_sub_content)
                    remaining = len(original_segments) - processed_segments if original_segments else 0
                    progress_pct = (processed_segments/len(original_segments)*100) if original_segments else 0
                    
                    st.warning("🛑 **AUTOPILOT MANUALLY STOPPED**")
                    progress_message = (
                        "CURRENT PROGRESS:\n"
                        f"- Segments completed: {processed_segments} of {len(original_segments) if original_segments else 0}\n"
                        f"- Segments remaining: {remaining}\n"
                        f"- Completion rate: {progress_pct:.1f}%\n\n"
                        "WORK PRESERVED: All processed segments are saved and ready for export\n"
                        "RESUME: Click 'Start Autopilot' to continue from where you stopped"
                    )
                    st.info(progress_message)
                    st.rerun()
            with col_continue:
                if st.button("▶️ Force Continue"):
                    run_autopilot_batch()
                    st.rerun()
    
    with col5:
        if st.button("🗑️ Delete Last Output"):
            delete_last_output()
    
    with col6:
        if st.button("🧹 Clear All"):
            if st.session_state.get('confirm_clear', False):
                st.session_state.conversation = []
                st.session_state.confirm_clear = False
                st.success("✅ Conversation cleared!")
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Click again to confirm clearing all conversation!")
    
    # Add backup restore button if backup exists
    if st.session_state.get('conversation_backup'):
        col_backup, _ = st.columns([1, 5])
        with col_backup:
            if st.button("🔄 Restore Backup", help="Restore conversation to state before last edit"):
                st.session_state.conversation = st.session_state.conversation_backup.copy()
                st.session_state.conversation_backup = None
                st.success("✅ Conversation restored from backup!")
                st.rerun()
    
    # Custom message input
            col1, col2 = st.columns(2)
    # Custom message input
    if st.session_state.get('show_custom_input', False):
        st.subheader("✏️ Add Custom Message")
        custom_message = st.text_area(
            "Enter your custom message:",
            height=150,
            help="Type your message and press Ctrl+Enter or click Send"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("📤 Send Message"):
                if custom_message.strip():
                    continue_conversation(custom_message.strip())
                    st.session_state.show_custom_input = False
                    st.rerun()
                else:
                    st.error("❌ Please enter a message!")
        with col2:
            if st.button("❌ Cancel"):
                st.session_state.show_custom_input = False
                st.rerun()
    
    # Instructions for editing
    st.info("💡 **Editing Tips**: You can directly edit the conversation text below. Changes to Assistant responses (subtitle translations) will be preserved and used in future exports. Use the separator lines (===) to distinguish between different messages. A backup is automatically created before each edit.")
    
    # Display conversation
    st.subheader("💭 Conversation History")
    
    # Create columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Show original files
        if st.session_state.english_sub_content:
            st.markdown("**📄 Original English Subtitles**")
            st.text_area(
                "English Subtitles",
                value=st.session_state.english_sub_content,
                height=300,
                disabled=True,
                label_visibility="collapsed"
            )
        
        if st.session_state.translation_content:
            st.markdown("**🌐 Translation Text**")
            st.text_area(
                "Translation",
                value=st.session_state.translation_content,
                height=300,
                disabled=True,
                label_visibility="collapsed"
            )
    
    with col2:
        st.markdown("**💬 Conversation with AI**")
        
        # Check if conversation needs refresh (after autopilot completion)
        if st.session_state.get('conversation_needs_refresh', False):
            st.info("🔄 Conversation updated - all autopilot results are now available for export!")
            st.session_state.conversation_needs_refresh = False
        
        # Display conversation messages in chronological order, excluding initial large input
        conversation_text = ""
        for i, entry in enumerate(st.session_state.conversation):
            # Skip the first user message if it contains the large input files (template placeholders)
            if (i == 0 and entry['speaker'] == 'User' and 
                ('{{ENGLISH_SUB}}' in st.session_state.prompt_template or 
                 len(entry['message']) > 2000)):  # Skip very long initial messages
                conversation_text += f"\n\n{'='*80}\n{entry['speaker']}:\n[Initial prompt with files submitted - see file previews on the left]"
            else:
                conversation_text += f"\n\n{'='*80}\n{entry['speaker']}:\n{entry['message']}"
        
        # Show processing status if autopilot just completed
        if (not st.session_state.autopilot_running and 
            len(st.session_state.conversation) > 0 and 
            st.session_state.conversation[-1]['speaker'] == 'Assistant'):
            
            # Count bilingual segments in final conversation
            all_assistant_content = "\n".join([entry['message'] for entry in st.session_state.conversation if entry['speaker'] == 'Assistant'])
            segments = extract_bilingual_segments(all_assistant_content)
            
            if segments:
                conversation_text += f"\n\n{'='*80}\n💡 PROCESSING COMPLETE:\nGenerated {len(segments)} bilingual segments ready for export!\n" + "─"*80
        
        # Always editable conversation - add invisible character at end to force scroll to bottom
        display_text = conversation_text + "\n\n" + "─" * 80 + "\n[End of conversation - scroll up to see earlier messages]"
        
        # Create a dynamic key that changes when conversation length changes or when refresh is needed
        # This forces the text area to re-render and scroll to bottom when new content is added
        refresh_indicator = "_refreshed" if st.session_state.get('conversation_needs_refresh', False) else ""
        conversation_key = f"conversation_editor_{len(st.session_state.conversation)}_{len(conversation_text)}{refresh_indicator}"
        
        updated_conversation = st.text_area(
            "Conversation with AI (editable - changes persist in memory)",
            value=display_text,
            height=600,
            key=conversation_key,
            help="✏️ EDITABLE: Make corrections to subtitle translations directly in this text. Your changes will be automatically saved and used for all future extractions. Keep the === separator lines intact."
        )
        
        # Check if user made changes and update conversation in memory
        if updated_conversation != display_text:
            debug_log("User edited conversation - updating session state", "INFO")
            parse_and_update_conversation(updated_conversation)
            st.success("✅ **Conversation changes saved!** Your edits will be used for future extractions.")
            # Force a rerun to show the changes have been applied
            st.rerun()
    
    # Simple export section for conversation tab
    st.subheader("💾 Download Options")
    
    if not st.session_state.conversation:
        st.info("📝 No conversation available for export. Start a conversation first!")
        return
    
    # Check if there are bilingual segments available
    segments, is_valid = validate_and_extract_segments()
    
    # Main download button - most prominent
    if len(segments) > 0:
        st.success(f"🎉 **Ready for Export!** {len(segments)} bilingual segments processed successfully.")
        
        # Multiple export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(
                "📦 **SRT + Quality Report**", 
                type="primary", 
                use_container_width=True,
                help="Downloads bilingual SRT file and comprehensive quality report"
            ):
                export_bilingual_srt()
                export_quality_report()
                st.success("✅ **Downloaded:** Bilingual SRT file and Quality Report!")
        
        with col2:
            if st.button(
                "🎬 **Download VTT**", 
                type="secondary", 
                use_container_width=True,
                help="Download bilingual subtitles in WebVTT format for web players"
            ):
                export_bilingual_vtt()
                st.success("✅ **Downloaded:** Bilingual VTT file!")
        
        with col3:
            if st.button(
                "🌐 **Target Language Only**", 
                type="secondary", 
                use_container_width=True,
                help="Download only the target language subtitles (SRT format)"
            ):
                export_target_only_srt()
                st.success("✅ **Downloaded:** Target language SRT file!")
    else:
        st.info("⚠️ No bilingual segments detected in the conversation yet.")
        
    # Always allow conversation download
    st.markdown("---")
    st.markdown("**📄 Additional Export Options**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💬 Download Conversation", type="secondary", use_container_width=True):
            export_conversation_txt()
    
    with col2:
        if len(segments) > 0 and st.button("🎬 Target VTT Only", type="secondary", use_container_width=True, help="Download only target language in VTT format"):
            export_target_only_vtt()
            st.success("✅ **Downloaded:** Target language VTT file!")


def continue_conversation(message: str):
    # Continue the conversation with a custom message
    if not st.session_state.conversation:
        st.error("❌ No conversation to continue")
        return
    
    # Add user message
    st.session_state.conversation.append({"speaker": "User", "message": message})
    
    # Prepare messages for API
    messages = []
    for entry in st.session_state.conversation:
        role = "user" if entry['speaker'] == "User" else "assistant"
        messages.append({"role": role, "content": entry['message']})
    
    # Make API call
    with st.spinner("🤔 AI is thinking..."):
        response = call_llm_api_with_messages(messages)
        assistant_message = extract_api_response(response)
        
        # Add response to conversation
        st.session_state.conversation.append({"speaker": "Assistant", "message": assistant_message})
        
        if "API Error:" in assistant_message:
            st.error(f"❌ {assistant_message}")
        else:
            st.success("✅ Response received!")

def continue_with_preset(preset_text: str):
    """Continue the conversation with next N segments using autopilot logic"""
    # Extract the number of segments from preset text
    if "next 5 segments" in preset_text.lower():
        batch_size = 5
    elif "next 20 segments" in preset_text.lower():
        batch_size = 20
    else:
        # Fallback to simple continue for other preset texts
        continue_conversation(preset_text)
        st.rerun()
        return
    
    debug_log(f"Processing next {batch_size} segments with autopilot logic", "PROCESS")
    
    try:
        # Check prerequisites
        if not st.session_state.inputs_validated or not st.session_state.english_sub_content:
            st.error("❌ English subtitle file not available")
            return
        
        if not st.session_state.conversation:
            st.error("❌ No conversation started - submit initial prompt first")
            return
        
        # Parse original English segments
        original_segments = parse_srt_segments(st.session_state.english_sub_content)
        if not original_segments:
            st.error("❌ Could not parse English subtitle segments")
            return
        
        # Count how many segments have been processed
        processed_segments = count_processed_segments()
        debug_log(f"Already processed {processed_segments} segments", "PROCESS")
        
        if processed_segments >= len(original_segments):
            st.warning("✅ All segments have already been processed!")
            return
        
        # Calculate actual batch size (don't exceed remaining segments)
        remaining_segments = len(original_segments) - processed_segments
        actual_batch_size = min(batch_size, remaining_segments)
        debug_log(f"Processing batch: {actual_batch_size} segments, {remaining_segments} remaining", "PROCESS")
        
        # Get the specific English segments for this batch
        batch_segments = original_segments[processed_segments:processed_segments + actual_batch_size]
        
        # Generate progressive word limits (calculated based on English text length)
        segment_word_limits = []
        for i, segment in enumerate(batch_segments):
            text_content = segment.get('text') or segment.get('english', '')
            eng_words = len(text_content.split())
            max_words = calculate_progressive_word_limit(eng_words)
            segment_word_limits.append(f"Segment {processed_segments + i + 1}: {eng_words} English words → MAX {max_words} target words")
        
        # Build detailed English segments text
        english_segments_text = ""
        for i, segment in enumerate(batch_segments, start=processed_segments + 1):
            # Normalize English text to single line (remove line breaks)
            english_text = segment['text'].replace('\n', ' ').replace('\r', ' ').strip()
            # Remove multiple spaces
            english_text = ' '.join(english_text.split())
            english_segments_text += f"\n{i}. {segment['start_time']} --> {segment['end_time']}\n{english_text}\n"
        
        # Create context message based on conversation history
        if processed_segments == 0:
            context_message = "CONTEXT: You have the complete translation and are starting with the first segments."
            action_verb = "Map"
        else:
            context_message = f"CONTEXT: You have the complete translation and have already mapped segments 1-{processed_segments}."
            action_verb = "Continue mapping"
        
        # Build the autopilot-style request message
        request_message = f"""{action_verb} segments {processed_segments + 1}-{processed_segments + actual_batch_size}.

{context_message}

SEGMENTS TO MAP:
{english_segments_text}

🎯 WORD LIMITS FOR {st.session_state.target_language.upper()}:
Maximum 15 words per segment to prevent content bleeding

SEGMENT-SPECIFIC LIMITS:
{chr(10).join(segment_word_limits)}

🚨 CRITICAL ANTI-BLEEDING RULES:
- Use the EXACT word limits specified above for each segment
- Count words carefully as you extract - STOP at the specified limit
- These limits prevent bleeding while respecting {st.session_state.target_language} language characteristics
- NEVER exceed the calculated word count for any segment

⚡ SINGLE BATCH REQUIREMENT:
- Generate complete bilingual mapping immediately without questions
- NO confirmations, clarifications, or interaction requests
- Handle all complex phrases and parenthetical content automatically
- Provide direct output in the specified format only

OUTPUT: Map these {actual_batch_size} segments using semantic alignment. Use STANDARD SRT format:

{processed_segments + 1}
[start_time] --> [end_time]
[English] || [Target within limit]

{processed_segments + 2}
[start_time] --> [end_time]
[English] || [Target within limit]

CRITICAL FORMAT REQUIREMENTS:
- Number each segment on its own line: "{processed_segments + 1}" then newline, "{processed_segments + 2}" then newline, etc.
- Use exact timestamp format: HH:MM:SS,mmm --> HH:MM:SS,mmm on next line
- Separate languages with " || " (space-pipe-pipe-space)
- Keep all original English text exactly as provided, but on a SINGLE LINE (no line breaks within the English text)
- Keep target language text on the SAME LINE as English text after the " || " separator
- Maintain chronological order

Continue for segments {processed_segments + 1} through {processed_segments + actual_batch_size}

Generate immediately without questions."""
        
        # Add user message to conversation
        st.session_state.conversation.append({"speaker": "User", "message": request_message})
        debug_log(f"Added autopilot-style request for {actual_batch_size} segments", "PROCESS")
        
        # Prepare messages for API
        messages = []
        for entry in st.session_state.conversation:
            role = "user" if entry['speaker'] == "User" else "assistant"
            messages.append({"role": role, "content": entry['message']})
        
        # Make API call with progress indicator
        with st.spinner(f"🤔 Processing next {actual_batch_size} segments..."):
            debug_log("Calling LLM API with autopilot-style messages", "API")
            response = call_llm_api_with_messages(messages)
            assistant_message = extract_api_response(response)
            
            # Add response to conversation
            st.session_state.conversation.append({"speaker": "Assistant", "message": assistant_message})
            debug_log(f"Received API response: {len(assistant_message)} characters", "PROCESS")
            
            if "API Error:" in assistant_message:
                st.error(f"❌ API Error: {assistant_message}")
                return
            
            # Check for AI refusal patterns and retry if needed
            refusal_patterns = [
                "apologize", "already mapped", "clarify", "clarification", "questions",
                "would you like me to", "could you please", "unclear", "not sure",
                "which option", "please specify"
            ]
            
            assistant_lower = assistant_message.lower()
            ai_refused = any(pattern in assistant_lower for pattern in refusal_patterns)
            
            # Extract segments to check if format was followed
            extracted_segments = extract_bilingual_segments(assistant_message)
            
            if ai_refused and len(extracted_segments) == 0:
                debug_log("AI refused, sending retry directive", "WARNING")
                
                # Remove refusal response
                st.session_state.conversation.pop()
                
                # Create direct retry prompt
                retry_message = f"""DIRECTIVE: Process segments {processed_segments + 1}-{processed_segments + actual_batch_size} immediately.

{context_message}

SEGMENTS TO PROCESS:
{english_segments_text}

STRICT REQUIREMENTS:
- Output ONLY the bilingual mapping format
- NO questions, clarifications, or confirmations
- Use the translation provided in the initial context
- Follow word limits: max 15 words per target segment

REQUIRED STANDARD SRT FORMAT:
{processed_segments + 1}
HH:MM:SS,mmm --> HH:MM:SS,mmm
[English] || [Target]

{processed_segments + 2}
HH:MM:SS,mmm --> HH:MM:SS,mmm
[English] || [Target]

FORMAT RULES:
- Number on its own line: "{processed_segments + 1}" then newline
- Timestamp on next line: "HH:MM:SS,mmm --> HH:MM:SS,mmm"
- Bilingual text on next line with " || " separator
- Preserve exact English text and timestamps

PROCESS NOW."""
                
                # Add retry message and make second API call
                st.session_state.conversation.append({"speaker": "User", "message": retry_message})
                
                with st.spinner(f"🔄 Retrying next {actual_batch_size} segments..."):
                    retry_response = call_llm_api_with_messages([
                        {"role": "user" if entry['speaker'] == "User" else "assistant", "content": entry['message']}
                        for entry in st.session_state.conversation
                    ])
                    retry_assistant_message = extract_api_response(retry_response)
                    
                    # Update conversation with retry response
                    st.session_state.conversation.append({"speaker": "Assistant", "message": retry_assistant_message})
                    
                    if "API Error:" in retry_assistant_message:
                        st.error(f"❌ Retry failed: {retry_assistant_message}")
                        return
                    
                    assistant_message = retry_assistant_message
            
            # Basic validation of the response
            final_extracted_segments = extract_bilingual_segments(assistant_message)
            
            if len(final_extracted_segments) > 0:
                st.success(f"✅ Successfully processed {len(final_extracted_segments)} segments!")
                if len(final_extracted_segments) != actual_batch_size:
                    st.warning(f"⚠️ Expected {actual_batch_size} segments, got {len(final_extracted_segments)}")
            else:
                st.warning("⚠️ No segments detected in response. Check the output in the conversation.")
            
    except Exception as e:
        debug_log(f"Error in continue_with_preset: {str(e)}", "ERROR")
        st.error(f"❌ Error processing segments: {str(e)}")
    
    st.rerun()

def delete_last_output():
    # Delete the last assistant response from the conversation
    if not st.session_state.conversation:
        st.error("❌ No conversation to modify")
        return
    
    # Check if the last message is from the assistant
    if st.session_state.conversation[-1]['speaker'] == "Assistant":
        # Remove the last message
        st.session_state.conversation.pop()
        
        # Ask if user wants to remove their message too
        if (len(st.session_state.conversation) > 0 and 
            st.session_state.conversation[-1]['speaker'] == "User"):
            
            if st.session_state.get('confirm_delete_user', False):
                st.session_state.conversation.pop()
                st.session_state.confirm_delete_user = False
                st.success("✅ Last assistant output and user message removed!")
            else:
                st.session_state.confirm_delete_user = True
                st.warning("⚠️ Click again to also remove your last message!")
        else:
            st.success("✅ Last assistant output removed!")
        
        st.rerun()
    else:
        st.info("ℹ️ The last message is not from the assistant")

def export_conversation_txt():
    # Export the complete conversation as a text file
    if not st.session_state.conversation:
        st.error("❌ No conversation to export")
        return
    
    try:
        # Generate conversation text with header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detected_lang = st.session_state.get('target_language', 'unknown')
        provider = st.session_state.get('llm_provider', 'unknown')
        model = st.session_state.get('model', 'unknown')
        
        conversation_text = (
            f"Bilingual Subtitle Generator - Conversation Export\n"
            f"Generated: {timestamp}\n"
            f"Target Language: {detected_lang}\n"
            f"AI Provider: {provider} ({model})\n"
            f"Total Messages: {len(st.session_state.conversation)}\n"
            f"{'='*100}\n"
        )
        
        for i, entry in enumerate(st.session_state.conversation, 1):
            conversation_text += f"\n\n[Message {i}] {entry['speaker']}:\n{'-'*50}\n{entry['message']}"
        
        # Create download with improved filename
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{detected_lang}_{timestamp_file}.txt"
        
        st.download_button(
            label="📥 Download Full Conversation",
            data=conversation_text.encode('utf-8'),
            file_name=filename,
            mime="text/plain",
            help="Download the complete conversation including metadata and all messages",
            use_container_width=True
        )
        st.success(f"✅ Conversation ready for download as `{filename}`")
        st.info(f"📊 **Export Summary:** {len(st.session_state.conversation)} messages, {len(conversation_text)} characters")
        
    except Exception as e:
        st.error(f"❌ Error exporting conversation: {str(e)}")

def export_quality_report():
    """Export the quality report generated during autopilot processing"""
    try:
        # Generate quality report content
        quality_report_content = generate_quality_report()
        
        if not quality_report_content or quality_report_content.strip() == "":
            st.warning("⚠️ No quality report data available. Run autopilot processing first.")
            return
        
        # Add export metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detected_lang = st.session_state.get('target_language', 'unknown')
        provider = st.session_state.get('llm_provider', 'unknown')
        model = st.session_state.get('model', 'unknown')
        
        # Enhanced quality report with metadata
        enhanced_report = f"""# Quality Report - Bilingual Subtitle Generator

**Generated:** {timestamp}
**Target Language:** {detected_lang}
**AI Provider:** {provider} ({model})
**Total Segments Processed:** {st.session_state.quality_report.get('total_segments_processed', 0)}
**Segments with Issues:** {st.session_state.quality_report.get('segments_with_issues', 0)}

---

{quality_report_content}

---
*This report was automatically generated during autopilot processing and contains detailed analysis of translation quality, content alignment, and potential issues.*
"""
        
        # Create download with proper filename
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quality_report_{detected_lang}_{timestamp_file}.md"
        
        st.download_button(
            label="📋 Download Quality Report",
            data=enhanced_report.encode('utf-8'),
            file_name=filename,
            mime="text/markdown",
            help="Download detailed quality report with autopilot analysis",
            use_container_width=True
        )
        st.success(f"✅ Quality report ready for download as `{filename}`")
        
        # Show summary statistics
        total_warnings = len(st.session_state.quality_report.get('warnings', []))
        severe_warnings = len(st.session_state.quality_report.get('severe_warnings', []))
        moderate_warnings = len(st.session_state.quality_report.get('moderate_warnings', []))
        
        st.info(f"📊 **Report Summary:** {total_warnings} total warnings ({severe_warnings} severe, {moderate_warnings} moderate)")
        
    except Exception as e:
        st.error(f"❌ Error exporting quality report: {str(e)}")

def export_bilingual_srt():
    # Export bilingual SRT content with enhanced validation
    debug_log("Starting bilingual SRT export", "PROCESS")
    
    if not st.session_state.conversation:
        debug_log("No conversation found for export", "ERROR")
        st.error("❌ No conversation to export")
        return
    
    debug_log(f"Processing conversation with {len(st.session_state.conversation)} entries", "PROCESS")
    
    try:
        # Extract SRT content
        srt_content = extract_srt_content(st.session_state.conversation)
        debug_log(f"Extracted SRT content: {len(srt_content)} characters", "PROCESS")
        
        if not srt_content:
            debug_log("No bilingual SRT content found", "ERROR")
            st.error("❌ No bilingual SRT content found in the conversation")
            st.info("💡 **Tip:** Make sure the AI has generated bilingual subtitles in the correct format (segments with || separator)")
            
            # Show what patterns we're looking for
            st.code("Expected pattern: timestamp --> timestamp\\nEnglish text || Translated text", language="text")
            return
        
        # Validate SRT format using helper function
        segments, is_valid = validate_and_extract_segments()
        
        if not is_valid:
            st.warning("⚠️ No valid bilingual segments detected. Proceeding with raw content.")
        
        # Create download with metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detected_lang = st.session_state.get('target_language', 'unknown')
        filename = f"bilingual_{detected_lang}_{timestamp}.srt"
        debug_log(f"Created export filename: {filename}", "PROCESS")
        
        st.download_button(
            label="📥 Download Bilingual SRT",
            data=srt_content.encode('utf-8'),
            file_name=filename,
            mime="application/x-subrip",
            help="Download bilingual subtitles in SRT format for video players",
            use_container_width=True
        )
        debug_log("SRT export completed successfully", "SUCCESS")
        st.success(f"✅ Bilingual SRT ready for download as `{filename}`")
        
        # Show export statistics
        if segments:
            st.info(f"📊 **Export Summary:** {len(segments)} bilingual segments, {len(srt_content)} characters")
            
            # Show preview of first few segments
            with st.expander("👀 Preview First 3 Segments"):
                preview_segments = segments[:3]
                for seg in preview_segments:
                    st.text(f"{seg['number']}\n{seg['start_time']} --> {seg['end_time']}\n{seg['english_text']} || {seg['target_text']}\n")
        
    except Exception as e:
        st.error(f"❌ Error exporting SRT: {str(e)}")

def export_bilingual_vtt():
    # Export bilingual VTT content with enhanced validation
    if not st.session_state.conversation:
        st.error("❌ No conversation to export")
        return
    
    try:
        # Extract VTT content
        vtt_content = extract_vtt_content(st.session_state.conversation)
        
        if not vtt_content or vtt_content.strip() == "WEBVTT":
            st.error("❌ No bilingual VTT content found in the conversation")
            st.info("💡 **Tip:** Make sure the AI has generated bilingual subtitles in the correct format (segments with || separator)")
            return
        
        # Validate VTT format using helper function
        segments, is_valid = validate_and_extract_segments()
        
        if not is_valid:
            st.warning("⚠️ No valid bilingual segments detected. Proceeding with raw content.")
        
        # Create download with metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detected_lang = st.session_state.get('target_language', 'unknown')
        filename = f"bilingual_{detected_lang}_{timestamp}.vtt"
        
        st.download_button(
            label="📥 Download Bilingual VTT",
            data=vtt_content.encode('utf-8'),
            file_name=filename,
            mime="text/vtt",
            help="Download bilingual subtitles in WebVTT format for web players",
            use_container_width=True
        )
        st.success(f"✅ Bilingual VTT ready for download as `{filename}`")
        
        # Show export statistics
        if segments:
            st.info(f"📊 **Export Summary:** {len(segments)} bilingual segments, {len(vtt_content)} characters")
            
            # Show preview of first few segments
            with st.expander("👀 Preview First 3 Segments"):
                preview_segments = segments[:3]
                for seg in preview_segments:
                    # Convert timestamps for VTT format
                    start_vtt = seg['start_time'].replace(',', '.')
                    end_vtt = seg['end_time'].replace(',', '.')
                    st.text(f"{start_vtt} --> {end_vtt}\n{seg['english_text']} || {seg['target_text']}\n")
        
    except Exception as e:
        st.error(f"❌ Error exporting VTT: {str(e)}")

def export_target_only_srt():
    # Export target language only SRT content (removes English text before ||)
    debug_log("Starting target-language-only SRT export", "PROCESS")
    
    if not st.session_state.conversation:
        debug_log("No conversation found for export", "ERROR")
        st.error("❌ No conversation to export")
        return
    
    debug_log(f"Processing conversation with {len(st.session_state.conversation)} entries", "PROCESS")
    
    try:
        # Extract target-only SRT content
        srt_content = extract_target_only_srt_content(st.session_state.conversation)
        debug_log(f"Extracted target-only SRT content: {len(srt_content)} characters", "PROCESS")
        
        if not srt_content:
            debug_log("No target-language SRT content found", "ERROR")
            st.error("❌ No target-language SRT content found in the conversation")
            st.info("💡 **Tip:** Make sure the AI has generated bilingual subtitles in the correct format (segments with || separator)")
            return
        
        # Validate SRT format
        all_assistant_text = "\n".join([entry['message'] for entry in st.session_state.conversation if entry['speaker'] == 'Assistant'])
        segments = extract_bilingual_segments(all_assistant_text)
        debug_log(f"Validation found {len(segments)} bilingual segments", "PROCESS")
        
        if not segments:
            debug_log("No valid bilingual segments detected", "WARNING")
            st.warning("⚠️ No valid bilingual segments detected. Cannot extract target language.")
            return
        
        # Create download with metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detected_lang = st.session_state.get('target_language', 'unknown')
        filename = f"target_{detected_lang}_{timestamp}.srt"
        debug_log(f"Created export filename: {filename}", "PROCESS")
        
        st.download_button(
            label="📥 Download Target Language SRT",
            data=srt_content.encode('utf-8'),
            file_name=filename,
            mime="application/x-subrip",
            help="Download target language only subtitles in SRT format",
            use_container_width=True
        )
        debug_log("Target-only SRT export completed successfully", "SUCCESS")
        st.success(f"✅ Target language SRT ready for download as `{filename}`")
        
        # Show export statistics
        st.info(f"📊 **Export Summary:** {len(segments)} target language segments, {len(srt_content)} characters")
        
        # Show preview of first few segments
        with st.expander("👀 Preview First 3 Target Language Segments"):
            preview_segments = segments[:3]
            for seg in preview_segments:
                st.text(f"{seg['number']}\n{seg['start_time']} --> {seg['end_time']}\n{seg['target_text']}\n")
        
    except Exception as e:
        debug_log(f"Error exporting target-only SRT: {str(e)}", "ERROR")
        st.error(f"❌ Error exporting target language SRT: {str(e)}")

def export_target_only_vtt():
    # Export target language only VTT content (removes English text before ||)
    debug_log("Starting target-language-only VTT export", "PROCESS")
    
    if not st.session_state.conversation:
        debug_log("No conversation found for export", "ERROR")
        st.error("❌ No conversation to export")
        return
    
    try:
        # Extract target-only VTT content
        vtt_content = extract_target_only_vtt_content(st.session_state.conversation)
        debug_log(f"Extracted target-only VTT content: {len(vtt_content)} characters", "PROCESS")
        
        if not vtt_content or vtt_content.strip() == "WEBVTT":
            debug_log("No target-language VTT content found", "ERROR")
            st.error("❌ No target-language VTT content found in the conversation")
            st.info("💡 **Tip:** Make sure the AI has generated bilingual subtitles in the correct format (segments with || separator)")
            return
        
        # Validate VTT format using helper function
        segments, is_valid = validate_and_extract_segments()
        debug_log(f"Validation found {len(segments)} bilingual segments", "PROCESS")
        
        if not is_valid:
            debug_log("No valid bilingual segments detected", "WARNING")
            st.warning("⚠️ No valid bilingual segments detected. Cannot extract target language.")
            return
        
        # Create download with metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detected_lang = st.session_state.get('target_language', 'unknown')
        filename = f"target_{detected_lang}_{timestamp}.vtt"
        debug_log(f"Created export filename: {filename}", "PROCESS")
        
        st.download_button(
            label="📥 Download Target Language VTT",
            data=vtt_content.encode('utf-8'),
            file_name=filename,
            mime="text/vtt",
            help="Download target language only subtitles in WebVTT format",
            use_container_width=True
        )
        debug_log("Target-only VTT export completed successfully", "SUCCESS")
        st.success(f"✅ Target language VTT ready for download as `{filename}`")
        
        # Show export statistics
        st.info(f"📊 **Export Summary:** {len(segments)} target language segments, {len(vtt_content)} characters")
        
        # Show preview of first few segments
        with st.expander("👀 Preview First 3 Target Language Segments"):
            preview_segments = segments[:3]
            for seg in preview_segments:
                # Convert timestamps for VTT format
                start_vtt = seg['start_time'].replace(',', '.')
                end_vtt = seg['end_time'].replace(',', '.')
                st.text(f"{start_vtt} --> {end_vtt}\n{seg['target_text']}\n")
        
    except Exception as e:
        debug_log(f"Error exporting target-only VTT: {str(e)}", "ERROR")
        st.error(f"❌ Error exporting target language VTT: {str(e)}")

def main():
    # Main application function
    # Initialize session state
    initialize_session_state()
    
    # Setup sidebar
    setup_sidebar()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📖 App Instructions", "📁 Prompt & Files", "💬 Conversation", "🚁 Autopilot"])
    
    with tab1:
        setup_tab()
    
    with tab2:
        prompt_files_tab()
    
    with tab3:
        conversation_tab()
    
    with tab4:
        autopilot_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 20px;'>"
        "Bilingual Subtitle Generator | Built with Streamlit & Multi-LLM Support<br>"
        "<small>Supports Claude, OpenAI GPT, and Google Gemini models</small>"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
