# 🎬 Bilingual Subtitle Generator

**AI-Powered Professional Subtitle Generation with Multi-Provider Support**

Generate bilingual SRT/VTT subtitle files using cutting-edge AI models with intelligent quality control and autopilot processing.

## ✨ Key Features

- **🤖 Latest AI Models**: GPT-5 (default), Claude 3.5 Sonnet, Gemini 2.0, and 50+ models via OpenRouter
- **🛡️ Smart Quality Control**: Prevents content bleeding, validates semantic consistency, adaptive processing
- **🚁 Autopilot Processing**: Fully automated batch processing with intelligent error handling
- **📁 Flexible Input**: SRT/VTT subtitles + TXT/DOCX translations with automatic timestamp cleaning  
- **🌍 Multi-Language**: 15+ languages with automatic detection and validation

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (Python 3.11+ recommended)
- **Git** (for cloning the repository)
- **API Key** from one of the supported providers:
  - [OpenRouter](https://openrouter.ai/keys) - Recommended (GPT-5 + 50+ models)
  - [Anthropic](https://console.anthropic.com/settings/keys) - Claude models
  - [OpenAI](https://platform.openai.com/api-keys) - GPT models
  - [Google AI](https://makersuite.google.com/app/apikey) - Gemini models

### Installation

#### 🪟 Windows
```cmd
# Clone the repository
git clone https://github.com/andersob0/subtitle-generator.git
cd subtitle-generator

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run src/subtitle_generator/app.py
```

#### 🐧 Linux
```bash
# Clone the repository
git clone https://github.com/andersob0/subtitle-generator.git
cd subtitle-generator

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run src/subtitle_generator/app.py
```

#### 🍎 macOS
```bash
# Clone the repository
git clone https://github.com/andersob0/subtitle-generator.git
cd subtitle-generator

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run src/subtitle_generator/app.py
```

### 🌐 Access the Application
After running the command above, open your web browser and go to:
**http://localhost:8501**

### 🔧 First-Time Setup
1. **Configure AI Provider**: In the sidebar, select your preferred provider (OpenRouter recommended)
2. **Enter API Key**: Add your API key in the configuration section
3. **Upload Files**: 
   - English SRT/VTT subtitle file
   - Translation file (TXT or DOCX format)
4. **Start Processing**: Choose from Manual, Batch, or Autopilot modes

### 💡 Usage Tips
- **OpenRouter** provides access to GPT-5 and 50+ other models with competitive pricing
- **Autopilot mode** handles large projects automatically with quality monitoring
- **Quality control** prevents content bleeding and ensures accurate translations
- **Multiple output formats** supported: SRT, VTT, bilingual, or target-language only

## 🏗️ Architecture & Code Structure

### Core Components

```
src/
└── subtitle_generator/
    ├── app.py          # Main Streamlit application (5000+ lines)
    └── prompts.py      # AI prompt templates and anti-bleeding logic
```

### Key Architecture Concepts

#### 1. **Multi-Provider AI Integration**
- **OpenRouter**: Primary gateway to 50+ models (GPT-5, Claude, Llama, etc.)
- **Direct APIs**: Claude (Anthropic), OpenAI, Google Gemini
- **Fallback System**: Automatic provider switching on failures
- **Quality Optimization**: Real-time processing optimization across providers

#### 2. **Processing Pipeline**
```
Input Validation → Context Setup → Batch Processing → Quality Validation → Export
     ↓               ↓              ↓                 ↓              ↓
  File Upload    AI Prompt       Sequential        Semantic       SRT/VTT
  Language       Generation      Alignment         Analysis       Output
  Detection      Full Context    Word Limits       Anti-Bleeding  Multiple Formats
```

#### 3. **Quality Control System**
- **Source Validation**: Ensures translations match source content
- **Anti-Bleeding Logic**: Prevents content mixing between segments  
- **Semantic Consistency**: Cross-language similarity validation
- **Progressive Limits**: Adaptive word count management
- **Severity Classification**: Automatic issue categorization (severe/moderate)

#### 4. **State Management**
```python
# Core session state variables
st.session_state = {
    'llm_provider': 'openrouter',           # Default: OpenRouter
    'model': 'openai/gpt-5',               # Default: GPT-5
    'conversation': [],                     # Processing history
    'quality_report': {},                   # Quality tracking
    'autopilot_running': False,             # Autopilot status
    'english_sub_content': '',              # Source subtitles
    'translation_content': '',              # Translation text
    'target_language': ''                   # Detected language
}
```

## 🔧 Development & Extension Guide

### Understanding the Codebase

#### Main Application Logic (`app.py`)
1. **Session State Init** (Lines 290-330): Initialize all application state
2. **Provider Setup** (Lines 380-610): Multi-provider configuration and API management
3. **Processing Core** (Lines 2500-3200): Autopilot, batch processing, and validation
4. **Quality System** (Lines 3200-4000): Anti-bleeding validation and quality scoring
5. **UI Tabs** (Lines 4000-5080): Streamlit interface with 4 main tabs

#### Key Functions to Understand
```python
# Core processing functions
def run_autopilot_batch()              # Batch processing logic
def validate_autopilot_response()      # Quality validation system
def call_llm_api_with_messages()       # Multi-provider API calls
def extract_bilingual_segments()       # Parse AI responses into segments

# Quality control functions  
def validate_source_content()          # Anti-content-generation validation
def check_batch_semantic_consistency() # Cross-segment consistency
def add_quality_warning()             # Issue tracking and severity classification

# Export functions
def export_bilingual_srt()            # Generate bilingual SRT files
def export_target_only_vtt()          # Generate target-language VTT files
```

### Adding New AI Providers

1. **Add Provider Constants** (Lines 28-70):
```python
# Add availability check
try:
    import your_provider_library
    YOUR_PROVIDER_AVAILABLE = True
except ImportError:
    YOUR_PROVIDER_AVAILABLE = False
```

2. **Update Provider Lists** (Lines 384-404):
```python
providers = ["openrouter", "claude", "openai", "gemini", "your_provider"]
provider_labels = {
    "your_provider": "Your Provider Name"
}
```

3. **Add API Key Handling** (Lines 420-470):
```python
elif selected_provider == "your_provider":
    api_key = st.text_input("Your Provider API Key", ...)
    st.session_state.your_provider_api_key = api_key
```

4. **Implement API Call Function**:
```python
def call_your_provider_api(messages, model, max_tokens):
    # Implement your provider's API call logic
    # Return response text or None on failure
```

### Extending Quality Control

The quality system uses a severity-based approach:

```python
# Add new validation rules in validate_autopilot_response()
def your_custom_validation(segments, context):
    issues = []
    for segment in segments:
        if your_condition(segment):
            issues.append({
                'type': 'your_issue_type',
                'severity': 'severe' or 'moderate',  # Affects autopilot stopping
                'details': 'Description of the issue'
            })
    return issues
```

### Testing & Debugging

#### Essential Test Files
```
tests/
├── run_tests.py           # Main test runner
├── test_integration.py    # UI and backend integration tests  
└── test_core_validation.py # Quality control and validation tests
```

#### Debug Tools
- **Debug Logging**: `debug_log(message, category)` - comprehensive logging system
- **Quality Reports**: Export detailed processing reports with issue analysis
- **Session State Inspector**: Built-in state debugging in UI

#### Running Tests
```bash
# Run all tests
python tests/run_tests.py

# Test specific functionality  
python tests/test_integration.py

# Test quality validation
python tests/test_core_validation.py
```

## 📁 File Formats & Support

### Input Formats
- **Subtitles**: SRT, VTT (with automatic format detection)
- **Translations**: TXT (plain text), DOCX (Microsoft Word)
- **Languages**: Auto-detection for 15+ languages with manual override

### Output Formats
- **Bilingual**: English || Target Language in single file
- **Target Only**: Pure target language subtitles
- **Formats**: SRT, VTT, TXT (conversation export)
- **Quality Reports**: Detailed processing analysis and issue tracking

## 🔧 Troubleshooting

### Common Issues

#### Virtual Environment Problems
```bash
# If venv creation fails, try:
python -m pip install --upgrade pip
python -m pip install virtualenv
python -m virtualenv venv

# On macOS/Linux, if permission denied:
sudo python3 -m venv venv
```

#### Dependencies Installation Issues
```bash
# If pip install fails, try upgrading pip first:
pip install --upgrade pip

# For Windows users, if Microsoft Visual C++ errors:
pip install --upgrade setuptools wheel

# For specific package issues:
pip install streamlit requests python-docx pandas
```

#### Streamlit Won't Start
```bash
# Check if streamlit is installed:
streamlit --version

# If not found, reinstall:
pip uninstall streamlit
pip install streamlit

# Try running with python -m:
python -m streamlit run src/subtitle_generator/app.py
```

#### Port Already in Use
```bash
# Run on different port:
streamlit run src/subtitle_generator/app.py --server.port 8502

# Or kill existing process:
# Windows: netstat -ano | findstr :8501
# macOS/Linux: lsof -i :8501
```

### API Issues
- **OpenRouter**: Verify API key format starts with `sk-or-`
- **Rate Limits**: Wait a few minutes if you hit API rate limits
- **Invalid Keys**: Double-check API key is correctly copied without spaces

### Performance Tips
- **Large Files**: Use Autopilot mode for files with 100+ segments
- **Memory**: Close other applications if processing large subtitle files
- **Network**: Ensure stable internet connection for API calls

## 🤝 Contributing

1. **Fork the repository**
2. **Understand the architecture** using this guide
3. **Follow the code patterns** established in existing functions
4. **Test thoroughly** with the provided test suite
5. **Document changes** in code comments and README updates

### Code Style
- **Functions**: Descriptive names with docstrings
- **Constants**: UPPER_CASE with clear meanings
- **State Management**: Use session state consistently
- **Error Handling**: Graceful degradation with user feedback
- **Logging**: Use debug_log() for troubleshooting

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**🚀 Ready to generate professional bilingual subtitles with AI?**  
Start with `streamlit run src/subtitle_generator/app.py` and explore the powerful features!
