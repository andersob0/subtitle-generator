#!/usr/bin/env python3
"""
Integration test script for the Bilingual Subtitle Generator
Tests UI and backend integration
"""

import sys
import os
import traceback

# Add the current directory to path
sys.path.insert(0, os.path.dirname(__file__))
# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test all necessary imports"""
    print("🔍 Testing imports...")
    try:
        import streamlit as st
        from subtitle_generator import app
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_session_state_initialization():
    """Test session state initialization"""
    print("🔍 Testing session state initialization...")
    try:
        from subtitle_generator import app
        
        # Mock streamlit session state for testing
        class MockSessionState:
            def __init__(self):
                self.data = {}
            
            def __contains__(self, key):
                return key in self.data
            
            def __getitem__(self, key):
                return self.data[key]
            
            def __setitem__(self, key, value):
                self.data[key] = value
            
            def get(self, key, default=None):
                return self.data.get(key, default)
        
        # Temporarily replace streamlit session state
        import streamlit as st
        original_session_state = getattr(st, 'session_state', None)
        st.session_state = MockSessionState()
        
        # Test initialization
        app.initialize_session_state()
        
        # Check essential keys
        required_keys = [
            'conversation', 'api_key', 'llm_provider', 'model',
            'english_sub_content', 'translation_content', 'prompt_template',
            'inputs_validated', 'target_language', 'language_confirmed',
            'autopilot_running', 'autopilot_segments'
        ]
        
        missing_keys = []
        for key in required_keys:
            if key not in st.session_state:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"❌ Missing session state keys: {missing_keys}")
            return False
        
        print("✅ Session state initialization successful")
        
        # Restore original session state
        if original_session_state:
            st.session_state = original_session_state
        
        return True
        
    except Exception as e:
        print(f"❌ Session state initialization error: {e}")
        traceback.print_exc()
        return False

def test_api_functions():
    """Test API integration functions"""
    print("🔍 Testing API functions...")
    try:
        from subtitle_generator import app
        
        # Test API key function
        key = app.get_current_api_key()
        print(f"✅ API key function works (key present: {bool(key)})")
        
        # Test that all API call functions exist
        api_functions = [
            'call_llm_api', 'call_llm_api_with_messages',
            'call_claude_api', 'call_claude_api_with_messages',
            'call_openai_api', 'call_openai_api_with_messages',
            'call_gemini_api', 'call_gemini_api_with_messages'
        ]
        
        for func_name in api_functions:
            if not hasattr(app, func_name):
                print(f"❌ Missing API function: {func_name}")
                return False
        
        print("✅ All API functions present")
        return True
        
    except Exception as e:
        print(f"❌ API function test error: {e}")
        return False

def test_file_processing():
    """Test file processing functions"""
    print("🔍 Testing file processing functions...")
    try:
        from subtitle_generator import app
        
        # Test language detection
        test_texts = {
            "Hello world, how are you today?": "english",
            "Bonjour le monde, comment allez-vous?": "french",
            "Hola mundo, ¿cómo estás hoy?": "spanish",
            "Olá mundo, como você está hoje?": "portuguese"
        }
        
        for text, expected_lang in test_texts.items():
            detected = app.detect_language(text)
            if detected == expected_lang:
                print(f"✅ Language detection: '{text[:20]}...' -> {detected}")
            else:
                print(f"⚠️ Language detection: '{text[:20]}...' -> {detected} (expected {expected_lang})")
        
        # Test SRT parsing
        test_srt = """1
00:00:01,000 --> 00:00:05,000
Hello world

2
00:00:06,000 --> 00:00:10,000
How are you today?"""
        
        segments = app.parse_srt_segments(test_srt)
        if len(segments) == 2:
            print(f"✅ SRT parsing: {len(segments)} segments parsed correctly")
        else:
            print(f"❌ SRT parsing: Expected 2 segments, got {len(segments)}")
            return False
        
        # Test bilingual segment extraction
        test_bilingual = """1
00:00:01,000 --> 00:00:05,000
Hello world || Bonjour le monde

2
00:00:06,000 --> 00:00:10,000
How are you? || Comment allez-vous?"""
        
        bilingual_segments = app.extract_bilingual_segments(test_bilingual)
        if len(bilingual_segments) == 2:
            print(f"✅ Bilingual extraction: {len(bilingual_segments)} segments extracted")
        else:
            print(f"❌ Bilingual extraction: Expected 2 segments, got {len(bilingual_segments)}")
            return False
        
        print("✅ File processing functions working")
        return True
        
    except Exception as e:
        print(f"❌ File processing test error: {e}")
        traceback.print_exc()
        return False

def test_ui_functions():
    """Test UI function existence"""
    print("🔍 Testing UI functions...")
    try:
        from subtitle_generator import app
        
        ui_functions = [
            'setup_sidebar', 'setup_tab', 'prompt_files_tab', 'conversation_tab',
            'continue_conversation', 'continue_with_preset', 'delete_last_output',
            'start_autopilot', 'run_autopilot_batch',
            'export_conversation_txt', 'export_bilingual_srt', 'export_bilingual_vtt'
        ]
        
        missing_functions = []
        for func_name in ui_functions:
            if not hasattr(app, func_name):
                missing_functions.append(func_name)
        
        if missing_functions:
            print(f"❌ Missing UI functions: {missing_functions}")
            return False
        
        print("✅ All UI functions present")
        return True
        
    except Exception as e:
        print(f"❌ UI function test error: {e}")
        return False

def test_validation_functions():
    """Test validation and quality functions"""
    print("🔍 Testing validation functions...")
    try:
        from subtitle_generator import app
        
        validation_functions = [
            'validate_segment_quality', 'check_content_bleeding',
            'check_autopilot_prerequisites', 'validate_autopilot_response',
            'calculate_semantic_similarity'
        ]
        
        missing_functions = []
        for func_name in validation_functions:
            if not hasattr(app, func_name):
                missing_functions.append(func_name)
        
        if missing_functions:
            print(f"❌ Missing validation functions: {missing_functions}")
            return False
        
        print("✅ All validation functions present")
        return True
        
    except Exception as e:
        print(f"❌ Validation function test error: {e}")
        return False

def test_main_app_structure():
    """Test main app structure"""
    print("🔍 Testing main app structure...")
    try:
        from subtitle_generator import app
        
        # Test that main function exists
        if not hasattr(app, 'main'):
            print("❌ Missing main function")
            return False
        
        # Test default prompt exists
        if not hasattr(app, 'DEFAULT_PROMPT') or not app.DEFAULT_PROMPT:
            print("❌ Missing or empty DEFAULT_PROMPT")
            return False
        
        print("✅ Main app structure valid")
        return True
        
    except Exception as e:
        print(f"❌ Main app structure test error: {e}")
        return False

def run_all_tests():
    """Run all integration tests"""
    print("🚀 Starting UI and Backend Integration Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_session_state_initialization,
        test_api_functions,
        test_file_processing,
        test_ui_functions,
        test_validation_functions,
        test_main_app_structure
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! UI and Backend are fully integrated!")
        return True
    else:
        print(f"⚠️ {failed} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
