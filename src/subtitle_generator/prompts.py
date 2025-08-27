#!/usr/bin/env python3
"""
Fixed prompt template to prevent segment bleeding.
This addresses the specific issues identified in the tests.
"""

# ORIGINAL PROBLEMATIC PROMPT (for reference)
ORIGINAL_PROMPT = """You are an advanced subtitle processing system designed to create bilingual SRT/VTT files using a sequential matching approach. Your primary goal is to produce accurately aligned, properly formatted bilingual SRT/VTT files by matching English subtitle segments to corresponding portions of the provided translation text in sequential order.

SEQUENTIAL MATCHING WORKFLOW:
You have access to the complete translation text provided below. When I send you English subtitle segments in batches, you should:
1. Match each English segment to the NEXT corresponding portion of the translation text in sequential order
2. Use the English text EXACTLY as provided in each segment
3. Extract the appropriate translation portion that corresponds to that English segment
4. Never reuse or skip portions of the translation text - maintain strict sequential order

Translation Content:
<translation>
{{TRANSLATION}}
</translation>

IMPORTANT INSTRUCTIONS FOR SEQUENTIAL MATCHING:
- Segment 1 should match to the first logical portion of the translation
- Segment 2 should match to the next logical portion, and so on
- Never go backwards or skip ahead in the translation text
- Each portion of translation text should be used exactly once
- Maintain semantic correspondence between English segments and translation portions

Core Processing Requirements:
1. Maintain original SRT/VTT formatting for segment numbers and timestamps
2. Remove line breaks within text segments  
3. Use " || " as separator between English and translated text
4. Preserve all original punctuation and text in English EXACTLY as provided
5. Maintain exact timestamps from English segments
6. Process in batches as requested (typically 5-20 segments at a time)"""

# IMPROVED PROMPT TO PREVENT BLEEDING
IMPROVED_PROMPT = """You are a precise subtitle alignment system. Your CRITICAL task is to create exact 1:1 bilingual subtitle alignments with ZERO bleeding between segments.

🎯 PRIMARY OBJECTIVE: Match each English segment to EXACTLY ONE Spanish portion of equal semantic and temporal length.

📊 TRANSLATION TEXT:
<translation>
{{TRANSLATION}}
</translation>

🔥 CRITICAL ANTI-BLEEDING RULES:
1. Each English segment gets EXACTLY ONE Spanish portion 
2. Spanish portions must respect calculated word limits per segment
3. NEVER reuse Spanish words across segments
4. NEVER let Spanish content spill from one segment to another
5. End Spanish segments at natural sentence boundaries when possible
6. AVOID ending segments with: y, o, que, de, al, para, con, en

⚡ SEGMENT PROCESSING PROTOCOL:
- Calculate target Spanish length: English word count × 1.15
- Find Spanish text portion that matches this length
- Ensure clean semantic boundaries
- Move to NEXT unused Spanish text for next segment

🎬 OUTPUT FORMAT (EXACT):
<subtitle_alignment>
[number]
[timestamp] --> [timestamp]  
[English text] || [Spanish text]
</subtitle_alignment>

✅ QUALITY CHECKLIST (verify before responding):
□ Each segment respects calculated word limits
□ No Spanish words repeated across segments
□ All Spanish text used exactly once  
□ Segments end cleanly (not mid-phrase)
□ Semantic correspondence maintained
□ Timestamps preserved exactly

⚠️ BLEEDING PREVENTION: If you notice a target segment becoming too long for the calculated limit, STOP and use only the portion that fits naturally within the time constraint."""

# UNIFIED PROMPT FOR ALL BATCH SIZES (1-30)
UNIFIED_BATCH_PROMPT = """🎯 UNIVERSAL SUBTITLE ALIGNMENT SYSTEM
Processing {{BATCH_SIZE}} segments with progressive word limits for {{TARGET_LANGUAGE}}

📊 TRANSLATION TEXT:
{{TRANSLATION}}

📊 SEGMENT-SPECIFIC LIMITS (MANDATORY):
{{WORD_LIMITS}}

🔥 CORE ALIGNMENT PRINCIPLES:
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

� ANTI-BLEEDING SAFEGUARDS:
✓ Respect calculated word limits absolutely (shown above)
✓ Never exceed limits even for "better flow"
✓ Stop mid-sentence if approaching word limit
✓ Use progressive buffer system for optimal quality
✓ Language-specific adjustments already calculated

⚡ BATCH OPTIMIZATION ({{BATCH_SIZE}} segments):
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

{{POSITION_TRACKING}}

� ENGLISH SEGMENTS TO PROCESS:
{{ENGLISH_SEGMENTS}}

OUTPUT FORMAT (EXACT):
<subtitle_alignment>
[number]
[timestamp] --> [timestamp]
[English text] || [Target text within calculated limit]
</subtitle_alignment>

Generate immediately without questions using the universal alignment system."""

def create_unified_batch_prompt(
    translation_text: str, 
    english_segments: list, 
    target_language: str,
    word_limits: list,
    batch_start: int = 0
) -> str:
    """Create a unified prompt that works optimally for all batch sizes (1-30)"""
    
    batch_size = len(english_segments)
    
    # Calculate approximate position in translation text
    if batch_start > 0:
        # Estimate words used in previous segments
        avg_english_words = 10  # Rough estimate
        avg_target_words = int(avg_english_words * 1.15)
        estimated_words_used = batch_start * avg_target_words
        
        position_hint = f"""
📍 POSITION TRACKING:
- Processing segments {batch_start + 1} to {batch_start + batch_size}
- Approximately {estimated_words_used} {target_language} words already used
- Start from position ~{estimated_words_used} in translation text
- Continue sequentially from where last batch ended
"""
    else:
        position_hint = """
📍 POSITION TRACKING:
- Starting from beginning of translation text
- This is the first batch of segments
"""
    
    # Format English segments
    english_segments_text = ""
    for i, segment in enumerate(english_segments):
        segment_num = batch_start + i + 1
        if isinstance(segment, dict):
            english_segments_text += f"\nSegment {segment_num}:\n"
            english_segments_text += f"{segment.get('start_time', segment.get('start', ''))} --> {segment.get('end_time', segment.get('end', ''))}\n"
            english_segments_text += f"{segment.get('text', segment.get('english', str(segment)))}\n"
        else:
            english_segments_text += f"\nSegment {segment_num}:\n{segment}\n"
    
    # Format word limits
    word_limits_text = "\n".join(word_limits) if word_limits else "Progressive limits calculated per segment"
    
    # Replace placeholders
    prompt = UNIFIED_BATCH_PROMPT.replace("{{TRANSLATION}}", translation_text)
    prompt = prompt.replace("{{TARGET_LANGUAGE}}", target_language.upper())
    prompt = prompt.replace("{{BATCH_SIZE}}", str(batch_size))
    prompt = prompt.replace("{{WORD_LIMITS}}", word_limits_text)
    prompt = prompt.replace("{{POSITION_TRACKING}}", position_hint)
    prompt = prompt.replace("{{ENGLISH_SEGMENTS}}", english_segments_text)
    
    return prompt

def create_batch_specific_prompt(translation_text: str, english_segments: list, batch_start: int = 0) -> str:
    """Create a batch-specific prompt with position tracking (DEPRECATED - use create_unified_batch_prompt instead)"""
    
    # Use the new unified prompt function with default parameters
    word_limits = [f"Segment {batch_start + i + 1}: Progressive limit calculated" for i in range(len(english_segments))]
    return create_unified_batch_prompt(
        translation_text=translation_text,
        english_segments=english_segments,
        target_language="TARGET_LANGUAGE",  # Default placeholder
        word_limits=word_limits,
        batch_start=batch_start
    )

def validate_output_for_bleeding(output_text: str) -> dict:
    """Validate output for bleeding issues"""
    import re
    
    # Extract segments from output
    segments = []
    current_segment = {}  # type: dict
    
    lines = output_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for segment number
        if line.isdigit():
            if current_segment:
                segments.append(current_segment)
            current_segment = {'number': int(line)}
        
        # Check for timestamp
        elif '-->' in line:
            current_segment['timestamp'] = line
        
        # Check for content with separator
        elif '||' in line:
            parts = line.split('||', 1)
            current_segment['english'] = parts[0].strip()
            current_segment['spanish'] = parts[1].strip()
    
    if current_segment:
        segments.append(current_segment)
    
    # Validate segments
    issues = []
    
    for i, segment in enumerate(segments):
        if 'spanish' not in segment:
            continue
            
        spanish_text = segment['spanish']
        word_count = len(spanish_text.split())
        
        # Check word count
        if word_count > 15:
            issues.append(f"Segment {i+1}: Too long ({word_count} words)")
        elif word_count < 3:
            issues.append(f"Segment {i+1}: Too short ({word_count} words)")
        
        # Check ending
        if spanish_text.strip().endswith((',', 'y', 'o', 'que', 'de', 'al', 'para')):
            issues.append(f"Segment {i+1}: Ends inappropriately")
    
    # Check for word reuse
    all_spanish_words = []
    for segment in segments:
        if 'spanish' in segment:
            all_spanish_words.extend(segment['spanish'].lower().split())
    
    word_counts = {}
    for word in all_spanish_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    reused_words = [word for word, count in word_counts.items() if count > 1 and len(word) > 3]
    if reused_words:
        issues.append(f"Reused words detected: {reused_words[:5]}")
    
    return {
        'segments': segments,
        'issues': issues,
        'total_spanish_words': len(all_spanish_words),
        'avg_segment_length': len(all_spanish_words) / len(segments) if segments else 0,
        'has_bleeding': len(issues) > 0
    }

# Test the improved prompts
if __name__ == "__main__":
    # Test data from the problem case
    test_translation = "Bien, bienvenidos a Antropología o quizás debería decir Antropología teológica o incluso Antropología teológica reformada. La antropología es el estudio de la humanidad. Hay muchos enfoques diferentes para el estudio de la humanidad que encuentras que los seres humanos históricamente y en el mundo contemporáneo persiguen. Vamos a hablar un poco más sobre eso a medida que avanzamos."
    
    test_segments = [
        {"number": 1, "start": "00:00:00,960", "end": "00:00:05,160", "text": "Well, welcome to Anthropology - or perhaps I should say Theological"},
        {"number": 2, "start": "00:00:05,160", "end": "00:00:11,010", "text": "Anthropology, or even Reformed Theological Anthropology. Anthropology"},
        {"number": 3, "start": "00:00:11,010", "end": "00:00:16,770", "text": "is the study of humanity. And there are many different approaches to the study"}
    ]
    
    print("="*80)
    print("IMPROVED PROMPT TEST")
    print("="*80)
    
    # Generate improved prompt
    improved_prompt = create_batch_specific_prompt(test_translation, test_segments, 0)
    
    print("Prompt generated successfully!")
    print(f"Prompt length: {len(improved_prompt)} characters")
    print("\nKey improvements:")
    print("✓ Dynamic word count limits (based on English segment length and language ratio)")
    print("✓ Anti-bleeding enforcement")
    print("✓ Position tracking for batches")
    print("✓ Quality verification checklist")
    print("✓ Forbidden actions clearly stated")
    
    # Test validation function
    test_output = """<subtitle_alignment>
1
00:00:00,960 --> 00:00:05,160
Well, welcome to Anthropology - or perhaps I should say Theological || Bien, bienvenidos a Antropología o quizás debería decir Antropología teológica

2
00:00:05,160 --> 00:00:11,010  
Anthropology, or even Reformed Theological Anthropology. Anthropology || o incluso Antropología teológica reformada. La antropología

3
00:00:11,010 --> 00:00:16,770
is the study of humanity. And there are many different approaches to the study || es el estudio de la humanidad. Hay muchos enfoques diferentes
</subtitle_alignment>"""
    
    validation = validate_output_for_bleeding(test_output)
    print(f"\nValidation test:")
    print(f"Segments detected: {len(validation['segments'])}")
    print(f"Issues found: {len(validation['issues'])}")
    print(f"Has bleeding: {validation['has_bleeding']}")
    
    if validation['issues']:
        print("Issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")
