"""
Debug script to test PDF processing and see exactly what data is extracted.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.pdf_processor import PDFProcessor
from src.config import get_config

def main():
    # Load config
    config = get_config()
    
    # Create PDF processor
    processor = PDFProcessor(config)
    
    # Ask user for PDF path
    pdf_path = input("Enter path to PDF file: ").strip()
    
    if not pdf_path or not Path(pdf_path).exists():
        print(f"File not found: {pdf_path}")
        return
    
    # Read PDF
    with open(pdf_path, 'rb') as f:
        pdf_content = f.read()
    
    print(f"\n📄 Processing: {pdf_path}")
    print("=" * 60)
    
    # Process PDF - extract text first
    text = processor.extract_text(pdf_content)
    print(f"✅ Extracted {len(text)} characters")
    print(f"First 500 chars:\n{text[:500]}\n")
    
    # Segment into clauses
    clauses = processor.segment_clauses(text)
    print(f"✅ Segmented into {len(clauses)} clauses")
    
    # Check what type of data we have
    print(f"\n🔍 Clause type: {type(clauses)}")
    if clauses:
        print(f"🔍 First clause type: {type(clauses[0])}")
        print(f"🔍 First clause content: {clauses[0]}")
    
    # Now check if we have strings or dicts
    string_count = 0
    dict_count = 0
    other_count = 0
    
    for c in clauses:
        if isinstance(c, str):
            string_count += 1
        elif isinstance(c, dict):
            dict_count += 1
        else:
            other_count += 1
    
    print(f"\n📊 Summary:")
    print(f"  - Strings: {string_count}")
    print(f"  - Dictionaries: {dict_count}")
    print(f"  - Other: {other_count}")
    
    # If there are strings, convert them and show preview
    if string_count > 0:
        print("\n⚠️ WARNING: Clauses contain strings instead of dictionaries!")
        print("Converting to dictionaries for testing...")
        
        converted = []
        for c in clauses:
            if isinstance(c, str):
                converted.append({'text': c, 'type': 'converted'})
            else:
                converted.append(c)
        
        clauses = converted
    
    # Now try to get clause summary
    try:
        summary = processor.get_clause_summary(clauses)
        print(f"\n✅ Clause Summary:")
        for key, value in summary.items():
            print(f"  - {key}: {value}")
    except Exception as e:
        print(f"\n❌ Error in get_clause_summary: {e}")
        import traceback
        traceback.print_exc()
    
    # Show all clauses
    print("\n📋 All Clauses:")
    for i, c in enumerate(clauses):
        if isinstance(c, dict):
            text = c.get('text', '')
            text_preview = text[:100] + "..." if len(text) > 100 else text
            print(f"  {i+1}: {text_preview}")
        else:
            print(f"  {i+1}: {str(c)[:100]}...")

if __name__ == "__main__":
    main()