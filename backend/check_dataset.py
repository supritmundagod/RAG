#!/usr/bin/env python3
"""
Dataset Diagnostic Tool
Checks all PDF files in the dataset for issues
"""

import os
import sys
import fitz
import pdfplumber
from pathlib import Path

def check_pdf_file(file_path, filename):
    """Check a single PDF file for issues"""
    issues = []
    info = {
        'filename': filename,
        'size': 0,
        'readable': False,
        'pages': 0,
        'text_extracted': False,
        'text_length': 0,
        'methods_working': []
    }
    
    # Check file size
    try:
        file_size = os.path.getsize(file_path)
        info['size'] = file_size
        
        if file_size == 0:
            issues.append("File is empty (0 bytes)")
            return info, issues
        
        if file_size > 100 * 1024 * 1024:  # 100MB
            issues.append(f"File is very large ({file_size / 1024 / 1024:.1f}MB)")
    except Exception as e:
        issues.append(f"Cannot access file: {e}")
        return info, issues
    
    # Try PyMuPDF (fitz)
    try:
        doc = fitz.open(file_path)
        info['pages'] = len(doc)
        info['readable'] = True
        
        text_content = []
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                page_text = page.get_text("text")
                if page_text and page_text.strip():
                    text_content.append(page_text)
            except Exception as e:
                issues.append(f"PyMuPDF: Error on page {page_num + 1}: {e}")
        
        doc.close()
        
        if text_content:
            full_text = " ".join(text_content)
            info['text_extracted'] = True
            info['text_length'] = len(full_text)
            info['methods_working'].append('PyMuPDF')
            
            if len(full_text.strip()) < 10:
                issues.append("PyMuPDF: Extracted text is too short (< 10 chars)")
        else:
            issues.append("PyMuPDF: No text extracted")
    except Exception as e:
        issues.append(f"PyMuPDF: Cannot open file - {e}")
    
    # Try pdfplumber
    try:
        with pdfplumber.open(file_path) as pdf:
            if info['pages'] == 0:
                info['pages'] = len(pdf.pages)
            
            text_content = []
            for page in pdf.pages:
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_content.append(page_text)
                except Exception as e:
                    pass  # Skip page errors
            
            if text_content:
                full_text = " ".join(text_content)
                if not info['text_extracted']:
                    info['text_extracted'] = True
                    info['text_length'] = len(full_text)
                if 'pdfplumber' not in info['methods_working']:
                    info['methods_working'].append('pdfplumber')
    except Exception as e:
        if 'PyMuPDF' not in info['methods_working']:
            issues.append(f"pdfplumber: Cannot open file - {e}")
    
    # Final checks
    if not info['readable']:
        issues.append("File is not readable by either PDF library")
    
    if not info['text_extracted']:
        issues.append("No text could be extracted (may be image-only PDF)")
    
    if info['pages'] == 0:
        issues.append("File appears to have 0 pages")
    
    return info, issues

def main():
    """Main diagnostic function"""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    dataset_dir = os.path.join(backend_dir, "dataset")
    
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("  PDF Dataset Diagnostic Tool")
    print("=" * 70)
    print()
    
    # Find all PDF files
    pdf_files = []
    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(dataset_dir, filename)
            pdf_files.append((file_path, filename))
    
    if not pdf_files:
        print(f"[WARNING] No PDF files found in {dataset_dir}")
        sys.exit(0)
    
    print(f"Found {len(pdf_files)} PDF file(s) to check...\n")
    
    # Check each file
    results = []
    total_issues = 0
    
    for file_path, filename in pdf_files:
        print(f"Checking: {filename}...", end=" ")
        info, issues = check_pdf_file(file_path, filename)
        results.append((info, issues))
        
        if issues:
            print(f"❌ {len(issues)} issue(s)")
            total_issues += len(issues)
        else:
            print("✅ OK")
    
    print("\n" + "=" * 70)
    print("  Summary Report")
    print("=" * 70)
    print()
    
    # Statistics
    total_files = len(results)
    files_with_issues = sum(1 for _, issues in results if issues)
    files_ok = total_files - files_with_issues
    
    print(f"Total files: {total_files}")
    print(f"Files OK: {files_ok} ✅")
    print(f"Files with issues: {files_with_issues} ❌")
    print(f"Total issues found: {total_issues}")
    print()
    
    # Detailed report
    if files_with_issues > 0:
        print("=" * 70)
        print("  Files with Issues")
        print("=" * 70)
        print()
        
        for info, issues in results:
            if issues:
                print(f"\n📄 {info['filename']}")
                print(f"   Size: {info['size'] / 1024:.1f} KB")
                print(f"   Pages: {info['pages']}")
                print(f"   Text extracted: {'Yes' if info['text_extracted'] else 'No'}")
                if info['text_extracted']:
                    print(f"   Text length: {info['text_length']} characters")
                if info['methods_working']:
                    print(f"   Working methods: {', '.join(info['methods_working'])}")
                print("   Issues:")
                for issue in issues:
                    print(f"     - {issue}")
    
    # Files that are OK
    if files_ok > 0:
        print("\n" + "=" * 70)
        print("  Files OK (No Issues)")
        print("=" * 70)
        print()
        for info, issues in results:
            if not issues:
                print(f"✅ {info['filename']} ({info['pages']} pages, {info['text_length']} chars)")
    
    print("\n" + "=" * 70)
    print("  Recommendations")
    print("=" * 70)
    print()
    
    if files_with_issues == 0:
        print("✅ All PDF files are healthy and ready to process!")
    else:
        print("⚠️  Some files have issues. Recommendations:")
        print()
        print("1. Files with 'No text extracted' may be image-only PDFs.")
        print("   Consider using OCR (Optical Character Recognition) tools.")
        print()
        print("2. Files that cannot be opened may be corrupted.")
        print("   Try re-downloading or re-creating these files.")
        print()
        print("3. Empty files should be removed from the dataset.")
        print()
        print("4. Very large files (>100MB) may cause memory issues.")
        print("   Consider splitting them into smaller files.")
    
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Diagnostic cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

