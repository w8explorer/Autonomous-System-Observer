"""
Java code parsing and metadata extraction utilities
"""

import re
from typing import Dict, Any


def extract_java_metadata(code: str, filepath: str) -> Dict[str, Any]:
    """
    Extract imports, classes, methods, and calls from Java code

    Args:
        code: Java source code
        filepath: Path to the source file

    Returns:
        Dictionary containing extracted metadata
    """
    metadata = {
        'imports': [],
        'classes': [],
        'methods': [],
        'method_calls': [],
        'extends': [],
        'implements': []
    }

    try:
        # Extract imports
        import_pattern = r'import\s+([\w.]+);'
        metadata['imports'] = re.findall(import_pattern, code)

        # Extract class names
        class_pattern = r'(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)'
        metadata['classes'] = re.findall(class_pattern, code)

        # Extract extends/implements
        extends_pattern = r'class\s+\w+\s+extends\s+(\w+)'
        implements_pattern = r'implements\s+([\w,\s]+)'
        metadata['extends'] = re.findall(extends_pattern, code)
        impl_matches = re.findall(implements_pattern, code)
        if impl_matches:
            metadata['implements'] = [
                i.strip() for match in impl_matches for i in match.split(',')
            ]

        # Extract method definitions
        method_pattern = r'(?:public|private|protected)\s+(?:static\s+)?(?:\w+\s+)?(\w+)\s*\([^)]*\)\s*\{'
        metadata['methods'] = re.findall(method_pattern, code)

        # Extract method calls (simplified - looks for pattern: identifier.method())
        call_pattern = r'(\w+)\.(\w+)\s*\('
        calls = re.findall(call_pattern, code)
        metadata['method_calls'] = list(set([f"{obj}.{method}" for obj, method in calls]))

    except Exception:
        # If parsing fails, return empty metadata
        pass

    return metadata


def extract_target_entity(question: str) -> str:
    """
    Extract the target class/method name from question

    Args:
        question: User's question

    Returns:
        Target entity name or None
    """
    # Common patterns: "What calls User?", "What uses AuthenticationService?"
    patterns = [
        r'(?:calls|uses|depends on|imports|extends|implements)\s+([A-Z]\w+)',
        r'([A-Z]\w+)\s+(?:class|method|function)',
    ]

    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return match.group(1)

    # Fallback: find capitalized words (likely class names)
    words = question.split()
    for word in words:
        if word and word[0].isupper() and len(word) > 2:
            return word.strip('?.,')

    return None
