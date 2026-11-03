
LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
}

NODE_TYPES = {
    "python": {
        "function_definition": "function",
        "class_definition": "class",
        "import_statement": "import",
        "import_from_statement": "import",
    },
    "javascript": {
        "function_declaration": "function",
        "method_definition": "method",
        "class_declaration": "class",
        "import_statement": "import",
    },
    "typescript": {
        "function_declaration": "function",
        "method_definition": "method",
        "class_declaration": "class",
        "import_statement": "import",
    },
    "tsx": {
        "function_declaration": "function",
        "method_definition": "method",
        "class_declaration": "class",
        "import_statement": "import",
    },
    "java": {
        "method_declaration": "method",
        "constructor_declaration": "constructor",
        "class_declaration": "class",
        "import_declaration": "import",
        "interface_declaration": "interface",
    },
}
