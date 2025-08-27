#!/usr/bin/env python3
"""
Setup script for the Bilingual Subtitle Generator.
Production-ready AI-powered subtitle generation system.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="bilingual-subtitle-generator",
    version="2.5.0",
    author="Anderson Oliveira",
    author_email="anderson@example.com",
    description="Production-ready AI-powered bilingual subtitle generation system with enterprise-grade quality control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andersob0/subtitle-generator",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.0",
        "requests>=2.31.0",
        "python-docx>=0.8.11",
        "pandas>=1.5.0",
    ],
    extras_require={
        "ai": [
            "openai>=1.0.0",
            "google-generativeai>=0.3.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "all": [
            "openai>=1.0.0",
            "google-generativeai>=0.3.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "subtitle-generator=subtitle_generator.app:main",
            "subtitle-cost=scripts.api_cost_calculator:main",
            "subtitle-cli=scripts.comprehensive_cost_cli:main",
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Video",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business",
        "Framework :: Streamlit",
    ],
    keywords=[
        "subtitle", "bilingual", "AI", "translation", "SRT", "VTT", 
        "claude", "openai", "gemini", "openrouter", "streamlit",
        "quality-control", "cost-analysis", "batch-processing",
        "enterprise-grade", "production-ready"
    ],
    project_urls={
        "Homepage": "https://github.com/andersob0/subtitle-generator",
        "Bug Reports": "https://github.com/andersob0/subtitle-generator/issues",
        "Source Code": "https://github.com/andersob0/subtitle-generator",
        "Documentation": "https://github.com/andersob0/subtitle-generator/tree/main/docs",
        "Changelog": "https://github.com/andersob0/subtitle-generator/blob/main/docs/IMPLEMENTATION_SUMMARY.md",
    },
    include_package_data=True,
    zip_safe=False,
)
