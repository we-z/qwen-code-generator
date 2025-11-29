import os
import time
import uuid
import json
import random
import logging
from pathlib import Path
from typing import List, Set, Dict

# Configuration
DATASET_ROOT = Path(os.path.expanduser("~/Desktop/generated_python_dataset"))
PROMPTS_DIR = DATASET_ROOT / "prompts"
SEEN_PROMPTS_FILE = DATASET_ROOT / "seen_prompts.log"
LOG_FILE = DATASET_ROOT / "pipeline.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PromptGenerator")

# Combinatorial Data
DOMAINS = [
    "Web Development", "Data Science", "Machine Learning", "System Administration",
    "Network Programming", "Game Development", "Embedded Systems", "Cryptography",
    "Audio Processing", "Image Processing", "Scientific Computing", "Database Management",
    "GUI Application", "CLI Tool", "Automation", "Security", "Finance", "Education"
]

PERSONAS = [
    "Senior Software Engineer", "Data Scientist", "DevOps Engineer", "Security Researcher",
    "Game Developer", "Systems Architect", "Machine Learning Engineer", "Python Tutor",
    "Bioinformatician", "Financial Analyst", "Network Engineer", "QA Engineer"
]

DIFFICULTIES = [
    "Beginner", "Intermediate", "Advanced", "Expert", "Master"
]

TOPICS = {
    "Algorithms": ["Sorting", "Searching", "Graph Theory", "Dynamic Programming", "Greedy Algorithms", "Backtracking", "Tree Traversal"],
    "Data Structures": ["Linked List", "Binary Tree", "Hash Map", "Heap", "Trie", "Graph", "Queue", "Stack"],
    "File System": ["File Organizer", "Duplicate Finder", "Log Parser", "Backup Utility", "Directory Tree Visualizer", "File Encryptor"],
    "Networking": ["Port Scanner", "HTTP Server", "DNS Resolver", "Packet Sniffer", "Chat Client", "Proxy Server"],
    "Math": ["Matrix Multiplication", "Prime Number Generator", "Fourier Transform", "Statistical Analysis", "Equation Solver", "Geometry Utils"],
    "ML/AI": ["Linear Regression", "Neural Network", "K-Means Clustering", "Decision Tree", "Image Classifier", "Text Generator"],
    "Utilities": ["Unit Converter", "Password Generator", "Markdown Parser", "URL Shortener", "Cron Job Scheduler", "Clipboard Manager"],
    "Games": ["Snake", "Tetris", "Sudoku Solver", "Chess Engine", "Text Adventure", "Minesweeper"],
    "Web": ["Web Scraper", "API Client", "Static Site Generator", "Flask App", "Django Model", "WebSocket Server"],
    "Audio/Video": ["Audio Visualizer", "Video Transcoder", "Metadata Extractor", "Tone Generator", "Speech Recognition Stub"],
    "Security": ["Password Cracker (Simulated)", "Caesar Cipher", "RSA Implementation", "Hash Cracker", "Steganography"],
}

CONSTRAINTS = [
    "Optimize for speed", "Optimize for memory usage", "Use object-oriented programming",
    "Use functional programming style", "Include comprehensive docstrings", "Include unit tests",
    "Handle edge cases robustly", "Use type hinting", "No external dependencies (standard library only)",
    "Use asyncio", "Use multiprocessing", "Follow PEP 8 strictly"
]

TEMPLATES = [
    "Write a Python script that implements {topic} for {domain} purposes. Act as a {persona}. Difficulty: {difficulty}. Constraint: {constraint}.",
    "Create a {difficulty} level Python program for {topic}. The program should be useful for a {persona} in the field of {domain}. Ensure you {constraint}.",
    "As a {persona}, develop a tool that performs {topic}. This tool is intended for {domain}. {constraint}.",
    "Implement a robust {topic} solution. Context: {domain}. Target Audience: {persona}. Level: {difficulty}. Requirement: {constraint}.",
    "Design a Python module for {topic} suitable for {domain} applications. Written by a {persona}. Complexity: {difficulty}. {constraint}.",
]

FORMATTING_OPTS = [
    "Use 4 spaces for indentation.",
    "Use tabs for indentation.",
    "Follow Google Python Style Guide.",
    "Follow NumPy docstring convention.",
    "Use snake_case for variable names.",
    "Include a main execution block.",
    "Add type hints to all functions.",
    "Use descriptive variable names."
]

class PromptGenerator:
    def __init__(self):
        self.seen_prompts: Set[str] = set()
        self.new_seen_hashes: List[str] = []
        self.load_seen_prompts()
        self.ensure_directories()

    def ensure_directories(self):
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    def load_seen_prompts(self):
        if SEEN_PROMPTS_FILE.exists():
            try:
                with open(SEEN_PROMPTS_FILE, "r") as f:
                    for line in f:
                        self.seen_prompts.add(line.strip())
                logger.info(f"Loaded {len(self.seen_prompts)} seen prompts.")
            except Exception as e:
                logger.error(f"Failed to load seen prompts: {e}")

    def save_prompt_hash(self, prompt_text: str):
        p_hash = str(hash(prompt_text))
        if p_hash not in self.seen_prompts:
            self.seen_prompts.add(p_hash)
            self.new_seen_hashes.append(p_hash)
            return True
        return False

    def flush_seen_prompts(self):
        if not self.new_seen_hashes:
            return
        try:
            with open(SEEN_PROMPTS_FILE, "a") as f:
                for h in self.new_seen_hashes:
                    f.write(h + "\n")
            self.new_seen_hashes = []
        except Exception as e:
            logger.error(f"Failed to flush seen prompts: {e}")

    def generate_prompt(self) -> str:
        attempts = 0
        while True:
            attempts += 1
            if attempts % 10000 == 0:
                logger.warning(f"High collision rate: {attempts} attempts to find unique prompt. (Total seen: {len(self.seen_prompts)})")

            category = random.choice(list(TOPICS.keys()))
            topic = random.choice(TOPICS[category])
            domain = random.choice(DOMAINS)
            persona = random.choice(PERSONAS)
            difficulty = random.choice(DIFFICULTIES)
            constraint = random.choice(CONSTRAINTS)
            template = random.choice(TEMPLATES)

            prompt = template.format(
                topic=topic,
                domain=domain,
                persona=persona,
                difficulty=difficulty,
                constraint=constraint
            )
            
            # Add random variation
            if random.random() < 0.3:
                prompt += " Please add detailed comments."
            if random.random() < 0.3:
                prompt += " The code should be modular."
            
            # Add formatting option to increase combinatorial space
            if random.random() < 0.5:
                prompt += " " + random.choice(FORMATTING_OPTS)
            
            # Check uniqueness
            p_hash = str(hash(prompt))
            if p_hash not in self.seen_prompts:
                self.save_prompt_hash(prompt)
                return prompt

    def run(self):
        logger.info("Starting Prompt Generator...")
        batch_size = 5000
        target_prompts = 50_000_000
        generated_count = len(self.seen_prompts)
        
        logger.info(f"Targeting {target_prompts:,} prompts. Already have {generated_count:,}.")

        while generated_count < target_prompts:
            logger.info(f"Generating batch starting at {generated_count:,}...")
            batch = []
            for _ in range(batch_size):
                if generated_count >= target_prompts:
                    break
                p = self.generate_prompt()
                batch.append(p)
                generated_count += 1
            
            if not batch:
                break

            # Write batch
            batch_id = uuid.uuid4()
            filename = PROMPTS_DIR / f"{batch_id}.json"
            
            try:
                with open(filename, "w") as f:
                    json.dump(batch, f)
                
                # Flush seen prompts to disk
                self.flush_seen_prompts()

                # Log progress
                logger.info(f"Progress: {generated_count:,}/{target_prompts:,} ({generated_count/target_prompts:.4%}) - Batch {batch_id} saved.")
            except Exception as e:
                logger.error(f"Failed to write prompt batch: {e}")


if __name__ == "__main__":
    generator = PromptGenerator()
    generator.run()
