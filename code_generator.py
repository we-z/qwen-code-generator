import os
import time
import uuid
import json
import logging
import requests
import threading
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

# Configuration
DATASET_ROOT = Path(os.path.expanduser("~/Desktop/generated_python_dataset"))
PROMPTS_DIR = DATASET_ROOT / "prompts"
FAILURES_DIR = DATASET_ROOT / "failures"
CHECKPOINT_FILE = DATASET_ROOT / "checkpoint.json"
LOG_FILE = DATASET_ROOT / "pipeline.log"
QUEUE_VALIDATION_DIR = DATASET_ROOT / "queue_validation"

LLM_ENDPOINT = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "qwen/qwen3-coder-30b"
NUM_WORKERS = 4

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CodeGenerator")

class CodeGenerator:
    def __init__(self):
        self.lock = threading.Lock()
        self.ensure_directories()
        self.stats = self.load_checkpoint()

    def ensure_directories(self):
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        FAILURES_DIR.mkdir(parents=True, exist_ok=True)
        QUEUE_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    def load_checkpoint(self) -> Dict[str, int]:
        if CHECKPOINT_FILE.exists():
            try:
                with open(CHECKPOINT_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
        return {"generated": 0, "failed_extraction": 0, "api_errors": 0}

    def save_checkpoint(self):
        with self.lock:
            try:
                with open(CHECKPOINT_FILE, "w") as f:
                    json.dump(self.stats, f)
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

    def get_next_prompt_batch(self) -> Optional[Path]:
        # Simple FIFO: grab the first json file in prompts dir
        # We need to be careful about race conditions if multiple workers were picking files,
        # but here the main thread or a shared queue manager could dispense them.
        # Since we have 4 workers, let's have them contend for files or use a queue.
        # A simple way is to list files and try to rename/move one to "processing".
        # Or just pick one, read it, and delete it.
        # To avoid contention, let's just list and pick random or first, but handle FileNotFoundError.
        try:
            files = list(PROMPTS_DIR.glob("*.json"))
            if not files:
                return None
            return files[0] # Pick first
        except Exception:
            return None

    def call_llm(self, prompt: str) -> Optional[str]:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096, # Adjust as needed
            "temperature": 0.7
        }
        try:
            response = requests.post(LLM_ENDPOINT, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM Call failed: {e}")
            with self.lock:
                self.stats["api_errors"] += 1
            return None

    def extract_python_code(self, text: str) -> Optional[str]:
        # Strategy: Look for ```python ... ``` blocks.
        # If multiple, combine them or take the largest? Usually the first or largest.
        # Let's take all python blocks.
        code_blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
        if not code_blocks:
            # Try without "python" tag
            code_blocks = re.findall(r"```(.*?)```", text, re.DOTALL)
        
        if not code_blocks:
            return None
        
        # Join blocks or just take the first one?
        # Often explanations are between blocks.
        # Let's join them with newlines.
        full_code = "\n\n".join(block.strip() for block in code_blocks)
        
        # Basic syntax check
        try:
            compile(full_code, "<string>", "exec")
            return full_code
        except SyntaxError:
            return None

    def process_prompt(self, prompt: str, worker_id: int):
        file_uuid = uuid.uuid4().hex
        
        # Call LLM
        raw_output = self.call_llm(prompt)
        if not raw_output:
            return

        # Extract Code
        code = self.extract_python_code(raw_output)
        if not code:
            logger.warning(f"Worker {worker_id}: Failed to extract code for {file_uuid}")
            with self.lock:
                self.stats["failed_extraction"] += 1
            # Save failure for inspection?
            fail_path = FAILURES_DIR / f"{file_uuid}_extraction_fail.txt"
            with open(fail_path, "w") as f:
                f.write(f"PROMPT:\n{prompt}\n\nRAW OUTPUT:\n{raw_output}")
            return

        # Determine Directory
        # Use first 4 chars of UUID for 2-level nesting: ab/cd/
        dir_1 = file_uuid[:2]
        dir_2 = file_uuid[2:4]
        target_dir = DATASET_ROOT / dir_1 / dir_2
        target_dir.mkdir(parents=True, exist_ok=True)

        # Write Files
        py_file = target_dir / f"{file_uuid}.py"
        raw_file = target_dir / f"{file_uuid}_raw_output.txt"
        prompt_file = target_dir / f"{file_uuid}_prompt_used.txt"
        meta_file = target_dir / f"{file_uuid}.json"

        try:
            with open(py_file, "w") as f:
                f.write(code)
            with open(raw_file, "w") as f:
                f.write(raw_output)
            with open(prompt_file, "w") as f:
                f.write(prompt)
            
            metadata = {
                "uuid": file_uuid,
                "prompt": prompt,
                "worker_id": worker_id,
                "timestamp": time.time(),
                "status": "generated",
                "syntax_check": "passed" # We did a compile check
            }
            with open(meta_file, "w") as f:
                json.dump(metadata, f)

            # Signal Validator
            # We can write a small file to queue_validation
            queue_item = QUEUE_VALIDATION_DIR / f"{file_uuid}.json"
            with open(queue_item, "w") as f:
                json.dump({"path": str(py_file), "meta_path": str(meta_file)}, f)

            with self.lock:
                self.stats["generated"] += 1
                if self.stats["generated"] % 10 == 0:
                    logger.info(f"Total Generated: {self.stats['generated']}")
                    self.save_checkpoint()

        except Exception as e:
            logger.error(f"Worker {worker_id}: File write error {e}")

    def worker_loop(self, worker_id: int):
        logger.info(f"Worker {worker_id} started.")
        while True:
            # Get a batch file
            batch_file = self.get_next_prompt_batch()
            if not batch_file:
                time.sleep(2)
                continue
            
            # Try to rename it to lock it (simple file locking)
            # Or just read and delete if we assume we are fast enough or don't care about slight race
            # Better: rename to .processing
            processing_file = batch_file.with_suffix(".processing")
            try:
                batch_file.rename(processing_file)
            except FileNotFoundError:
                # Another worker took it
                continue
            except Exception as e:
                logger.error(f"Error renaming batch file: {e}")
                continue

            # Process batch
            try:
                with open(processing_file, "r") as f:
                    prompts = json.load(f)
                
                for prompt in prompts:
                    self.process_prompt(prompt, worker_id)
                
                # Done with batch
                processing_file.unlink()
            except Exception as e:
                logger.error(f"Error processing batch {batch_file}: {e}")
                # Move to failures?
                try:
                    processing_file.rename(FAILURES_DIR / processing_file.name)
                except:
                    pass

    def run(self):
        logger.info("Starting Code Generator...")
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for i in range(NUM_WORKERS):
                executor.submit(self.worker_loop, i)

if __name__ == "__main__":
    generator = CodeGenerator()
    generator.run()
