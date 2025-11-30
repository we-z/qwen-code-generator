import os
import time
import uuid
import json
import logging
import requests
import re
from pathlib import Path
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
        try:
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump(self.stats, f)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def get_next_prompt_batch(self) -> Optional[Path]:
        try:
            files = list(PROMPTS_DIR.glob("*.json"))
            if not files:
                return None
            return files[0]
        except Exception:
            return None

    def call_llm(self, prompt: str) -> Optional[str]:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 262144,
            "temperature": 0.7,
            "stream": True
        }
        try:
            logger.info("Calling LLM API with streaming enabled...")
            response = requests.post(LLM_ENDPOINT, json=payload, timeout=300, stream=True)
            response.raise_for_status()
            
            full_content = ""
            print(f"\n{'='*80}")
            print("LLM Response (streaming):")
            print(f"{'='*80}")
            
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        data_str = line_text[6:]
                        if data_str == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    print(content, end='', flush=True)
                                    full_content += content
                        except json.JSONDecodeError:
                            continue
            
            print(f"\n{'='*80}\n")
            logger.info(f"LLM returned {len(full_content)} characters")
            return full_content
        except Exception as e:
            logger.error(f"LLM Call failed: {e}")
            self.stats["api_errors"] += 1
            return None

    def extract_python_code(self, text: str) -> Optional[str]:
        # Strategy 1: Look for properly closed ```python ... ``` blocks
        code_blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
        
        # Strategy 2: If no properly closed blocks, try without "python" tag
        if not code_blocks:
            code_blocks = re.findall(r"```(.*?)```", text, re.DOTALL)
        
        # Strategy 3: If still nothing, try to extract unclosed code fences
        if not code_blocks:
            # Look for ```python or ``` at start, capture everything after
            match = re.search(r"```(?:python)?\s*\n(.*)", text, re.DOTALL)
            if match:
                code_blocks = [match.group(1)]
        
        # Strategy 4: If all else fails, check if entire response looks like Python
        if not code_blocks:
            # If text starts with common Python patterns, treat it as code
            if any(text.strip().startswith(pattern) for pattern in 
                   ['import ', 'from ', 'def ', 'class ', '#!', '"""', "'''"]):
                code_blocks = [text]
        
        if not code_blocks:
            return None
        
        # Join all blocks
        full_code = "\n\n".join(block.strip() for block in code_blocks)
        
        # Basic syntax check
        try:
            compile(full_code, "<string>", "exec")
            return full_code
        except SyntaxError as e:
            logger.warning(f"Syntax error in extracted code: {e}")
            # Try to salvage partial code by truncating at the error
            # This handles cases where the stream was cut off mid-line
            if hasattr(e, 'lineno') and e.lineno:
                lines = full_code.split('\n')
                # Try removing lines from the error point backward
                for trim_point in range(e.lineno - 1, max(0, e.lineno - 10), -1):
                    truncated = '\n'.join(lines[:trim_point])
                    try:
                        compile(truncated, "<string>", "exec")
                        logger.info(f"Salvaged code by truncating to line {trim_point}")
                        return truncated
                    except:
                        continue
            return None

    def process_prompt(self, prompt: str):
        file_uuid = uuid.uuid4().hex
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing new prompt (UUID: {file_uuid})")
        logger.info(f"{'='*80}")
        logger.info(f"Prompt: {prompt[:200]}...")
        
        # Call LLM
        raw_output = self.call_llm(prompt)
        if not raw_output:
            logger.error("LLM returned no output, skipping this prompt")
            return

        # Extract Code
        logger.info("Extracting Python code from LLM response...")
        code = self.extract_python_code(raw_output)
        if not code:
            logger.warning(f"Failed to extract code for {file_uuid}")
            self.stats["failed_extraction"] += 1
            fail_path = FAILURES_DIR / f"{file_uuid}_extraction_fail.txt"
            with open(fail_path, "w") as f:
                f.write(f"PROMPT:\n{prompt}\n\nRAW OUTPUT:\n{raw_output}")
            return
        
        logger.info(f"Successfully extracted {len(code)} characters of Python code")
        
        # Print generated code in real-time
        print(f"\n{'='*80}")
        print(f"Extracted Python Code for {file_uuid}:")
        print(f"{'='*80}")
        print(code)
        print(f"{'='*80}\n")

        # Determine Directory
        logger.info(f"Writing files to dataset...")
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
                "timestamp": time.time(),
                "status": "generated",
                "syntax_check": "passed"
            }
            with open(meta_file, "w") as f:
                json.dump(metadata, f)

            # Signal Validator
            queue_item = QUEUE_VALIDATION_DIR / f"{file_uuid}.json"
            with open(queue_item, "w") as f:
                json.dump({"path": str(py_file), "meta_path": str(meta_file)}, f)

            self.stats["generated"] += 1
            logger.info(f"✓ Successfully saved {file_uuid}.py to {dir_1}/{dir_2}/")
            logger.info(f"Progress: {self.stats['generated']} files generated total")
            
            if self.stats["generated"] % 10 == 0:
                self.save_checkpoint()

        except Exception as e:
            logger.error(f"File write error {e}")

    def run(self):
        logger.info("Starting Code Generator...")
        while True:
            batch_file = self.get_next_prompt_batch()
            if not batch_file:
                time.sleep(2)
                continue
            
            # Rename to claim the batch
            processing_file = batch_file.with_suffix(".processing")
            try:
                batch_file.rename(processing_file)
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.error(f"Error renaming batch file: {e}")
                continue

            # Process batch
            try:
                with open(processing_file, "r") as f:
                    prompts = json.load(f)
                
                for prompt in prompts:
                    self.process_prompt(prompt)
                
                processing_file.unlink()
            except Exception as e:
                logger.error(f"Error processing batch {batch_file}: {e}")
                try:
                    processing_file.rename(FAILURES_DIR / processing_file.name)
                except:
                    pass

if __name__ == "__main__":
    generator = CodeGenerator()
    generator.run()