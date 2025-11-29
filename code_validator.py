import os
import time
import json
import logging
import subprocess
import resource
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any

# Configuration
DATASET_ROOT = Path(os.path.expanduser("~/Desktop/generated_python_dataset"))
QUEUE_VALIDATION_DIR = DATASET_ROOT / "queue_validation"
FAILURES_DIR = DATASET_ROOT / "failures"
LOG_FILE = DATASET_ROOT / "pipeline.log"
NUM_WORKERS = 4

# Sandbox Limits
TIME_LIMIT_SECONDS = 5
MEMORY_LIMIT_MB = 256

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CodeValidator")

def set_limits():
    # Set CPU time limit
    resource.setrlimit(resource.RLIMIT_CPU, (TIME_LIMIT_SECONDS, TIME_LIMIT_SECONDS + 1))
    # Set Memory limit (AS - Address Space)
    mem_bytes = MEMORY_LIMIT_MB * 1024 * 1024
    # On macOS, RLIMIT_AS might not be strictly enforced or behave differently, but it's good practice.
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except ValueError:
        pass # Ignore if system doesn't allow

def validate_file(queue_item_path: Path):
    try:
        with open(queue_item_path, "r") as f:
            data = json.load(f)
        
        py_path = Path(data["path"])
        meta_path = Path(data["meta_path"])
        
        if not py_path.exists():
            logger.error(f"File not found: {py_path}")
            queue_item_path.unlink()
            return

        # Execute in sandbox
        start_time = time.time()
        result = "failed"
        stdout = ""
        stderr = ""
        exit_code = -1
        
        try:
            # We run the file as a subprocess
            # We use a separate process for safety and limits
            proc = subprocess.run(
                ["python3", str(py_path)],
                capture_output=True,
                text=True,
                preexec_fn=set_limits,
                timeout=TIME_LIMIT_SECONDS + 2 # slightly more than CPU limit to catch it
            )
            stdout = proc.stdout
            stderr = proc.stderr
            exit_code = proc.returncode
            
            if exit_code == 0:
                result = "passed"
            else:
                result = "failed_runtime_error"
                
        except subprocess.TimeoutExpired:
            result = "failed_timeout"
            stderr = "Timeout expired"
        except Exception as e:
            result = f"failed_exception_{type(e).__name__}"
            stderr = str(e)
            
        execution_time = time.time() - start_time

        # Update Metadata
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        except:
            metadata = {}

        metadata.update({
            "validation_result": result,
            "execution_time": execution_time,
            "exit_code": exit_code,
            "stdout": stdout[:1000], # Truncate logs
            "stderr": stderr[:1000],
            "validation_timestamp": time.time()
        })

        # Move files based on result
        if result == "passed":
            # It's already in the target directory (ab/cd/uuid.py), so we just update the json
            # and maybe log it.
            # The prompt said: "Move both the .py file and its metadata JSON into the final directory hierarchy"
            # But the generator already put them there!
            # "If accepted: ... Move both ... into the final directory hierarchy"
            # "If failed: Move the file into ... failures/"
            
            # So if passed, we just update the JSON in place.
            with open(meta_path, "w") as f:
                json.dump(metadata, f)
            logger.info(f"Validated PASS: {py_path.name}")
            
        else:
            # Move to failures
            fail_dest = FAILURES_DIR / py_path.name
            meta_dest = FAILURES_DIR / meta_path.name
            
            shutil.move(str(py_path), str(fail_dest))
            shutil.move(str(meta_path), str(meta_dest))
            
            # Also write failure report? Metadata has it.
            logger.info(f"Validated FAIL: {py_path.name} ({result})")

        # Remove queue item
        queue_item_path.unlink()

    except Exception as e:
        logger.error(f"Validator error processing {queue_item_path}: {e}")
        # Try to remove queue item to avoid loop
        try:
            queue_item_path.unlink()
        except:
            pass

class CodeValidator:
    def __init__(self):
        self.ensure_directories()

    def ensure_directories(self):
        QUEUE_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        FAILURES_DIR.mkdir(parents=True, exist_ok=True)

    def run(self):
        logger.info("Starting Code Validator...")
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            while True:
                # Scan queue
                queue_items = list(QUEUE_VALIDATION_DIR.glob("*.json"))
                if not queue_items:
                    time.sleep(1)
                    continue
                
                # Submit tasks
                # We can submit all, but let's limit to avoid memory explosion if queue is huge
                # ProcessPoolExecutor manages queue size somewhat, but let's be nice.
                futures = []
                for item in queue_items[:100]: # Take batch
                    # Move item to .processing to claim it?
                    # Or just rely on the fact that we are the only validator process (if we run one script)
                    # But we have internal workers.
                    # If we have multiple validator SCRIPTS running, we need locking.
                    # Assuming one validator script with 4 workers.
                    # We can just submit the path.
                    # But if we want to be safe against crashes, we might want to rename.
                    # For now, let's just submit.
                    # Wait, if we submit `item` and then `glob` again, we might pick it up again before it's deleted.
                    # So we should rename it or move it to a 'processing' list.
                    
                    try:
                        processing_path = item.with_suffix(".processing")
                        item.rename(processing_path)
                        futures.append(executor.submit(validate_file, processing_path))
                    except FileNotFoundError:
                        continue
                
                # Wait for this batch? Or just keep going?
                # If we wait, we are synchronous in batches.
                # If we don't, we need to manage the futures.
                # Let's wait for the batch to keep logic simple and robust.
                for f in futures:
                    f.result()

if __name__ == "__main__":
    validator = CodeValidator()
    validator.run()
