"""
prepare_smart.py
----------------
Data Preparation Script for the AI Supervisor Model.

This script processes raw therapy conversation data and transforms it into
context-aware training pairs suitable for supervised learning. Each training
example includes the conversational history (context window) to enable the
model to understand the therapeutic trajectory.

Author: Clarina AI Team
"""

import json
import csv

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_FILE = "conversations.json"       # Source: HuggingFace mental health dataset
OUTPUT_FILE = "raw_data_with_context.csv"
HISTORY_WINDOW = 3                      # Number of previous turns to include as context

# =============================================================================
# MAIN PROCESSING LOGIC
# =============================================================================

print("[INFO] Starting conversation preprocessing with context extraction...")

rows = []

try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for conv in data:
        # Parse conversation structure: {"id": "...", "messages": [...]}
        msgs = conv.get('messages', [])
        
        # Extract sequential USER -> ASSISTANT turn pairs
        for i in range(len(msgs) - 1):
            if msgs[i]['role'] == 'user' and msgs[i+1]['role'] == 'assistant':
                
                # Extract the current interaction to be evaluated
                patient_txt = msgs[i]['content']
                therapist_txt = msgs[i+1]['content']
                
                # Extract conversational context (sliding window of previous messages)
                # This provides the supervisor model with therapeutic trajectory information
                start_idx = max(0, i - HISTORY_WINDOW)
                context_msgs = msgs[start_idx:i]
                
                # Format context as human-readable string for model input
                context_str = ""
                for m in context_msgs:
                    role_label = "Patient" if m['role'] == 'user' else "Therapist"
                    context_str += f"{role_label}: {m['content']}\n"

                rows.append([context_str, patient_txt, therapist_txt])

    # Persist processed data to CSV format
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["history", "patient_text", "therapist_text"])
        writer.writerows(rows)

    print(f"[SUCCESS] Created {len(rows)} context-aware training pairs.")
    print(f"[OUTPUT] Data saved to: {OUTPUT_FILE}")

except FileNotFoundError:
    print(f"[ERROR] Input file not found: {INPUT_FILE}")
except json.JSONDecodeError as e:
    print(f"[ERROR] Failed to parse JSON: {e}")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")