import json
import re

log_path = r"C:\Users\Anusha\.gemini\antigravity-ide\brain\ea75b8af-5869-4b13-89c0-83a20bfdbef2\.system_generated\logs\transcript_full.jsonl"
target_file = r"c:\Users\Anusha\engression\engression-ts\engressionts\base\base_engression.py"

lines_by_num = {}

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        step_index = obj.get("step_index")
        if step_index in [17, 19, 21, 23]:
            content = obj.get("content", "")
            for l in content.splitlines():
                m = re.match(r"^(\d+):(.*)", l)
                if m:
                    line_num = int(m.group(1))
                    raw_content = m.group(2)
                    if raw_content.startswith(" "):
                        line_content = raw_content[1:]
                    else:
                        line_content = raw_content
                    lines_by_num[line_num] = line_content

sorted_keys = sorted(lines_by_num.keys())
print(f"Total lines reconstructed: {len(sorted_keys)}")

reconstructed_lines = [lines_by_num[k] + "\n" for k in sorted_keys]

with open(target_file, "w", encoding="utf-8", newline="") as out:
    out.writelines(reconstructed_lines)

print("Reconstruction complete!")
