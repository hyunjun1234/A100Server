#!/usr/bin/env python3
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); a=p.parse_args()
text=Path(a.input).read_text(errors='ignore') if Path(a.input).exists() else ''
patterns={
 'latency_ms':r'(?i)(?:latency|time|runtime|exec(?:ution)? time)[^\n\d+-]*([-+]?\d+(?:\.\d+)?)\s*ms',
 'latency_us':r'(?i)(?:latency|time|runtime|exec(?:ution)? time)[^\n\d+-]*([-+]?\d+(?:\.\d+)?)\s*(?:us|µs)',
 'gflops':r'(?i)([-+]?\d+(?:\.\d+)?)\s*GFLOP(?:/s|S)?',
 'tflops':r'(?i)([-+]?\d+(?:\.\d+)?)\s*TFLOP(?:/s|S)?',
 'speedup':r'(?i)(?:speedup|speed-up)[^\n\d+-]*([-+]?\d+(?:\.\d+)?)\s*[x×]?',
 'qps':r'(?i)([-+]?\d+(?:\.\d+)?)\s*(?:QPS|queries/s)',
}
result={'source':a.input,'matches':{}}
for key,pat in patterns.items(): result['matches'][key]=[float(x) for x in re.findall(pat,text)]
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)); print(out)
