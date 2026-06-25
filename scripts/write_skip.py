#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--name',required=True); p.add_argument('--reason',required=True); a=p.parse_args()
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'name':a.name,'status':'skipped','reason':a.reason,'epoch':time.time()},indent=2))
print(out)
