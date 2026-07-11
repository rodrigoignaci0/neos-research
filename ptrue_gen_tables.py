#!/usr/bin/env python3
"""Genera todas las tablas del paper desde results.json.
Uso: python3 gen_tables.py results.json > tablas.md
Regla de oro: ningún número del paper se edita a mano; si cambia el run,
se regenera results.json y se vuelven a pegar estas tablas."""
import json, sys
from decimal import Decimal, ROUND_HALF_UP

def r3(x): return str(Decimal(str(x)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))
def sign(x, nd=3):
    s = str(Decimal(str(x)).quantize(Decimal('0.'+'0'*nd), rounding=ROUND_HALF_UP))
    return ('+'+s) if x >= 0 else s.replace('-', '−')

d = json.load(open(sys.argv[1] if len(sys.argv)>1 else 'results.json'))
DOMS = ['history','geography','entertainment','sports','literature']
FAMS = ['Llama3-8B','Gemma2-2B','Gemma2-9B','Qwen25-7B']
DISPLAY = {'Qwen25-7B':'Qwen2.5-7B'}
disp = lambda f: DISPLAY.get(f, f)

print("### 4.1 TriviaQA Accuracy\n")
print("| Model | Base Acc | Instruct Acc |\n|---|---|---|")
for f in FAMS:
    print(f"| {disp(f)} | {r3(d['triviaqa'][f+'-base']['accuracy'])} | {r3(d['triviaqa'][f+'-instruct']['accuracy'])} |")

print("\n### 4.2 ECE\n")
print("| Model | Base ECE | Instruct ECE | Full-set direction | Matched ΔECE | Matched direction |\n|---|---|---|---|---|---|")
for f in FAMS:
    b, i = d['triviaqa'][f+'-base']['ece_full'], d['triviaqa'][f+'-instruct']['ece_full']
    m = d['item_matched'][f]['delta_ece_matched']
    fd = "instruct better ↓" if i < b else "instruct worse ↑"
    md = "instruct worse ↑" if m > 0 else "instruct better ↓"
    print(f"| {disp(f)} | {r3(b)} | {r3(i)} | {fd} | {sign(m)} | {md} |")

print("\n### 4.3 Domain-Stratified LOCO AUC\n")
print("| Model | Hist. | Geo. | Ent. | Spt. | Lit. | LOCO Avg |\n|---|---|---|---|---|---|---|")
def loco_row(name, rec):
    L = rec['loco_by_domain']
    print(f"| {name} | " + " | ".join(r3(L[k]) for k in DOMS) + f" | {r3(L['macro5'])} |")
for f in FAMS:
    loco_row(disp(f)+'-base', d['triviaqa'][f+'-base']); loco_row(disp(f)+'-instruct', d['triviaqa'][f+'-instruct'])
loco_row('OLMo-2-7B-base', d['olmo_pythia']['OLMo-2-7B-base']); loco_row('Pythia-6.9B-base', d['olmo_pythia']['Pythia-6.9B-base'])

print("\n### 4.4 Item-Matched AUC (Main Results)\n")
print("| Model Family | n | AUC Base | AUC Inst | Delta | 98.75% CI | Sig. |\n|---|---|---|---|---|---|---|")
for f in FAMS:
    v = d['item_matched'][f]
    print(f"| {disp(f)} | {v['n_matched']} | {r3(v['auc_base'])} | {r3(v['auc_instruct'])} | {sign(v['delta'])} | [{sign(v['ci_98_75_lo'])}, {sign(v['ci_98_75_hi'])}] | {v['significant']} |")

print("\n### 4.5 Chat Template Ablation\n")
print("| Model | Base LOCO | Instruct LOCO (raw) | Instruct LOCO (chat) | Chat − Raw | Chat vs. Base |\n|---|---|---|---|---|---|")
for f in FAMS:
    b = d['triviaqa'][f+'-base']['loco_by_domain']['macro5']
    raw = d['triviaqa'][f+'-instruct']['loco_by_domain']['macro5']
    chat = d['triviaqa_chat'][f+'-instruct-chat']['loco_by_domain']['macro5']
    rel = "chat-instruct > base" if chat > b else "base > chat-instruct"
    print(f"| {disp(f)} | {r3(b)} | {r3(raw)} | {r3(chat)} | {sign(chat-raw)} | {rel} |")

print("\n### 4.6 SciQ Replication\n")
print("| Model | Base AUC | Instruct AUC | ΔAUC | Base ECE | Instruct ECE |\n|---|---|---|---|---|---|")
for f in FAMS:
    b, i = d['sciq'][f+'-base'], d['sciq'][f+'-instruct']
    print(f"| {disp(f)} | {r3(b['auc_overall'])} | {r3(i['auc_overall'])} | {sign(i['auc_overall']-b['auc_overall'])} | {r3(b['ece_full'])} | {r3(i['ece_full'])} |")

print("\n### 4.7 Pretraining Data Control\n")
o, p = d['olmo_pythia']['OLMo-2-7B-base'], d['olmo_pythia']['Pythia-6.9B-base']
print("| Model | Alignment | Pretraining | Accuracy | LOCO AUC |\n|---|---|---|---|---|")
print(f"| OLMo-2-7B-1124 | None (base only) | Dolma2 + QA-rich Dolmino mid-train (~3T tok) | {r3(o['accuracy'])} | {r3(o['loco_by_domain']['macro5'])} |")
print(f"| Pythia-6.9B | None (base only) | The Pile (~300B tok) | {r3(p['accuracy'])} | {r3(p['loco_by_domain']['macro5'])} |")
print(f"| Llama3-8B-instruct | RLHF/DPO | — | {r3(d['triviaqa']['Llama3-8B-instruct']['accuracy'])} | {r3(d['triviaqa']['Llama3-8B-instruct']['loco_by_domain']['macro5'])} |")
print(f"| Gemma2-9B-instruct | SFT/RLHF | — | {r3(d['triviaqa']['Gemma2-9B-instruct']['accuracy'])} | {r3(d['triviaqa']['Gemma2-9B-instruct']['loco_by_domain']['macro5'])} |")
print(f"| Qwen2.5-7B-base | None (base) | Qwen2.5 QA-rich pretraining | {r3(d['triviaqa']['Qwen25-7B-base']['accuracy'])} | {r3(d['triviaqa']['Qwen25-7B-base']['loco_by_domain']['macro5'])} |")
