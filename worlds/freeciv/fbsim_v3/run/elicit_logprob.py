#!/usr/bin/env python3
"""elicit_logprob.py — token-probability vs verbalized elicitation on the same items (2026-10-02).

  uv run python elicit_logprob.py --out DIR [--models id,id] [--sample "bank=200,tails=60"] [--workers 8]

Three probabilities per (model, item), all from models that return logprobs through OpenRouter, thinking off everywhere:
  verbal   the standard prompt (report, question, criteria, <probability> tag). value = the stated number.
           The same call is made with logprobs on, so the token after "<probability>0." also gives the model's own
           distribution over the two-digit answer: ev_verbal = its expectation, numeric_mass = the share of that
           distribution that is a number (second-order / "distribution over stated numbers").
  token    the same report, question and criteria, then "Answer with a single word, Yes or No." max_tokens 2,
           temperature 0, top_logprobs 20. p_yes = P(yes variants) / (P(yes variants) + P(no variants)), summing over
           surface forms (Yes, yes, YES, " Yes", "Yes."), as surface-form competition requires; coverage = P(yes)+P(no)
           before normalisation, so a model that mostly wants to say something else is visible.
Rows: DIR/<slug>/results.jsonl, resumable. Score with analyze_logprob.py.
"""
import argparse, json, math, os, re, sys, time, threading, collections, urllib.request, urllib.error, random
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from elicit_v1 import build_jobs, sample_filter, KEY
TOKEN_TAIL = """
Question Title: {question}

Resolution Criteria: {criteria}

Based on the report above, answer the question with a single word, Yes or No. Do not explain."""
MODELS = {   # id -> (provider order or None, extra body)
    'openai/gpt-4.1': (['OpenAI'], {}), 'openai/gpt-4.1-mini': (['OpenAI'], {}), 'openai/gpt-4o-mini': (['OpenAI'], {}),
    # hosts pinned to the ones that return top-20 logprobs (probed 2026-10-02); DigitalOcean avoided (dropped 8k prompts in Sept)
    'qwen/qwen3-235b-a22b-2507': (['Parasail', 'Google'], {}), 'deepseek/deepseek-chat-v3-0324': (['GMICloud'], {}), 'qwen/qwen3-next-80b-a3b-instruct': (['Parasail', 'Google'], {}),
    'deepseek/deepseek-v4-flash-0731': (['StreamLake', 'Sail Research', 'Reka'], {"reasoning": {"enabled": False}}),
}
IGNORE = ['Novita', 'OpenInference', 'DigitalOcean']   # hosts that dropped the question on 8k prompts (2026-09-09)
lock = threading.Lock()
def call(mid, messages, max_tokens, extra):
    prov, body_extra = MODELS[mid]
    body = {"model": mid, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "logprobs": True, "top_logprobs": 20, "usage": {"include": True}, **body_extra, **extra}
    body["provider"] = {"order": prov, "allow_fallbacks": len(prov) > 1} if prov else {"ignore": IGNORE}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    err = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=300) as r: resp = json.load(r)
            ch = resp['choices'][0]; lp = (ch.get('logprobs') or {}).get('content') or []
            return dict(text=(ch.get('message') or {}).get('content') or '', lp=[dict(token=t['token'], logprob=t['logprob'], top=[(x['token'], x['logprob']) for x in (t.get('top_logprobs') or [])]) for t in lp],
                        provider=resp.get('provider'), cost=float((resp.get('usage') or {}).get('cost') or 0), tokens_out=(resp.get('usage') or {}).get('completion_tokens'), finish=ch.get('finish_reason'), error=None)
        except urllib.error.HTTPError as e:
            err = f'HTTP {e.code}: ' + e.read().decode(errors='replace')[:200]
            if e.code in (400, 404) and attempt >= 1: break
            time.sleep(10 + 10 * attempt)
        except Exception as e: err = str(e)[:200]; time.sleep(5 * (attempt + 1))
    return dict(text='', lp=[], error=err, cost=0)
YES = {'yes', 'y', 'true'}; NO = {'no', 'n', 'false'}
def norm(tok): return re.sub(r'[^a-z]', '', tok.lower())
def yes_no(lp):
    """P(yes) over surface forms from the first generated token's top-20."""
    if not lp: return None, 0.0, []
    top = lp[0]['top'] or [(lp[0]['token'], lp[0]['logprob'])]
    py = sum(math.exp(l) for t, l in top if norm(t) in YES); pn = sum(math.exp(l) for t, l in top if norm(t) in NO)
    return (py / (py + pn) if py + pn > 0 else None), py + pn, [(t, round(l, 2)) for t, l in top[:8]]
def verbal_dist(lp):
    """Distribution over the stated number: look at the token following '<probability>' and the one following '<probability>0.'."""
    text = ''
    best = None
    for i, t in enumerate(lp):
        text += t['token']; s = re.sub(r'\s', '', text)
        nxt = lp[i + 1] if i + 1 < len(lp) else None
        if not nxt or not nxt['top']: continue
        if s.endswith('<probability>0.') or s.endswith('<probability>.'):
            if nxt['top'] == t['top']: continue                      # host repeated the previous token's list: no digit distribution available
            digs = [(tok.strip(), math.exp(l)) for tok, l in nxt['top'] if re.fullmatch(r'\s*\d{1,3}\s*', tok)]
            single = digs and all(len(t) == 1 for t, _ in digs) and len(digs) >= 5          # digit-by-digit tokenizer (Qwen, DeepSeek): tenths only
            cand = [((int(t) + 0.5) / 10 if single else float('0.' + t), w) for t, w in digs]
            mass = sum(w for _, w in cand)
            if cand and mass > 0 and (best is None or mass > best[1]): best = (sum(v * w for v, w in cand) / mass, mass, 'after 0. (tenths)' if single else 'after 0.')
        elif s.endswith('<probability>'):
            cand = []
            for tok, l in nxt['top']:
                tk = tok.strip()
                if re.fullmatch(r'(0?\.\d+|1(\.0+)?|0)', tk): cand.append((float(tk), math.exp(l)))
                elif tk == '0': cand.append((None, math.exp(l)))        # continues with '.' -> handled by the branch above
            num = [(v, w) for v, w in cand if v is not None]; mass = sum(w for v, w in num)
            if num and mass > 0.5 and (best is None or mass > best[1]): best = (sum(v * w for v, w in num) / mass, mass, 'after tag')
    return best
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--models', default=','.join(MODELS)); ap.add_argument('--sample', default='bank=200,tails=60'); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    jobs = sample_filter(build_jobs('bank,tails'), a.sample)['t1']
    print(f"{len(jobs)} items ({collections.Counter(j['set'] for j in jobs)}) x 2 arms x {len(a.models.split(','))} models", flush=True)
    if a.dry_run: return
    for mid in a.models.split(','):
        slug = mid.replace('/', '_').replace('.', '_'); d = f'{a.out}/{slug}'; os.makedirs(d, exist_ok=True); ck = f'{d}/results.jsonl'
        done = {}
        if os.path.exists(ck):
            for line in open(ck): r = json.loads(line); done[(r['item'], r['arm'])] = r
        out = open(ck, 'a'); t0 = time.time(); spent = [0.0]
        def run(j):
            rows = []
            if (j['item'], 'verbal') not in done:
                r = call(mid, [{"role": "user", "content": j['prefix'] + j['tail']}], 4000, {})
                m = re.findall(r'<probability>\s*([0-9]*\.?[0-9]+)\s*</probability>', r['text']); v = float(m[-1]) if m else None
                dist = verbal_dist(r['lp'])
                tag_lp = next(([(t['token'], t['top'][:12]) for t in r['lp'][i:i+5]] for i in range(len(r['lp']) - 1, -1, -1) if re.sub(r'\s','',''.join(t['token'] for t in r['lp'][:i+1])).endswith('<probability>')), None)   # LAST tag occurrence
                rows.append(dict(model=mid, item=j['item'], set=j['set'], arm='verbal', value=(v / 100 if v and v > 1 else v), ev_verbal=dist[0] if dist else None, numeric_mass=dist[1] if dist else None, dist_where=dist[2] if dist else None, tag_lp=tag_lp,
                                 text=r['text'], provider=r.get('provider'), cost=r['cost'], finish=r.get('finish'), error=r.get('error')))
            if (j['item'], 'token') not in done:
                q = re.search(r'Question Title: (.*)', j['tail']).group(1); c = re.search(r'Resolution Criteria: (.*)', j['tail']).group(1)
                r = call(mid, [{"role": "user", "content": j['prefix'] + TOKEN_TAIL.format(question=q, criteria=c)}], 2, {})
                p, cov, top = yes_no(r['lp'])
                rows.append(dict(model=mid, item=j['item'], set=j['set'], arm='token', value=p, coverage=cov, top=top, text=r['text'], provider=r.get('provider'), cost=r['cost'], error=r.get('error')))
            with lock:
                for row in rows: out.write(json.dumps(row) + '\n'); spent[0] += row['cost']
                out.flush()
            return rows
        with ThreadPoolExecutor(a.workers) as ex: list(ex.map(run, jobs))
        rows = [json.loads(l) for l in open(ck)]
        print(f"{mid}: {len(rows)} rows, verbal unparsed {sum(1 for r in rows if r['arm']=='verbal' and r['value'] is None)}, token no-yes/no {sum(1 for r in rows if r['arm']=='token' and r['value'] is None)}, errors {sum(1 for r in rows if r.get('error'))}, ${spent[0]:.2f}, {time.time()-t0:.0f}s, providers {dict(collections.Counter(r.get('provider') for r in rows))}", flush=True)
if __name__ == '__main__': main()
