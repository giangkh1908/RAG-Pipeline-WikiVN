import gzip
import json
import os
from collections import Counter

count = 0
doc_ids = set()
section_counts = Counter()
ref_count = 0
max_tokens = 0
total_tokens = 0
sample_texts = []

with gzip.open('chunks/chunks_50pct.jsonl.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        count += 1
        doc_ids.add(obj['doc_id'])
        section_counts['/'.join(obj['section_path'][:2])] += 1
        if obj.get('is_reference_section'):
            ref_count += 1
        tokens = obj['token_count']
        max_tokens = max(max_tokens, tokens)
        total_tokens += tokens
        if count <= 3:
            sample_texts.append((obj['title'], obj['section_path'], obj['text'][:200]))

print(f'Total chunks: {count:,}')
print(f'Unique docs: {len(doc_ids):,}')
print(f'Reference chunks: {ref_count:,} ({ref_count/count*100:.1f}%)')
print(f'Max tokens/chunk: {max_tokens}')
print(f'Avg tokens/chunk: {total_tokens/count:.1f}')
size_mb = os.path.getsize('chunks/chunks_50pct.jsonl.gz') / 1024 / 1024
print(f'File size MB: {size_mb:.2f}')
print()
print('Top 5 sections:')
for sec, c in section_counts.most_common(5):
    print(f'  {sec}: {c:,}')
print()
print('Sample chunks:')
for title, path, text in sample_texts:
    path_str = ' > '.join(path)
    print(f'  {title} | {path_str} | {text}...')
