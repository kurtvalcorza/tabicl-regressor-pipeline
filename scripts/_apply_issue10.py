#!/usr/bin/env python3
import json
from pathlib import Path

MAIN = Path('tutorials/tabiclv2_regressor_colab.ipynb')
INFERENCE = Path('tutorials/tabiclv2_regressor_artifact_inference_colab.ipynb')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, notebook):
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def source_text(cell):
    source = cell.get('source', '')
    return ''.join(source) if isinstance(source, list) else str(source)


def find_cell(notebook, marker):
    matches = [cell for cell in notebook['cells'] if marker in source_text(cell)]
    if len(matches) != 1:
        raise SystemExit(f'Expected one cell containing {marker!r}; found {len(matches)}')
    return matches[0]


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{label}: expected one occurrence, found {count}')
    return text.replace(old, new, 1)


main = load(MAIN)
data_cell = find_cell(main, 'def apply_encoder(frame,encoders):')
text = source_text(data_cell)
old = '''def apply_encoder(frame,encoders):
    out=frame.copy()
    for col,cats in encoders.items():
        lookup,unknown={c:i for i,c in enumerate(cats)},len(cats)
        out[col]=[unknown if pd.isna(v) else lookup.get(str(v),unknown) for v in out[col]]
    return out
'''
new = '''def apply_encoder(frame,encoders,label="data"):
    out=frame.copy()
    for col,cats in encoders.items():
        lookup,unknown={c:i for i,c in enumerate(cats)},len(cats)
        encoded=[]; unseen=0
        for value in out[col]:
            if pd.isna(value):
                encoded.append(unknown); continue
            key=str(value)
            if key not in lookup: unseen+=1
            encoded.append(lookup.get(key,unknown))
        out[col]=encoded
        if unseen: print(f"⚠ {label}: {unseen} unseen categorical value(s) in {col!r} encoded as unknown.")
    return out
'''
text = replace_once(text, old, new, 'main encoder')
for old_call, new_call in (
    ('train_encoded=apply_encoder(train_data,CATEGORICAL_ENCODERS)', 'train_encoded=apply_encoder(train_data,CATEGORICAL_ENCODERS,"training data")'),
    ('holdout_encoded=apply_encoder(holdout_data,CATEGORICAL_ENCODERS)', 'holdout_encoded=apply_encoder(holdout_data,CATEGORICAL_ENCODERS,"holdout")'),
    ('test_encoded=apply_encoder(test_data,CATEGORICAL_ENCODERS) if test_data is not None else None', 'test_encoded=apply_encoder(test_data,CATEGORICAL_ENCODERS,"independent test") if test_data is not None else None'),
):
    text = replace_once(text, old_call, new_call, 'main encoder call')
data_cell['source'] = text

fine_markdown = find_cell(main, '## 4. Evaluate pretrained TabICLv2')
text = source_text(fine_markdown)
if 'fine-tuned `best.ckpt` may remain larger' not in text:
    text += '\n**Fine-tuning disk usage:** TabICL writes epoch checkpoints while tuning, so temporary disk use scales with `FINE_TUNE_EPOCHS`. After the best checkpoint is loaded and evaluated, this notebook deletes non-best epoch checkpoints and retains `best.ckpt` only. The fine-tuned `best.ckpt` may remain larger than the base checkpoint because upstream training state can be embedded; this notebook preserves the upstream checkpoint format for checkpoint compatibility rather than rewriting serialized state.\n'
fine_markdown['source'] = text

fine_cell = find_cell(main, 'FINE_TUNE_EPOCHS,FINE_TUNE_TIME_LIMIT,FINE_TUNE_PATIENCE')
text = source_text(fine_cell)
anchor = '    if candidate_test_metrics: show("Candidate checkpoint test",candidate_test_metrics)\n'
block = anchor + '''    transient_checkpoints=[
        checkpoint for checkpoint in ft_dir.rglob("*.ckpt")
        if checkpoint.resolve()!=candidate_checkpoint.resolve()
    ]
    for checkpoint in transient_checkpoints:
        checkpoint.unlink()
    if transient_checkpoints:
        print(f"✓ Pruned {len(transient_checkpoints)} non-best fine-tuning checkpoint(s); retained best.ckpt.")
'''
text = replace_once(text, anchor, block, 'checkpoint pruning')
fine_cell['source'] = text

manifest_cell = find_cell(main, '"artifactFormat":"tabicl-dimer-regressor-v1"')
text = source_text(manifest_cell)
old = '    "tabiclVersion":TABICL_VERSION,"mode":ACTIVE_MODE,"selectionBasis":SELECTION_BASIS,\n'
new = '''    "tabiclVersion":TABICL_VERSION,"mode":ACTIVE_MODE,"selectionBasis":SELECTION_BASIS,
    "checkpointSource":CHECKPOINT_SOURCE,
    "metrics":{"selectionMetric":EVAL_METRIC,
               "pretrainedHoldout":baseline_metrics,"fineTunedHoldout":candidate_metrics,
               "pretrainedIndependentTest":baseline_test_metrics,"fineTunedIndependentTest":candidate_test_metrics},
'''
text = replace_once(text, old, new, 'artifact provenance')
manifest_cell['source'] = text
save(MAIN, main)

inference = load(INFERENCE)
cell = find_cell(inference, 'def apply_encoder(frame,encoders):')
text = source_text(cell)
text = replace_once(text, old='''def apply_encoder(frame,encoders):
    out=frame.copy()
    for col,cats in encoders.items():
        lookup,unknown={c:i for i,c in enumerate(cats)},len(cats)
        out[col]=[unknown if pd.isna(v) else lookup.get(str(v),unknown) for v in out[col]]
    return out
''', new='''def apply_encoder(frame,encoders,label="data"):
    out=frame.copy()
    for col,cats in encoders.items():
        lookup,unknown={c:i for i,c in enumerate(cats)},len(cats)
        encoded=[]; unseen=0
        for value in out[col]:
            if pd.isna(value):
                encoded.append(unknown); continue
            key=str(value)
            if key not in lookup: unseen+=1
            encoded.append(lookup.get(key,unknown))
        out[col]=encoded
        if unseen: print(f"⚠ {label}: {unseen} unseen categorical value(s) in {col!r} encoded as unknown.")
    return out
''', label='inference encoder')
text = replace_once(
    text,
    'X=apply_encoder(rows[FEATURE_COLUMNS],inference.get("categoricalEncoders",{}))',
    'X=apply_encoder(rows[FEATURE_COLUMNS],inference.get("categoricalEncoders",{}),"inference CSV")',
    'inference encoder call',
)
cell['source'] = text
save(INFERENCE, inference)
