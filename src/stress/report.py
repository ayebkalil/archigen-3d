"""Reports distinguish observation from extrapolation and technical success from quality."""
from collections import Counter, defaultdict
import statistics


def render(campaign):
    records = campaign['records']
    lines = ['# ArchiGen 3D — stress campaign', '', f"Parent MLflow run: `{campaign['run_id']}`.",
             f"Elapsed: {campaign.get('elapsed_seconds',0):.1f} s. Job memory cap: {campaign.get('memory_limit_mib',0):.0f} MiB.",
             '', 'CPU/GPU comparison was not repeated. Warmups and recovery probes are excluded from measurement summaries.',
             'Memory samples describe the shared service process tree and global device; they are not per-request GPU allocations.',
             '', '| Model | Case | N | Technical outcomes | Quality | Median / min / max HTTP seconds |',
             '|---|---|---:|---|---|---|']
    grouped = defaultdict(list)
    for r in records:
        if r['run_kind']=='measurement': grouped[(r['model'],r['case']['id'])].append(r)
    for (model,case), group in grouped.items():
        durations = [r['metrics']['http_total_seconds'] for r in group if 'http_total_seconds' in r['metrics']]
        duration = f'{statistics.median(durations):.3f} / {min(durations):.3f} / {max(durations):.3f}' if durations else 'unavailable'
        lines.append(f"| {model} | {case} | {len(group)} | {dict(Counter(r['technical_status'] for r in group))} | {dict(Counter(r['quality'] for r in group))} | {duration} |")
    models = ['floorplan','facade','furniture']
    if any(r['model']=='furniture_preprocessing' for r in records): models.append('furniture_preprocessing')
    for model in models:
        measured = [r for r in records if r['model']==model and r['run_kind']=='measurement']
        lines += ['', f'## {model}', '', f'{len(measured)} measured attempts.']
        if model=='furniture_preprocessing':
            lines += ['**Component tests only. No TripoSR neural reconstruction was executed in these runs.**']
        rejected = [r for r in measured if r['technical_status']!='success']
        lines += [f'{len(rejected)} non-success responses. No claim of perceptual accuracy follows from a valid output.',
                  'Ground-truth metrics are absent unless matching annotations were actually available and evaluated.']
        if measured:
            reliable = [name for (m,name),group in grouped.items() if m==model and len(group)>=3 and all(
                        r['technical_status']=='success' and r['quality']=='usable' for r in group)]
            lines += ['Verified usable conditions: '+(', '.join(reliable) if reliable else
                      'none established by the current visual/ground-truth review; consult structural checks and artifacts.')]
            for r in measured:
                if r['technical_status']!='success' or r['quality'] in ('degraded','unusable'):
                    lines.append(f"- `{r['case']['id']}` / `{r['run_id']}`: {r['technical_status']}, quality={r['quality']}, stage={r.get('error_stage','indeterminate')}.")
        blocked = [r for r in campaign['skipped'] if r.get('model')==model]
        if blocked: lines += ['', 'Not executed / blocked:', *[f'- {r.get("case","startup")}: {r["reason"]}' for r in blocked]]
    lines += ['', '## Interpretation and recommendations', '',
              '- Treat observed model tensor sizes separately from file decoding and resizing costs. Source pixels do not establish neural inference cost.',
              '- Keep the TripoSR sequential GPU lock and existing 3500 MiB guard; concurrent clients may queue rather than execute inference together.',
              '- Use the tested timeout ceilings as campaign protections, not a measured production SLA. No p95 is inferred from three repetitions.',
              '- Reject empty/corrupt images and unsupported content before expensive work; use byte and decoded-pixel bounds independently.',
              '- Select production size and concurrency limits only from completed, reviewed cases; do not extrapolate to larger untested inputs.',
              '- Investigate traces for each failure stage and use recovery probes before declaring a service recovered.',
              '- Endurance samples can flag growth but do not prove absence of a memory leak during longer use.',
              '- Uniform or missing-foreground images producing geometry are semantic failures even when GLB validation succeeds.',
              '', '## Tracking limitations', '', *['- '+w for w in campaign.get('tracking_warnings',[])]]
    return '\n'.join(lines)+'\n'
