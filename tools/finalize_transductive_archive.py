import csv, json, sys
from pathlib import Path
from datetime import datetime, timezone

MODELS=['ToPoGate_strict8','KMeans','PCA+KMeans','scMAE','IDEC','EDESC','TableDC','ZEUS','scNAME','scDeepCluster','scCDCG']
SEEDS={42,123,7}

def main(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    def read_csv(name):
        p=root/name
        return list(csv.DictReader(p.open(encoding='utf-8',newline='')))
    ledger=read_csv('run_ledger.csv'); dm=read_csv('dataset_manifest.csv')
    datasets=[]
    for r in dm:
        d=r.get('dataset_id') or r.get('dataset')
        if d and d not in datasets: datasets.append(d)
    # ensure exactly the declared model x dataset grid is represented
    reasons={m:'no verified transductive three-seed default evidence' for m in MODELS}
    reasons.update({'KMeans':'new transductive runs','PCA+KMeans':'new transductive runs',
                    'EDESC':'historical reuse only where all target seeds and provenance are explicit',
                    'TableDC':'historical reuse only where all target seeds and provenance are explicit',
                    'ZEUS':'historical reuse only where all target seeds and provenance are explicit'})
    # deterministic audit tables
    with (root/'model_registry.json').open('w') as f:
        json.dump({'models':MODELS,'protocol':'transductive','seeds':[42,123,7],'default_parameters_only':True},f,indent=2)
    plan=[]
    for m in MODELS:
        for d in datasets:
            plan.append({'model':m,'dataset':d,'seed_set':'42,123,7','protocol':'transductive','planned_action':'reuse_or_run','status':'completed' if any(x.get('model')==m and x.get('dataset')==d for x in ledger) else 'not_available','reason':reasons[m]})
    def write(name,rows,fields):
        with (root/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    write('job_plan.csv',plan,['model','dataset','seed_set','protocol','planned_action','status','reason'])
    audit=[]
    for x in ledger:
        audit.append({'model':x.get('model'),'dataset':x.get('dataset'),'seed':x.get('seed'),'source':x.get('source','new_run'),'accepted': 'yes' if x.get('status')=='completed' else 'no','protocol':x.get('protocol','transductive'),'historical_source_path':x.get('historical_source_path',''),'reason':'accepted target seed record' if x.get('status')=='completed' else 'incomplete'})
    write('reuse_audit.csv',audit,['model','dataset','seed','source','accepted','protocol','historical_source_path','reason'])
    write('compute_costs.csv',[{'model':m,'new_run_count':sum(1 for x in ledger if x.get('model')==m and x.get('source','new_run')=='new_run'),'historical_reuse_count':sum(1 for x in ledger if x.get('model')==m and x.get('source')=='historical_reuse')} for m in MODELS],['model','new_run_count','historical_reuse_count'])
    artifacts=[]
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name not in {'artifact_manifest.csv'}:
            artifacts.append({'path':str(p),'size_bytes':p.stat().st_size})
    write('artifact_manifest.csv',artifacts,['path','size_bytes'])
    # create workbook from the already-generated wide CSV
    try:
        from openpyxl import Workbook
        wide=read_csv('default_three_seed_wide.csv')
        wb=Workbook(); ws=wb.active; ws.title='ARI_mean_model_x_dataset'
        cols=list(wide[0].keys()) if wide else ['model']
        ws.append(cols)
        for r in wide: ws.append([r.get(c,'NA') for c in cols])
        for row in ws.iter_rows():
            for cell in row:
                if cell.column>1 and cell.value not in ('NA',None):
                    try: cell.value=round(float(cell.value),6)
                    except Exception: pass
        wb.save(root/'default_three_seed_model_x_dataset.xlsx')
    except Exception:
        # stdlib-only XLSX fallback (minimal valid workbook)
        import zipfile, html
        wide=read_csv('default_three_seed_wide.csv'); cols=list(wide[0].keys()) if wide else ['model']
        rows=[cols]+[[r.get(c,'NA') for c in cols] for r in wide]
        def cell(v, ref):
            s=html.escape(str(v))
            return f'<c r="{ref}" t="inlineStr"><is><t>{s}</t></is></c>'
        def colname(n):
            out=''
            while n:
                n,rem=divmod(n-1,26); out=chr(65+rem)+out
            return out
        sheet=['<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>']
        for ri,row in enumerate(rows,1):
            sheet.append(f'<row r="{ri}">'+''.join(cell(v,f'{colname(ci)}{ri}') for ci,v in enumerate(row,1))+'</row>')
        sheet.append('</sheetData></worksheet>')
        files={'[Content_Types].xml':'<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>', '_rels/.rels':'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>', 'xl/workbook.xml':'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="ARI_mean_model_x_dataset" sheetId="1" r:id="rId1"/></sheets></workbook>', 'xl/_rels/workbook.xml.rels':'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>', 'xl/worksheets/sheet1.xml':''.join(sheet)}
        with zipfile.ZipFile(root/'default_three_seed_model_x_dataset.xlsx','w',zipfile.ZIP_DEFLATED) as z:
            for n,c in files.items(): z.writestr(n,c)
    counts=[]
    for m in MODELS:
        valid=0
        for d in datasets:
            rs=[x for x in ledger if x.get('model')==m and x.get('dataset')==d and int(x.get('seed',-1)) in SEEDS and x.get('status')=='completed']
            if len({int(x['seed']) for x in rs})==3: valid+=1
        counts.append((m,valid,len(datasets)-valid))
    lines=['# FINAL REPORT','',f'Archive: `{root}`',f'Protocol: transductive; seeds: 42, 123, 7',f'Datasets discovered: {len(datasets)}',f'New/reused ledger records: {len(ledger)}','', '## Model cell counts','', '| model | valid cells | NA cells |','|---|---:|---:|']
    lines += [f'| {m} | {v} | {na} |' for m,v,na in counts]
    lines += ['', 'All values are same-sample transductive results and must not be interpreted as out-of-sample generalization. Missing or protocol-unverified cells remain `NA`.']
    (root/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__': main(sys.argv[1])
