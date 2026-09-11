/** Synchronize three-seed final-test metrics into the formal workbook. */
import fs from 'node:fs/promises';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const [workbookPath, finalsPath] = process.argv.slice(2);
if (!workbookPath || !finalsPath) {
  throw new Error('usage: node sync_final_metrics_workbook.mjs <workbook.xlsx> <finals.json>');
}

const finals = JSON.parse(await fs.readFile(finalsPath, 'utf8'));
const valid = finals.filter((final) => {
  const count = final.final_records ?? (
    Array.isArray(final.seeds) ? final.seeds.length : final.per_seed?.length ?? 0
  );
  return count === 3 && (
    (final.final_test_mean && final.final_test_std) ||
    (final.test_ari_mean !== undefined && final.test_nmi_mean !== undefined && final.test_acc_mean !== undefined) ||
    final.final_test_metrics
  );
});

function summaryFromRecords(final) {
  if (final.final_test_mean && final.final_test_std) {
    return [final.final_test_mean, final.final_test_std];
  }
  if (final.test_ari_mean !== undefined) {
    return [
      { ari: final.test_ari_mean, nmi: final.test_nmi_mean, acc: final.test_acc_mean },
      { ari: final.test_ari_std, nmi: final.test_nmi_std, acc: final.test_acc_std },
    ];
  }
  const records = Object.values(final.final_test_metrics);
  const mean = {};
  const std = {};
  for (const metric of ['ari', 'nmi', 'acc', 'ami', 'f1_macro', 'fmi']) {
    const values = records.map((record) => Number(record[metric])).filter(Number.isFinite);
    if (!values.length) continue;
    mean[metric] = values.reduce((total, value) => total + value, 0) / values.length;
    std[metric] = values.length > 1
      ? Math.sqrt(values.reduce((total, value) => total + (value - mean[metric]) ** 2, 0) / (values.length - 1))
      : 0;
  }
  return [mean, std];
}

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const sheet = workbook.worksheets.getItem('Results');
const range = sheet.getRange('A1:AO151');
const rows = range.values;
let updated = 0;
for (const final of valid) {
  const rowIndex = rows.findIndex((row, index) => index >= 4 && row[0] === final._path);
  if (rowIndex < 0) continue;
  const [mean, std] = summaryFromRecords(final);
  rows[rowIndex][2] = final.status || 'completed_runtime_audit_pending';
  for (const [column, metric] of [[7, 'ari'], [9, 'nmi'], [11, 'acc']]) {
    rows[rowIndex][column] = mean[metric];
    rows[rowIndex][column + 1] = std[metric];
  }
  rows[rowIndex][38] = 'final_summary_verified';
  rows[rowIndex][40] = `final test mean/std | AMI ${mean.ami ?? 'n/a'} +/- ${std.ami ?? 'n/a'} | F1_macro ${mean.f1_macro ?? 'n/a'} +/- ${std.f1_macro ?? 'n/a'} | FMI ${mean.fmi ?? 'n/a'} +/- ${std.fmi ?? 'n/a'}`;
  updated += 1;
}
range.values = rows;
workbook.recalculate();
await (await SpreadsheetFile.exportXlsx(workbook)).save(workbookPath);
console.log(JSON.stringify({ accepted: valid.length, updated }));
