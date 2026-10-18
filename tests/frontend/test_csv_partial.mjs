import {readFileSync} from 'node:fs';
import assert from 'node:assert/strict';
// Execute the actual client normalizer with only HTTP/storage substituted.
const source = readFileSync('aria-frontend/src/services/api.js', 'utf8');
const start = source.indexOf('export const analyzeQuestion = ');
const end = source.indexOf('\n};', start) + 3;
const fn = source.slice(start, end).replace('export const analyzeQuestion = ', '');
const deriveStart = source.indexOf('const deriveGoalType = ');
const deriveEnd = source.indexOf('\n};', deriveStart) + 3;
const derive = source.slice(deriveStart, deriveEnd);
let payload;
const analyze = new Function('request', `${derive} return (${fn.replace(/;$/, '')})`)(async () => payload);
payload = {status:'partial', analysis_source:'csv_llm', execution:{success:true}, narrative_status:'unavailable',
  answer:'Explanation unavailable.', analysis_results:[{data:[{profit:150}]}], data:[{profit:150}], columns:['profit']};
const result = await analyze('profit');
assert.equal(result.data.resultStatus, 'partial');
assert.equal(result.data.narrative_status, 'unavailable');
assert.equal(result.data.analysis_results[0].data[0].profit, 150);
payload = {...payload, execution:{success:false}};
await assert.rejects(() => analyze('profit'));
payload = {...payload, status:'error'};
await assert.rejects(() => analyze('profit'));
console.log('CSV partial-result client contract: passed');
