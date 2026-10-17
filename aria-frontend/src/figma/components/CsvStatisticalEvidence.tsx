interface Props { result: any; showTable?: boolean }

export default function CsvStatisticalEvidence({ result, showTable = false }: Props) {
  const evidence = result.statistical_evidence;
  if (!evidence) return null;
  const rows: any[] = result.data || [];
  const columns: string[] = result.columns || [];
  return <div className="space-y-2 mt-3 text-xs">
    <p className="font-medium">Exploratory statistical evidence</p>
    <p>{evidence.method}</p>
    <p className="text-muted-foreground">Examined {evidence.sample_count?.toLocaleString()} matching records{evidence.sampled ? " (bounded sample; not randomly selected)" : ""}.</p>
    {evidence.status === "insufficient_data" && <p>Not enough usable observations for this check.</p>}
    {evidence.status === "no_candidates" && <p>No candidates met this method’s criteria. Other patterns may still exist.</p>}
    {showTable && rows.length > 0 && <div className="max-h-80 overflow-auto"><table className="w-full text-left"><thead><tr>{columns.map(col => <th className="p-2 border-b" key={col}>{col.replace(/_/g, " ")}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index}>{columns.map(col => <td key={col} className="p-2 border-b">{row[col] == null ? "Unavailable" : typeof row[col] === "number" ? row[col].toLocaleString(undefined, { maximumSignificantDigits: 10 }) : String(row[col])}</td>)}</tr>)}</tbody></table></div>}
    {(evidence.limitations || []).map((limit: string, index: number) => <p key={index} className="text-muted-foreground">{limit}</p>)}
    <details><summary className="cursor-pointer">Calculation inputs</summary><pre className="mt-2 overflow-auto whitespace-pre-wrap">{JSON.stringify({ tool: evidence.tool, arguments: evidence.arguments, source_query: evidence.sample_sql, parameters: evidence.sql_parameters }, null, 2)}</pre></details>
  </div>;
}
