import type { SchemaField } from "@/lib/types";

interface Props {
  schema: SchemaField[];
  rowCount: number;
  colCount: number;
  format: string;
  path: string;
}

export default function TabularPreview({ schema, rowCount, colCount, format, path }: Props) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-4 text-xs text-zinc-500">
        <span className="bg-zinc-800 px-2 py-0.5 rounded font-mono">{format.toUpperCase()}</span>
        <span>{rowCount.toLocaleString()} rows</span>
        <span>{colCount} columns</span>
        <span className="font-mono text-zinc-600 truncate">{path}</span>
      </div>
      <div className="overflow-auto max-h-80">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="border-b border-zinc-800">
              <th className="text-left px-3 py-2 text-zinc-500 font-medium">Column</th>
              <th className="text-left px-3 py-2 text-zinc-500 font-medium">Type</th>
            </tr>
          </thead>
          <tbody>
            {schema.map((f) => (
              <tr key={f.name} className="border-b border-zinc-900 hover:bg-zinc-800/40">
                <td className="px-3 py-1.5 font-mono text-zinc-200">{f.name}</td>
                <td className="px-3 py-1.5 text-zinc-500">{f.type}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
