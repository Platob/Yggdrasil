// Minimal ambient declarations for the Office.js globals the Excel
// task-pane uses. Office.js is loaded at runtime from Microsoft's CDN
// (see the taskpane page); we don't bundle @types/office-js, so the
// surface is typed loosely as `any` — enough to keep tsc happy.
declare const Office: any;
declare const Excel: any;

interface Window {
  Office?: any;
  Excel?: any;
}
