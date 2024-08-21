// find the output element
const output = document.getElementById("output");

// initialize codemirror and pass configuration to support Python and the dracula theme
const editor = CodeMirror.fromTextArea(
  document.getElementById("code"), {
    mode: {
      name: "python",
      version: 3,
      singleLineStringErrors: false,
    },
    theme: "dracula",
    lineNumbers: true,
    indentUnit: 4,
    matchBrackets: true,
  }
);
// set the initial value of the editor
editor.setValue("print('Hello world')");
output.value = "Initializing...\n";

// add pyodide returned value to the output
function addToOutput(stdout) {
  output.value += ">>> " + "\n" + stdout + "\n";
}

// clean the output section
function clearHistory() {
  output.value = "";
}

// init pyodide and show sys.version when it's loaded successfully
async function main() {
  let pyodide = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.20.0/full/",
  });
  output.value = pyodide.runPython(`
    import sys
    sys.version
  `);
  output.value += "\n" + "Python Ready !" + "\n";
  return pyodide;
}

// run the main function
let pyodideReadyPromise = main();

// pass the editor value to the pyodide.runPython function and show the result in the output section
async function evaluatePython() {
  let pyodide = await pyodideReadyPromise;
  try {
    pyodide.runPython(`
      import io
      sys.stdout = io.StringIO()
    `);
    let result = pyodide.runPython(editor.getValue());
    let stdout = pyodide.runPython("sys.stdout.getvalue()");
    addToOutput(stdout);
  } catch (err) {
    addToOutput(err);
  }
}