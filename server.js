const express = require("express");
const { spawn } = require("child_process");
const cors = require("cors");

const app = express();
app.use(express.json());
app.use(cors());

app.post("/run-python", (req, res) => {
    const userInput = req.body.input || "";  // Get user input from request body
    const pythonProcess = spawn("python", ["script.py", userInput]);

    let outputData = "";

    pythonProcess.stdout.on("data", (data) => {
        outputData += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
        console.error(`Python Error: ${data}`);
    });

    pythonProcess.on("close", () => {
        res.json({ output: outputData });  // Send Python output to frontend
    });
});

app.listen(3000, () => console.log("Server running on http://localhost:3000"));
