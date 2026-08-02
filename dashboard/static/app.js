async function loadDashboard() {

    try {

        const response = await fetch("/api/dashboard");

        const data = await response.json();

        document.getElementById("accuracy").innerHTML =
            (data.accuracy * 100).toFixed(2) + "%";

        document.getElementById("precision").innerHTML =
            (data.precision * 100).toFixed(2) + "%";

        document.getElementById("recall").innerHTML =
            (data.recall * 100).toFixed(2) + "%";

        document.getElementById("f1").innerHTML =
            (data.f1 * 100).toFixed(2) + "%";

        document.getElementById("roc").innerHTML =
            (data.roc_auc * 100).toFixed(2) + "%";

        document.getElementById("folds").innerHTML = data.folds;

        document.getElementById("confidence").innerHTML =
            (data.avg_confidence * 100).toFixed(2) + "%";

        document.getElementById("calibration").innerHTML = data.calibration;

        document.getElementById("total").innerHTML = data.total_predictions;

        document.getElementById("t15").innerHTML = data.thresholds["15m"] ?? "--";
        document.getElementById("t30").innerHTML = data.thresholds["30m"] ?? "--";
        document.getElementById("t60").innerHTML = data.thresholds["60m"] ?? "--";
        document.getElementById("t120").innerHTML = data.thresholds["120m"] ?? "--";
        document.getElementById("t240").innerHTML = data.thresholds["240m"] ?? "--";

        const tbody = document.querySelector("#predictionTable tbody");

        tbody.innerHTML = "";

        data.recent_predictions.forEach(row => {

            const tr = document.createElement("tr");

            tr.innerHTML = `
                <td>${row.timestamp ?? ""}</td>
                <td>${row.asset ?? ""}</td>
                <td>${row.target_price ?? ""}</td>
                <td>${row.minutes ?? ""}</td>
                <td>${(row.probability * 100).toFixed(2)}%</td>
                <td>${row.prediction ? "YES ✅" : "NO ❌"}</td>
            `;

            tbody.appendChild(tr);

        });

    }
    catch (err) {
        console.error(err);
    }

}

loadDashboard();

setInterval(loadDashboard, 5000);


// ==========================================
// PREDICTION FORM
// ==========================================

document
    .getElementById("predictForm")
    .addEventListener("submit", async function (e) {

        e.preventDefault();

        const resultBox = document.getElementById("predictionResult");

        resultBox.innerText = "Running prediction...";

        try {

            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    asset: document.getElementById("asset").value,
                    target: document.getElementById("target").value,
                    minutes: document.getElementById("minutes").value,
                }),
            });

            const result = await response.json();

            if (result.error) {
                resultBox.innerText = "Error: " + result.error;
                return;
            }

            resultBox.innerText = JSON.stringify(result, null, 4);

        } catch (err) {

            resultBox.innerText = "Request failed: " + err;

        }

    });