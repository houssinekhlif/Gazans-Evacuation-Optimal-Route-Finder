// =============== LOAD LOCATIONS ===============
fetch("/get_nodes")
    .then(res => res.json())
    .then(nodes => {
        // Sort alphabetically
        nodes.sort((a, b) => a.localeCompare(b));

        let select = document.getElementById("locationSelect");
        nodes.forEach(n => {
            let opt = document.createElement("option");
            opt.value = n;
            opt.textContent = n;
            select.appendChild(opt);
        });
    });


// =============== COMPUTE PATH ===============
function computePath() {
    let source = document.getElementById("locationSelect").value;
    let scenario = document.getElementById("scenarioSelect").value;

    fetch("/compute_best_paths", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({source: source, scenario: scenario})
    })
    .then(res => res.json())
    .then(data => {
        let div = document.getElementById("results");
        div.innerHTML = "";

        data.forEach((item, idx) => {
            div.innerHTML += `
                <div class="card">
                    <h3>#${idx+1} → ${item.destination}</h3>
                    <p><b>Path:</b> ${item.path.join(" → ")}</p>
                    <p><b>Algorithm:</b> ${item.algorithm}</p>
                    <p><b>Cost:</b> ${item.cost.toFixed(3)}</p>
                    

                    <p><b>Total Distance:</b> 
                        ${item.total_distance !== null ? item.total_distance.toFixed(2) + " km" : "N/A"}
                    </p>

                    <p><b>Risk Rating:</b> 
                        ${item.avg_risk !== null ? item.avg_risk.toFixed(3) : "N/A"}
                    </p>

                    <p><b>Humanitarian Aid Rating:</b>
                        ${item.avg_aid !== null ? item.avg_aid.toFixed(3) : "N/A"}
                    </p>
                </div>
                <hr>
            `;

        })
    });
}
