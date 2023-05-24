function fetchlogs() {
    fetch('/logs')
        .then(response => {
            return response.json();

        })
        .then(data => {
            for (let i = previousCount; i < data.length; i++) {
                let current = color[count % rowCount];
                count = count + 1;

                details = JSON.parse(data[i]);

                let row = document.createElement("tr");
                row.className = current;
                let head = document.createElement("th");
                head.scope = "row";
                head.innerHTML = details['user'];
                let col1 = document.createElement("td");
                col1.innerHTML = details['movement'];
                let col2 = document.createElement("td");
                col2.innerHTML = details['success'];

                row.appendChild(head);
                row.appendChild(col1);
                row.appendChild(col2);

                document.querySelector('.table > tbody').append(row);
            }
            previousCount = data.length;

        })
}

function moves(direction, num, user) {
    let steps = document.getElementById("steps").value;
    // console.log("number of steps is " + steps);
    fetch('/' + direction + '/' + user + '/' + steps)
        .then(response => {
            return response.json();

        })
        .then(result => {
            // console.log(result);
            fetchlogs();

        })
}

let count = 0;
let color = ['table-primary', 'table-secondary', 'table-success', 'table-danger', 'table-warning', 'table-info', 'table-light', 'table-dark'];
let rowCount = color.length;

let previousCount = 0;

function addRow() {
    fetch('/logs')
        .then((response) => response.json())
        .then((data) => {
        });
}


function refresh_page() {
    window.location.reload();
};

// setInterval(fetchlogs, 5000);