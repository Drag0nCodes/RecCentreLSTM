let myChartInstance = null; // To store the Chart.js instance

const int1label = document.getElementById('int1label');
const int2label = document.getElementById('int2label');
const int3label = document.getElementById('int3label');
const int4label = document.getElementById('int4label');
const int5label = document.getElementById('int5label');

// --- Helper function to display messages to the user ---
const formMessage = document.getElementById('formMessage');

function changeHour() {
    var hour = document.getElementById('inputHour').value;
    if (hour) {
        hour = parseInt(hour);
        int1label.innerHTML = "Hour " + (hour - 4) + ":";
        int2label.innerHTML = "Hour " + (hour - 3) + ":";
        int3label.innerHTML = "Hour " + (hour - 2) + ":";
        int4label.innerHTML = "Hour " + (hour - 1) + ":";
        int5label.innerHTML = "Hour " + hour + ":";
    }
}

// Event Listener for the Tweet Fetching Button 
document.getElementById('fetchTweetsBtn').addEventListener('click', async function () {
    const fetchButton = this;
    const fetchSpinner = fetchButton.querySelector('.spinner-border');

    // Show loading state
    fetchButton.disabled = true;
    fetchSpinner.style.display = 'inline-block';
    formMessage.textContent = ''; // Clear previous messages

    try {
        // Fetch at /gettweets endpoint
        //const response = await fetch('http://127.0.0.1:8080/gettweets', { // Local testing
        const response = await fetch('https://rec-centre-lstm.camdvr.org/gettweets', { // Server deployment
            method: 'GET'
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();

        // Populate the form fields with the fetched data
        document.getElementById('inputDate').value = data.date;
        document.getElementById('inputHour').value = data.hour;

        for (let i = 0; i < 5; i++) {
            document.getElementById(`integer${i + 1}`).value = data.values[i];
        }

        // Update the hour labels after populating the form
        changeHour();

    } catch (error) {
        console.error('Error fetching tweet data:', error);
        formMessage.textContent = `Failed to fetch data: ${error.message}`;
    } finally {
        // Hide loading state
        fetchButton.disabled = false;
        fetchSpinner.style.display = 'none';
    }
});


// Event Listener for the Prediction Form Submission 
document.getElementById('dataForm').addEventListener('submit', async function (event) {
    event.preventDefault(); // Prevent default form submission

    const form = event.target;
    const formData = new FormData(form);
    const formUrlEncoded = new URLSearchParams(formData).toString();

    const submitButton = form.querySelector('button[type="submit"]');
    const loadingSpinner = form.querySelector('.loading-spinner');
    const graphArea = document.getElementById('graphArea');
    const myChartCanvas = document.getElementById('myChart');

    // Show loading state
    submitButton.disabled = true;
    loadingSpinner.style.display = 'inline-block';
    formMessage.textContent = ''; // Clear previous messages
    graphArea.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Generating forecast...</span></div><p class="mt-2">Generating forecast...</p>';
    myChartCanvas.style.display = 'none'; // Hide canvas during loading

    try {
        //fetch at /predict endpoint
        //const response = await fetch('http://127.0.0.1:8080/predict', { // Local testing
        const response = await fetch('https://rec-centre-lstm.camdvr.org/predict', { // Server deployment
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: formUrlEncoded
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! Status: ${response.status}. Response: ${errorText}`);
        }

        const data = await response.json();

        if (data.graph_data && Array.isArray(data.graph_data) && data.graph_data.length > 0) {
            const predictedHours = data.graph_data.map(item => item.hour);
            const predictedValues = data.graph_data.map(item => item.value);
            const dotwAverageData = data.dotw_average_data;
            const date = new Date(data.submitted_date);

            const dayNames = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
            const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

            const lastKnownHour = parseInt(formData.get('submissionHour'));
            const inputHours = [];
            const inputValues = [];

            for (let i = 0; i < 5; i++) {
                const hour = lastKnownHour - 4 + i;
                const value = parseFloat(formData.get(`integer${i + 1}`));
                inputHours.push(hour);
                inputValues.push(value);
            }

            const allHours = [...new Set([...inputHours, ...predictedHours])].sort((a, b) => a - b);
            const allLabels = allHours;

            if (myChartInstance) {
                myChartInstance.destroy();
            }

            graphArea.innerHTML = '';
            graphArea.appendChild(myChartCanvas);
            myChartCanvas.style.display = 'block';

            // Prepare the dotw data for charting
            const dotwHours = allHours.map(hour => {
                return {
                    x: hour,
                    y: dotwAverageData[hour]
                };
            });

            myChartInstance = new Chart(myChartCanvas, {
                type: 'line',
                data: {
                    labels: allLabels,
                    datasets: [{
                        label: 'Input Values',
                        data: inputHours.map((h, i) => ({ x: h, y: inputValues[i] })),
                        borderColor: 'rgb(79, 38, 131)',
                        backgroundColor: 'rgba(79, 38, 131, 0.2)',
                        fill: false,
                        tension: 0.4,
                        pointBackgroundColor: 'rgb(79, 38, 131)',
                        pointRadius: 5,
                        pointHoverRadius: 7,
                        spanGaps: true
                    }, {
                        label: 'Predicted Values',
                        data: predictedHours.map((h, i) => ({ x: h, y: predictedValues[i] })),
                        borderColor: 'rgb(154, 100, 246)',
                        backgroundColor: 'rgba(154, 100, 246, 0.2)',
                        fill: false,
                        tension: 0.4,
                        borderDash: [5, 5],
                        pointRadius: 5,
                        pointHoverRadius: 7,
                        spanGaps: true
                    }, {
                        label: `Average for ${dayNames[date.getDay()]} in ${monthNames[date.getMonth()]}`,
                        data: dotwHours,
                        borderColor: 'rgb(129, 130, 132)',
                        backgroundColor: 'rgba(129, 130, 132, 0.2)',
                        fill: false,
                        tension: 0.4,
                        borderDash: [2, 2],
                        pointRadius: 0, // No points for this line
                        spanGaps: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: `WR Value Forecast for ${data.submitted_date}`
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Hour of Day'
                            },
                            min: allLabels[0],
                            max: allLabels[allLabels.length - 1],
                            type: 'linear',
                            ticks: {
                                stepSize: 1,
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'WR Value'
                            },
                            beginAtZero: true,
                            min: 0
                        }
                    }
                }
            });

            // Increase prediction count by one
            const countheader = document.getElementById('forecastCount');
            countheader.innerHTML = parseInt(countheader.innerHTML) + 1

        } else {
            graphArea.innerHTML = `<p class="text-warning">No valid forecast data received from the server.</p>`;
        }

    } catch (error) {
        console.error('Error submitting data:', error);
        formMessage.textContent = `Error: ${error.message}. Please check the server connection and refresh your browser.`;
        graphArea.innerHTML = `<p class="error-message">Failed to load forecast data. ${error.message}</p>`;
        myChartCanvas.style.display = 'none';
    } finally {
        submitButton.disabled = false;
        loadingSpinner.style.display = 'none';
    }
});

function setDefaultDate() {
    const today = new Date();
    const year = today.getFullYear();
    const month = (today.getMonth() + 1).toString().padStart(2, '0');
    const day = today.getDate().toString().padStart(2, '0');
    document.getElementById('inputDate').value = `${year}-${month}-${day}`;
}


document.addEventListener('DOMContentLoaded', async function (event) {
    setDefaultDate(); // Call on page load
    changeHour(); // Call on page load to set initial labels

    // Get the prediction counter number and set the header value
    const countheader = document.getElementById('forecastCount');
    try {
        //const response = await fetch('http://127.0.0.1:8080/getforecastcount', { // Local testing
        const response = await fetch('https://rec-centre-lstm.camdvr.org/getforecastcount', { // Server deployment
            method: 'GET'
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();

        // Populate fields with fetched data
        countheader.innerHTML = data.count;

    } catch (error) {
        console.error('Error fetching tweet data:', error);
        countheader.innerHTML = `N/A`;
    }
});
