let myChartInstance = null; // To store the Chart.js instance

const int1label = document.getElementById('int1label');
const int2label = document.getElementById('int2label');
const int3label = document.getElementById('int3label');
const int4label = document.getElementById('int4label');
const int5label = document.getElementById('int5label');

function changeHour(){
    var hour = document.getElementById('inputHour').value;
    int1label.innerHTML = "Hour " + (hour-4) + ":";
    int2label.innerHTML = "Hour " + (hour-3) + ":";
    int3label.innerHTML = "Hour " + (hour-2) + ":";
    int4label.innerHTML = "Hour " + (hour-1) + ":";
    int5label.innerHTML = "Hour " + hour + ":";
}

document.getElementById('dataForm').addEventListener('submit', async function (event) {
    event.preventDefault(); // Prevent default form submission

    const form = event.target;
    const formData = new FormData(form);
    const formUrlEncoded = new URLSearchParams(formData).toString();

    const submitButton = form.querySelector('button[type="submit"]');
    const loadingSpinner = form.querySelector('.loading-spinner');
    const formMessage = document.getElementById('formMessage');
    const graphArea = document.getElementById('graphArea');
    const myChartCanvas = document.getElementById('myChart');

    // Show loading state
    submitButton.disabled = true;
    loadingSpinner.style.display = 'inline-block';
    formMessage.textContent = ''; // Clear previous messages
    graphArea.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Generating forecast...</span></div><p class="mt-2">Generating forecast...</p>';
    myChartCanvas.style.display = 'none'; // Hide canvas during loading

    try {
        // Use fetch API to send a POST request
        const response = await fetch('http://127.0.0.1:8080', {
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
        console.log('Server response:', data);

        // Check if graph data is present and valid
        if (data.graph_data && Array.isArray(data.graph_data) && data.graph_data.length > 0) {
            // Extract predicted hours and values from the server response
            const predictedHours = data.graph_data.map(item => item.hour);
            const predictedValues = data.graph_data.map(item => item.value);

            // --- Get Input WR Values and Hours from the Form ---
            const lastKnownHour = parseInt(formData.get('submissionHour'));
            const inputHours = [];
            const inputValues = [];

            // The form provides values for the 5 hours leading up to and including the last known hour
            for (let i = 0; i < 5; i++) {
                const hour = lastKnownHour - 4 + i;
                const value = parseFloat(formData.get(`integer${i + 1}`));
                inputHours.push(hour);
                inputValues.push(value);
            }

            // To plot both lines, we need a continuous set of labels for the x-axis
            // The predicted data already contains the last known hour, so we don't need to append it again.
            const allHours = [...new Set([...inputHours, ...predictedHours])].sort((a, b) => a - b);
            const allLabels = allHours;

            // Destroy existing chart instance if it exists
            if (myChartInstance) {
                myChartInstance.destroy();
            }

            // Re-add the canvas to the graphArea after clearing previous content
            graphArea.innerHTML = ''; // Clear loading message
            graphArea.appendChild(myChartCanvas);
            myChartCanvas.style.display = 'block'; // Show canvas

            // Create the chart with two datasets
            myChartInstance = new Chart(myChartCanvas, {
                type: 'line',
                data: {
                    labels: allLabels,
                    datasets: [
                        {
                            label: 'Input WR Values',
                            data: inputHours.map((h, i) => ({ x: h, y: inputValues[i] })),
                            borderColor: 'rgb(255, 99, 132)', // Distinct color for input
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            fill: false,
                            tension: 0.4,
                            pointBackgroundColor: 'rgb(255, 99, 132)',
                            pointRadius: 5,
                            pointHoverRadius: 7,
                            spanGaps: true // Ensures line continues smoothly
                        },
                        {
                            label: 'Predicted WR Values',
                            data: predictedHours.map((h, i) => ({ x: h, y: predictedValues[i] })),
                            borderColor: 'rgb(54, 162, 235)', // Distinct color for predictions
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            fill: false,
                            tension: 0.4,
                            borderDash: [5, 5], // Dashed line for forecast
                            pointRadius: 3,
                            pointHoverRadius: 5,
                            spanGaps: true // Ensures line continues smoothly
                        }
                    ]
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
                            beginAtZero: true,
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
                            beginAtZero: true
                        }
                    }
                }
            });
        } else {
            graphArea.innerHTML = `<p class="text-warning">No valid forecast data received from the server.</p>`;
        }

    } catch (error) {
        console.error('Error submitting data:', error);
        formMessage.textContent = `Error: ${error.message}. Please check the server connection and ensure the model files are present.`;
        graphArea.innerHTML = `<p class="error-message">Failed to load forecast data. ${error.message}</p>`;
        myChartCanvas.style.display = 'none'; // Ensure canvas is hidden on error
    } finally {
        // Hide loading state
        submitButton.disabled = false;
        loadingSpinner.style.display = 'none';
    }
});

// Function to set default date to today
function setDefaultDate() {
    const today = new Date();
    const year = today.getFullYear();
    const month = (today.getMonth() + 1).toString().padStart(2, '0');
    const day = today.getDate().toString().padStart(2, '0');
    document.getElementById('inputDate').value = `${year}-${month}-${day}`;
}
setDefaultDate(); // Call on page load