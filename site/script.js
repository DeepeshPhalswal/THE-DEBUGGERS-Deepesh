// Initialize Globe
const myGlobe = Globe()(document.getElementById('globe-container'))
    .globeImageUrl('//unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
    .bumpImageUrl('//unpkg.com/three-globe/example/img/earth-topology.png');

// Fetch real-time data from NASA POWER API
async function fetchWeatherData(lat, lon) {
    const today = new Date();
    today.setDate(today.getDate() - 4);
    const formattedDate = today.toISOString().split('T')[0].replace(/-/g, '');

    const url = `https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,WS10M,RH2M&community=RE&longitude=${lon}&latitude=${lat}&format=JSON&start=${formattedDate}&end=${formattedDate}`;

    try {
        const response = await fetch(url);
        const data = await response.json();

        if (!data.properties || !data.properties.parameter) {
            throw new Error("Invalid response format from NASA POWER API");
        }

        const lastHourKey = Object.keys(data.properties.parameter.T2M || {}).pop();
        return lastHourKey
            ? {
                temperature: `${data.properties.parameter.T2M[lastHourKey] || 'N/A'}°C`,
                windSpeed: `${data.properties.parameter.WS10M[lastHourKey] || 'N/A'} m/s`,
                humidity: `${data.properties.parameter.RH2M[lastHourKey] || 'N/A'}%`
            }
            : { temperature: "N/A", windSpeed: "N/A", humidity: "N/A" };
    } catch (error) {
        console.error("Error fetching NASA POWER data:", error);
        return { temperature: "N/A", windSpeed: "N/A", humidity: "N/A" };
    }
}



// Update Forest Data
async function updateForestData() {
    await Promise.all(
        forests.map(async (forest) => {
            const weatherData = await fetchWeatherData(forest.lat, forest.lng);
            Object.assign(forest, weatherData);
        })
    );

    myGlobe
        .pointsData([...forests]) // Ensure re-rendering
        .pointLat((d) => d.lat)
        .pointLng((d) => d.lng)
        .pointColor(() => 'red')
        .pointRadius(0.5)
        .pointLabel(
            (d) => `${d.name}
            \nTemperature: ${d.temperature}
            \nWind Speed: ${d.windSpeed}
            \nHumidity: ${d.humidity}`
        );

    updateForestDashboard();
}

function updateForestDashboard() {
    const forestInfo = document.getElementById("forest-info");
    if (!forestInfo) return;

    forestInfo.innerHTML = forests
        .map(
            (forest) => `<p><strong>${forest.name}</strong><br>
        Temperature: ${forest.temperature}<br>
        Wind Speed: ${forest.windSpeed}<br>
        Humidity: ${forest.humidity}</p>`
        )
        .join('');
}

// Initialize Updates
document.addEventListener("DOMContentLoaded", updateForestData);


const forests = [
    { name: "Siberian Taiga", lat: 60.0, lng: 105.0 },
    { name: "Boreal Forest", lat: 56.0, lng: -106.0 },
    { name: "Scandinavian Taiga", lat: 64.0, lng: 15.0 },
    { name: "Alaska Boreal", lat: 64.0, lng: -150.0},
    { name: "Black Forest", lat: 48.0, lng: 8.0 },
    { name: "Daintree Rainforest", lat: -16.0, lng: 145.0},
    { name: "Great Smoky Mountains", lat: 35.6, lng: -83.5},
    { name: "Jiuzhaigou Valley", lat: 33.3, lng: 104.2 },
    { name: "Amazon Rainforest", lat: -3.4653, lng: -62.2159 },
    { name: "Congo Rainforest", lat: -1.5, lng: 23.5 },
    { name: "Sundaland Rainforest", lat: -2.5, lng: 102.0},
    { name: "Gir Forest", lat: 21.124, lng: 70.824 },
    { name: "Chamela-Cuixmala Reserve", lat: 19.5, lng: -105.0},
    { name: "Sundarbans", lat: 21.9497, lng: 89.1833},
    { name: "Everglades Mangroves", lat: 25.2866, lng: -80.8987},
    { name: "Bwindi Impenetrable Forest", lat: -1.0575, lng: 29.6048}

];

  