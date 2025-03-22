<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
  .chart-pair {
    display: flex;
    justify-content: space-around;
    margin-bottom: 2em;
    flex-wrap: wrap;
  }
  .chart-container {
    flex: 0 0 auto;
    width: 500px;
    margin: 0.5em;
  }
</style>

The performance of our approach is outlined below.

## Proximal Policy Optimization with Stable-Baselines3
The approach utilizing Stable-Baselines3 demonstrated the strongest performance in our comparison. Over the course of 1,000,000 timesteps, it reached a peak survival time of 33 seconds, with an average performance of up to 16 seconds. Notably, even the minimum survival time improved, rising to nearly 3 seconds by the end of training.


<div class="chart-pair">
  <div class="chart-container">
    <canvas id="ppo_sb3_reward" width="500" height="400"></canvas>
  </div>
  <div class="chart-container">
    <canvas id="ppo_sb3_episode" width="500" height="400"></canvas>
  </div>
</div>

## Proximal Policy Optimization without Stable-Baselines3
Without the use of Stable-Baselines3, the PPO implementation reached a maximum survival time of 25 seconds. The average value was around 9 seconds. Compared to the SB3-based approach, this version showed weaker learning progress with lower overall stability and performance.

<div class="chart-pair">
  <div class="chart-container">
    <canvas id="ppo_no_sb3_reward" width="500" height="400"></canvas>
  </div>
  <div class="chart-container">
    <canvas id="ppo_no_sb3_episode" width="500" height="400"></canvas>
  </div>
</div>

## Deep Q Network
The DQN approach briefly reached a maximum survival time of 25 seconds. However, the average remained around 8 seconds. After an early peak in performance, learning collapsed significantly. Both reward and episode length decreased and then stagnated at a lower level for the remainder of training.

<div class="chart-pair">
  <div class="chart-container">
    <canvas id="dqn_reward" width="500" height="400"></canvas>
  </div>
  <div class="chart-container">
    <canvas id="dqn_episode" width="500" height="400"></canvas>
  </div>
</div>

<script>

  function parseCSV(csv) {
    const lines = csv.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    const data = [];
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',');
      const obj = {};
      headers.forEach((header, j) => {

        obj[header] = parseFloat(values[j]);
      });
      data.push(obj);
    }
    return data;
  }


  function loadCSV(url, callback) {
    fetch(url)
      .then(response => response.text())
      .then(text => {
        const data = parseCSV(text);
        callback(data);
      })
      .catch(error => console.error('Fehler beim Laden von CSV:', error));
  }


  const commonOptions = {
    plugins: {
      legend: {
        labels: { color: '#fff' }
      },
      title: {
        display: true,
        color: '#fff'
      }
    },
    scales: {
      x: {
        ticks: { color: '#fff' },
        title: { display: true, text: 'Minibatch', color: '#fff' },
        grid: { color: 'rgba(255, 255, 255, 0.2)' }
      },
      y: {
        ticks: { color: '#fff' },
        title: { display: true, text: '', color: '#fff' },
        grid: { color: 'rgba(255, 255, 255, 0.2)' }
      }
    }
  };


  function createMultiDatasetChart(ctx, titleText, labels, minData, avgData, maxData, yAxisTitle) {

    const options = JSON.parse(JSON.stringify(commonOptions));
    options.plugins.title.text = titleText;
    options.scales.y.title.text = yAxisTitle;
    return new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Min',
            data: minData,
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'transparent',
            fill: false,
            tension: 0.4,
            pointRadius: 0
          },
          {
            label: 'Avg',
            data: avgData,
            borderColor: 'rgba(54, 162, 235, 1)',
            backgroundColor: 'transparent',
            fill: false,
            tension: 0.4,
            pointRadius: 0
          },
          {
            label: 'Max',
            data: maxData,
            borderColor: 'rgba(75, 192, 75, 1)',
            backgroundColor: 'transparent',
            fill: false,
            tension: 0.4,
            pointRadius: 0
          }
        ]
      },
      options: options
    });
  }


  function createChartsFromData(data, rewardCanvasId, episodeCanvasId) {
    const labels = data.map(row => row['Minibatch']);

    const minReward = data.map(row => row['Min Reward']);
    const avgReward = data.map(row => row['Avg Reward']);
    const maxReward = data.map(row => row['Max Reward']);

    const minEpisode = data.map(row => row['Min Episode Length']);
    const avgEpisode = data.map(row => row['Avg Episode Length']);
    const maxEpisode = data.map(row => row['Max Episode Length']);

    const ctxReward = document.getElementById(rewardCanvasId).getContext('2d');
    createMultiDatasetChart(ctxReward, 'Reward per Minibatch', labels, minReward, avgReward, maxReward, 'Reward');

    const ctxEpisode = document.getElementById(episodeCanvasId).getContext('2d');
    createMultiDatasetChart(ctxEpisode, 'Episode Length per Minibatch', labels, minEpisode, avgEpisode, maxEpisode, 'Frames');
  }

  loadCSV('ppo_sb3.csv', function(data) {
    createChartsFromData(data, 'ppo_sb3_reward', 'ppo_sb3_episode');
  });

  loadCSV('ppo_no_sb3.csv', function(data) {
    createChartsFromData(data, 'ppo_no_sb3_reward', 'ppo_no_sb3_episode');
  });

  loadCSV('dqn.csv', function(data) {
    createChartsFromData(data, 'dqn_reward', 'dqn_episode');
  });
</script>
