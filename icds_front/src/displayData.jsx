import React, { useState, useEffect } from 'react';
import Stack from 'react-bootstrap/Stack';
import Plot from 'react-plotly.js';

const PointCloudVisualization = ({ fileId }) => {
  const [pointCloudData, setPointCloudData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!fileId) return;

    setLoading(true);
    setError(null);

    fetch('http://127.0.0.1:8000/api/process-point-cloud/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ file_id: fileId })
    })
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error! Status: ${res.status}`);
        return res.json();
      })
      .then(data => {
        if (data.status === 'success') {
          setPointCloudData(data);
        } else {
          throw new Error(data.message || 'Failed to load visualization data');
        }
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, [fileId]);

  if (loading) return <div>Loading visualization...</div>;
  if (error) return <div className="text-danger">{error}</div>;
  if (!pointCloudData) return <div>Select a point cloud file to visualize</div>;

  const x = pointCloudData.points.map(p => p[0]);
  const y = pointCloudData.points.map(p => p[1]);
  const z = pointCloudData.points.map(p => p[2]);

  let markerConfig = { size: 2 };
  if (pointCloudData.colors) {
    markerConfig.color = pointCloudData.colors.map(c => `rgb(${c.map(n => Math.floor(n * 255)).join(',')})`);
  } else {
    markerConfig.color = 'rgb(0,100,255)';
  }

  const data = [{
    type: 'scatter3d',
    mode: 'markers',
    x, y, z,
    marker: markerConfig,
    hoverinfo: 'none'
  }];

  const layout = {
    autosize: true,
    height: 600,
    scene: {
      aspectratio: { x: 1, y: 1, z: 0.5 },
      camera: { eye: { x: 1.5, y: 1.5, z: 1 } },
      xaxis: { showticklabels: false },
      yaxis: { showticklabels: false },
      zaxis: { showticklabels: false }
    },
    margin: { l: 0, r: 0, b: 0, t: 0 }
  };

  return (
    <div className="visualization-container">
      <Plot data={data} layout={layout} style={{ width: '100%', height: '100%' }} useResizeHandler config={{ responsive: true }} />
    </div>
  );
};

function DisplayData() {
  // 🚀 Just use the file name!
  const [selectedFile, setSelectedFile] = useState("main.pcd");

  return (
    <div className="container">
      <h2>Point Cloud Visualization</h2>
      <Stack direction="horizontal" gap={3} className="mb-3">
        <div className="p-2">Control Panel</div>
        <div className="p-2 ms-auto">Options</div>
        <div className="vr" />
        <div className="p-2">View Settings</div>
      </Stack>
      <div className="visualization-container" style={{ height: "600px" }}>
        <PointCloudVisualization fileId={selectedFile} />
      </div>
    </div>
  );
}

export default DisplayData;