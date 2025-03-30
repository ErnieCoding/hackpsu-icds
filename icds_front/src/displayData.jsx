import React, { useState, useEffect } from 'react';
import { Card } from 'react-bootstrap';
import Stack from 'react-bootstrap/Stack';
import Plot from 'react-plotly.js';
import './styles/App.css';

const PointCloudVisualization = ({ fileId }) => {
  const [pointCloudData, setPointCloudData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!fileId) return;
  
    const formData = new FormData();
    formData.append("file", new File([], fileId)); // dummy file name (you don’t have access to real file)
  
    // Instead, make a special endpoint for reading uploaded file by ID:
    fetch(`http://127.0.0.1:8000/api/view-point-cloud/?file_id=${fileId}`, {
      method: 'GET'
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

function DisplayData({ fileId }) {
  useEffect(() => {
    console.log("Visualizing file ID:", fileId);
  }, [fileId]);

  return (
    <Card className="card text-center">
      <Card.Header className='card-header'>
      <h2>Point Cloud Visualization</h2>
      <div className='dropdown'>
        <button class="btn btn-secondary dropdown-toggle" type="button" id="dropdownMenu2" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
          Plot Options
        </button>
        <div className='dropdown-menu' aria-labelledby='dropdownMenu2'>
          <a className='dropdown-item' href='#'>No Object Plot</a>
          <a className='dropdown-item' href='#'>Object Detection Plot</a>
        </div>
      </div>
      </Card.Header>
      <Stack direction="horizontal" gap={3} className="mb-3">
        <div className="p-2">Control Panel</div>
        <div className="p-2 ms-auto">Options</div>
        <div className="vr" />
        <div className="p-2">View Settings</div>
      </Stack>
      <div className="visualization-container" style={{ height: "600px" }}>
        <PointCloudVisualization fileId={fileId} />
      </div>
    </Card>
  );
}

export default DisplayData;