import React, { useState, useEffect } from 'react';
import Stack from 'react-bootstrap/Stack';
import Plot from 'react-plotly.js';

const PointCloudVisualization = ({ fileId }) => {
  const [pointCloudData, setPointCloudData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      if (!fileId) return;
      
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch('http://127.0.0.1:8000/api/process-point-cloud/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCsrfToken()
          },
          body: JSON.stringify({
            is_pcd: true,
            file_path: fileId,
          })
        });
        
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'success') {
          setPointCloudData(data);
        } else {
          setError(data.message || 'Failed to load visualization data');
        }
      } catch (err) {
        setError(err.message || 'Error fetching point cloud data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [fileId]);

  // Helper function to get CSRF token
  const getCsrfToken = () => {
    return document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
  };

  if (loading) {
    return <div className="flex justify-center items-center h-64">Loading visualization...</div>;
  }

  if (error) {
    return <div className="text-red-500 p-4 border border-red-300 rounded bg-red-50">{error}</div>;
  }

  if (!pointCloudData) {
    return <div className="p-4">Select a point cloud file to visualize</div>;
  }

  // Prepare data for Plotly
  const x = pointCloudData.points.map(point => point[0]);
  const y = pointCloudData.points.map(point => point[1]);
  const z = pointCloudData.points.map(point => point[2]);

  // Configure marker properties
  let markerConfig = { size: 2 };
  if (pointCloudData.colors) {
    // Convert RGB values to Plotly format if available
    const colors = pointCloudData.colors.map(color => 
      `rgb(${Math.floor(color[0]*255)},${Math.floor(color[1]*255)},${Math.floor(color[2]*255)})`
    );
    markerConfig.color = colors;
  } else {
    markerConfig.color = 'rgb(0,100,255)';
  }

  const data = [{
    type: 'scatter3d',
    mode: 'markers',
    x: x,
    y: y,
    z: z,
    marker: markerConfig,
    hoverinfo: 'none',
  }];

  const layout = {
    autosize: true,
    height: 600,
    scene: {
      aspectratio: { x: 1, y: 1, z: 0.5 },
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1 }
      },
      xaxis: { showticklabels: false },
      yaxis: { showticklabels: false },
      zaxis: { showticklabels: false }
    },
    margin: { l: 0, r: 0, b: 0, t: 0, pad: 0 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
  };
  return (
    <div className="w-full h-full border rounded-lg overflow-hidden bg-white shadow-lg">
      <Plot
        data={data}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
      />
    </div>
  );
};

function DisplayData(){
  const [selectedFile, setSelectedFile] = useState("path/to/your/example.pcd");
  
  return (
    <div className="container">
      <h2>Point Cloud Visualization</h2>
      
      {/* File selection could go here */}
      {/* <input type="file" onChange={handleFileChange} /> */}
      
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