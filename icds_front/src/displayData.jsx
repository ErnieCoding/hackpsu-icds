import React, { useState, useEffect } from 'react';
import { Card, Dropdown, DropdownButton, Spinner } from 'react-bootstrap';
import Plot from 'react-plotly.js';
import './styles/App.css';

const DisplayData = ({ fileId }) => {
  const [plotType, setPlotType] = useState("raw");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!fileId) return;
    setLoading(true);
    setError(null);

    const fetchData = async () => {
      try {
        const endpoint = plotType === "detection"
          ? `http://127.0.0.1:8000/api/run-detection/?file_id=${fileId}`
          : `http://127.0.0.1:8000/api/view-point-cloud/?file_id=${fileId}`;
    
        const response = await fetch(endpoint);
        const result = await response.json();
    
        if (result.status !== "success") {
          throw new Error(result.message || "Failed to fetch data");
        }
    
        setData(result);
    
        // ✅ Dispatch to App.js (match App.js listener exactly)
        if (plotType === "detection" && result.boxes && result.class_names && result.scores) {
          window.dispatchEvent(new CustomEvent("detectionResults", {
            detail: {
              boxes: result.boxes,
              class_names: result.class_names,
              scores: result.scores
            }
          }));
        }
      } catch (err) {
        console.error(err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [fileId, plotType]);

  const getPlotData = () => {
    if (!data?.points) return [];

    const x = data.points.map(p => p[0]);
    const y = data.points.map(p => p[1]);
    const z = data.points.map(p => p[2]);

    let marker = { size: 2 };

    if (data.colors) {
      marker.color = data.colors.map(c => `rgb(${c.map(v => Math.floor(v * 255)).join(',')})`);
    } else {
      marker.color = 'rgb(0,100,255)';
    }

    const trace = {
      type: 'scatter3d',
      mode: 'markers',
      x, y, z,
      marker,
      hoverinfo: 'none'
    };

    const traces = [trace];

    // If we are in detection mode, render boxes as well
    if (plotType === "detection" && data.boxes) {
      for (let i = 0; i < data.boxes.length; i++) {
        const box = data.boxes[i];
        const label = data.class_names?.[data.labels[i] - 1] || `Object ${i + 1}`;
        const score = data.scores[i]?.toFixed(2);

        const traceBox = {
          type: 'scatter3d',
          mode: 'markers+text',
          x: [box[0]],
          y: [box[1]],
          z: [box[2]],
          marker: { size: 4, color: 'red' },
          text: [`${label} (${score})`],
          textposition: 'top center',
          hoverinfo: 'text',
        };

        traces.push(traceBox);
      }
    }

    return traces;
  };

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
    <Card className="mt-4">
      <Card.Header className="d-flex justify-content-between align-items-center">
        <h5 className="mb-0">Visualization</h5>
        <DropdownButton
          variant="secondary"
          title="Plot Options"
          onSelect={(e) => setPlotType(e)}
        >
          <Dropdown.Item eventKey="raw">No Object Plot</Dropdown.Item>
          <Dropdown.Item eventKey="detection">Object Detection Plot</Dropdown.Item>
        </DropdownButton>
      </Card.Header>
      <Card.Body style={{ height: 600 }}>
        {loading ? (
          <div className="d-flex justify-content-center align-items-center h-100">
            <Spinner animation="border" />
          </div>
        ) : error ? (
          <div className="text-danger text-center">{error}</div>
        ) : (
          <Plot
            data={getPlotData()}
            layout={layout}
            style={{ width: '100%', height: '100%' }}
            useResizeHandler
            config={{ responsive: true }}
          />
        )}
      </Card.Body>
    </Card>
  );
};

export default DisplayData;
